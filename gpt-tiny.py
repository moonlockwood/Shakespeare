import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import trange

# -----------------------------------------------------
#  hyperparameters

batch_size = 128 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 1000 # number of training steps
eval_interval = 300 # how often to evaluate the model and checkpoint
learning_rate = 8e-4 # 3e-4 is a good value for AdamW
eval_iters = 200 # how many iterations to use for evaluation
n_embed = 480 # the dimension of the embeddings, transformers from 100-500 are common
n_head = 8 # how many attention heads to use
n_layer = 6 # how many transformer blocks to use
weight_decay = 2e-3 # weight decay strength 1e-3 moderately powerful

# -----------------------------------------------------
torch.manual_seed(42)
# -----------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using {device}")

# -----------------------------------------------------
# source file path - set this to valid text file path to train on the rest should work
data_file_name = 'shakespeare.txt'                     # 1mb file is ~15 minutes to train
model_name = data_file_name.split('/')[-1].split('.')[0]    # -> 'shakespeare' for model name

# -----------------------------------------------------
with open(data_file_name, 'r', encoding='utf-8') as f:
    text = f.read()

# get all unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]              # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])     # decoder: take a list of integers, output a string

# generate train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# create simplest data loader
def get_batch(split):
    data = train_data if split == 'train' else val_data        # generate batch of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

print(f'* data loaded\ntrain characters: {len(train_data)} - val characters: {len(val_data)}')

# ---------------------------------------------------------------------------------------
#  super simple bigram model 

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.positiion_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embed, vocab_size) # linear layer to produce logits

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.positiion_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = token_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the index to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            # stream the new character
            print(decode([idx_next.item()]), end='', flush=True)
        return idx

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Block(nn.Module):
    '''Transformer block: communication followed by computation'''
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
      
class MultiHeadAttention(nn.Module):
    '''multiple head self attention in parallel'''
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class Head(nn.Module):
    '''one head self attention'''
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        # compute attention scores
        wei = q @ k.transpose(-2, -1) * (C**-0.5) # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # perform weighted aggregation
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    
class FeedForward(nn.Module):
    '''simple layer followed by a non-linearity'''
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
            nn.Linear(n_embed, n_embed)
        )
    def forward(self, x):
        return self.net(x)
    
# ----------------------------------------------------------------------------------------------------------
#   main
    
# create the model
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create the optimizer and the learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,  total_steps=max_iters, pct_start=0.01)

# ---------- training loop -------------
log=[]
bar = trange(1, max_iters+1, desc=f"Training '{model_name}'", leave=True)
for step in bar:
    if step % eval_interval == 0:       # evaluate the loss on train and val sets at interval
        losses = estimate_loss()
        print(f'iter {step}, train loss {losses["train"]:.2f}, val loss {losses["val"]:.2f}')
        torch.save(model.state_dict(), f"{model_name}_{step}.ckpt")   # save checkpoint
    # ----------------------------------------------------
    xb, yb = get_batch('train')             # sample a batch of data
    logits, loss = model(xb, yb)            # evaluate the loss
    optimizer.zero_grad(set_to_none=True)   # clear gradients
    loss.backward()                         # backprop
    optimizer.step()                        # update the weights
    scheduler.step()                        # update the learning rate
    # ----------------------------------------------------  
    cur_learn = f'{scheduler.get_last_lr()[0]:.6f}'
    cur_loss = f'{loss.item():.4f}'
    log.append([step, cur_learn, cur_loss])
    bar.set_postfix(learn=cur_learn, loss=cur_loss) # Update the progress bar for loose feedback

# show final losses
losses = estimate_loss()
print(f'\n> final train loss {losses["train"]:.2f} - final val loss {losses["val"]:.2f} : {step} iterations\n')

# save the log
with open('log.txt', 'w', encoding='utf-8') as f:
    f.write(str(log))
      
# save the final model
torch.save(model.state_dict(), f"{model_name}_{step}.ckpt")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)     # single token to start generation
output = m.generate(context, max_new_tokens=1000)                  # call generate
output = decode(output[0].tolist())                                # decode the output to chars

# write output to file
with open(f'{model_name}-output.txt', 'w', encoding='utf-8') as f:
    f.write(output)
