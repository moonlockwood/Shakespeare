
# Shakespeare

Andrej Karpathy's "makemore" code. 

The 'gpt-tiny.py' script and a text file are all you need to make a tiny gpt that will output text very roughly in the style of the text it was trained on.

Running the script will train a tiny transformer, save the weights, then generate some output. Default configuration is to train from shakespeare.txt in the same directory. Just change the filename in the top of the script to use a different file.

The script is laid out in a straightforward way that goes through each stage of the process sequentially in very simple python. The simpler stuff in the code is documented clearly, so changing the source file etc is super easy even if you don't know much python.

The more complex areas are pretty self explanatory if you know enough to understand what they do.

It's configured to be quick to train and run on an average gpu. In this state it still gets pretty good results. Things like; context length, parameter count, learning rate and weight decay can probably get you better results. With a big text file and settings maxed out it should be able to do a reasonable job of generating nonsense that is very similar to the source file.

## Things to experiment with:
- batch_size - ram usage
- block_size - context. Larger contexts make it learn better, but slows things right down
- max_iters - training steps. small batches need lots of steps
- learning_rate - it's cycled with the value given being it's peak near the start, then fading to 0
- n_embed - number of parameters! this dimension sets the scale of the model
- n_head - number of attention heads, 6 should work too.
- n_layers - number of layers, how deep the network is.
- weight decay - regularises things so that you are less prone to overfitting

default settings:
```
batch_size = 256 # number of independent sequences to process in parallel
block_size = 400 # maximum context length for predictions
max_iters = 1000 # number of training steps
learning_rate = 8e-4 # 3e-4 is a good value for AdamW
n_embed = 480 # the dimension of the embeddings, transformers from 100-500 are common
n_head = 8 # how many attention heads to use
n_layer = 6 # how many transformer blocks to use
weight_decay = 2e-3 # weight decay strength 1e-3 moderately powerful
```

### Usage
It's a one file simple as it gets transformer, you can just run it. As long as shakespeare.txt is present it will perform all necessary steps and update it's progress then start streaming text when it's done.

### Prerequisites
Python 3.6+

- torch
- tqdm

```
pip install torch
pip install tqdm
```
Should run on pretty much any computer, will revert to cpu if no cuda is found.

~6gb vram  
~15 minutes

As configured on a low/medium grade consumer gpu to train on the 1mb shakespeare file.

## Acknowledgements
- Andrej Karpathy for the original "makemore"

This is just an adaptation of his work with very small changes.
