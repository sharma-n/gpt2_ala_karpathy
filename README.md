# GPT2 from Scratch - à la Andrej Karpathy
This is my attempt to follow along (and re-produce in my own coding style) the lecture from Andrej Karpathy below:

[Let's reproduce GPT-2 (124M) by Andrej Karpathy on YouTube](https://youtu.be/l8pRSuU81PU?si=ubLsdiqtkD4MMyUv)

This is going to mostly be a redo of the [build-nanogpt by Andrej Karpathy](https://github.com/karpathy/build-nanogpt) codebase, but I will be adding a lot more comments to keep track of my learnings throughout the process. Therefore, please use the original [build-nanogpt by Andrej Karpathy](https://github.com/karpathy/build-nanogpt). This repo is just for my own learnings.

## Changes I made on top of Karpathy's code
- Keeping track of runs, losses, configs, weights using `wandb`
- Used the torch `Dataset` class so that we can have proper shuffling. This means I do DDP slightly differently on the data side.
- Refactoring the code into three main files:
  - `gpt.py`: The model definition
  - `dataset.py`: The dataset
  - `train.py`: The training loop
- Single `config.yaml` file that can control the multiple model and training hyper-parameters

## Notes
- Differences between GPT2 and original transformer model from 'Attention is all you need' paper:
  - Layer norm is moved to the input of each sub-block
  - additional layer normalization was added after the final attention block.
  - Weights of the residual layers at initialization were scaled by a factor of $1/\sqrt{N}$, where $N$ is the number of residual layers.
- Sanity check of loss at initialization: For cross entropy, we assume that all possible tokens are equally likely. So, the loss should be around $-\ln{\frac{1}{\text{\# of tokens}}} = -\ln{\frac{1}{\text{50257}}} \approx 10.825$.
- The gains in loss function seen within the first 100 of so iterations are all from driving the "low-probability" tokens to almost zero probability.
- **Weight Sharing Scheme**: The original transformer paper shares the weights between the token embedding layer at the bottom and the final classifier layer at the top. Specially for shallow LLMs, these two correspond to almost $30\%$ of all weights! The idea behind the weight sharing is that two vectors in the embedding space that are close to each other are roughly the same word. And at the output, we should also see that this relationship should hold. In fact, the output embeddings behave like word embeddings!
- The growth of the gradients inside the residual stream is controlled by multiplying them with $\frac{1}{\sqrt{N}}$, where $N$ is the number of residual streams (which is $2\times$ the number of attention blocks).
- **Gradient clipping**: Usually a good idea, as it protects you from really weird training samples that would get a very high loss and disturb the learning process and shock the model. It's also a good idea to plot the gradient norm over time. If it increases over time, there is something wrong. If there's a spike in it, there's an issue of stability.
- **Weight Decaying**: A good regularization technique that forces the model to use more of its channels, instead of focussing on only a few. It is common to *not* weight decay bias vectors or any other one-dimensional vectors (e.g. layer norms).
- **Gradient Accumulation**: The way to get very large batch sizes (for example, for GPT2-124M the batch size was 0.5M tokens). Remember that when doing gradient accumulation, you need to scale the loss at each steps by the number of accumulation steps! This will make training more stable, and also leads to some performance improvements since you're not doing backprop so often.
- Steps to speed up training (at the start, on my RTX 3060 Laptop GPU, I was getting a measly $215$ tok/sec 🥲):
  - Use Float32 by adding `torch.set_float32_matmul_precision('high')` ($215\to 274$ tok/sec, $+27.4\%$)
  - [BF16 Mixed Precision](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) ($274\to 327, +19.3\%$)
  - Compiling the torch model using `torch.compile(model)`. This requires the `triton` package, which is only on Linux, so will turn it on when using the GPU box.
  - Using flash attention 2 ($327\to 1100, +236.4\%$ 😍)
  - Using powers of 2 everywhere ($1100\to 1150, +4.35\%$; additionally, at least on my laptop GPU, this led to more consistent tok/sec numbers across iterations 🤔)
  - Use kernel fusion for AdamW