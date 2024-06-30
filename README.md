# GPT2 from Scratch - à la Andrej Karpathy
This is my attempt to follow along (and re-produce in my own coding style) the lecture from Andrej Karpathy below:
<iframe width="560" height="315" src="https://www.youtube.com/embed/l8pRSuU81PU?si=KDl_uFi2CL-RMPw6" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

This is going to mostly be a redo of the [build-nanogpt by Andrej Karpathy](https://github.com/karpathy/build-nanogpt) codebase, but I will be adding a lot more comments to keep track of my learnings throughout the process. Therefore, please use the original [build-nanogpt by Andrej Karpathy](https://github.com/karpathy/build-nanogpt). This repo is just for my own learnings.

## Notes
- Differences between GPT2 and original transformer model from 'Attention is all you need' paper:
  - Layer norm is moved to the input of each sub-block
  - additional layer normalization was added after the final attention block.
  - Weights of the residual layers at initialization were scaled by a factor of $1/\sqrt{N}$, where $N$ is the number of residual layers.
- Sanity check of loss at initialization: For cross entropy, we assume that all possible tokens are equally likely. So, the loss should be around $-\ln{\frac{1}{\text{\# of tokens}}} = -\ln{\frac{1}{\text{50257}}} \approx 10.825$.