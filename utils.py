import torch
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)
import tiktoken
import math
from dataclasses import dataclass, fields, field

def get_device_type():
    """
    Returns the device type (CUDA or MPS) if available, otherwise returns "cpu"
    """
    if torch.cuda.is_available():
        logger.info('[UTILS\t] Using device: cuda')
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info('[UTILS\t] Using device: mps')
        return "mps"
    else:
        logger.info('[UTILS\t] Using device: cpu')
        return "cpu"

def sample(model, prompt: str, reps: int = 5, max_total_tokens: int = 30, topk: int =50, device = 'cuda'):
    '''
    Runs the GPT model on a given string to get back samples of outputs.

    args:
        model (GPT): the GPT model
        prompt (string): the initial tokens passed to the model to extend
        reps (int): number of samples to draw
        max_total_tokens (int): number of new tokens in each sample
        topk (int): Top-K sampling
        device (str): device to use
    '''
    enc = tiktoken.get_encoding('gpt2')
    tokens = torch.tensor(enc.encode(prompt), dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(reps, 1)    # (reps, len(prompt))
    x = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)
    while x.size(1) < max_total_tokens:
        with torch.no_grad():
            logits, _ = model(x)    # (B, T, vocab_size)
            logits = logits[:, -1, :]   # take only the predictions for T+1
            probs = F.softmax(logits, dim=-1)
            # topk sampling where you only sample from the k most probable tokens
            # topk_probs and topk_indices are both (B,topk)
            topk_probs, topk_indices = torch.topk(probs, topk, dim=-1)
            # select a single token based on the probabilities
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)   #(B,1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)
            # append to the sequence for next prediction
            x = torch.cat((x, xcol), dim=1)
    
    samples = [[enc.decode(x[i,:].tolist())] for i in range(reps)]
    return samples

@dataclass
class TrainConfig:
    batch_size: int = 524288
    minibatch_size: int = 8
    max_steps: int = 19073
    eval_interval: int = 500
    adam_betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    adam_eps: float = 1e-8
    lr_max: float = 3.0e-4
    lr_min: float = 3.0e-5
    lr_warmup_steps: int = 10
    weight_decay: float = 0.1
    val_steps: int = 20
    dataloader_nworkers: int = 4

    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

@dataclass
class GPTConfig:
    context_len: int = 1024  # called block_size in the video
    n_embed: int = 768
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    init_linear_std: float = 0.02

    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

def cosine_lr_scheduler(it: int, config: TrainConfig):
    '''Implements the cosine learning rate scheduler
    
    args:
        it (int): iteration
        config (TrainConfig): parameters of learning rate scheduler
    '''
    if it < config.lr_warmup_steps:
        return config.lr_max * (it+1) / config.lr_warmup_steps
    if it > config.max_steps:
        return config.lr_min
    decay_ratio = (it-config.lr_warmup_steps) / (config.max_steps - config.lr_warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))   #coeff starts at 1 and goes to 0
    return config.lr_min + coeff * (config.lr_max - config.lr_min)

def get_most_likely_row(tokens, mask, logits):
    '''
    Helper function that returns the index of the completion with the lowest loss
    '''
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm