import torch.utils
import torch.utils.data
import yaml
import torch
from torch.utils.data import DataLoader
import tiktoken
import logging
from dataclasses import dataclass, fields
from gpt import GPT, GPTConfig
from dataset import ShakespeareDataset

logging.basicConfig(level=logging.INFO)

def get_device_type():
    """
    Returns the device type (CUDA or MPS) if available, otherwise returns "cpu"
    """
    if torch.cuda.is_available():
        logging.info('Using device: cuda')
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logging.info('Using device: mps')
        return "mps"
    else:
        logging.info('Using device: cpu')
        return "cpu"
DEVICE = get_device_type()

def sample(model: GPT, prompt: str, reps: int = 5, max_total_tokens: int = 30, topk: int =50):
    '''
    Runs the GPT model on a given string to get back samples of outputs.

    args:
        model (GPT): the GPT model
        prompt (string): the initial tokens passed to the model to extend
        reps (int): number of samples to draw
        max_total_tokens (int): number of new tokens in each sample
        topk (int): Top-K sampling
    '''
    model.eval()
    enc = tiktoken.get_encoding('gpt2')
    tokens = torch.tensor(enc.encode(prompt), dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(reps, 1)    # (reps, len(prompt))
    x = tokens.to(DEVICE)

    while x.size(1) < max_total_tokens:
        with torch.no_grad():
            logits = model(x)    # (B, T, vocab_size)
            logits = logits[:, -1, :]   # take only the predictions for T+1
            probs = F.softmax(logits, dim=-1)
            # topk sampling where you only sample from the k most probable tokens
            # topk_probs and topk_indices are both (B,topk)
            topk_probs, topk_indices = torch.topk(probs, topk, dim=-1)
            # select a single token based on the probabilities
            ix = torch.multinomial(topk_probs, 1)   #(B,1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)
            # append to the sequence for next prediction
            x = torch.cat((x, xcol), dim=1)
    
    samples = [enc.decode(x[i,:].tolist()) for i in range(reps)]
    return samples

@dataclass
class TrainConfig:
    batch_size: int = 64
    steps: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    learning_rate: float = 3e-4
    dropout: float = 0.2

    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

def train(model: GPT, data: torch.utils.data.Dataset, config: TrainConfig):
    '''
    Train a GPT model on given data.

    args:
        model (GPT): GPT model to train
        dataloader (torch.utils.data.Dataset): Torch data
        config (TrainConfig): Training config
    '''
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    dataloader = DataLoader(data, batch_size=config.batch_size, shuffle=True)
    for i, data in enumerate(dataloader):
        x, y = data
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()   # always remember to zero the grads at the start! because loss.backwards is an accumulation.
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        logging.info(f'step {i}: loss = {loss.item()}')

if __name__ == '__main__':
    CONFIG_PATH = 'config.yaml'
    config = yaml.safe_load(open(CONFIG_PATH, 'r'))
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    gpt = GPT(GPTConfig(**config))
    logging.info('GPT2 model initialized successfully using YAML file!')
    gpt.to(DEVICE)
    # print('\n> '.join(gpt.sample("Hello, I'm a language model")))

    data = ShakespeareDataset(config['context_len'])
    train(gpt, data, TrainConfig(**config))