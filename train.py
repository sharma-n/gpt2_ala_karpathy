import torch.utils
import torch.utils.data
import yaml
import torch
from torch.utils.data import DataLoader
from time import time
import logging
logging.basicConfig(level=logging.INFO)

from gpt import GPT
from dataset import ShakespeareDataset
from utils import DEVICE, TrainConfig, GPTConfig, cosine_lr_scheduler


def train(model: GPT, data: torch.utils.data.Dataset, config: TrainConfig):
    '''
    Train a GPT model on given data.

    args:
        model (GPT): GPT model to train
        dataloader (torch.utils.data.Dataset): Torch data
        config (TrainConfig): Training config
    '''
    optimizer = model.configure_optimizers(config)
    dataloader = DataLoader(data, batch_size=config.minibatch_size, shuffle=True)
    assert config.batch_size % (config.minibatch_size * model.config.context_len) == 0, "batch size should be divisible by (minibatch_size * context_len)"
    grad_accum_steps = config.batch_size // (config.minibatch_size * model.config.context_len)
    logging.info(f'[TRAIN\t] For a total batch size of {config.batch_size}, doing gradient application over {grad_accum_steps} steps...')
    loss_accum, t0 = 0.0, time()
    for step, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss /= grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

        if (step+1)%grad_accum_steps == 0:      # gradient accumulation: do backward pass only every grad_accum_steps
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping!
            lr = cosine_lr_scheduler(step, config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
            optimizer.zero_grad()   # always remember to zero the grads! loss.backwards is an accumulation.
            torch.cuda.synchronize()
            t1 = time()
            tokens_per_sec = config.batch_size / (t1-t0)
            if USE_WANDB:
                wandb.log({'loss': loss_accum, 'norm': norm ,'tok/sec': tokens_per_sec, 'lr': lr})
            else:
                logging.info(f'step {step}: loss = {loss_accum}, norm = {norm:.4f}, tok/sec = {tokens_per_sec:.2f}, lr = {lr:.5f}')
            t0 = time()
            loss_accum = 0.0

USE_WANDB = False
if __name__ == '__main__':
    CONFIG_PATH = 'config.yaml'
    config = yaml.safe_load(open(CONFIG_PATH, 'r'))
    if config['use_wandb']:
        import wandb
        wandb.init(project='gpt2_ala_karpathy', config=config)
        USE_WANDB = True
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision('high')

    gpt = GPT(GPTConfig(**config))
    logging.info('[TRAIN\t] GPT2 model initialized successfully using YAML file!')
    gpt.to(DEVICE)
    ## TODO: When you train on the GPU box, make sure to change the following!
    # - Compile the model
    # - remove the cuda.synchronize()
    # gpt = torch.compile(gpt)
    # print('\n> '.join(gpt.sample("Hello, I'm a language model")))

    data = ShakespeareDataset(config['context_len'])
    train(gpt, data, TrainConfig(**config))