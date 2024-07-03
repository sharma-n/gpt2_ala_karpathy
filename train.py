import torch.utils
import torch.utils.data
import yaml
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from time import time
import logging
logger = logging.getLogger(__name__)

from gpt import GPT
from dataset import ShakespeareDataset
from utils import get_device_type, TrainConfig, GPTConfig, cosine_lr_scheduler


def train(model: GPT, data: torch.utils.data.Dataset, config: TrainConfig):
    '''
    Train a GPT model on given data.

    args:
        model (GPT): GPT model to train
        dataloader (torch.utils.data.Dataset): Torch data
        config (TrainConfig): Training config
    '''
    raw_model = model.module if IS_DDP else model
    optimizer = raw_model.configure_optimizers(config, DEVICE)
    dataloader = DataLoader(data, batch_size=config.minibatch_size, shuffle=True)
    assert config.batch_size % (config.minibatch_size * raw_model.config.context_len * DDP_WORLD_SIZE) == 0, "batch size should be divisible by (minibatch_size * context_len * DDP_WORLD_SIZE)"
    grad_accum_steps = config.batch_size // (config.minibatch_size * raw_model.config.context_len * DDP_WORLD_SIZE)
    logger.info(f'[TRAIN\t] For a total batch size of {config.batch_size}, doing gradient application over {grad_accum_steps} steps...')
    loss_accum, t0 = 0.0, time()
    device_type = "cuda" if DEVICE.startswith("cuda") else "cpu"
    for _ in range(config.epochs):
        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            if IS_DDP: model.require_backward_grad_sync = (step+1)%grad_accum_steps == 0
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss /= grad_accum_steps
            loss_accum += loss.detach()
        
            loss.backward()

            if (step+1)%grad_accum_steps == 0:      # gradient accumulation: do backward pass only every grad_accum_steps
                if IS_DDP: dist.all_reduce(loss_accum, dist.ReduceOp.AVG)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping!
                lr = cosine_lr_scheduler(step, config)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                optimizer.step()
                optimizer.zero_grad()   # always remember to zero the grads! loss.backwards is an accumulation.
                torch.cuda.synchronize()
                t1 = time()
                tokens_per_sec = (config.batch_size * DDP_WORLD_SIZE) / (t1-t0)
                if USE_WANDB and MASTER_PROCESS:
                    wandb.log({'loss': loss_accum, 'norm': norm ,'tok/sec': tokens_per_sec, 'lr': lr})
                else:
                    logger.info(f'step {step}: loss = {loss_accum}, norm = {norm:.4f}, tok/sec = {tokens_per_sec:.2f}, lr = {lr:.5f}')
                t0 = time()
                loss_accum = 0.0

if __name__ == '__main__':
    ##########
    # Simple launch: python train.py
    # DDP launch: torchrun --standalone --nproc_per_node=8 train.py
    CONFIG_PATH = 'config.yaml'
    USE_WANDB = False
    IS_DDP = int(os.environ.get('RANK', -1)) != -1 # check if using DDP
    config = yaml.safe_load(open(CONFIG_PATH, 'r'))

    if config['use_wandb']:
        import wandb
        wandb.init(project='gpt2_ala_karpathy', config=config)
        USE_WANDB = True
    if IS_DDP:
        assert torch.cuda.is_available()
        dist.init_process_group(backend='nccl')
        DDP_RANK, DDP_LOCAL_RANK, DDP_WORLD_SIZE = int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
        DEVICE = f'cuda:{DDP_LOCAL_RANK}'
        torch.cuda.set_device(DEVICE)
        MASTER_PROCESS = DDP_RANK == 0  # process that does logging, checkpointing etc
    else:
        DEVICE = get_device_type()
        DDP_RANK, DDP_LOCAL_RANK, DDP_WORLD_SIZE = 0, 0, 1
        MASTER_PROCESS = True

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision('high')
    if MASTER_PROCESS: logging.basicConfig(level=logging.INFO)

    gpt = GPT(GPTConfig(**config))
    logger.info('[TRAIN\t] GPT2 model initialized successfully using YAML file!')
    gpt.to(DEVICE)
    ## TODO: When you train on the GPU box, make sure to change the following!
    # - Compile the model
    # - remove the cuda.synchronize()
    # gpt = torch.compile(gpt)
    if IS_DDP: gpt = DDP(gpt, device_ids=[DDP_LOCAL_RANK])

    data = ShakespeareDataset(config['context_len'], DDP_RANK, DDP_WORLD_SIZE)
    train(gpt, data, TrainConfig(**config))
    if IS_DDP: dist.destroy_process_group()