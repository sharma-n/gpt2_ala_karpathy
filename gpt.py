import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
import yaml
import logging
logger = logging.getLogger(__name__)
from utils import GPTConfig, TrainConfig

class CausalSelfAttention(nn.Module):
    '''
    Causal self-attention
    '''
    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed, 3*config.n_embed) # build qkv as a single linear layer, split it in forward
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NORMALIZE_RESIDUAL_GRADIENT_HACK = 1
        self.n_head, self.n_embed = config.n_head, config.n_embed
        # called 'bias' in OpenAI/HF naming convention, but essentially the mask. It is a registered buffer because it's not a trainable tensor.
        # self.register_buffer('bias', torch.tril(torch.ones(config.context_len, config.context_len))
        #                              .view(1, 1, config.context_len, config.context_len))

    def forward(self, x):
        B, T, C = x.size()  # batch size, seq length, n_embed
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        # reshape tensors so that each head of the multi-head is calculated separately
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)   # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)   # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)   # (B, nh, T, hs)
        
        # attention (materializes the large (T,T) matrix for all queries and keys)
        # att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))   # auto-regressive mask
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs), essentially a "weighted sum" of the values
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1,2).contiguous().view(B,T,C)   # brings the different computations for each head back together
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    '''
    Simple MLP with two layers and one GeLU
    '''
    def __init__(self, config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU() #gelu is better because it doesn't go to exactly zero  on the left side.
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.NORMALIZE_RESIDUAL_GRADIENT_HACK = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class TransformerBlock(nn.Module):
    '''
    The auto-regressive self-attention block
    '''
    def __init__(self, config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        '''
        Forward pass of a transformer block is essentially a "map-reduce".
        '''
        x = x + self.attn(self.ln_1(x))     # reduce
        x = x + self.mlp(self.ln_2(x))      # map
        return x

class GPT(nn.Module):
    '''
    The full GPT-2 model.
    '''
    def __init__(self, config: GPTConfig) -> None:
        '''
        Constructor

        args:
            config_path (GPTConfig): config object
        '''
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embed),    # weights for token embeddings
            wpe = nn.Embedding(self.config.context_len, self.config.n_embed),   # weights for position embeddings
            h = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.n_layer)]), # transformer blocks
            ln_f = nn.LayerNorm(self.config.n_embed)                            # layer-norm
        ))
        self.lm_head = nn.Linear(self.config.n_embed, self.config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight   # weight sharing scheme!
        self.apply(self._init_weights)  # init params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = self.config.init_linear_std
            if hasattr(module, 'NORMALIZE_RESIDUAL_GRADIENT_HACK'):
                std = (2*self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_linear_std)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.context_len, f"Input exceeds the maximum context length of the model"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_embed = self.transformer.wpe(pos)   # (T, n_embed)
        tok_embed = self.transformer.wte(idx)   # (B, T, n_embed)
        x = pos_embed + tok_embed
        # pass through transformer layers
        for block in self.transformer.h:
            x = block(x)
        # final layernorm (different from original transformer)
        x = self.transformer.ln_f(x)
        # final classifier layer
        logits = self.lm_head(x)    # (B, T, vocab_size)
        loss = None if targets is None else F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, train_config: TrainConfig, device: str = 'cuda'):
        '''
        Setup the optimizer to have weight decay
        '''
         # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': train_config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.info(f"[GPT\t] num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        logger.info(f"[GPT\t] num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"
        logger.info(f"[GPT\t] using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.lr_min, betas=tuple(train_config.adam_betas), eps=train_config.adam_eps, fused=use_fused)
        return optimizer

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface
        
        args:
            model_type (str): One of gpt2 | gpt2-medium | gpt2-large | gpt2-xl
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        logger.info(f'[GPT\t] Loading weights from pretrained GPT: {model_type}')

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embed=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embed=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['context_len'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    CONFIG_PATH = 'config.yaml'
    config = yaml.safe_load(open(CONFIG_PATH, 'r'))
    gpt = GPT(GPTConfig(**config))
    logger.info('[GPT\t] GPT2 model initialized successfully using YAML file!')

    # gpt = GPT.from_pretrained('gpt2')
    # logger.info('[GPT\t]GPT2 model weights loaded successfully from HF!')