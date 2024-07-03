from os.path import exists
import requests
import tiktoken
import torch
import logging
logger = logging.getLogger(__name__)
from torch.utils.data import Dataset

def get_tiny_shakespeare(save_path='tiny_shakespeare.txt'):
    '''
    Loads tiny shakespeare dataset. Downloads it if not found.

    args:
        save_path (str): Path to save the text file at.
    '''
    if not exists(save_path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
    with open(save_path, 'r') as f:
        text = f.read()
    enc = tiktoken.get_encoding('gpt2')
    tokens = torch.tensor(enc.encode(text))
    return tokens

# Hopefully using torch dataset gives some gains, and is not just adding lag due to boilerplate...
class ShakespeareDataset(Dataset):
    '''
    A small dataset that contains all the plays of shakespeare
    '''
    def __init__(self, T: int, proc_rank: int = 0, n_procs: int = 1) -> None:
        '''
        Constructor

        args:
            T (int): context length
            proc_rank (int): For DDP, rank of current process
            n_procs (int): For DDP, total number of processes
        '''
        self.T = T
        self.tokens = get_tiny_shakespeare()
        if not len(self.tokens)%T:
            self.tokens = self.token[:-(len(self.tokens)%T)]    #remove the "left-over" tokens
        logger.info(f'[DATALOADER\t] The dataset has {self.tokens.size(0)} tokens and {len(self)} samples, before splitting across processes.')
        tokens_per_proc = len(self.tokens)//n_procs
        self.tokens = self.tokens[proc_rank * tokens_per_proc : (proc_rank+1) * tokens_per_proc]

    def __len__(self):
        return len(self.tokens)//self.T
    
    def __getitem__(self, i):
        x = self.tokens[i*self.T: (i+1)*self.T]
        y = self.tokens[(i*self.T)+1: ((i+1)*self.T)+1]
        return x,y