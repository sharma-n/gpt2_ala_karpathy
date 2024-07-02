from os.path import exists
import requests
import tiktoken
import torch
import logging
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
    def __init__(self, T: int) -> None:
        '''
        Constructor

        args:
            T (int): context length
        '''
        self.T = T
        self.tokens = get_tiny_shakespeare()
        if not len(self.tokens)%T:
            self.tokens = self.token[:-(len(self.tokens)%T)]    #remove the "left-over" tokens
        logging.info(f'[DATALOADER  ] The dataset has {self.tokens.size()} tokens and {len(self)} samples')

    def __len__(self):
        return len(self.tokens)//self.T
    
    def __getitem__(self, i):
        x = self.tokens[i*self.T: (i+1)*self.T]
        y = self.tokens[(i*self.T)+1: ((i+1)*self.T)+1]
        return x,y