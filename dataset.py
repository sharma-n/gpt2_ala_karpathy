from os.path import exists
import requests
import tiktoken

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
    tokens = enc.encode(text)
    return tokens