"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import itertools
import os
import multiprocessing as mp
import numpy as np
import tiktoken
import h5py
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

def tokenize(doc):
    # init the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>'] # end of text token
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def download_fineweb(hdf5_path='edu_fineweb10B.hdf5', remote_name='sample-10BT', shard_size :int = 1024):
    '''
    Download the fineweb dataset and store it locally into a HDF5 dataset after sharding

    args:
        hdf5_path (str): path to local HDF5 file where the data is saved
        remote_name (str): name of the dataset type to download. Check options at https://huggingface.co/datasets/HuggingFaceFW/fineweb
        shard_size (int): Number of tokens per shard
    '''
    # download the dataset
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
    h5f = h5py.File(hdf5_path, 'w')
    write_n_tokens = int(1e8)
    # HDF5 is nice enough that it will fill your memory to the limit, and no more...i'm sure there is a way to set a manual limit, but can't be bothered.
    dset_train = h5f.create_dataset('edu_fineweb_train', shape=(write_n_tokens, ), maxshape=(None,), dtype=np.uint16, chunks=(shard_size,))
    dset_val = h5f.create_dataset('edu_fineweb_val', shape=(write_n_tokens, ), dtype=np.uint16, chunks=(shard_size,))
    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count()-2)
    with mp.Pool(nprocs) as pool:
        progress_bar = tqdm(total=len(fw), unit="samples", desc='Downloading FineWeb', dynamic_ncols=True)
        token_count = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((write_n_tokens,), dtype=np.uint16)
        save_iter, fw_iter = 0, 0
        for tokens in pool.imap(tokenize, fw, chunksize=32):
            fw_iter += 1
            if token_count + len(tokens) < write_n_tokens:  # if still space in buffer
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
            else:                
                # split the document into whatever fits in the buffer; the remainder will be written next time
                remainder = write_n_tokens - token_count
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                if save_iter == 0:  #save val dataset
                    dset_val[:] = all_tokens_np
                elif save_iter == 1:    # no resizing on the first save
                    dset_train[write_n_tokens*(save_iter-1):] = all_tokens_np
                else:
                    dset_train.resize(dset_train.shape[0]+write_n_tokens, axis=0)       # don't resize the first time for training
                    dset_train[write_n_tokens*(save_iter-1):] = all_tokens_np
                save_iter += 1
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder
                progress_bar.update(fw_iter)
                fw_iter = 0
        # write any remaining tokens as the last shard
        if token_count != 0:
            dset_train.resize(dset_train.shape[0]+token_count, axis=0)
            dset_train[write_n_tokens*(save_iter-1):] = all_tokens_np[:token_count]
            progress_bar.update(token_count)
        progress_bar.close()
    h5f.close()
    

if __name__ == '__main__':
    download_fineweb()
