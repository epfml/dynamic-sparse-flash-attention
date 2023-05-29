import os
import zipfile
import urllib
import numpy as np
import tiktoken


WIKITEXT_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/wikitext/")


def get_wikitext_data():
    """ Inspired from https://github.com/tysam-code/hlb-gpt """
    if not os.path.exists(WIKITEXT_DATA_PATH):
        os.makedirs(WIKITEXT_DATA_PATH, exist_ok=True)
        print("downloading data and tokenizing (1-2 min)")
        raw_data_source = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip'
        urllib.request.urlretrieve(raw_data_source, os.path.join(WIKITEXT_DATA_PATH,'data.zip'))

        with zipfile.ZipFile(os.path.join(WIKITEXT_DATA_PATH, "data.zip"), 'r') as zip_ref:
            zip_ref.extractall(WIKITEXT_DATA_PATH)

        with open(os.path.join(WIKITEXT_DATA_PATH, "wikitext-103-raw/wiki.train.raw"), 'r') as data_file:
            raw_train_data = data_file.read()

        with open(os.path.join(WIKITEXT_DATA_PATH, "wikitext-103-raw/wiki.valid.raw"), 'r') as data_file:
            raw_eval_data = data_file.read()

        tokenizer = tiktoken.get_encoding("gpt2")
        raw_tokenized_train = tokenizer.encode_ordinary(raw_train_data)
        raw_tokenized_eval = tokenizer.encode_ordinary(raw_eval_data)

        train_tokenized = np.array(raw_tokenized_train, dtype=np.uint16) 
        eval_tokenized = np.array(raw_tokenized_eval, dtype=np.uint16)

        train_tokenized.tofile(os.path.join(WIKITEXT_DATA_PATH, 'train.bin'))
        eval_tokenized.tofile(os.path.join(WIKITEXT_DATA_PATH, 'val.bin'))
        print("completed the tokenization process!")

    train_data = np.memmap(os.path.join(WIKITEXT_DATA_PATH, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(WIKITEXT_DATA_PATH, 'val.bin'), dtype=np.uint16, mode='r')

    return {'train': train_data, 'val': val_data}
