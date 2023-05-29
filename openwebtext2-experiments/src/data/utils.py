import numpy as np
from typing import Dict

from .shakespeare import get_shakespeare_data
from .wikitext import get_wikitext_data
from .arxiv import get_arxiv_2000, get_arxiv_full
from .openwebtext2 import get_openwebtext2_data


def get_dataset(args) -> Dict[str, np.ndarray]:
    """ Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
     contained in its own python file. The expected format at the moment is a dictionary of np.memmap
     containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data. """
    if args.dataset == 'wikitext':
        return get_wikitext_data()
    if args.dataset == "shakespeare-char":
        return get_shakespeare_data()
    if args.dataset == "arxiv2000":
        return get_arxiv_2000()
    if args.dataset == "arxiv":
        return get_arxiv_full()
    if args.dataset == "arxiv+wiki":
        arxiv_data = get_arxiv_full()
        wiki_data = get_wikitext_data()
        train_data = np.concatenate((arxiv_data['train'], wiki_data['train']))
        val_data = np.concatenate((arxiv_data['val'], wiki_data['val']))
        return {'train': train_data, 'val': val_data}
    if args.dataset == 'openwebtext2':
        return get_openwebtext2_data()
    else:
        raise NotImplementedError(f"Unknow dataset key '{args.dataset}'")
