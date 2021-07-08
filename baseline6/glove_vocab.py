from collections import defaultdict
from typing import Callable, Dict

import numpy as np
from tqdm import tqdm
from texar.torch.hyperparams import HParams
from texar.torch.utils import utils
def load_glove(filename: str) -> np.ndarray:
    r"""Loads embeddings in the glove text format in which each line is
    ``<word-string> <embedding-vector>``. Dimensions of the embedding vector
    are separated with whitespace characters.

    Args:
        filename (str): Path to the embedding file.
        vocab (dict): A dictionary that maps token strings to integer index.
            Tokens not in :attr:`vocab` are not read.
        word_vecs: A 2D numpy array of shape `[vocab_size, embed_dim]`
            which is updated as reading from the file.

    Returns:
        The updated :attr:`word_vecs`.
    """
    word_vecs = defaultdict()
    with open(filename) as fin:
        for line in tqdm(fin):
            vec = line.strip().split()
            if len(vec) == 0:
                continue
            word, vec = vec[0], vec[1:]
            word_vecs[word] = np.array([float(v) for v in vec])
    return word_vecs

word_vecs = load_glove("/apdcephfs/share_916081/jamgao/projects/event-ib/ft_local/glove.6B.100d.ext.txt")

print(len(list(word_vecs.keys())))

for w in list(word_vecs.keys()):
    print(w)
