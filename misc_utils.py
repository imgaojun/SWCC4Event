# -*- coding: utf-8 -*-
from __future__ import absolute_import
from typing import (
    Any, Callable, Generic, Iterator, List, Optional, Tuple, TypeVar, Union)

import torch
from texar.torch.data import BatchingStrategy
import logging
from scipy import spatial, stats
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy.linalg import norm
Example = TypeVar('Example')


logger = logging.getLogger()


def init_logger(log_file=None):
    log_format = logging.Formatter("[%(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def cosine_similarity(a, b):
    # return 1 - spatial.distance.cosine(a, b)
    return np.inner(a, b) / (norm(a) * norm(b))

def spearmanr(a,b):
    return stats.spearmanr(a,b)