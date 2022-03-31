from logging import BASIC_FORMAT
from typing import List, Optional, Tuple, Iterator, Dict

import numpy as np
import torch

import texar.torch as tx
from texar.torch.hyperparams import HParams
import random
import json
from tqdm import tqdm
from pathlib import Path
import pickle
# from nltk import WordNetLemmatizer
Example = Tuple[np.ndarray, np.ndarray]

tokenizer = tx.data.BERTTokenizer(pretrained_model_name="bert-base-uncased")
pad_token_id = tokenizer.map_token_to_id(tokenizer.pad_token)


def freq_norm(val):
    max_val = 38354
    min_val = 3
    return (val - min_val) / (max_val - min_val)


def map_evt_to_tokens(evt):
    # evt ='<cls> ' + ' <sep> '.join(evt) + ' <sep>'
    evt = [evt[1], evt[0], evt[2]]
    evt = ' '.join(evt)
    evt = evt.replace("''", "").lower().split()
    evt = ' '.join([tokenizer.cls_token] + evt + [tokenizer.sep_token])
    return evt


def map_evt_to_tokens_for_text(evt):
    # evt ='<cls> ' + ' <sep> '.join(evt) + ' <sep>'
    # evt = [evt[1],evt[0],evt[2]]
    evt = ' '.join(evt)
    evt = evt.replace("''", "").lower().split()
    evt = ' '.join([tokenizer.cls_token] + evt + [tokenizer.sep_token])
    return evt


class TrainDataSource(tx.data.TextLineDataSource):

    def __iter__(self) -> Iterator[List[str]]:
        path = Path(self._file_paths[0])

        with self._open_file(path) as f:
            for line in f:
                ex = json.loads(line.strip())
                evt_q = ex['evt_q'].split('\t')
                evt_k = evt_q
                sampled_evt = random.choice(ex['evt_k'])
                evt_freq = freq_norm(sampled_evt[1])
                evt_p = sampled_evt[0].split('\t')
                yield evt_q, evt_k, evt_p, evt_freq


class ValidDataSource(tx.data.TextLineDataSource):

    def __iter__(self) -> Iterator[List[str]]:
        path = Path(self._file_paths[0])
        with self._open_file(path) as f:
            for line in f:
                ex = json.loads(line.strip())
                evt_q = ex['evt_q'].split('\t')
                evt_k = evt_q
                evt_p = ex['evt_k'][0]
                evt_freq = freq_norm(evt_p[1])
                evt_p = evt_p[0].split('\t')
                yield evt_q, evt_k, evt_p, evt_freq


class HardDataSource(tx.data.TextLineDataSource):

    def __iter__(self) -> Iterator[List[str]]:
        for path in self._file_paths:
            with self._open_file(path) as f:
                for line in f:
                    tokens = line.strip().split(' | ')
                    evt_a = [tokens[0], tokens[1], tokens[2]]
                    evt_b = [tokens[3], tokens[4], tokens[5]]
                    evt_c = [tokens[6], tokens[7], tokens[8]]
                    evt_d = [tokens[9], tokens[10], tokens[11]]
                    yield evt_a, evt_b, evt_c, evt_d


class TransDataSource(tx.data.TextLineDataSource):

    def __iter__(self) -> Iterator[List[str]]:
        for path in self._file_paths:
            with self._open_file(path) as f:
                for line in f:
                    tokens = line.strip().split(' | ')
                    evt_a = [tokens[0], tokens[1], tokens[2]]
                    evt_b = [tokens[3], tokens[4], tokens[5]]
                    score = float(tokens[6])
                    # if score < 1.5:
                    #     continue
                    yield evt_a, evt_b, score


class TrainData(tx.data.DatasetBase[Example, Example]):

    def __init__(self, hparams=None, device: Optional[torch.device] = None):
        self._hparams = HParams(hparams, self.default_hparams())
        data_source = TrainDataSource(
            self._hparams.dataset.files,
            compression_type=self._hparams.dataset.compression_type)
        # self._vocab = Vocab(self._hparams.dataset.vocab_file)
        self.vocab_size = tokenizer.vocab_size

        super().__init__(data_source, hparams, device=device)

    @staticmethod
    def default_hparams():
        return {
            **tx.data.DatasetBase.default_hparams(),
            'dataset': {
                'files': 'data.txt',
                'compression_type': None,
                'vocab_file': 'vocab.txt'
            },
        }

    def process(self, raw_example):

        evt_q, evt_k_str, evt_p, evt_freq = raw_example[0], raw_example[
            1], raw_example[2], raw_example[3]

        evt_q = map_evt_to_tokens(evt_q)

        evt_q_ids = tokenizer.map_text_to_id(evt_q)
        mask_pos = random.randint(1, len(evt_q_ids) - 2)
        mask_id = evt_q_ids[mask_pos]
        evt_q_ids[mask_pos] = tokenizer.map_token_to_id(tokenizer.mask_token)

        evt_k = map_evt_to_tokens(evt_k_str)
        evt_k_ids = tokenizer.map_text_to_id(evt_k)

        evt_p = map_evt_to_tokens(evt_p)
        evt_p_ids = tokenizer.map_text_to_id(evt_p)

        return {
            "evt_k": ' '.join(evt_k_str),
            "evt_q_ids": evt_q_ids,
            "evt_k_ids": evt_k_ids,
            "evt_p_ids": evt_p_ids,
            "mask_pos": mask_pos,
            "mask_id": mask_id,
            "evt_freq": evt_freq,
        }

    def collate(self, examples: List[Example]) -> tx.data.Batch:
        # evt_q = [ex["evt_q"] for ex in examples]
        evt_q_ids, evt_q_lengths = tx.data.padded_batch(
            [ex["evt_q_ids"] for ex in examples], pad_value=pad_token_id)

        evt_k = [ex["evt_k"] for ex in examples]
        evt_k_ids, evt_k_lengths = tx.data.padded_batch(
            [ex["evt_k_ids"] for ex in examples], pad_value=pad_token_id)

        evt_p_ids, evt_p_lengths = tx.data.padded_batch(
            [ex["evt_p_ids"] for ex in examples], pad_value=pad_token_id)

        evt_freq = [ex["evt_freq"] for ex in examples]
        mask_id = [ex["mask_id"] for ex in examples]
        mask_pos = [ex["mask_pos"] for ex in examples]

        return tx.data.Batch(
            len(examples),
            # evt_q=evt_q,
            evt_q_ids=torch.from_numpy(evt_q_ids),
            evt_q_lengths=torch.tensor(evt_q_lengths),
            evt_k=evt_k,
            evt_k_ids=torch.from_numpy(evt_k_ids),
            evt_k_lengths=torch.tensor(evt_k_lengths),
            evt_p_ids=torch.from_numpy(evt_p_ids),
            evt_p_lengths=torch.tensor(evt_p_lengths),
            evt_freq=torch.tensor(evt_freq),
            mask_id=torch.tensor(mask_id),
            mask_pos=torch.tensor(mask_pos),
        )


class ValidData(tx.data.DatasetBase[Example, Example]):

    def __init__(self, hparams=None, device: Optional[torch.device] = None):
        self._hparams = HParams(hparams, self.default_hparams())
        data_source = ValidDataSource(
            self._hparams.dataset.files,
            compression_type=self._hparams.dataset.compression_type)
        # self._vocab = Vocab(self._hparams.dataset.vocab_file)
        self.vocab_size = tokenizer.vocab_size

        super().__init__(data_source, hparams, device=device)

    @staticmethod
    def default_hparams():
        return {
            **tx.data.DatasetBase.default_hparams(),
            'dataset': {
                'files': 'data.txt',
                'compression_type': None,
                'vocab_file': 'vocab.txt'
            },
        }

    def process(self, raw_example):

        evt_q, evt_k_str, evt_p, evt_freq = raw_example[0], raw_example[
            1], raw_example[2], raw_example[3]

        evt_q = map_evt_to_tokens(evt_q)

        evt_q_ids = tokenizer.map_text_to_id(evt_q)
        mask_pos = random.randint(1, len(evt_q_ids) - 2)
        mask_id = evt_q_ids[mask_pos]
        evt_q_ids[mask_pos] = tokenizer.map_token_to_id(tokenizer.mask_token)

        evt_k = map_evt_to_tokens(evt_k_str)
        evt_k_ids = tokenizer.map_text_to_id(evt_k)

        evt_p = map_evt_to_tokens(evt_p)
        evt_p_ids = tokenizer.map_text_to_id(evt_p)

        return {
            "evt_k": ' '.join(evt_k_str),
            "evt_q_ids": evt_q_ids,
            "evt_k_ids": evt_k_ids,
            "evt_p_ids": evt_p_ids,
            "mask_pos": mask_pos,
            "mask_id": mask_id,
            "evt_freq": evt_freq,
        }

    def collate(self, examples: List[Example]) -> tx.data.Batch:
        # evt_q = [ex["evt_q"] for ex in examples]
        evt_q_ids, evt_q_lengths = tx.data.padded_batch(
            [ex["evt_q_ids"] for ex in examples], pad_value=pad_token_id)

        evt_k = [ex["evt_k"] for ex in examples]
        evt_k_ids, evt_k_lengths = tx.data.padded_batch(
            [ex["evt_k_ids"] for ex in examples], pad_value=pad_token_id)

        evt_p_ids, evt_p_lengths = tx.data.padded_batch(
            [ex["evt_p_ids"] for ex in examples], pad_value=pad_token_id)

        evt_freq = [ex["evt_freq"] for ex in examples]
        mask_id = [ex["mask_id"] for ex in examples]
        mask_pos = [ex["mask_pos"] for ex in examples]

        return tx.data.Batch(
            len(examples),
            # evt_q=evt_q,
            evt_q_ids=torch.from_numpy(evt_q_ids),
            evt_q_lengths=torch.tensor(evt_q_lengths),
            evt_k=evt_k,
            evt_k_ids=torch.from_numpy(evt_k_ids),
            evt_k_lengths=torch.tensor(evt_k_lengths),
            evt_p_ids=torch.from_numpy(evt_p_ids),
            evt_p_lengths=torch.tensor(evt_p_lengths),
            evt_freq=torch.tensor(evt_freq),
            mask_id=torch.tensor(mask_id),
            mask_pos=torch.tensor(mask_pos),
        )


class HardData(tx.data.DatasetBase[Example, Example]):

    def __init__(self, hparams=None, device: Optional[torch.device] = None):
        self._hparams = HParams(hparams, self.default_hparams())
        data_source = HardDataSource(
            self._hparams.dataset.files,
            compression_type=self._hparams.dataset.compression_type)
        # self._vocab = Vocab(self._hparams.dataset.vocab_file)
        super().__init__(data_source, hparams, device=device)

    @staticmethod
    def default_hparams():
        return {
            **tx.data.DatasetBase.default_hparams(),
            'dataset': {
                'files': 'data.txt',
                'compression_type': None,
                'vocab_file': 'vocab.txt'
            },
        }

    def process(self, raw_example):

        evt_a, evt_b, evt_c, evt_d  = \
                    raw_example[0], raw_example[1], raw_example[2],raw_example[3]
        # evt_a = ['<cls>']+[evt_a[0]] +['<sub>']+ [evt_a[1]] +['<pred>']+ [evt_a[2]]+['<obj>']
        evt_a = map_evt_to_tokens(evt_a)
        evt_a_ids = tokenizer.map_text_to_id(evt_a)

        # evt_b = ['<cls>']+[evt_b[0]] +['<sub>']+ [evt_b[1]] +['<pred>']+ [evt_b[2]]+['<obj>']
        evt_b = map_evt_to_tokens(evt_b)
        evt_b_ids = tokenizer.map_text_to_id(evt_b)

        # evt_c = ['<cls>']+[evt_c[0]] +['<sub>']+ [evt_c[1]] +['<pred>']+ [evt_c[2]]+['<obj>']
        evt_c = map_evt_to_tokens(evt_c)
        evt_c_ids = tokenizer.map_text_to_id(evt_c)

        # evt_d = ['<cls>']+[evt_d[0]] +['<sub>']+ [evt_d[1]] +['<pred>']+ [evt_d[2]]+['<obj>']
        evt_d = map_evt_to_tokens(evt_d)
        evt_d_ids = tokenizer.map_text_to_id(evt_d)

        return {
            "evt_a": evt_a,
            "evt_a_ids": evt_a_ids,
            "evt_b": evt_b,
            "evt_b_ids": evt_b_ids,
            "evt_c": evt_c,
            "evt_c_ids": evt_c_ids,
            "evt_d": evt_d,
            "evt_d_ids": evt_d_ids,
        }

    def collate(self, examples: List[Example]) -> tx.data.Batch:

        evt_a = [ex["evt_a"] for ex in examples]
        evt_a_ids, evt_a_lengths = tx.data.padded_batch(
            [ex["evt_a_ids"] for ex in examples], pad_value=pad_token_id)

        evt_b = [ex["evt_b"] for ex in examples]
        evt_b_ids, evt_b_lengths = tx.data.padded_batch(
            [ex["evt_b_ids"] for ex in examples], pad_value=pad_token_id)

        evt_c = [ex["evt_c"] for ex in examples]
        evt_c_ids, evt_c_lengths = tx.data.padded_batch(
            [ex["evt_c_ids"] for ex in examples], pad_value=pad_token_id)

        evt_d = [ex["evt_d"] for ex in examples]
        evt_d_ids, evt_d_lengths = tx.data.padded_batch(
            [ex["evt_d_ids"] for ex in examples], pad_value=pad_token_id)

        return tx.data.Batch(
            len(examples),
            evt_a=evt_a,
            evt_a_ids=torch.from_numpy(evt_a_ids),
            evt_a_lengths=torch.tensor(evt_a_lengths),
            evt_b=evt_b,
            evt_b_ids=torch.from_numpy(evt_b_ids),
            evt_b_lengths=torch.tensor(evt_b_lengths),
            evt_c=evt_c,
            evt_c_ids=torch.from_numpy(evt_c_ids),
            evt_c_lengths=torch.tensor(evt_c_lengths),
            evt_d=evt_d,
            evt_d_ids=torch.from_numpy(evt_d_ids),
            evt_d_lengths=torch.tensor(evt_d_lengths),
        )


class TransData(tx.data.DatasetBase[Example, Example]):

    def __init__(self, hparams=None, device: Optional[torch.device] = None):
        self._hparams = HParams(hparams, self.default_hparams())
        data_source = TransDataSource(
            self._hparams.dataset.files,
            compression_type=self._hparams.dataset.compression_type)
        # self._vocab = Vocab(self._hparams.dataset.vocab_file)
        super().__init__(data_source, hparams, device=device)

    @staticmethod
    def default_hparams():
        return {
            **tx.data.DatasetBase.default_hparams(),
            'dataset': {
                'files': 'data.txt',
                'compression_type': None,
                'vocab_file': 'vocab.txt'
            },
        }

    def process(self, raw_example):

        evt_a, evt_b, score = raw_example[0], raw_example[1], raw_example[2]
        evt_a = map_evt_to_tokens(evt_a)

        evt_a_ids = tokenizer.map_text_to_id(evt_a)
        evt_b = map_evt_to_tokens(evt_b)

        evt_b_ids = tokenizer.map_text_to_id(evt_b)

        return {
            "evt_a": evt_a,
            "evt_a_ids": evt_a_ids,
            "evt_b": evt_b,
            "evt_b_ids": evt_b_ids,
            "score": score,
        }

    def collate(self, examples: List[Example]) -> tx.data.Batch:
        evt_a = [ex["evt_a"] for ex in examples]

        evt_a_ids, evt_a_lengths = tx.data.padded_batch(
            [ex["evt_a_ids"] for ex in examples], pad_value=pad_token_id)

        evt_b = [ex["evt_b"] for ex in examples]
        evt_b_ids, evt_b_lengths = tx.data.padded_batch(
            [ex["evt_b_ids"] for ex in examples], pad_value=pad_token_id)

        score = [ex["score"] for ex in examples]

        return tx.data.Batch(
            len(examples),
            evt_a=evt_a,
            evt_a_ids=torch.from_numpy(evt_a_ids),
            evt_a_lengths=torch.tensor(evt_a_lengths),
            evt_b=evt_b,
            evt_b_ids=torch.from_numpy(evt_b_ids),
            evt_b_lengths=torch.tensor(evt_b_lengths),
            score=score,
        )


class Vocab(tx.data.Vocab):
    def load(self, filename: str) \
            -> Tuple[Dict[int, str], Dict[str, int]]:

        with open(filename, "r") as vocab_file:
            vocab = list(line.strip() for line in vocab_file)
        added_special_tokens = ['<sep>', '<cls>']
        for w in [
                self._pad_token, self._bos_token, self._eos_token,
                self._unk_token
        ] + added_special_tokens:
            if w in vocab:
                raise BaseException(
                    f"special token {w} has been already involved in Vocab...")
        # Places _pad_token at the beginning to make sure it take index 0.
        vocab = [
            self._pad_token, self._bos_token, self._eos_token, self._unk_token
        ] + added_special_tokens + vocab
        # Must make sure this is consistent with the above line
        vocab_size = len(vocab)

        # Creates python maps to interface with python code
        id_to_token_map_py = dict(zip(range(vocab_size), vocab))
        token_to_id_map_py = dict(zip(vocab, range(vocab_size)))

        return id_to_token_map_py, token_to_id_map_py