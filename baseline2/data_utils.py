from typing import List, Optional, Tuple, Iterator, Dict

import numpy as np
import torch

import texar.torch as tx
from texar.torch.hyperparams import HParams
import random
import json
from tqdm import tqdm
from pathlib import Path
Example = Tuple[np.ndarray, np.ndarray]

class TrainDataSource(tx.data.TextLineDataSource):
    def __iter__(self) -> Iterator[List[str]]:
        path = Path(self._file_paths[0])
        with self._open_file(path) as f:
            for line in f:
                example = json.loads(line.strip())
                if len(example['events']) < 2:
                    continue
                for evt in example['events']:
                    raw_text = example['text']
                    if len(raw_text.split(' ')) > 128:
                        continue
                    evt_q = evt
                    evt_k = random.choice(example['events'])
                    yield evt_q,evt_k 

class HardDataSource(tx.data.TextLineDataSource):
    def __iter__(self) -> Iterator[List[str]]:
        for path in self._file_paths:
            with self._open_file(path) as f:
                for line in f:
                    tokens = line.strip().split(' | ')
                    evt_a = [tokens[0], tokens[1], tokens[2]]
                    evt_b = [tokens[3], tokens[4], tokens[5]]
                    evt_c = [tokens[9], tokens[10], tokens[11]]
                    yield evt_a, evt_b, evt_c

class TransDataSource(tx.data.TextLineDataSource):
    def __iter__(self) -> Iterator[List[str]]:
        for path in self._file_paths:
            with self._open_file(path) as f:
                for line in f:
                    tokens = line.strip().split(' | ')
                    evt_a = [tokens[0], tokens[1], tokens[2]]
                    evt_b = [tokens[3], tokens[4], tokens[5]]
                    score = float(tokens[6])
                    yield evt_a, evt_b, score

class TrainData(tx.data.DatasetBase[Example, Example]):
    def __init__(self, hparams=None,
                 device: Optional[torch.device] = None):
        self._hparams = HParams(hparams, self.default_hparams())
        data_source = TrainDataSource(
                    self._hparams.dataset.files,
                    compression_type=self._hparams.dataset.compression_type)
        self._vocab = Vocab(self._hparams.dataset.vocab_file)
        self.vocab_size = self._vocab.size
     
        super().__init__(data_source, hparams, device=device)

    
    @staticmethod
    def default_hparams():
        return {
            **tx.data.DatasetBase.default_hparams(),
            'dataset': { 'files': 'data.txt',
                        'compression_type':None,
                        'vocab_file':'vocab.txt'},
        }

    def process(self, raw_example):
    
        evt_q,evt_k = raw_example[0], raw_example[1]
        # evt_q = ['<cls>']+[evt_q[0]] +['<sub>']+ [evt_q[1]] +['<pred>']+ [evt_q[2]]+['<obj>']
        evt_q = ['<cls>']+[evt_q[0]] +[evt_q[1]] + [evt_q[2]]
        evt_q = ' '.join(evt_q).lower().split(' ')

        evt_q_ids = self._vocab.map_tokens_to_ids_py(evt_q)

        # evt_k = ['<cls>']+[evt_k[0]] +['<sub>']+ [evt_k[1]] +['<pred>']+ [evt_k[2]]+['<obj>']
        evt_k = ['<cls>']+[evt_k[0]] + [evt_k[1]] + [evt_k[2]]

        evt_k = ' '.join(evt_k).lower().split(' ')
        evt_k_ids = self._vocab.map_tokens_to_ids_py(evt_k)
        
        return {
            "evt_q": evt_q,
            "evt_q_ids": evt_q_ids,
            "evt_k": evt_k,
            "evt_k_ids": evt_k_ids,
      
        }


    def collate(self, examples: List[Example]) -> tx.data.Batch:
        evt_q = [ex["evt_q"] for ex in examples]
        evt_q_ids, evt_q_lengths = tx.data.padded_batch(
            [ex["evt_q_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)

        evt_k = [ex["evt_k"] for ex in examples]
        evt_k_ids, evt_k_lengths = tx.data.padded_batch(
            [ex["evt_k_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)


        return tx.data.Batch(
            len(examples),
            evt_q=evt_q,
            evt_q_ids = torch.from_numpy(evt_q_ids),
            evt_q_lengths = torch.tensor(evt_q_lengths),
            evt_k=evt_k,
            evt_k_ids = torch.from_numpy(evt_k_ids),
            evt_k_lengths = torch.tensor(evt_k_lengths),
        )

class HardData(tx.data.DatasetBase[Example, Example]):
    def __init__(self, hparams=None,
                 device: Optional[torch.device] = None):
        self._hparams = HParams(hparams, self.default_hparams())
        data_source = HardDataSource(
                    self._hparams.dataset.files,
                    compression_type=self._hparams.dataset.compression_type)
        self._vocab = Vocab(self._hparams.dataset.vocab_file)
        super().__init__(data_source, hparams, device=device)

    
    @staticmethod
    def default_hparams():
        return {
            **tx.data.DatasetBase.default_hparams(),
            'dataset': { 'files': 'data.txt',
                        'compression_type':None,
                        'vocab_file':'vocab.txt'},
        }

    def process(self, raw_example):
        
        evt_a,evt_b,evt_c  = raw_example[0], raw_example[1], raw_example[2]
        # evt_a = ['<cls>']+[evt_a[0]] +['<sub>']+ [evt_a[1]] +['<pred>']+ [evt_a[2]]+['<obj>']
        evt_a = ['<cls>']+[evt_a[0]] +[evt_a[1]] + [evt_a[2]]
        evt_a = ' '.join(evt_a).lower().split(' ')
        evt_a_ids = self._vocab.map_tokens_to_ids_py(evt_a)

        # evt_b = ['<cls>']+[evt_b[0]] +['<sub>']+ [evt_b[1]] +['<pred>']+ [evt_b[2]]+['<obj>']
        evt_b = ['<cls>']+[evt_b[0]] +[evt_b[1]] + [evt_b[2]]
        evt_b = ' '.join(evt_b).lower().split(' ')
        evt_b_ids = self._vocab.map_tokens_to_ids_py(evt_b)

        # evt_c = ['<cls>']+[evt_c[0]] +['<sub>']+ [evt_c[1]] +['<pred>']+ [evt_c[2]]+['<obj>']
        evt_c = ['<cls>']+[evt_c[0]] +[evt_c[1]] + [evt_c[2]]
        evt_c = ' '.join(evt_c).lower().split(' ')
        evt_c_ids = self._vocab.map_tokens_to_ids_py(evt_c)
        
        return {
            "evt_a": evt_a,
            "evt_a_ids": evt_a_ids,
            "evt_b": evt_b,
            "evt_b_ids": evt_b_ids,
            "evt_c": evt_c,
            "evt_c_ids": evt_c_ids,
        }


    def collate(self, examples: List[Example]) -> tx.data.Batch:
        evt_a = [ex["evt_a"] for ex in examples]
        evt_a_ids, evt_a_lengths = tx.data.padded_batch(
            [ex["evt_a_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)

        evt_b = [ex["evt_b"] for ex in examples]
        evt_b_ids, evt_b_lengths = tx.data.padded_batch(
            [ex["evt_b_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)

        evt_c = [ex["evt_c"] for ex in examples]
        evt_c_ids, evt_c_lengths = tx.data.padded_batch(
            [ex["evt_c_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)


        return tx.data.Batch(
            len(examples),
            evt_a=evt_a,
            evt_a_ids = torch.from_numpy(evt_a_ids),
            evt_a_lengths = torch.tensor(evt_a_lengths),
            evt_b=evt_b,
            evt_b_ids = torch.from_numpy(evt_b_ids),
            evt_b_lengths = torch.tensor(evt_b_lengths),
            evt_c=evt_c,
            evt_c_ids = torch.from_numpy(evt_c_ids),
            evt_c_lengths = torch.tensor(evt_c_lengths),
        )

class TransData(tx.data.DatasetBase[Example, Example]):
    def __init__(self, hparams=None,
                 device: Optional[torch.device] = None):
        self._hparams = HParams(hparams, self.default_hparams())
        data_source = TransDataSource(
                    self._hparams.dataset.files,
                    compression_type=self._hparams.dataset.compression_type)
        self._vocab = Vocab(self._hparams.dataset.vocab_file)
        super().__init__(data_source, hparams, device=device)

    
    @staticmethod
    def default_hparams():
        return {
            **tx.data.DatasetBase.default_hparams(),
            'dataset': { 'files': 'data.txt',
                        'compression_type':None,
                        'vocab_file':'vocab.txt'},
        }

    def process(self, raw_example):
        
        evt_a,evt_b,score  = raw_example[0], raw_example[1], raw_example[2]
        # evt_a = ['<cls>']+[evt_a[0]] +['<sub>']+ [evt_a[1]] +['<pred>']+ [evt_a[2]]+['<obj>']
        evt_a = ['<cls>']+[evt_a[0]] +[evt_a[1]] + [evt_a[2]]
        evt_a = ' '.join(evt_a).lower().split(' ')

        evt_a_ids = self._vocab.map_tokens_to_ids_py(evt_a)

        # evt_b = ['<cls>']+[evt_b[0]] +['<sub>']+ [evt_b[1]] +['<pred>']+ [evt_b[2]]+['<obj>']
        evt_b = ['<cls>']+[evt_b[0]] +[evt_b[1]] + [evt_b[2]]
        evt_b = ' '.join(evt_b).lower().split(' ')

        evt_b_ids = self._vocab.map_tokens_to_ids_py(evt_b)

        
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
            [ex["evt_a_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)

        evt_b = [ex["evt_b"] for ex in examples]
        evt_b_ids, evt_b_lengths = tx.data.padded_batch(
            [ex["evt_b_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)

        score = [ex["score"] for ex in examples]
        


        return tx.data.Batch(
            len(examples),
            evt_a=evt_a,
            evt_a_ids = torch.from_numpy(evt_a_ids),
            evt_a_lengths = torch.tensor(evt_a_lengths),
            evt_b=evt_b,
            evt_b_ids = torch.from_numpy(evt_b_ids),
            evt_b_lengths = torch.tensor(evt_b_lengths),
            score=score,
        )

class Vocab(tx.data.Vocab):
    def load(self, filename: str) \
            -> Tuple[Dict[int, str], Dict[str, int]]:

        with open(filename, "r") as vocab_file:
            vocab = list(line.strip() for line in vocab_file)
        added_special_tokens=['<sub>','<pred>','<obj>','<cls>']
        # Places _pad_token at the beginning to make sure it take index 0.
        vocab = [self._pad_token, self._bos_token, self._eos_token,
                 self._unk_token] + added_special_tokens + vocab
        # Must make sure this is consistent with the above line
        vocab_size = len(vocab)

        # Creates python maps to interface with python code
        id_to_token_map_py = dict(zip(range(vocab_size), vocab))
        token_to_id_map_py = dict(zip(vocab, range(vocab_size)))

        return id_to_token_map_py, token_to_id_map_py

if __name__ == "__main__":
    hparams={
        'dataset': { 'files': '/apdcephfs/share_916081/changlongyu/nyt_corpus/nyt_events_all.jsonl'},
        'batch_size': 10,
        'lazy_strategy': 'all',
        'num_parallel_calls': 10,
        'shuffle': False
    }
    data = TrainData(hparams)
    iterator = tx.data.DataIterator(data)

    for batch in iterator:
        print(batch)
        break