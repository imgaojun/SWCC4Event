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
            for cnt,line in enumerate(f):
                if cnt >= 100000:
                    break
                example = json.loads(line.strip())
                if len(example['events']) < 2:
                    continue
                for idx,evt in enumerate(example['events']):

                    evt_q = evt
                    for i in range(len(example['events'])-idx-1):
                        evt_k = example['events'][idx+1+i]
                        yield evt_q,evt_k
                # for evt in example['events']:
                #     evt_q = evt
                #     evt_k = random.choice(example['events'])
                #     yield evt_q,evt_k 

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

    @property
    def vocab(self):
        return self._vocab
    
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

        evtq_s,evtq_p,evtq_o = evt_q[0].lower().split(' '), evt_q[1].lower().split(' '), evt_q[2].lower().split(' ')
        evtq_s_ids = self._vocab.map_tokens_to_ids_py(evtq_s)
        evtq_p_ids = self._vocab.map_tokens_to_ids_py(evtq_p)
        evtq_o_ids = self._vocab.map_tokens_to_ids_py(evtq_o)

        evtk_s,evtk_p,evtk_o = evt_k[0].lower().split(' '), evt_k[1].lower().split(' '), evt_k[2].lower().split(' ')
        evtk_s_ids = self._vocab.map_tokens_to_ids_py(evtk_s)
        evtk_p_ids = self._vocab.map_tokens_to_ids_py(evtk_p)
        evtk_o_ids = self._vocab.map_tokens_to_ids_py(evtk_o)
        
        return {
            "evtq_s": evtq_s,
            "evtq_s_ids": evtq_s_ids,
            "evtq_p": evtq_p,
            "evtq_p_ids": evtq_p_ids,
            "evtq_o": evtq_o,
            "evtq_o_ids": evtq_o_ids,
            "evtk_s": evtk_s,
            "evtk_s_ids": evtk_s_ids,
            "evtk_p": evtk_p,
            "evtk_p_ids": evtk_p_ids,
            "evtk_o": evtk_o,
            "evtk_o_ids": evtk_o_ids,
         
      
        }


    def collate(self, examples: List[Example]) -> tx.data.Batch:
        evtq_s = [ex["evtq_s"] for ex in examples]
        evtq_s_ids, evtq_s_lengths = tx.data.padded_batch(
            [ex["evtq_s_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)
        evtq_p = [ex["evtq_p"] for ex in examples]
        evtq_p_ids, evtq_p_lengths = tx.data.padded_batch(
            [ex["evtq_p_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)
        evtq_o = [ex["evtq_o"] for ex in examples]
        evtq_o_ids, evtq_o_lengths = tx.data.padded_batch(
            [ex["evtq_o_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)

        evtk_s = [ex["evtk_s"] for ex in examples]
        evtk_s_ids, evtk_s_lengths = tx.data.padded_batch(
            [ex["evtk_s_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)
        evtk_p = [ex["evtk_p"] for ex in examples]
        evtk_p_ids, evtk_p_lengths = tx.data.padded_batch(
            [ex["evtk_p_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)
        evtk_o = [ex["evtk_o"] for ex in examples]
        evtk_o_ids, evtk_o_lengths = tx.data.padded_batch(
            [ex["evtk_o_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)



        return tx.data.Batch(
            len(examples),
            evtq_s=evtq_s,
            evtq_s_ids = torch.from_numpy(evtq_s_ids),
            evtq_s_lengths = torch.tensor(evtq_s_lengths),
            evtq_p=evtq_p,
            evtq_p_ids = torch.from_numpy(evtq_p_ids),
            evtq_p_lengths = torch.tensor(evtq_p_lengths),
            evtq_o=evtq_o,
            evtq_o_ids = torch.from_numpy(evtq_o_ids),
            evtq_o_lengths = torch.tensor(evtq_o_lengths),
            evtk_s=evtk_s,
            evtk_s_ids = torch.from_numpy(evtk_s_ids),
            evtk_s_lengths = torch.tensor(evtk_s_lengths),
            evtk_p=evtk_p,
            evtk_p_ids = torch.from_numpy(evtk_p_ids),
            evtk_p_lengths = torch.tensor(evtk_p_lengths),
            evtk_o=evtk_o,
            evtk_o_ids = torch.from_numpy(evtk_o_ids),
            evtk_o_lengths = torch.tensor(evtk_o_lengths),
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
        
        
        evta_s,evta_p,evta_o = evt_a[0].lower().split(' '), evt_a[1].lower().split(' '), evt_a[2].lower().split(' ')
        evta_s_ids = self._vocab.map_tokens_to_ids_py(evta_s)
        evta_p_ids = self._vocab.map_tokens_to_ids_py(evta_p)
        evta_o_ids = self._vocab.map_tokens_to_ids_py(evta_o)

        evtb_s,evtb_p,evtb_o = evt_b[0].lower().split(' '), evt_b[1].lower().split(' '), evt_b[2].lower().split(' ')
        evtb_s_ids = self._vocab.map_tokens_to_ids_py(evtb_s)
        evtb_p_ids = self._vocab.map_tokens_to_ids_py(evtb_p)
        evtb_o_ids = self._vocab.map_tokens_to_ids_py(evtb_o)

        evtc_s,evtc_p,evtc_o = evt_c[0].lower().split(' '), evt_c[1].lower().split(' '), evt_c[2].lower().split(' ')
        evtc_s_ids = self._vocab.map_tokens_to_ids_py(evtc_s)
        evtc_p_ids = self._vocab.map_tokens_to_ids_py(evtc_p)
        evtc_o_ids = self._vocab.map_tokens_to_ids_py(evtc_o)
        
        return {
            "evta_s": evta_s,
            "evta_s_ids": evta_s_ids,
            "evta_p": evta_p,
            "evta_p_ids": evta_p_ids,
            "evta_o": evta_o,
            "evta_o_ids": evta_o_ids,
            "evtb_s": evtb_s,
            "evtb_s_ids": evtb_s_ids,
            "evtb_p": evtb_p,
            "evtb_p_ids": evtb_p_ids,
            "evtb_o": evtb_o,
            "evtb_o_ids": evtb_o_ids,
            "evtc_s": evtc_s,
            "evtc_s_ids": evtc_s_ids,
            "evtc_p": evtc_p,
            "evtc_p_ids": evtc_p_ids,
            "evtc_o": evtc_o,
            "evtc_o_ids": evtc_o_ids,
        }


    def collate(self, examples: List[Example]) -> tx.data.Batch:
        evta_s = [ex["evta_s"] for ex in examples]
        evta_s_ids, evta_s_lengths = tx.data.padded_batch(
            [ex["evta_s_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)
        evta_p = [ex["evta_p"] for ex in examples]
        evta_p_ids, evta_p_lengths = tx.data.padded_batch(
            [ex["evta_p_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)
        evta_o = [ex["evta_o"] for ex in examples]
        evta_o_ids, evta_o_lengths = tx.data.padded_batch(
            [ex["evta_o_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)

        evtb_s = [ex["evtb_s"] for ex in examples]
        evtb_s_ids, evtb_s_lengths = tx.data.padded_batch(
            [ex["evtb_s_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)
        evtb_p = [ex["evtb_p"] for ex in examples]
        evtb_p_ids, evtb_p_lengths = tx.data.padded_batch(
            [ex["evtb_p_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)
        evtb_o = [ex["evtb_o"] for ex in examples]
        evtb_o_ids, evtb_o_lengths = tx.data.padded_batch(
            [ex["evtb_o_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)

        evtc_s = [ex["evtc_s"] for ex in examples]
        evtc_s_ids, evtc_s_lengths = tx.data.padded_batch(
            [ex["evtc_s_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)
        evtc_p = [ex["evtc_p"] for ex in examples]
        evtc_p_ids, evtc_p_lengths = tx.data.padded_batch(
            [ex["evtc_p_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)
        evtc_o = [ex["evtc_o"] for ex in examples]
        evtc_o_ids, evtc_o_lengths = tx.data.padded_batch(
            [ex["evtc_o_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)


        return tx.data.Batch(
            len(examples),
            evta_s=evta_s,
            evta_s_ids = torch.from_numpy(evta_s_ids),
            evta_s_lengths = torch.tensor(evta_s_lengths),
            evta_p=evta_p,
            evta_p_ids = torch.from_numpy(evta_p_ids),
            evta_p_lengths = torch.tensor(evta_p_lengths),
            evta_o=evta_o,
            evta_o_ids = torch.from_numpy(evta_o_ids),
            evta_o_lengths = torch.tensor(evta_o_lengths),
            evtb_s=evtb_s,
            evtb_s_ids = torch.from_numpy(evtb_s_ids),
            evtb_s_lengths = torch.tensor(evtb_s_lengths),
            evtb_p=evtb_p,
            evtb_p_ids = torch.from_numpy(evtb_p_ids),
            evtb_p_lengths = torch.tensor(evtb_p_lengths),
            evtb_o=evtb_o,
            evtb_o_ids = torch.from_numpy(evtb_o_ids),
            evtb_o_lengths = torch.tensor(evtb_o_lengths),
            evtc_s=evtc_s,
            evtc_s_ids = torch.from_numpy(evtc_s_ids),
            evtc_s_lengths = torch.tensor(evtc_s_lengths),
            evtc_p=evtc_p,
            evtc_p_ids = torch.from_numpy(evtc_p_ids),
            evtc_p_lengths = torch.tensor(evtc_p_lengths),
            evtc_o=evtc_o,
            evtc_o_ids = torch.from_numpy(evtc_o_ids),
            evtc_o_lengths = torch.tensor(evtc_o_lengths),
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
        
        evta_s,evta_p,evta_o = evt_a[0].lower().split(' '), evt_a[1].lower().split(' '), evt_a[2].lower().split(' ')
        evta_s_ids = self._vocab.map_tokens_to_ids_py(evta_s)
        evta_p_ids = self._vocab.map_tokens_to_ids_py(evta_p)
        evta_o_ids = self._vocab.map_tokens_to_ids_py(evta_o)

        evtb_s,evtb_p,evtb_o = evt_b[0].lower().split(' '), evt_b[1].lower().split(' '), evt_b[2].lower().split(' ')
        evtb_s_ids = self._vocab.map_tokens_to_ids_py(evtb_s)
        evtb_p_ids = self._vocab.map_tokens_to_ids_py(evtb_p)
        evtb_o_ids = self._vocab.map_tokens_to_ids_py(evtb_o)

        
        return {
            "evta_s": evta_s,
            "evta_s_ids": evta_s_ids,
            "evta_p": evta_p,
            "evta_p_ids": evta_p_ids,
            "evta_o": evta_o,
            "evta_o_ids": evta_o_ids,
            "evtb_s": evtb_s,
            "evtb_s_ids": evtb_s_ids,
            "evtb_p": evtb_p,
            "evtb_p_ids": evtb_p_ids,
            "evtb_o": evtb_o,
            "evtb_o_ids": evtb_o_ids,
            "score": score,
        }


    def collate(self, examples: List[Example]) -> tx.data.Batch:
        evta_s = [ex["evta_s"] for ex in examples]
        evta_s_ids, evta_s_lengths = tx.data.padded_batch(
            [ex["evta_s_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)
        evta_p = [ex["evta_p"] for ex in examples]
        evta_p_ids, evta_p_lengths = tx.data.padded_batch(
            [ex["evta_p_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)
        evta_o = [ex["evta_o"] for ex in examples]
        evta_o_ids, evta_o_lengths = tx.data.padded_batch(
            [ex["evta_o_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)

        evtb_s = [ex["evtb_s"] for ex in examples]
        evtb_s_ids, evtb_s_lengths = tx.data.padded_batch(
            [ex["evtb_s_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)
        evtb_p = [ex["evtb_p"] for ex in examples]
        evtb_p_ids, evtb_p_lengths = tx.data.padded_batch(
            [ex["evtb_p_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)
        evtb_o = [ex["evtb_o"] for ex in examples]
        evtb_o_ids, evtb_o_lengths = tx.data.padded_batch(
            [ex["evtb_o_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)

        score = [ex["score"] for ex in examples]
        


        return tx.data.Batch(
            len(examples),
            evta_s=evta_s,
            evta_s_ids = torch.from_numpy(evta_s_ids),
            evta_s_lengths = torch.tensor(evta_s_lengths),
            evta_p=evta_p,
            evta_p_ids = torch.from_numpy(evta_p_ids),
            evta_p_lengths = torch.tensor(evta_p_lengths),
            evta_o=evta_o,
            evta_o_ids = torch.from_numpy(evta_o_ids),
            evta_o_lengths = torch.tensor(evta_o_lengths),
            evtb_s=evtb_s,
            evtb_s_ids = torch.from_numpy(evtb_s_ids),
            evtb_s_lengths = torch.tensor(evtb_s_lengths),
            evtb_p=evtb_p,
            evtb_p_ids = torch.from_numpy(evtb_p_ids),
            evtb_p_lengths = torch.tensor(evtb_p_lengths),
            evtb_o=evtb_o,
            evtb_o_ids = torch.from_numpy(evtb_o_ids),
            evtb_o_lengths = torch.tensor(evtb_o_lengths),
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