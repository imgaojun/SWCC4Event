from typing import List, Optional, Tuple, Iterator

import numpy as np
import torch

import texar.torch as tx
from texar.torch.hyperparams import HParams
import random
import json
from tqdm import tqdm
Example = Tuple[np.ndarray, np.ndarray]

class TextLineDataSource(tx.data.TextLineDataSource):
    def __iter__(self) -> Iterator[List[str]]:
        for path in self._file_paths:
            with self._open_file(path) as f:
                for line in f:
                    example = json.loads(line.strip())
                    if len(example['events']) < 2:
                        continue
                    for evt in example['events']:
                        raw_text = example['text']
                        if len(raw_text.split(' ')) > 512:
                            continue
                        evt_q = evt
                        evt_k = random.choice(example['events'])
                        yield evt_q,evt_k 
                        

class TrainData(tx.data.DatasetBase[Example, Example]):
    def __init__(self, hparams=None,
                 device: Optional[torch.device] = None):
        self._hparams = HParams(hparams, self.default_hparams())
        data_source = TextLineDataSource(
                    self._hparams.dataset.files,
                    compression_type=self._hparams.dataset.compression_type)
        self.tokenizer = tx.data.GPT2Tokenizer(pretrained_model_name='gpt2-small')
        special_tokens=['<sub>','<pred>','<obj>','<cls>']
        self.tokenizer.add_tokens(special_tokens)
        self.vocab_size = len(self.tokenizer.encoder) + len(self.tokenizer.added_tokens_encoder)
     
        super().__init__(data_source, hparams, device=device)

    
    @staticmethod
    def default_hparams():
        return {
            **tx.data.DatasetBase.default_hparams(),
            'dataset': { 'files': 'data.txt',
                        'compression_type':None},
        }

    def process(self, raw_example):
        # sub_ids, pred_ids, obj_ids = self.tokenizer.map_text_to_id(raw_example[0][0]), \
        #     self.tokenizer.map_text_to_id(raw_example[0][1]), \
        #     self.tokenizer.map_text_to_id(raw_example[0][2])
        
        evt_q,evt_k = raw_example[0], raw_example[1]
        evt_q = ['<cls>']+[evt_q[0]] +['<sub>']+ [evt_q[1]] +['<pred>']+ [evt_q[2]]+['<obj>']
        evt_q = ' '.join(evt_q)
        evt_q_ids = self.tokenizer.map_text_to_id(evt_q)

        evt_k = ['<cls>']+[evt_k[0]] +['<sub>']+ [evt_k[1]] +['<pred>']+ [evt_k[2]]+['<obj>']
        evt_k = ' '.join(evt_k)
        evt_k_ids = self.tokenizer.map_text_to_id(evt_k)
        
        return {
            "evt_q": evt_q,
            "evt_q_ids": evt_q_ids,
            "evt_k": evt_k,
            "evt_k_ids": evt_k_ids,
      
        }


    def collate(self, examples: List[Example]) -> tx.data.Batch:
        evt_q = [ex["evt_q"] for ex in examples]
        evt_q_ids, evt_q_lengths = tx.data.padded_batch(
            [ex["evt_q_ids"] for ex in examples], pad_value=self.tokenizer.map_token_to_id(self.tokenizer.pad_token))

        evt_k = [ex["evt_k"] for ex in examples]
        evt_k_ids, evt_k_lengths = tx.data.padded_batch(
            [ex["evt_k_ids"] for ex in examples], pad_value=self.tokenizer.map_token_to_id(self.tokenizer.pad_token))


        return tx.data.Batch(
            len(examples),
            evt_q=evt_q,
            evt_q_ids = torch.tensor(evt_q_ids),
            evt_q_lengths = torch.tensor(evt_q_lengths),
            evt_k=evt_k,
            evt_k_ids = torch.tensor(evt_k_ids),
            evt_k_lengths = torch.tensor(evt_k_lengths),
        )

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