import argparse
from functools import reduce
import importlib
from typing import Any
from warnings import catch_warnings

import torch
import torch.nn as nn
from texar.torch.run import *
from pathlib import Path
import misc_utils
from misc_utils import init_logger, logger

import texar.torch as tx

from model import AdCo
import data_utils
import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

import pickle
from tqdm import tqdm
import csv
parser = argparse.ArgumentParser()
parser.add_argument(
    '--config-model', type=str, default="config_model",
    help="The model config.")
parser.add_argument(
    '--config-data', type=str, default="config_data",
    help="The dataset config.")
parser.add_argument(
    "--do-train", action="store_true", help="Whether to run training.")
parser.add_argument(
    "--do-eval", action="store_true",
    help="Whether to run eval on the dev set.")
parser.add_argument(
    "--output-dir", type=str, default="./outputs1/",
    help="Path to save the trained model and logs.")
parser.add_argument(
    "--log-file", type=str, default="exp.log",
    help="Path to save the trained model and logs.")

parser.add_argument(
    '--checkpoint', type=str, default=None,
    help="Model checkpoint to load model weights from.")
args = parser.parse_args()

config_model: Any = importlib.import_module(args.config_model)
config_data: Any = importlib.import_module(args.config_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

make_deterministic(config_model.random_seed)

train_data = data_utils.TrainData(config_data.train_hparams,device=device)
model = AdCo(config_model=config_model, config_data=config_data)

if args.checkpoint:
    print(f"loading checkpoint {args.checkpoint}...")
    model.load_state_dict(torch.load(args.checkpoint)['model'])
model.to(device)

tokenizer = tx.data.BERTTokenizer(pretrained_model_name="bert-base-uncased")
test_data = pickle.load(open("/data_utils/mcnc/corpus_index_test", "rb"))
# cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
def get_events(raw_list):
    all_events = []
    for ex in raw_list:
        event = []
        selected_id = [3,0,4,5]
        for i in selected_id:
            if ex[i] is not None:
                event.append(ex[i].replace('+',' '))
        all_events.append(event)
    return all_events

def convert_events_to_embeddings(events):
    events = [data_utils.map_evt_to_tokens_for_text(evt) for evt in events]    
    evt_ids = [tokenizer.map_text_to_id(evt) for evt in events]
    evt_ids, evt_lengths = tx.data.padded_batch(
            evt_ids, pad_value=tokenizer.map_token_to_id(tokenizer.pad_token))

    evt_ids = torch.from_numpy(evt_ids).to(device)
    evt_lengths = torch.tensor(evt_lengths).to(device)
    # evt_emb = model.encoder_q.get_glove_embedding(evt_ids,evt_lengths)
    evt_emb = model.encoder_q(evt_ids,evt_lengths)
    return evt_emb

model.eval()
results = {1:[],2:[],3:[],4:[],5:[]}

with open('outputs.csv', 'w', newline='') as csvfile:
    fieldnames = ['context', 'choices', 'answer', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for example in tqdm(test_data):
        context = get_events(example[0])
        candidates = get_events(example[1])
        label = example[2]

        context_embs = convert_events_to_embeddings(context)
        cand_embs = convert_events_to_embeddings(candidates)
        context_emb = context_embs.sum(dim=0,keepdim=True)
        # sim_scores = cosine_similarity(cand_embs.tolist(), cand_embs.tolist())
        sim_scores = cosine_similarity(context_emb.tolist(), cand_embs.tolist())

        # sim_scores = sim_scores.mean(axis=1)
        
        topk_preds = np.argsort(-sim_scores[0])
        # print(topk_preds)
        # raise BaseException("break")
        pred = topk_preds[0]
        if pred == label:
            writer.writerow({'context': '\n'.join([' '.join(ex) for ex in context]), 'choices': '\n'.join([' '.join(ex) for ex in candidates]), 'answer': ' '.join(candidates[label]),'label':1})
        else:
            writer.writerow({'context': '\n'.join([' '.join(ex) for ex in context]), 'choices': '\n'.join([' '.join(ex) for ex in candidates]), 'answer': ' '.join(candidates[label]),'label':0})
        for k in range(1,6):
            if label in topk_preds[:k]:
                results[k].append(1)
            else:
                results[k].append(0)
    print(sum(results[1])/len(results[1]))
    print(sum(results[2])/len(results[2]))
    print(sum(results[3])/len(results[3]))
    print(sum(results[4])/len(results[4]))
    print(sum(results[5])/len(results[5]))

