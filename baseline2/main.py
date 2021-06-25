import argparse
import importlib
from typing import Any

import torch
import torch.nn as nn
from texar.torch.run import *
from pathlib import Path
import misc_utils
from misc_utils import init_logger, logger

import texar.torch as tx

from model import AdCo, Adversary_Negatives
import data_utils
import numpy as np


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
    "--output-dir", type=str, default="./outputs/",
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

output_dir = Path(args.output_dir)
tx.utils.maybe_create_dir(output_dir)

init_logger(output_dir/args.log_file)

def main() -> None:
    

    train_data = data_utils.TrainData(config_data.train_hparams,device=device)
    train_data_iterator = tx.data.DataIterator(train_data)

    hard_data = data_utils.HardData(config_data.hard_hparams,device=device)
    hard_data_iterator = tx.data.DataIterator(hard_data)

    

    hardext_data = data_utils.HardData(config_data.hardext_hparams,device=device)
    hardext_data_iterator = tx.data.DataIterator(hardext_data)

    trans_data = data_utils.TransData(config_data.trans_hparams,device=device)
    trans_data_iterator = tx.data.DataIterator(trans_data)

    Memory_Bank = Adversary_Negatives(config_model.bank_size,config_model.hidden_dim)
    Memory_Bank.to(device)
    model = AdCo(config_model=config_model, vocab_size=train_data.vocab_size)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)

    optim = torch.optim.SGD(model.parameters(), config_model.lr,
                                momentum=config_model.momentum,
                                weight_decay=config_model.weight_decay)

    def _update(batch):
        q, k = model(batch)
        
        l_pos = torch.einsum('nc,ck->nk', [q, k.T])
        
        d_norm, d, l_neg = Memory_Bank(q)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= config_model.moco_t

        
        batch_size = batch.batch_size
        labels = torch.arange(0, batch_size, dtype=torch.long).to(device)

        loss = loss_fn(logits, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()

        with torch.no_grad():
            logits = torch.cat([l_pos, l_neg], dim=1) / config_model.mem_t
            p_qd=nn.functional.softmax(logits, dim=1)[:,batch_size:]
            g = torch.einsum('cn,nk->ck',[q.T,p_qd])/logits.shape[0] - torch.mul(torch.mean(torch.mul(p_qd,l_neg),dim=0),d_norm)
            g = -torch.div(g,torch.norm(d,dim=0))/config_model.mem_t # c*k
            Memory_Bank.v.data = config_model.momentum * Memory_Bank.v.data + g + config_model.mem_wd * Memory_Bank.W.data
            Memory_Bank.W.data = Memory_Bank.W.data - config_model.memory_lr * Memory_Bank.v.data
        # logits=torch.softmax(logits,dim=1)
        # batch_prob=torch.sum(logits[:,:logits.size(0)],dim=1)
        # batch_prob=torch.mean(batch_prob)
        # return l_neg, logits, loss.item()
        return {'loss': loss.item()}
    
    def _save_epoch(epoch):
        logger.info("saving model...")
        torch.save(model.state_dict(), output_dir/f"checkpoint{epoch}.pt")
    def _eval_model():
        model.eval()
        hard_results = []
        for batch in hard_data_iterator:
            evt_a = model.encoder_q(batch.evt_a_ids, batch.evt_a_lengths).tolist()[0]
            evt_b = model.encoder_q(batch.evt_b_ids, batch.evt_b_lengths).tolist()[0]
            evt_c = model.encoder_q(batch.evt_c_ids, batch.evt_c_lengths).tolist()[0]
            ab_sim = misc_utils.cosine_similarity(evt_a, evt_b)
            ac_sim = misc_utils.cosine_similarity(evt_a, evt_c)
            if ab_sim > ac_sim:
                hard_results.append(1)
            else:
                hard_results.append(0)
        hard_results = np.array(hard_results)
        hardext_results = []
        for batch in hardext_data_iterator:
            evt_a = model.encoder_q(batch.evt_a_ids, batch.evt_a_lengths).tolist()[0]
            evt_b = model.encoder_q(batch.evt_b_ids, batch.evt_b_lengths).tolist()[0]
            evt_c = model.encoder_q(batch.evt_c_ids, batch.evt_c_lengths).tolist()[0]
            ab_sim = misc_utils.cosine_similarity(evt_a, evt_b)
            ac_sim = misc_utils.cosine_similarity(evt_a, evt_c)
            if ab_sim > ac_sim:
                hardext_results.append(1)
            else:
                hardext_results.append(0)
        hardext_results = np.array(hardext_results)
        trans_results = []
        human_scores = []
        for batch in trans_data_iterator:
            evt_a = model.encoder_q(batch.evt_a_ids, batch.evt_a_lengths).tolist()[0]
            evt_b = model.encoder_q(batch.evt_b_ids, batch.evt_b_lengths).tolist()[0]
            ab_sim = misc_utils.cosine_similarity(evt_a, evt_b)
            trans_results.append(ab_sim)
            human_scores.append(batch.score)
        trans_corr, _ = misc_utils.spearmanr(trans_results,human_scores)
        logger.info(f"Hard: {hard_results.mean():.4f} | Hard Ext: {hardext_results.mean():.4f} | Transitive: {trans_corr}")
    def _train_epoch(epoch):
        model.train()
        step = 0
        avg_rec = tx.utils.AverageRecorder()
        for batch in train_data_iterator:
            return_dict = _update(batch)
            avg_rec.add(return_dict)
            if step % config_data.display_steps == 0:
                logger.info(f"epoch: {epoch} | step: {step} | {avg_rec.to_str(precision=4, delimiter=' | ')}")
                avg_rec.reset()
            if step % config_data.eval_steps == 0:
                _eval_model()
                model.train()
            step += 1
    if args.do_train:
        logger.info(f"start training...")

        for epoch in range(config_data.max_train_epoch):
            _train_epoch(epoch)
            _save_epoch(epoch)

    elif args.do_eval:
        print(f"start testing...")
        model.load_state_dict(torch.load(args.checkpoint))
        model.eval()
        hard_results = []
        for batch in hard_data_iterator:
            evt_a = model.encoder_q(batch.evt_a_ids, batch.evt_a_lengths).tolist()[0]
            evt_b = model.encoder_q(batch.evt_b_ids, batch.evt_b_lengths).tolist()[0]
            evt_c = model.encoder_q(batch.evt_c_ids, batch.evt_c_lengths).tolist()[0]
            ab_sim = misc_utils.cosine_similarity(evt_a, evt_b)
            ac_sim = misc_utils.cosine_similarity(evt_a, evt_c)
            if ab_sim > ac_sim:
                hard_results.append(1)
            else:
                hard_results.append(0)
        hard_results = np.array(hard_results)
        hardext_results = []
        for batch in hardext_data_iterator:
            evt_a = model.encoder_q(batch.evt_a_ids, batch.evt_a_lengths).tolist()[0]
            evt_b = model.encoder_q(batch.evt_b_ids, batch.evt_b_lengths).tolist()[0]
            evt_c = model.encoder_q(batch.evt_c_ids, batch.evt_c_lengths).tolist()[0]
            ab_sim = misc_utils.cosine_similarity(evt_a, evt_b)
            ac_sim = misc_utils.cosine_similarity(evt_a, evt_c)
            if ab_sim > ac_sim:
                hardext_results.append(1)
            else:
                hardext_results.append(0)
        hardext_results = np.array(hardext_results)
        trans_results = []
        human_scores = []
        for batch in trans_data_iterator:
            evt_a = model.encoder_q(batch.evt_a_ids, batch.evt_a_lengths).tolist()[0]
            evt_b = model.encoder_q(batch.evt_b_ids, batch.evt_b_lengths).tolist()[0]
            ab_sim = misc_utils.cosine_similarity(evt_a, evt_b)
            trans_results.append(ab_sim)
            human_scores.append(batch.score)
        trans_corr, _ = misc_utils.spearmanr(trans_results,human_scores)
        print(f"Hard: {hard_results.mean():.4f} | Hard Ext: {hardext_results.mean():.4f} | Transitive: {trans_corr}")

if __name__ == '__main__':
    main()