import argparse
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

from model import AdCo, Adversary_Negatives, MarginLoss
import data_utils
import numpy as np
import torch.nn.functional as F

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
    "--output-dir", type=str, default="./outputs5/",
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

    # Memory_Bank = Adversary_Negatives(config_model.bank_size,config_model.hidden_dim)
    # Memory_Bank.to(device)
    model = AdCo(config_model=config_model, config_data=config_data, vocab=train_data.vocab)
    model.to(device)

    nce_loss_fn = nn.CrossEntropyLoss()
    nce_loss_fn.to(device)

    margin_loss_fn = MarginLoss(0.5)
    margin_loss_fn.to(device)
    error_case = {}

    # optim = torch.optim.SGD(model.parameters(), config_model.lr,
    #                             momentum=config_model.momentum,
    #                             weight_decay=config_model.weight_decay)
    optim = torch.optim.Adagrad(
            model.parameters(), lr=config_model.learning_rate, initial_accumulator_value=config_model.initial_accumulator_value,weight_decay=config_model.weight_decay)

    def _update(batch):
        q, k, n = model(batch)
        l_pos = torch.einsum('nc,nc->n', [q, k])
        # weights = F.softmax(torch.einsum('nc,ck->nk', [k, n.T]), dim=-1)
        l_neg = torch.einsum('nc,ck->nk', [q, n.T])
        # weighted_l_neg = weights * l_neg
        # d_norm, d, l_neg = Memory_Bank(q)
        # logits = torch.cat([l_pos.unsqueeze(-1), l_neg.clone().detach(), l_neg_ext], dim=1)
        logits = torch.cat([l_pos.unsqueeze(-1), l_neg], dim=1)

        logits /= config_model.moco_t


        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        nce_loss = nce_loss_fn(logits, labels)
        loss = nce_loss
        optim.zero_grad()
        loss.backward()
        optim.step()

        # with torch.no_grad():
        #     logits = torch.cat([l_pos, l_neg], dim=1) / config_model.mem_t
        #     p_qd=nn.functional.softmax(logits, dim=1)[:,batch_size:]
        #     g = torch.einsum('cn,nk->ck',[q.T,p_qd])/logits.shape[0] - torch.mul(torch.mean(torch.mul(p_qd,l_neg),dim=0),d_norm)
        #     g = -torch.div(g,torch.norm(d,dim=0))/config_model.mem_t # c*k
        #     Memory_Bank.v.data = config_model.momentum * Memory_Bank.v.data + g + config_model.mem_wd * Memory_Bank.W.data
        #     Memory_Bank.W.data = Memory_Bank.W.data - config_model.memory_lr * Memory_Bank.v.data
        # # logits=torch.softmax(logits,dim=1)
        # batch_prob=torch.sum(logits[:,:logits.size(0)],dim=1)
        # batch_prob=torch.mean(batch_prob)
        # return l_neg, logits, loss.item()
        return {'loss': loss.item()}
    
    def _save_step(step):
        logger.info(f"saving model...step {step}")
        torch.save(model.state_dict(), output_dir/f"checkpoint_last.pt")
    def _save_best_model(results):
        logger.info(f"saving model... {results[0]:.2f}_{results[1]:.2f}")
        torch.save(model.state_dict(), output_dir/f"checkpoint_{results[0]:.2f}_{results[1]:.2f}.pt")
        torch.save(model.state_dict(), output_dir/f"best_last.pt")
    def _eval_model():
        model.eval()

        cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        hard_results = []
        for batch in hard_data_iterator:
            evt_a = model.encoder_q(batch.evt_a_ids, batch.evt_a_lengths)
            evt_b = model.encoder_q(batch.evt_b_ids, batch.evt_b_lengths)
            evt_c = model.encoder_q(batch.evt_c_ids, batch.evt_c_lengths)
            evt_d = model.encoder_q(batch.evt_d_ids, batch.evt_d_lengths)
            ab_sim = cosine_similarity(evt_a, evt_b)
            cd_sim = cosine_similarity(evt_c, evt_d)

            ret = ab_sim > cd_sim
            hard_results += ret.tolist()
        hard_results = np.array(hard_results)
    
        hardext_results = []
        for batch in hardext_data_iterator:
            evt_a = model.encoder_q(batch.evt_a_ids, batch.evt_a_lengths)
            evt_b = model.encoder_q(batch.evt_b_ids, batch.evt_b_lengths)
            evt_c = model.encoder_q(batch.evt_c_ids, batch.evt_c_lengths)
            evt_d = model.encoder_q(batch.evt_d_ids, batch.evt_d_lengths)
            ab_sim = cosine_similarity(evt_a, evt_b)
            cd_sim = cosine_similarity(evt_c, evt_d)
            ret = ab_sim > cd_sim
            hardext_results += ret.tolist()
        hardext_results = np.array(hardext_results)
        trans_results = []
        human_scores = []
        for batch in trans_data_iterator:
            evt_a = model.encoder_q(batch.evt_a_ids, batch.evt_a_lengths)
            evt_b = model.encoder_q(batch.evt_b_ids, batch.evt_b_lengths)
            ab_sim = cosine_similarity(evt_a, evt_b)
            trans_results += ab_sim.tolist()
            human_scores  += batch.score
        trans_corr, _ = misc_utils.spearmanr(trans_results,human_scores)

        all_results = [hard_results.mean(), hardext_results.mean()]
        logger.info(f"Hard: {hard_results.mean():.4f} | Hard Ext: {hardext_results.mean():.4f} | Transitive: {trans_corr:.4f}")
        return all_results
    def _train_epoch(epoch):
        model.train()
        step = 0
        best_eval_results = np.array([0.0,0.0])
        avg_rec = tx.utils.AverageRecorder()
        for batch in train_data_iterator:
            return_dict = _update(batch)
            avg_rec.add(return_dict)
            if step % config_data.display_steps == 0:
                logger.info(f"epoch: {epoch} | step: {step} | {avg_rec.to_str(precision=4, delimiter=' | ')}")
                avg_rec.reset()
            if step % config_data.eval_steps == 0:
                eval_results = _eval_model()
                eval_results = np.array(eval_results)
                if all(eval_results > best_eval_results):
                    best_eval_results = eval_results
                    _save_best_model(best_eval_results)
                model.train()
            # if step % config_data.save_steps == 0:
            #     _save_step(step)
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