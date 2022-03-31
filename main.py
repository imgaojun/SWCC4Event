import argparse
import importlib
from logging import BASIC_FORMAT
from typing import Any
from warnings import catch_warnings

import torch
import torch.nn as nn
from texar.torch.run import *
from pathlib import Path
import misc_utils
from misc_utils import init_logger, logger

import texar.torch as tx

from model import SWCC,Memory_Bank
import data_utils
import numpy as np
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist

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
    "--do-tsne", action="store_true",
    help="Output tsne metadata.")
parser.add_argument(
    "--output-dir", type=str, default="./tau_03/",
    help="Path to save the trained model and logs.")
parser.add_argument(
    "--log-file", type=str, default="exp.log",
    help="Path to save the trained model and logs.")

parser.add_argument(
    '--checkpoint', type=str, default=None,
    help="Model checkpoint to load model weights from.")
parser.add_argument(
    '--baseline', type=str, default=None,
    help="Model checkpoint to load model weights from.")

args = parser.parse_args()

config_model: Any = importlib.import_module(args.config_model)
config_data: Any = importlib.import_module(args.config_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

make_deterministic(config_model.random_seed)

output_dir = Path(args.output_dir)
tx.utils.maybe_create_dir(output_dir)

init_logger(output_dir/args.log_file)

def safe_log(val):
    eps=1e-7
    return torch.log(val +eps)

@torch.no_grad()
def distributed_sinkhorn(out):
    Q = torch.exp(out / config_model.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(config_model.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

def main() -> None:
    

    train_data = data_utils.TrainData(config_data.train_hparams,device=device)
    train_data_iterator = tx.data.DataIterator(train_data)

    hard_data = data_utils.HardData(config_data.hard_hparams,device=device)
    hard_data_iterator = tx.data.DataIterator(hard_data)


    hardext_data = data_utils.HardData(config_data.hardext_hparams,device=device)
    hardext_data_iterator = tx.data.DataIterator(hardext_data)

    trans_data = data_utils.TransData(config_data.trans_hparams,device=device)
    trans_data_iterator = tx.data.DataIterator(trans_data)

    model = SWCC(config_model=config_model, config_data=config_data)
    # if args.checkpoint:
    #     logger.info(f"loading checkpoint {args.checkpoint}...")
    #     model.load_state_dict(torch.load(args.checkpoint))
    model.to(device)
    
    memory_bank = Memory_Bank(config_model.bank_size,dim=config_model.hidden_dim)
    memory_bank.to(device)

    # ce_loss_fn = nn.CrossEntropyLoss()
    # ce_loss_fn.to(device)
    optim = torch.optim.Adam(
        model.parameters(), lr=config_model.lr, betas=(0.9, 0.997), eps=1e-9)
    mem_optim = torch.optim.Adam(
        memory_bank.parameters(), lr=config_model.mem_lr, betas=(0.9, 0.997), eps=1e-9)
    
    def get_pos_neg_mask(logits):
        mask = torch.ones(logits.shape[0])
        mask = torch.diag_embed(mask)
        # mask = torch.cat([mask, torch.zeros((logits.shape[0],config_model.bank_size))],dim=-1)
        return mask.to(device), (1-mask).to(device)

    def _update(batch):
        q, k1, k2, p, mlm_logits= model(batch)
        k1_topic_logits = memory_bank(k1)
        k2_topic_logits = memory_bank(k2)
        k1_topic_logits = k1_topic_logits/config_model.moco_t
        k2_topic_logits = k2_topic_logits/config_model.moco_t
        l_pos1 = torch.einsum('nc,ck->nk', [q, k1.T])
        l_pos2 = torch.einsum('nc,ck->nk', [q, k2.T])
        l_pos3 = torch.einsum('nc,nc->n', [q, p])
        
        
        labels = torch.arange(0, l_pos1.shape[0], dtype=torch.long).to(device)
        l_pos1 = l_pos1/config_model.moco_t
        l_pos2 = l_pos2/config_model.moco_t
        l_pos3 = l_pos3/config_model.moco_t
        
        
        pos1_loss = F.cross_entropy(l_pos1, labels)
        pos2_loss = F.cross_entropy(l_pos2, labels)
        # pos3_loss = F.cross_entropy(l_pos3, labels, reduction='none')
        
        
        pos_mask, neg_mask = get_pos_neg_mask(l_pos1)
        
        neg = l_pos1.masked_select(neg_mask.bool())
        
        neg = torch.reshape(neg, (l_pos1.shape[0], l_pos1.shape[1]-1))
        neg = torch.exp(neg)
        pos3 = torch.exp(l_pos3)
        
        
        # pos = 0.5*pos1+0.5*pos2
        # reweight = neg/neg.mean(dim=1,keepdim=True)
        # Ng = (reweight*neg).sum(dim = -1)
        Ng = neg.sum(dim = -1)
        # self_sup_loss = 0.5*(- safe_log(pos1 / (pos1 + Ng) )).mean() + 0.5*(- safe_log(pos2 / (pos2 + Ng) )).mean()
        self_sup_loss = (pos1_loss+pos2_loss)/2
        weakly_sup_loss = (batch.evt_freq*(- safe_log(pos3 / (pos3 + Ng) ))).mean()
        # weakly_sup_loss = (batch.evt_freq*pos3_loss).mean()
        nce_loss =  self_sup_loss + weakly_sup_loss
        
        sub_cluster_loss1 = distributed_sinkhorn(k1_topic_logits)*torch.log_softmax(k2_topic_logits,dim=-1)
        sub_cluster_loss1 = sub_cluster_loss1.mean()
        
        sub_cluster_loss2 = distributed_sinkhorn(k2_topic_logits)*torch.log_softmax(k1_topic_logits,dim=-1)
        sub_cluster_loss2 = sub_cluster_loss2.mean()
        
        sub_cluster_ent1 = torch.softmax(k1_topic_logits,dim=-1)*torch.log_softmax(k1_topic_logits,dim=-1)
        sub_cluster_ent1 = sub_cluster_ent1.mean()
        sub_cluster_ent2 = torch.softmax(k2_topic_logits,dim=-1)*torch.log_softmax(k2_topic_logits,dim=-1)
        sub_cluster_ent2 = sub_cluster_ent2.mean()
        
        cluster_loss = -(sub_cluster_loss1+sub_cluster_loss2)/2
        cluster_ent = -(sub_cluster_ent1+sub_cluster_ent2)/2
        mlm_loss = F.cross_entropy(mlm_logits, batch.mask_id)
        loss = nce_loss + mlm_loss + 0.3*cluster_loss + config_model.epsilon*cluster_ent

        optim.zero_grad()
        mem_optim.zero_grad()
        loss.backward()
        mem_optim.step()
        optim.step()

        return {'nce_loss': nce_loss.item(),'mlm_loss':mlm_loss.item(),'cluster_loss':cluster_loss.item()}
    
    def _save_epoch(epoch):
        logger.info(f"saving model...epoch {epoch}")
        torch.save(model.state_dict(), output_dir/f"checkpoint{epoch}.pt")
    def _save_best_model(results):
        logger.info(f"saving model... {results[0]:.2f}_{results[1]:.2f}")
        ckpt = {
            "model": model.state_dict(),
            "memory_bank":memory_bank.state_dict(),
        }
        torch.save(ckpt, output_dir/f"checkpoint_{results[0]:.2f}_{results[1]:.2f}.pt")
        torch.save(ckpt, output_dir/f"best_last.pt")
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
            
            # print(memory_bank(F.normalize(evt_a,dim=-1))[:10])

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
        logger.info(f"Hard: {hard_results.mean():.4f} | Hard Ext: {hardext_results.mean():.4f} | Transitive: {trans_corr:.4f} ({len(trans_results)})")
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
                if eval_results.mean() > best_eval_results.mean():
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