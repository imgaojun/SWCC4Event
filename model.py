from logging import BASIC_FORMAT
from numpy.core.numeric import base_repr
import torch
import torch.nn as nn
import texar.torch as tx
from texar.torch.utils.utils import sequence_mask
import numpy as np
import torch.nn.functional as F

class MarginLoss(nn.Module):
    def __init__(self, margin):
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, pos_score, neg_score):
        '''
        pos_score: (batch)
        neg_score: (batch)
        '''
        return torch.mean(self.relu(neg_score - pos_score + self.margin))
    

class SWCC(nn.Module):
    def __init__(self, config_model, config_data):
        super(SWCC, self).__init__()
        self.config_model = config_model
        self.moco_m = config_model.moco_m
        self.vocab_size = config_model.encoder['vocab_size']
        
        self.encoder_q = EventEncoder(config_model, config_data)

        
        self.memory_bank = Memory_Bank(self.config_model.bank_size,dim=self.config_model.hidden_dim)
        
        self.mask_fc = nn.Sequential(
            nn.Linear(config_model.hidden_dim, self.vocab_size))

    def forward(self, batch):
        evt_q, evt_q_lengths = batch.evt_q_ids, batch.evt_q_lengths
        evt_k, evt_k_lengths = batch.evt_k_ids, batch.evt_k_lengths
        evt_p, evt_p_lengths = batch.evt_p_ids, batch.evt_p_lengths
        mask_pos, mask_id = batch.mask_pos, batch.mask_id
        q, q_outputs = self.encoder_q(evt_k, evt_k_lengths, is_train=True)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        k1, k1_outputs = self.encoder_q(evt_k, evt_k_lengths, is_train=True)
        k1 = nn.functional.normalize(k1, dim=1)
        
        k2, k2_outputs = self.encoder_q(evt_k, evt_k_lengths, is_train=True)
        k2 = nn.functional.normalize(k2, dim=1)
        
        p, p_outputs = self.encoder_q(evt_p, evt_p_lengths, is_train=True)
        p = nn.functional.normalize(p, dim=1)
        
        m, m_outputs = self.encoder_q(evt_q, evt_q_lengths, is_train=True)
        mlm_logits = [ex[mask_pos[idx],:].unsqueeze(0) for idx,ex in enumerate(m_outputs)]
        mlm_logits = torch.cat(mlm_logits,dim=0)
        mlm_logits = self.mask_fc(mlm_logits)
        return q, k1, k2, p, mlm_logits


        

class EventEncoder(nn.Module):
    def __init__(self,config_model, config_data):
        super(EventEncoder, self).__init__()
        self.config_model = config_model
        self.config_data = config_data
        self.input_fc = nn.Linear(config_model.word_dim, config_model.hidden_dim)
        self.encoder = tx.modules.BERTEncoder(
            hparams=self.config_model.encoder)

        self.fc = nn.Sequential(
            nn.Linear(config_model.hidden_dim, config_model.hidden_dim))


    def _embedding_fn(self, tokens: torch.LongTensor,
                      positions: torch.LongTensor) -> torch.Tensor:
        
        word_embed = self.word_embedder(tokens)
        pos_embed = self.pos_embedder(positions)

        return word_embed + pos_embed
    def forward(self, event_ids, event_lengths, is_train=False):
        encoder_input, event_lengths = event_ids, event_lengths
        batch_size = encoder_input.size(0)
        
        encoder_input_length = event_lengths
        outputs, pooled_output = self.encoder(
            inputs=encoder_input, sequence_length=encoder_input_length)
        inputs_padding = sequence_mask(
            event_lengths, event_ids.size()[1]).float()
        
        event_embedding = outputs[:,0,:]
        if is_train:
            return event_embedding, outputs
        else:
            return event_embedding



class Memory_Bank(nn.Module):
    def __init__(self, bank_size, dim):
        super(Memory_Bank, self).__init__()
        self.W = nn.Parameter(torch.randn(dim, bank_size))
    def forward(self, q):
        memory_bank = self.W
        memory_bank = nn.functional.normalize(memory_bank, dim=0)

        logit=torch.einsum('nc,ck->nk', [q, memory_bank])
        return logit

class LabelSmoothingLoss(nn.Module):
    r"""With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    Args:
        label_confidence: the confidence weight on the ground truth label.
        tgt_vocab_size: the size of the final classification.
        ignore_index: The index in the vocabulary to ignore weight.
    """
    one_hot: torch.Tensor

    def __init__(self, label_confidence, tgt_vocab_size, ignore_index=0):
        super().__init__()
        self.ignore_index = ignore_index
        self.tgt_vocab_size = tgt_vocab_size

        label_smoothing = 1 - label_confidence
        assert 0.0 < label_smoothing <= 1.0
        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))
        self.confidence = label_confidence

    def forward(self,  # type: ignore
                output: torch.Tensor,
                target: torch.Tensor,
                label_lengths: torch.LongTensor) -> torch.Tensor:
        r"""Compute the label smoothing loss.
        Args:
            output (FloatTensor): batch_size x seq_length * n_classes
            target (LongTensor): batch_size * seq_length, specify the label
                target
            label_lengths(torch.LongTensor): specify the length of the labels
        """
        orig_shapes = (output.size(), target.size())
        output = output.view(-1, self.tgt_vocab_size)
        target = target.view(-1)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob = model_prob.to(device=target.device)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        output = output.view(orig_shapes[0])
        model_prob = model_prob.view(orig_shapes[0])

        return tx.losses.sequence_softmax_cross_entropy(
            labels=model_prob,
            logits=output,
            sequence_length=label_lengths,
            average_across_batch=False,
            sum_over_timesteps=False,
        )