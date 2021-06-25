import torch
import torch.nn as nn
import texar.torch as tx
from texar.torch.utils.utils import sequence_mask
from texar.torch.data import embedding
import numpy as np
class AdCo(nn.Module):
    def __init__(self, config_model, config_data, vocab):
        super(AdCo, self).__init__()
        self.config_model = config_model
        self.m = config_model.m
        self.vocab=vocab
        self.vocab_size = vocab.size
        
        self.encoder_q = EventEncoder(config_model, config_data, self.vocab)
        self.encoder_k = EventEncoder(config_model, config_data, self.vocab)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient



    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, batch):
        evtq_s_ids, evtq_s_lengths, evtq_p_ids, evtq_p_lengths, evtq_o_ids, evtq_o_lengths \
                 = batch.evtq_s_ids, batch.evtq_s_lengths, batch.evtq_p_ids, batch.evtq_p_lengths, batch.evtq_o_ids, batch.evtq_o_lengths
        
        evtk_s_ids, evtk_s_lengths, evtk_p_ids, evtk_p_lengths, evtk_o_ids, evtk_o_lengths \
                 = batch.evtk_s_ids, batch.evtk_s_lengths, batch.evtk_p_ids, batch.evtk_p_lengths, batch.evtk_o_ids, batch.evtk_o_lengths
        q = self.encoder_q(evtq_s_ids, evtq_s_lengths, evtq_p_ids, evtq_p_lengths, evtq_o_ids, evtq_o_lengths)  # queries: NxC
        # q = nn.functional.normalize(q, dim=1)

        k = self.encoder_q(evtk_s_ids, evtk_s_lengths, evtk_p_ids, evtk_p_lengths, evtk_o_ids, evtk_o_lengths)
        # k = nn.functional.normalize(k, dim=1)
        # compute key features
        # with torch.no_grad():  # no gradient to keys
        #     # if update_key_encoder:
        #     self._momentum_update_key_encoder()  # update the key encoder

        #     k = self.encoder_k(evt_k, evt_k_lengths)  # keys: NxC
        #     k = nn.functional.normalize(k, dim=1)
            
        #     k = k.detach()

        return q, k
        

class EventEncoder(nn.Module):
    def __init__(self,config_model, config_data, vocab):
        super(EventEncoder, self).__init__()
        self.config_model = config_model
        self.config_data = config_data
        self.vocab = vocab
        self.vocab_size = vocab.size
        self.k = config_model.k
        self.emb_dim = config_model.emb_dim
        glove_embedding = np.random.rand(self.vocab_size, self.config_model.word_dim).astype('f')
        glove_embedding = embedding.load_glove(self.config_data.glove_file, self.vocab.token_to_id_map_py, glove_embedding)

        self.word_embedder = tx.modules.WordEmbedder(
            init_value=torch.from_numpy(glove_embedding),
            vocab_size=self.vocab_size,
            hparams=self.config_model.emb)

        self.subj_verb_comp = TensorComposition(self.k, self.emb_dim, self.emb_dim)
        self.verb_obj_comp = TensorComposition(self.k, self.emb_dim, self.emb_dim)
        self.final_comp = TensorComposition(self.k, self.k, self.k)
        self.fc = nn.Linear(self.emb_dim*3,self.emb_dim)
        self.linear1 = nn.Linear(2 * self.emb_dim, self.k)
        self.linear2 = nn.Linear(2 * self.emb_dim, self.k)
        self.linear3 = nn.Linear(2 * self.k, self.k)
        self.tanh = nn.Tanh()
        

    def _embedding_fn(self, tokens: torch.LongTensor,
                      positions: torch.LongTensor) -> torch.Tensor:
        
        word_embed = self.word_embedder(tokens)
        pos_embed = self.pos_embedder(positions)
        scale = self.config_model.hidden_dim ** 0.5
        return word_embed#*scale + pos_embed
    def forward(self, evt_s_ids, evt_s_lengths, 
                    evt_p_ids, evt_p_lengths, 
                    evt_o_ids, evt_o_lengths):
        
        evt_s_emb = self.word_embedder(evt_s_ids)
        evt_s_mask = sequence_mask(
             evt_s_lengths, evt_s_ids.size()[1]).float()
        evt_s_emb = (evt_s_mask.unsqueeze(-1) * evt_s_emb).mean(1)

        evt_p_emb = self.word_embedder(evt_p_ids)
        evt_p_mask = sequence_mask(
             evt_p_lengths, evt_p_ids.size()[1]).float()
        evt_p_emb = (evt_p_mask.unsqueeze(-1) * evt_p_emb).mean(1)

        evt_o_emb = self.word_embedder(evt_o_ids)
        evt_o_mask = sequence_mask(
             evt_o_lengths, evt_o_ids.size()[1]).float()
        evt_o_emb = (evt_o_mask.unsqueeze(-1) * evt_o_emb).mean(1)
        final_emb = self.fc(torch.cat([evt_s_emb,evt_p_emb,evt_o_emb],dim=-1))
        return final_emb
        # r1 = subj_verb_comp(subj, verb)
        tensor_comp = self.subj_verb_comp(evt_s_emb, evt_p_emb)   # (batch, k)
        cat = torch.cat((evt_s_emb, evt_p_emb), dim=1)    # (batch, 2*emb_dim)
        linear = self.linear1(cat)              # (batch, k)
        r1 = self.tanh(tensor_comp + linear)    # (batch, k)
        # r2 = verb_obj_comp(verb, obj)
        tensor_comp = self.verb_obj_comp(evt_p_emb, evt_o_emb)     # (batch, k)
        cat = torch.cat((evt_p_emb, evt_o_emb), dim=1)     # (batch, 2*emb_dim)
        linear = self.linear2(cat)              # (batch, k)
        r2 = self.tanh(tensor_comp + linear)    # (batch, k)
        # r3 = final_comp(r1, r2)
        tensor_comp = self.final_comp(r1, r2)   # (batch, k)
        cat = torch.cat((r1, r2), dim=1)        # (batch, 2*k)
        linear = self.linear3(cat)              # (batch, k)
        r3 = self.tanh(tensor_comp + linear)    # (batch, k)
        return r3
        # encoder_input, event_lengths = event_ids, event_lengths
        # batch_size = encoder_input.size(0)
        
        # encoder_input_length = event_lengths
        # positions = torch.arange(
        #     encoder_input_length.max(), dtype=torch.long,
        #     device=encoder_input.device).unsqueeze(0).expand(batch_size, -1)

        # enc_input_embedding = self._embedding_fn(encoder_input, positions)

        # enc_output = self.encoder(
        #     inputs=enc_input_embedding, sequence_length=encoder_input_length)
        # inputs_padding = sequence_mask(
        #     event_lengths, event_ids.size()[1]).float()
        
        # # event_embedding = (enc_output * inputs_padding.unsqueeze(-1)).mean(1)
        # event_embedding = enc_output[:,0,:]
        # return event_embedding

class TensorComposition(nn.Module):
    def __init__(self, k, n1, n2):
        super(TensorComposition, self).__init__()
        self.k = k
        self.n1 = n1
        self.n2 = n2
        self.t = nn.Parameter(torch.FloatTensor(k, n1, n2))
        # torch.nn.init.xavier_uniform_(self.t, gain=1)
        torch.nn.init.normal_(self.t, std=0.01)

    def forward(self, a, b):
        '''
        a: (*, n1)
        b: (*, n2)
        '''
        k = self.k
        n1 = self.n1
        n2 = self.n2
        output_shape = tuple(a.size()[:-1] + (k,))  # (*, k)
        a = a.view(-1, n1)      # (m, n1)
        b = b.view(-1, n2)      # (m, n2)
        o = torch.einsum('ijk,cj,ck->ci', [self.t, a, b])
        return o.view(output_shape)

class Adversary_Negatives(nn.Module):
    def __init__(self, bank_size, dim):
        super(Adversary_Negatives, self).__init__()
        self.register_buffer("W", torch.randn(dim, bank_size))
        self.register_buffer("v", torch.zeros(dim, bank_size))
    def forward(self, q):
        memory_bank = self.W
        memory_bank = nn.functional.normalize(memory_bank, dim=0)

        logit=torch.einsum('nc,ck->nk', [q, memory_bank])
        return memory_bank, self.W, logit

    def update(self, m, lr, weight_decay, g):
        g = g + weight_decay * self.W
        self.v = m * self.v +g
        self.W = self.W - lr * self.v

