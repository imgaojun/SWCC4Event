import torch
import torch.nn as nn
import texar.torch as tx

class AdCo(nn.Module):
    def __init__(self, config_model, vocab_size):
        super(AdCo, self).__init__()
        self.config_model = config_model
        self.m = config_model.m
        
        self.encoder_q = EventEncoder(config_model, vocab_size)
        self.encoder_k = EventEncoder(config_model, vocab_size)

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
        evt_q, evt_q_lengths = batch.evt_q_ids, batch.evt_q_lengths
        evt_k, evt_k_lengths = batch.evt_k_ids, batch.evt_k_lengths
        q = self.encoder_q(evt_q, evt_q_lengths)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        # compute key features
        with torch.no_grad():  # no gradient to keys
            # if update_key_encoder:
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(evt_k, evt_k_lengths)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            
            k = k.detach()

        return q, k

class EventEncoder(nn.Module):
    def __init__(self,config_model, vocab_size):
        super(EventEncoder, self).__init__()
        self.config_model = config_model
        self.vocab_size = vocab_size
        self.word_embedder = tx.modules.WordEmbedder(
            vocab_size=self.vocab_size,
            hparams=self.config_model.emb)
        self.pos_embedder = tx.modules.SinusoidsPositionEmbedder(
            position_size=self.config_model.max_positon_length,
            hparams=self.config_model.position_embedder_hparams)
        self.encoder = tx.modules.TransformerEncoder(
            hparams=self.config_model.encoder)

    def _embedding_fn(self, tokens: torch.LongTensor,
                      positions: torch.LongTensor) -> torch.Tensor:
        
        word_embed = self.word_embedder(tokens)
        pos_embed = self.pos_embedder(positions)
        scale = self.config_model.hidden_dim ** 0.5
        return word_embed*scale + pos_embed
    def forward(self, event_ids, event_lengths):
        encoder_input, event_lengths = event_ids, event_lengths
        batch_size = encoder_input.size(0)
        
        encoder_input_length = event_lengths
        positions = torch.arange(
            encoder_input_length.max(), dtype=torch.long,
            device=encoder_input.device).unsqueeze(0).expand(batch_size, -1)

        enc_input_embedding = self._embedding_fn(encoder_input, positions)

        enc_output = self.encoder(
            inputs=enc_input_embedding, sequence_length=encoder_input_length)
        
        event_embedding = enc_output[:,0]
        return event_embedding

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

