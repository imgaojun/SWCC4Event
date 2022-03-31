import copy

import texar.torch as tx

random_seed = 1234#1234

lr=2e-7

word_dim=768
max_positon_length=1024
hidden_dim = word_dim
moco_t=0.3
moco_m=0.9

sinkhorn_iterations=3
epsilon=0.05

mem_lr=5e-4
mem_t=0.3
bank_size=10

num_topics=50
emb = {
    "name": "lookup_table",
    "dim": word_dim,
    "initializer": {
        "type": "normal_",
        "kwargs": {"mean": 0.0, "std": word_dim ** -0.5},
    },
    "trainable": True,
}


position_embedder_hparams = {"dim": word_dim}


encoder = {
    "pretrained_model_name": "bert-base-uncased",
    "embed": {
        "dim": 768,
        "name": "word_embeddings"
    },
    "vocab_size": 30522,
    "segment_embed": {
        "dim": 768,
        "name": "token_type_embeddings"
    },
    "type_vocab_size": 2,
    "position_embed": {
        "dim": 768,
        "name": "position_embeddings"
    },
    "position_size": 512,

    "encoder": {
        "dim": 768,
        "embedding_dropout": 0.1,
        "multihead_attention": {
            "dropout_rate": 0.1,
            "name": "self",
            "num_heads": 12,
            "num_units": 768,
            "output_dim": 768,
            "use_bias": True
        },
        "name": "encoder",
        "num_blocks": 12,
        "eps": 1e-12,
        "poswise_feedforward": {
            "layers": [
                {
                    "kwargs": {
                        "in_features": 768,
                        "out_features": 3072,
                        "bias": True
                    },
                    "type": "Linear"
                },
                {"type": "BertGELU"},
                {
                    "kwargs": {
                        "in_features": 3072,
                        "out_features": 768,
                        "bias": True
                    },
                    "type": "Linear"
                }
            ]
        },
        "residual_dropout": 0.1,
        "use_bert_config": True
        },
    "hidden_size": 768,
    "initializer": None,
    "name": "bert_encoder",
}



