import copy

import texar.torch as tx

random_seed = 1234

max_positon_length=1024
hidden_dim = 256
moco_t=0.07
m=0.9
bank_size=7
emb = {
    "name": "lookup_table",
    "dim": hidden_dim,
    "initializer": {
        "type": "normal_",
        "kwargs": {"mean": 0.0, "std": hidden_dim ** -0.5},
    },
}

position_embedder_hparams = {"dim": hidden_dim}

encoder = {
    "dim": hidden_dim,
    "num_blocks": 2,
    "multihead_attention": {
        "num_heads": 2,
        "output_dim": hidden_dim
        # See documentation for more optional hyperparameters
    },
    "initializer": {
        "type": "variance_scaling_initializer",
        "kwargs": {"factor": 1.0, "mode": "FAN_AVG", "uniform": True},
    },
    "poswise_feedforward": tx.modules.default_transformer_poswise_net_hparams(
        input_dim=hidden_dim,
        output_dim=hidden_dim
    ),
}

lr=0.03
momentum=0.9
weight_decay=1e-4
moco_t=0.12


memory_lr=3
mem_t=0.02
mem_wd=1e-4