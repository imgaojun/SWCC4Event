import copy

import texar.torch as tx

random_seed = 1234

learning_rate=0.0004
word_dim=100
emb_dim=word_dim
k=100
max_positon_length=1024
hidden_dim = word_dim
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
    "trainable": False,
}

position_embedder_hparams = {"dim": hidden_dim}


poswise_feedforward_hparams = {
        "layers": [
            {
                "type": "Linear",
                "kwargs": {
                    "in_features": hidden_dim,
                    "out_features": hidden_dim,
                    "bias": True,
                }
            },
            {
                "type": "Dropout",
                "kwargs": {
                    "p": 0.1,
                }
            },
            {
                "type": "Linear",
                "kwargs": {
                    "in_features": hidden_dim,
                    "out_features": hidden_dim,
                    "bias": True,
                }
            },

        ],
        "name": "ffn"
    }


encoder = {
    "dim": hidden_dim,
    "num_blocks": 6,
    "multihead_attention": {
        "num_heads": 8,
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
    # "poswise_feedforward":poswise_feedforward_hparams,
}

lr=0.03
momentum=0.9
weight_decay=1e-4
moco_t=0.12


memory_lr=3
mem_t=0.02
mem_wd=1e-4