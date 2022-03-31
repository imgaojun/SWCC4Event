max_train_epoch = 10
display_steps = 100
eval_steps = 1000
save_steps = 1000
vocab_file=""

train_hparams={
        'dataset': { 'files': 'data_utils/multi_pos_train3.json',
                     'vocab_file': vocab_file},
        'batch_size': 256,
        'lazy_strategy': 'all',
        'num_parallel_calls': 10,
        'shuffle_buffer_size': 50000,
        
        "allow_smaller_final_batch": False,
        "cache_strategy": "none",
        'shuffle': True,

}

hard_hparams={
        'dataset': { 'files': 'data_utils/event_eval/hard.txt',
        'vocab_file': vocab_file},
        'batch_size': 128,
        'shuffle': False
}

hardext_hparams={
        'dataset': { 'files': 'data_utils/event_eval/hard_extend.txt',
        'vocab_file': vocab_file},
        'batch_size': 128,
        'shuffle': False
}

trans_hparams={
        'dataset': { 'files': 'data_utils/event_eval/transitive.txt',
        'vocab_file': vocab_file},
        'batch_size': 128,
        'shuffle': False
}