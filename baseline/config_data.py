max_train_epoch = 40
display_steps = 100
eval_steps = 100

train_hparams={
        'dataset': { 'files': '/apdcephfs/share_916081/changlongyu/nyt_corpus/nyt_events_all.jsonl'},
        'batch_size': 10,
        'lazy_strategy': 'all',
        'num_parallel_calls': 10,
        'shuffle_buffer_size': 50000,
        'shuffle': True
}