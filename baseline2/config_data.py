max_train_epoch = 10
display_steps = 100
eval_steps = 1000

train_hparams={
        'dataset': { 'files': '/apdcephfs/share_916081/changlongyu/nyt_corpus/nyt_events_all.jsonl',
                     'vocab_file': 'vocab50k.txt'},
        'batch_size': 256,
        'lazy_strategy': 'all',
        'num_parallel_calls': 10,
        'shuffle_buffer_size': 50000,
        'shuffle': True
}

hard_hparams={
        'dataset': { 'files': '/apdcephfs/share_916081/changlongyu/nyt_corpus/event_eval/hard.txt',
        'vocab_file': 'vocab30k.txt'},
        'batch_size': 1,
        'shuffle': False
}

hardext_hparams={
        'dataset': { 'files': '/apdcephfs/share_916081/changlongyu/nyt_corpus/event_eval/hard_extend.txt',
        'vocab_file': 'vocab30k.txt'},
        'batch_size': 1,
        'shuffle': False
}

trans_hparams={
        'dataset': { 'files': '/apdcephfs/share_916081/changlongyu/nyt_corpus/event_eval/transitive.txt',
        'vocab_file': 'vocab30k.txt'},
        'batch_size': 1,
        'shuffle': False
}