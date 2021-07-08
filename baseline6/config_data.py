max_train_epoch = 10
display_steps = 100
eval_steps = 1000
save_steps = 1000
glove_file = '/apdcephfs/share_916081/jamgao/projects/event-ib/ft_local/glove.6B.100d.ext.txt'

train_hparams={
        'dataset': { 'files': '/apdcephfs/share_916081/changlongyu/nyt_corpus/nyt_events_all.jsonl',
                     'vocab_file': '/apdcephfs/share_916081/jamgao/projects/event-ib/exp0629/glove_vocab.txt'},
        'batch_size': 128,
        'lazy_strategy': 'all',
        'num_parallel_calls': 4,
        'shuffle_buffer_size': 50000,
        'shuffle': True,
        "allow_smaller_final_batch": False,

}

hard_hparams={
        'dataset': { 'files': '/apdcephfs/share_916081/changlongyu/nyt_corpus/event_eval/hard.txt',
        'vocab_file': '/apdcephfs/share_916081/jamgao/projects/event-ib/exp0629/glove_vocab.txt'},
        'batch_size': 64,
        'shuffle': False
}

hardext_hparams={
        'dataset': { 'files': '/apdcephfs/share_916081/changlongyu/nyt_corpus/event_eval/hard_extend.txt',
        'vocab_file': '/apdcephfs/share_916081/jamgao/projects/event-ib/exp0629/glove_vocab.txt'},
        'batch_size': 64,
        'shuffle': False
}

trans_hparams={
        'dataset': { 'files': '/apdcephfs/share_916081/changlongyu/nyt_corpus/event_eval/transitive.txt',
        'vocab_file': '/apdcephfs/share_916081/jamgao/projects/event-ib/exp0629/glove_vocab.txt'},
        'batch_size': 64,
        'shuffle': False
}