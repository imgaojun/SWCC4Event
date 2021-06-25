from pathlib import Path
import json
from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool
path = Path("/apdcephfs/share_916081/changlongyu/nyt_corpus/nyt_events_all.jsonl")
c = Counter()
vocab_file = open("vocab_freq.txt",'w',encoding='utf8')

def process(example):
    for evt in example['events']:
        evt = ' '.join(evt).lower()
        words = word_tokenize(evt)
        return words
num_workers=20
buffer = []
with open(path,'r',encoding='utf8') as f, \
    Pool(num_workers) as p:

    for line in tqdm(f):
        example = json.loads(line.strip())
        buffer.append(example)
        if len(buffer) >= num_workers:
            rets = p.map(process, buffer)
            for ret in rets:
                for w in ret:
                    if w:
                        c[w]+=1
            buffer = []
    if len(buffer) > 0:
        rets = p.map(process, buffer)
        for ret in rets:
            for w in ret:
                if w:
                    c[w]+=1
        buffer = []
            
for key,f in sorted(c.items(), key=lambda x: x[1], reverse=True):
    vocab_file.write(key+" "+ str(f) + "\n")

vocab_file.close()

