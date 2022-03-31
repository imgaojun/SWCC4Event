
### SWCC: a Simultaneous Weakly supervised Contrastive learning and Clustering framework for event representation learning.
---

Official implementation of our paper "Improving Event Representation via Simultaneous Weakly Supervised Contrastive Learning  and Clustering".

### Note.
The Event triples we use for the training data are extracted from  the `New York Times Gigaword Corpus` using the Open Information Extraction system `Ollie`.
Our event representation model is implemented using the `Texar-PyTorch` package. Our model starts from pre-trained checkpoints of `BERT-based-uncased` and we use the `CLS` token representation as the event representation. We train our model with a batch size of $256$ using an Adam optimizer. The learning rate is set as 2e-7 for the event representation model and 5e-4 for the prototype memory. We adopt the temperature $\tau=0.3$ and the number of prototypes used in our experiment is $10$.

### Dataset
We recommend you use [gdown](https://github.com/wkentaro/gdown) to download our [data]() from Google Drive:
```shell
pip install gdown
gdown 
```

### Model
Coming soon....

### Quick Start
```shell
conda create -n swcc python=3.8
conda activate swcc
pip install -r requirements.txt
```

#### Training/Testing
To train and test a specific model, run the bash files `train.sh` and `test.sh`. For example, to train a new model and test a specific model, do the following:

```shell
// Training
// sh train.sh
CUDA_VISIBLE_DEVICES=0 python3 main.py --do-train 

// Testing
//sh test.sh
CUDA_VISIBLE_DEVICES=3 python3 main.py --do-eval --checkpoint ./outputs/checkpoint0.pt 
```

### Citation
```
@inproceedings{gao2022improving,
 author = {Jun Gao and Wei Wang and Changlong Yu and Huan Zhao and Wilfred Ng and Ruifeng Xu},
 booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
 title = {Improving Event Representation via Simultaneous Weakly Supervised Contrastive Learning and Clustering},
 year = {2022}
}
```