
### SWCC: a Simultaneous Weakly supervised Contrastive learning and Clustering framework for event representation learning.
---

Official implementation of our paper "Improving Event Representation via Simultaneous Weakly Supervised Contrastive Learning  and Clustering".

### Note.
The Event triples we use for the training data are extracted from  the `New York Times Gigaword Corpus` using the Open Information Extraction system `Ollie`.
Our event representation model is implemented using the `Texar-PyTorch` package. Our model starts from pre-trained checkpoints of `BERT-based-uncased` and we use the `CLS` token representation as the event representation. We train our model with a batch size of $256$ using an Adam optimizer. The learning rate is set as 2e-7 for the event representation model and 5e-4 for the prototype memory. We adopt the temperature $\tau=0.3$ and the number of prototypes used in our experiment is $10$.

### Training/Testing.
To train and test a specific model, run the bash files `train.sh` and `test.sh`. For example, to train a new model and test a specific model, do the following:

```shell
// Training
// sh train.sh
CUDA_VISIBLE_DEVICES=0 python3 main.py --do-train 

// Testing
//sh test.sh
CUDA_VISIBLE_DEVICES=3 python3 main.py --do-eval --checkpoint ./outputs/checkpoint0.pt 
```