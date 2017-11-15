# Self-Attentive-Tensorflow

![model image of TagSpace](https://raw.githubusercontent.com/flrngel/Self-Attentive-tensorflow/master/resources/self-attentive-model.png)

Tensorflow implementation of **A Structured Self-Attentive Sentence Embedding**

You can read more about concept from [this paper](https://arxiv.org/abs/1703.03130)

## Key Concept

Frobenius norm with attention

## Usage

Download [ag news dataset](https://github.com/mhjabreel/CharCNN/tree/master/data/ag_news_csv) as below

```
$ tree ./data
./data
└── ag_news_csv
    ├── classes.txt
    ├── readme.txt
    ├── test.csv
    ├── train.csv
    └── train_mini.csv
```

and then

```
$ python train.py
```

## Result

Accuracy 0.895

## To-do list

- support multiple dataset

## Notes

This implementation does not use pretrained GloVe or Word2vec.
