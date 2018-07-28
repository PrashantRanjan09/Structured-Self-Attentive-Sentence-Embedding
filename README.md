# Structured Self-Attentive Sentence Embedding
This is an implementation of the paper: https://arxiv.org/pdf/1703.03130.pdf published in ICLR 2017.
This paper proposes a new model for extracting an interpretable sentence embedding by introducing _self-attention_. Instead of using a vector, the paper use a 2-D matrix to represent the embedding, with each row of the matrix attending on a different part of the sentence.It also propose a self-attention mechanism.

![Optional Text](../master/self-attention.png)

The implementation is done on the imdb dataset with the following parameters:

    top_words = 10000
    learning_rate =0.001
    max_seq_len = 200
    emb_dim = 300
    batch_size=500
    u=64
    da = 32
    r= 16
    
**top_words** : only consider the top 10,000 most common words <br>
**u**: hidden unit number for each unidirectional LSTM<br>
**da** : a hyperparameter we can set arbitrarily. <br>
**r** : no. of different parts to be extracted from the sentence.<br>

### To Run:

    python self-attention.py

Running this for 4 epochs gives a training accuracy of `94%` and test accuracy of `87%`.

### To Do :
Penalization term <br>
results on other datasets
