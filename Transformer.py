import numpy as np
import math


seq_len = 16     # number of time steps
d_model = 128    # embedding dimension
num_heads = 4
d_ff = 256       # hidden size in FFN

data = np.random.uniform(low=0, high=1, size=(seq_len, d_model))
d_k = np.size(data, axis=0)

Qw = np.random.randn(d_model, d_model) / math.sqrt(d_model)
Kw = np.random.randn(d_model, d_model) / math.sqrt(d_model)
Vw = np.random.randn(d_model, d_model) / math.sqrt(d_model)
W0 = np.random.randn(d_model * num_heads, d_model) / math.sqrt(d_model)
W1 = np.random.randn(d_model, d_ff) / math.sqrt(d_model)
W2 = np.random.randn(d_ff, d_model) / math.sqrt(d_model)

def softmax(QK):
    QK = QK - np.max(QK, axis=-1, keepdims=True)
    QK = np.exp(QK)
    QK = QK / np.sum(QK, axis=-1, keepdims=True)
    return QK

def layernorm(x, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

def combine_heads(x):
    # (num_heads, seq_len, d_head) -> (seq_len, d_model)
    num_heads, seq_len, d_head = x.shape
    return x.transpose(1, 0, 2).reshape(seq_len, num_heads * d_head)


def ffnn(inputlayer, wt1, wt2):
    hiddenlayer = inputlayer @ wt1
    outputlayer = hiddenlayer @ wt2
    return outputlayer


def multiheadattention(qw, kw, vw, w0, data):

    Q_heads = []
    K_heads = []
    V_heads = []

    for h in range(num_heads):
        Q_head = data @ qw#[h]
        K_head = data @ kw#[h]
        V_head = data @ vw#[h]

        Q_heads.append(Q_head)
        K_heads.append(K_head)
        V_heads.append(V_head)

    Q = np.stack(Q_heads, axis=0)
    K = np.stack(K_heads, axis=0)
    V = np.stack(V_heads, axis=0)

    weights = softmax((Q @ K.transpose(0, 2, 1)) / math.sqrt(d_model))
    attention = weights @ V

    concat = combine_heads(attention)  # (seq, d_model)
    return concat @ w0


def encoderlayer(qw, kw, vw, w0, w1, w2, data):
    weighted = multiheadattention(qw, kw, vw, w0, data)
    weighted = layernorm(weighted + data)
    linear = ffnn(weighted, w1, w2)
    linear = layernorm(linear + weighted)
    return linear


def mlp(w0, w1, w2, w3, encoded):
    encoded[:,-1] = 1
    encoded = max(0,layernorm(encoded @ w0))
    encoded = max(0,layernorm(encoded @ w1))
    encoded = max(0,layernorm(encoded @ w2))
    output = softmax(encoded @ w3)
    return output


def full_architecture(data, qw, kw, vw, w0, w1, w2, fw0, fw1, fw2, ow):
    encoded = np.mean(encoderlayer(qw, kw, vw, w0, w1, w2, data), axis=0)
    output = mlp(fw0, fw1, fw2, ow, encoded)
    return output


x = encoderlayer(Qw, Kw, Vw, W0, W1, W2, data) # 16x128
print(x)