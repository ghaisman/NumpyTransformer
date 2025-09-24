import numpy as np
import math

seq_len = 16     # number of time steps
d_model = 128    # embedding dimension
num_heads = 8
d_ff = 256       # hidden size in FFN


data = np.random.uniform(low=0, high=1, size=(seq_len, d_model))
d_k = np.size(data, axis=0)
Qw = np.random.uniform(low=-1, high=1, size=(d_k, d_k))
Kw = np.random.uniform(low=-1, high=1, size=(d_k, d_k))
Vw = np.random.uniform(low=0, high=1, size=(d_k, d_k))

#Prep input vector
data[-1,:] = 1

def softmax(QK):
    QK = QK - np.max(QK, axis=-1, keepdims=True)
    QK = np.exp(QK)
    QK = QK / np.sum(QK, axis=1, keepdims=True)
    return QK

def QKV_Combine(Q, K, V):
    QK = np.matmul(Q, np.transpose(K))
    QK = QK / math.sqrt(d_k-1)
    QK = softmax(QK)
    QKV = np.matmul(QK, V)
    return QKV

def split_heads(x, num_heads):
    # (seq_len, d_model) -> (num_heads, seq_len, d_head)
    d_head = d_model // num_heads
    return x.reshape(seq_len, num_heads, d_head).transpose(1, 0, 2)

def combine_heads(x):
    # (num_heads, seq_len, d_head) -> (seq_len, d_model)
    num_heads, seq_len, d_head = x.shape
    return x.transpose(1, 0, 2).reshape(seq_len, num_heads * d_head)

def attention(qw, kw, vw, tensor):
    Q = np.matmul(qw, tensor)
    K = np.matmul(kw, tensor)
    V = np.matmul(vw, tensor)
    Combined = QKV_Combine(Q,K,V)
    return Combined


def ffnn(inputlayer, wt1, wt2):
    hiddenlayer = np.matmul(wt1, inputlayer)
    outputlayer = np.matmul(hiddenlayer, wt2)
    return outputlayer


output = attention(Qw, Kw, Vw, data)
print(output)