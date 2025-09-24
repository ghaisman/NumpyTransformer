import numpy as np
import math

data = np.random.uniform(low=0, high=1, size=(128+1, 16))
#data = np.array(([1.0,2,1],[2,3,5],[1,2,3]))
#Tensor sizes
d_k = np.size(data, axis=0)
Qw = np.random.uniform(low=-1, high=1, size=(d_k, d_k))
Kw = np.random.uniform(low=-1, high=1, size=(d_k, d_k))
Vw = np.random.uniform(low=0, high=1, size=(d_k, d_k))

#Prep input vector
data[-1,:] = 0

def softmax(QK):
    QK = np.exp(QK)
    QK = QK / np.sum(QK, axis=1, keepdims=True)
    #for i in range (np.size(QK, axis=0)):
        #QK[i,:] = QK[i,:] / np.sum(QK[i,:])
    return QK

def QKV_Combine(Q, K, V):
    QK = np.matmul(Q, np.transpose(K))
    QK = QK / math.sqrt(d_k-1)
    QK = softmax(QK)
    QKV = np.matmul(QK, V)
    return QKV


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