import numpy as np
from sklearn import datasets

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

o = np.array([0.7, 1.9, 0.1, 0.8])
print('#1')
s = softmax(o)
print(s)
print(sum(s))
print()

def crossentropy(y, o):
    return -np.sum(y*np.log2(o))

y = np.array([0, 1, 0, 0])
print('#2')
print(crossentropy(y, s))
print()


def negative_log_likelihood(o):
    return -np.log2(o)

print('#3')
print(negative_log_likelihood(s[1]))
print()

iris = datasets.load_iris()
x = iris.data
y = iris.target
print('#4')
print(x.shape, y.shape)
M = np.mean(x, axis=0)
S = np.std(x, axis=0)
print(M)
print(S)

a = np.array([4.1, 4.5, 1.8, 1.2])
def standardization(a, m, s):
    return (a-m)/s

print(standardization(a, np.array(M), np.array(S)))
print()

print('#5')
print(np.unique(y))
def my_one_hot_encoder(int_labels):
    one_hot_labels = np.eye(3)[int_labels]
    return one_hot_labels

int_labels = [2,1,0,1,2,0]
print(my_one_hot_encoder(int_labels))
print()