# TODO: 4-2
# from operator import add, mul
#
# data = [2, 7, 12, 7, 4, 5, 8, 2]
# kernel = [0.4, 0.2, 0.6]
#
#
# for i in range(len(data)-2):
#     temp = data[i:i+3]
#     res = [mul(x, y) for x, y in zip(temp, kernel)]
#     print(sum(res), res)

# TODO: 4-3
# import numpy as np
#
# def conv(X, filters, stride=1, pad=0):
#     n, c, h, w = X.shape
#     n_f, _, filter_h, filter_w = filters.shape
#     out_h = (h + 2 * pad - filter_h) // stride + 1
#     out_w = (w + 2 * pad - filter_w) // stride + 1
#
#     # add padding to height and width.
#     in_X = np.pad(X, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
#     out = np.zeros((n, n_f, out_h, out_w))
#
#     for i in range(n): # for each image.
#         for c in range(n_f): # for each channel.
#             for h in range(out_h): # slide the filter vertically.
#                 h_start = h * stride
#                 h_end = h_start + filter_h
#                 for w in range(out_w): # slide the filter horizontally.
#                     w_start = w * stride
#                     w_end = w_start + filter_w
#                     # Element-wise multiplication.
#                     out[i, c, h, w] = np.sum(in_X[i, :, h_start:h_end, w_start:w_end] * filters[c])
#
#     return out
#
#
#
#
# X = np.array([[[[1,0,2,1,1],
#              [2,0,2,0,2],
#              [2,1,2,1,1],
#              [1,0,1,1,2],
#              [1,2,1,0,1]]]])
#
# kernel = np.array([[[[1,0,-1],
#                    [1,0,-1],
#                    [1,0,-1]]]])
#
# print('Data:', X.shape)
# print('Filters:', kernel.shape)
# out = conv(X, kernel, stride=2, pad=1)
# print('Output:', out.shape)
# print(out)


# TODO: 4-[7~10]
# import numpy as np
#
# U1= np.array([[-0.3, 1.0, 1.2],
#               [1.6, -1.0, -1.1]])
# U2= np.array([[1.0, 1.0, -1.0],
#               [0.7, 0.5, 1.0]])
# U3= np.array([[0.5, -0.8, 1.0],
#               [-0.1, 0.3, 0.4]])
# U4= np.array([[1.0, 0.1, -0.2],
#               [-0.2, 1.3, -0.4]])
# UU = [U1, U2, U3, U4]
# data = np.array([[1, 0]])
# x0 = 1 # bias
#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
# def relu(x):
#     return np.maximum(0, x)
#
# def activation(arr, act):
#     result = []
#     for a in arr[0]:
#         if act == 'sigmoid':
#             logit = sigmoid(a)
#             result.append(logit)
#         elif act == 'relu':
#             logit = relu(a)
#             result.append(logit)
#     result = np.array(result).reshape(1, -1)
#     return result
#
# for i in range(4):
#     data = np.insert(data, 0, x0)
#     data = data.reshape(1, -1)
#     data = np.matmul(data, UU[i].T)
#     data = activation(data, 'relu')
#     print(data)

# TODO: [6-3~6-7] (p.373 - 5)
import numpy as np

data = np.array([[3, 2], [2, 1], [2, 3], [1, 4]])
len_data = data.shape[0]

def distance(a, b):
    dist_x = pow(a[0]-b[0], 2)
    dist_y = pow(a[1]-b[1], 2)
    return dist_x + dist_y

def similarity(arr, len_data=len_data):
    S = np.ones((len_data, len_data)) * 99
    for i in range(len_data):
        for j in range(len_data):
            if i == j:
                pass
            else:
                S[i, j] = -distance(data[i], data[j])
    for i in range(len_data):
        for j in range(len_data):
            if i == j:
                S[i, j] = np.amin(S)
    return S

# S = similarity(data)
# print('S: \n', S)

def max_as(S, A, row, index, len_data=len_data):
    values = []
    for k in range(len_data):
        if k == index:
            pass
        else:
            value = A[row, k] + S[row, k]
            values.append(value)
    return max(values)

def responsibility(S, A, len_data=len_data):
    R = np.zeros((len_data, len_data))
    for i in range(len_data):
        for j in range(len_data):
            R[i, j] = S[i, j] - max_as(S, A, row=i, index=j)
    return R

# A = np.zeros((len_data, len_data))
# R = np.zeros((len_data, len_data))
# R = responsibility(S, A)
# print('R: \n', R)

def max_r(R, row, index, len_data=len_data):
    values = []
    for i in range(len_data):
        if (i == row) or (i == index):
            pass
        else:
            value = max(0, R[i, index])
            values.append(value)
    return sum(values)

def available(R, len_data=len_data):
    A = np.zeros((len_data, len_data))
    for i in range(len_data):
        for j in range(len_data):
            if i == j:
                A[i, j] = max_r(R, row=i, index=j)
            else:
                A[i, j] = min(0, R[j, j] + max_r(R, row=i, index=j))
    return A

print('-'*10 + ' t = 0 ' + '-'*10)
S = similarity(data)
print('S: \n', S)
A = np.zeros((len_data, len_data))
R = np.zeros((len_data, len_data))
R = responsibility(S, A)
print('R: \n', R)
A = available(R)
print('A: \n', A)
print('-'*10 + ' t = 1 ' + '-'*10)
R = responsibility(S, A)
A = available(R)
print('R: \n', R)
print('A: \n', A)

# # TODO: [6-8~6-10] (p.373 - 9)
import numpy as np

# A = np.array([[2, 1], [2, 4], [4, 1], [4, 3]])
# A = np.array([[-1, -1.25], [-1, 1.75], [1, -1.25], [1, 0.75]])

from pandas import DataFrame
import pandas as pd

datas:DataFrame = pd.DataFrame({'x':[3, 2, 2, 1],
                                'y':[2, 1, 3, 4]})

A = np.array([[3, 2], [2, 1], [2, 3], [1, 4]])
mean = np.mean(A, axis=0)
A = A - mean

# x = datas['x']
# y = datas['y']

# N = len(datas)

# mu_x = np.mean(x)
# mu_y = np.mean(y)
# print(mu_x, mu_y)

# cov_sample = sum((x-mu_x)*(y-mu_y))/N
# print('{0:.3f}'.format(cov_sample))
# cov = sum((x-mu_x)*(y-mu_y))/(N-1)
# print('{0:.3f}'.format(cov))

# B = np.cov(x, y, ddof=0)
B = np.cov(A[:, 0], A[:, 1], ddof=0)
print('Cov: \n', B)
print()

eigenvalue, eigenvector = np.linalg.eig(B)
eigenvalue = np.round(eigenvalue, 5) # 소수점 5자리로 반올림
eigenvector = np.round(eigenvector, 5) # 소수점 5자리로 반올림
print("Eigen Value: \n", eigenvalue)

print("Eigen Vector Matix P: \n", eigenvector)
print()

print("Dimensionality Reduction with PCA: ")
W = np.array([eigenvector[:, 1]]).T#, eigenvector[:, 1]]).T
print("W = \n", W)
print("A = \n", A)

print("W.T@A.T = Z.T : \n", W.T@A.T)
ZT = W.T@A.T
print("Z = \n", ZT.T)


# import matplotlib.pyplot as plt
#
# plt.plot(ZT[:,0], ZT[:,1])
# plt.show()