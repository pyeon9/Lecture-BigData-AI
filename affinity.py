from sklearn.cluster import AffinityPropagation
from sklearn.metrics import *
from sklearn import datasets

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle


iris = datasets.load_iris()
X = iris.data
labels_true = iris.target
print(X.shape, labels_true.shape)

model = AffinityPropagation(preference=-50).fit(X)

cluster_centers_indices = model.cluster_centers_indices_
n_clusters_ = len(cluster_centers_indices)
labels = model.labels_

print('Estimated number of clusters: %d' % n_clusters_)
print("Adjusted Rand Index: %0.3f" % adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f" % adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f" % silhouette_score(X, labels, metric='sqeuclidean'))

colors = cycle('rgb')
plt.figure(figsize=(6, 4))
plt.title('Affinity')
for k, col in zip(range(n_clusters_), colors):
    class_members = (labels == k)
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col, alpha=0.25)
    plt.plot(cluster_center[0], cluster_center[1], 'o', mec='k', mew=3, markersize=7)

plt.show()


def similarity(xi, xk):
    return -((xi - xk)**2).sum()

S = np.ones((X.shape[0], X.shape[0])) * 99
for i in range(X.shape[0]):
    for k in range(X.shape[0]):
        S[i, k] = similarity(X[i], X[k])

for i in range(X.shape[0]):
    for k in range(X.shape[0]):
        if (i==k) : S[i, k] = -10#np.amin(S)

print("유사도 최솟값 : %0.5f" % S.min())
print("유사도 중간값 : %0.5f" % np.median(S))
print("유사도 최댓값 : %0.5f" % S.max())

print(S)