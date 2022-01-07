import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import *
from sklearn import datasets


iris = datasets.load_iris()
X = iris.data
labels_true = iris.target
print(X.shape, labels_true.shape)

n_clusters = 3

k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
k_means.fit(X)

k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

print('Estimated number of clusters: %d' % n_clusters)
print("Adjusted Rand Index: %0.3f" % adjusted_rand_score(labels_true, k_means_labels))
print("Adjusted Mutual Information: %0.3f" % adjusted_mutual_info_score(labels_true, k_means_labels))
print("Silhouette Coefficient: %0.3f" % silhouette_score(X, k_means_labels, metric='sqeuclidean'))


# fig = plt.figure(figsize=(8, 3))
# fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)

plt.figure(figsize=(6, 4))
colors = ['r', 'g', 'b']
plt.title('K-Means')
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.',  markersize=10)
    for x in X[my_members]:
       plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col, alpha=0.25)
    plt.plot(cluster_center[0], cluster_center[1], 'o', markeredgecolor='k', mew=3, markersize=7)
plt.show()


# def similarity(xi, xk):
#     return -((xi - xk)**2).sum()
#
# S = np.zeros((X.shape[0], X.shape[0]))
#
# for i in range(X.shape[0]):
#     for k in range(X.shape[0]):
#         S[i, k] = similarity(X[i], X[k])
#
# for i in range(X.shape[0]):
#     for k in range(X.shape[0]):
#         if (i==k) : S[i, k] = -10
#
# print("유사도 최솟값 : %0.5f" % S.min())
# print("유사도 중간값 : %0.5f" % np.median(S))
# print("유사도 최댓값 : %0.5f" % S.max())