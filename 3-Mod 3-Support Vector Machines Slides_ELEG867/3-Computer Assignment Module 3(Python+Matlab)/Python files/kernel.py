import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# define the data points
X = np.array([[2, 1], [6, 2], [2, 5], [4, 4]])
y = np.array([1, -1, 1, -1])

# define the RBF kernel function
def rbf_kernel(X1, X2, gamma=1):
    pairwise_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * pairwise_dists)

# compute the kernel matrix
K = rbf_kernel(X, X)

# fit the SVM with the RBF kernel
clf = SVC(kernel='precomputed')
clf.fit(K, y)

# plot the decision boundary
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))
Z = clf.predict(rbf_kernel(np.c_[xx1.ravel(), xx2.ravel()], X))
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.show()