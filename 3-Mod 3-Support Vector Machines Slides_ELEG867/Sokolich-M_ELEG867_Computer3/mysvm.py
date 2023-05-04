import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Define the data
X = np.array([[-2, 0], [0, 1], [0, -1], [2, 0]])
y = np.array([1, -1, -1, 1])

# Define the SVM with RBF kernel
clf = svm.SVC(kernel='rbf', gamma=1)
clf.fit(X, y)

# Plot the samples
plt.scatter(X[:, 0], X[:, 1], c=y)

# Create a grid of points to evaluate the decision function
xx, yy = np.meshgrid(np.linspace(-4, 4, 50), np.linspace(-4, 4, 50))
xy = np.c_[xx.ravel(), yy.ravel()]

# Evaluate the decision function on the grid
Z = clf.decision_function(xy).reshape(xx.shape)

# Plot the decision boundary and the margins
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')
plt.show()