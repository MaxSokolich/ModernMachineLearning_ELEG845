

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from tqdm import tqdm
#############################################################################
# Generate three seperate clusters
#############################################################################
n = 1000
x1 = np.random.randn(n,2) + np.matlib.repmat([4,4], n, 1)
x2 = np.random.randn(n,2) + np.matlib.repmat([4,12], n, 1)
x3 = np.random.randn(n,2) + np.matlib.repmat([10,8], n, 1)
data = np.concatenate((x1, x2, x3), axis=0)
print(data.shape)

#############################################################################
#                       KMEANS Class
#############################################################################
class Kmeans:

    def dist(self, x1, xc):
        d =  np.sqrt((x1[0]-xc[0])**2 + (x1[1]-xc[1])**2)
        return d

    def get_centroids(self, data, K, centroids):
        """
        data: mx2 containg x and y data
        K: number of clusters
        centroids: Lx2 dimenion containg the centroid of each cluster
        """
        C = centroids
        cluster1 = []
        cluster2 = []
        cluster3 = []
        classes = []
        for i in range(len(data)):
            
            distances = []
            for k in range(len(C)):
                ci = self.dist(data[i],C[k])
                distances.append(ci)
            
            classification = np.argmin(distances)
            if classification == 0:
                cluster1.append(data[i])
                classes.append(0)
            elif classification == 1:
                cluster2.append(data[i])
                classes.append(1)      
            else:
                cluster3.append(data[i])
                classes.append(2)
             
        newC = []
        myclusters = np.array([cluster1, cluster2, cluster3])


        for k in tqdm(range(len(C))):
            muk = 1/len(myclusters[k])   * np.sum(myclusters[k], axis=0)
            newC.append(muk)
    

        return newC, classes


#############################################################################
#                       Main Function
#############################################################################
# Hyper-parameters
K = 3
epochs = 100
Kmeans = Kmeans()
# Initialize the centroids
data_min = np.min(data)
data_max = np.max(data)
print('the min is %.2f' % (data_min))
print('the max is %.2f' % (data_max))
centroids = np.random.randint(low=data_min,high=data_max,size = (K,2))
print(centroids)

# Define the centroids vector for visualizing the training procedure
centroids_vector = np.zeros((epochs,K,2))

# Training loop for K-means clustering
for epoch in range(epochs):
	centroids,classes = Kmeans.get_centroids(data,K,centroids)
	centroids_vector[epoch] = centroids

#############################################################################
# Visualizing the dataset and the learned K-mean model
group_colors = ['skyblue', 'coral', 'lightgreen']
colors = [group_colors[j] for j in classes]

fig, _axs = plt.subplots(nrows=1, ncols=2,figsize=(8,6))
ax = _axs.flatten()

ax[0].scatter(data[:,0], data[:,1])
ax[0].set_title('The orignial dataset with 3 clusters')


ax[1].scatter(data[:,0], data[:,1], color=colors, alpha=0.5)
ax[1].scatter(centroids_vector[epochs-1][:,0], centroids_vector[epochs-1][:,1], color=['blue', 'darkred', 'green'], marker='o', lw=2)

l31 = ax[1].scatter(centroids_vector[:,0,0],centroids_vector[:,0,1], c=range(epochs), vmin=1, vmax=epochs, cmap='autumn',marker = 'v')
l32 = ax[1].scatter(centroids_vector[:,1,0],centroids_vector[:,1,1], c=range(epochs), vmin=1, vmax=epochs, cmap='autumn',marker = 'v')
l33 = ax[1].scatter(centroids_vector[:,2,0],centroids_vector[:,2,1], c=range(epochs), vmin=1, vmax=epochs, cmap='autumn',marker = 'v')

ax[1].plot(centroids_vector[:,0,0],centroids_vector[:,0,1],alpha=0.5,color ='k')
ax[1].plot(centroids_vector[:,1,0],centroids_vector[:,1,1],alpha=0.5,color ='k')
ax[1].plot(centroids_vector[:,2,0],centroids_vector[:,2,1],alpha=0.5,color ='k')
ax[1].set_xlabel('$x_0$')
ax[1].set_ylabel('$x_1$');
cbar = fig.colorbar(l31, ax=ax[1])
cbar.set_label('Epoch')
ax[1].set_title('The final dataset with 3 clusters')
#cbar.ax.set_yticklabels(['Start', '0', 'End'])


plt.show()