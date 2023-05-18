
import scipy.io as sio
import numpy as np
from numpy import mean,cov,cumsum,dot,linalg,size
import matplotlib.pyplot as plt


#############################################################################
# 					PCA function
#############################################################################
# Input: Data with dimension M \times N, where M is the number of samples, N is the number of features
#		 For MNIST dataset, M = 1000 and N = 784
# Input: numpc is the number of principal components
# Output: evectors consists of 'numpc' eigenvectors corresponding to the 'numpc' largest eigenvalues, the dimension of which is M \times numpc
# Output: representation is the sparse representation of Data, the dimension of which is numpc \times N
def princomp(Data,numpc=0):
    M = Data.shape[0] #number of samples
    N = Data.shape[1] #number of features
    
    #Step1: mean normalization
    for i in range(M):
        xi = Data[i,:] - ((1/M) * np.sum(Data, axis=0))
        Data[i] = xi

    test = np.sum(Data[:,]) #should be zero but its not
    print("test for mean centered = ", test)
    
    #Step2: eigendecomposition
    C = (1/M) * dot(Data, Data.T)
    lmbda, v = linalg.eig(C)

    #Step3: sort the eigenvectors and eigenvalues in the descending order of eigenvalues
    lmbda = np.real(lmbda)
    maxindices = np.argpartition(lmbda, -numpc)[-numpc:]

    #Step4: sort eigenvectors according to the sorted eigenvalues
    evectors = np.zeros((M,numpc))
    for i in range(len(maxindices)):
        evectors[:,i] = v[:,maxindices[i]]

    print("evectors shape NxK(U_k[1000 x k])= ", evectors.shape)

    #Step5: generate the 'numpc' dimensional representation
    representation = dot(evectors.T, Data)
    print("representation shape KxM (Y[kx784])= ",representation.shape)
        
    return evectors,representation
#############################################################################
# 					Preparing dataset for PCA
#############################################################################
dataset = sio.loadmat('/Users/bizzarohd/Desktop/ModernMachineLearning_ELEG845/7-Mod 7 Dimensionality Reduction-PCA+KPCA/7-Computer Assignment Module 7(Python+Matlab)/Python files/data_minst.mat')
PCA_examples = 1000
MNIST = dataset['train_feats'][:PCA_examples,:]
print('The dimension of MNIST is (%d, %d)'%(MNIST.shape[0],MNIST.shape[1]))

full_pc = size(MNIST,axis=1) # numbers of all the principal components
print('The number of all the principal compoents is %d'%(full_pc))

#############################################################################
#                       Main Function
#############################################################################
i = 0
# Generate an error vector to contain the error between the original image and the restored image form PCA
pc_num = []
error_vector = []
# Generate a vector to show the number of principal components we want to calculate
start_pc = 10
step_pc = 10
end_pc = 320
print('The number of principal components we want to calculate')
print(range(start_pc,end_pc,step_pc))
num_pc = len(range(start_pc,end_pc,step_pc))
# Generate an array to contain the image restoration given different principal components
MNIST_vector = np.zeros((num_pc,MNIST.shape[0],MNIST.shape[1]))
print(MNIST.shape)

#############################################################################
# Generate a for loop for realizing PCA image restoration given different principal components
for numpc in range(start_pc,end_pc,step_pc):

#############################################################################
# You are required to complete the PCA function based on the definition of the inputs and outputs
# PCA function: 'princomp'
# Input: 'MNIST' with dimension M \times N, where M is the number of samples, N is the number of features. Here M = 1000 and N = 784
# Input: 'numpc' is the number of principal components
# Output: 'evectors' consists of 'numpc' eigenvectors corresponding to the 'numpc' largest eigenvalues, the dimension of which is M \times numpc
# Output: 'representation' is the representation of MNIST, the dimension of which is numpc \times N
# You are required to complete the PCA function based on the above defintion of the inputs and outputs

    evectors, representation = princomp(MNIST,numpc)
    #############################################################################

    # Image restoration based on the eigenvectors and the representation
    MNIST_r = dot(evectors,representation) + mean(MNIST,axis=0) # image reconstruction
    # The image restoration could be complex values, we derive the magnitute to visualize the image.
    MNIST_vector[i] = np.absolute(MNIST_r)
    i = i + 1

    # Calculate the error between the original image and the restored image
    error = linalg.norm(MNIST-MNIST_r)/PCA_examples
    print('The number components is %03d, error is %.2f'%(numpc,error))
    error_vector.append(error)
    pc_num.append(numpc)

#############################################################################
#                      Visualize the PCA restoration
#############################################################################
# Each row shows three different restored images, the original one is randomly chosen from the MNIST dataset
# Each column shows the restored images given different principal components
"""num_images_display = 10
index = np.random.randint(low=0, high=PCA_examples, size=num_images_display)
for i in range(num_pc):
	for j in range(num_images_display):
		#print(j+i*num_images_display)
		plt.subplot(num_pc,num_images_display,j+i*num_images_display+1)
		plt.imshow(np.reshape(MNIST_vector[i][index[j]],(28,28)).T)
		plt.xticks([])
		plt.yticks([])
		if j == 1:
			plt.xlabel('PC %3d Error %.2f'%(start_pc+step_pc*(i),error_vector[i]))
		plt.gray()
plt.show()
plt.tight_layout()"""

fix, ax = plt.subplots()
ax.plot(pc_num, error_vector)
ax.set_xlabel("Principle Component Number")
ax.set_ylabel("Error")
ax.set_title("Principle Component Analysis on MNIST Dataset")
plt.show()



