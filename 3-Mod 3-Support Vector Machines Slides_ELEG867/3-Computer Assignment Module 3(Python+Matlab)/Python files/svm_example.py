

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import cvxopt
import cvxopt.solvers
np.random.seed(0)

#############################################################################
# The function to generate the dataset
#############################################################################
def generate_dataset(W,b,n):
    #x_batch = np.linspace(0, 2, n)
    x_batch = np.random.randn(n)
    y_batch = W * x_batch + b + 2 * np.random.randn(*x_batch.shape)
    #y_batch = W * x_batch + b
    return x_batch, y_batch

#############################################################################
# Preparing the dataset for SVM classifcation
#############################################################################
# The dimension of the dataset
n = 200
# The weight and the bias of the linear model1 y = Wx + b
Weight1 = 4
bias1 = -2
# The weight and the bias of the linear model2 y = Wx + b
Weight2 = 4
bias2 = 2


# Generating the dataset of two linear models
x_batch1, y_batch1 = generate_dataset(Weight1,bias1, n)
x_batch2, y_batch2 = generate_dataset(Weight2,bias2, n)

data_x = np.concatenate((x_batch1, x_batch2), axis=0)
data_y = np.concatenate((y_batch1, y_batch2), axis=0)

# Generating the training dataset based on the two linear modesl
data = np.append([data_x],[data_y],axis=0)
data = np.append(data,np.ones([1,data.shape[1]]),axis=0)
data = np.transpose(data)
print('The dimension of dataset is [%d, %d]' %(data.shape[0],data.shape[1]))

# Generating the training labels 
label = np.zeros(data.shape[0])
label[0:x_batch1.shape[0]] = 1
label[x_batch1.shape[0]:] = -1
print('The dimension of label is [%d, 1]' %(label.shape[0]))


#############################################################################
# The SVM Wolfe dual problem solution
# The input dataset: x_batch with dimesnion (n_samples, n_features) 
# The input dataset: y_batch with dimension (n_samples,1)
# The input value: C is the margin defined in 3.2 slides page 8 to 9
# The output vector: alpha with deimension (n_smaples,1) is defined in 3.2 slides page 4
# Note: the svm_dual function is implement based on the cone programming function in the cvxopt package (https://cvxopt.org/userguide/coneprog.html). 
# Note: comparing 3.2 slide page 4, you can understand how to solve SVM Wolfe dual based on the cone programming.
#############################################################################
print(data.shape[0])
def svm_dual(x_batch, y_batch, C):

    n_samples = x_batch.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = np.dot(x_batch[i,:], x_batch[j,:])
    
    P = cvxopt.matrix(np.outer(y_batch,y_batch) * K)
    q = cvxopt.matrix(np.ones(n_samples) * -1)
    A = cvxopt.matrix(y_batch, (1,n_samples))
    b = cvxopt.matrix(0.0)

    tmp1 = np.diag(np.ones(n_samples) * -1)
    tmp2 = np.identity(n_samples)
    G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
    tmp1 = np.zeros(n_samples)
    tmp2 = np.ones(n_samples) * C
    h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

    # solve QP problem
    print('SVM Wolfe Dual problem is solving ...')
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    alpha = np.ravel(solution['x'])

    return alpha


#############################################################################
# The classifcation_ratio function
# The input weights: beta with dimension (n_features, 1). It includes the bias
# The input dataset: x_batch with dimesnion (n_samples, n_features) 
# The input dataset: y_batch with dimension (n_samples,1)
# The output scalar: accuracy is the SVM classifcation accuracy
# The output vector: svm_classifer is a vector wit dimension (n_samples,1). If a sample belongs to the first linear model, svm_classier = 1, otherwise svm_classifer = 0
# Note: 3.1 slide page 5 shows how to derive SVM classifcation based on signed distances
#############################################################################

def classifcation_ratio(beta,x_batch,y_batch):
    svm_classifer = y_batch * np.sign(np.matmul(x_batch,beta))
    right_classication = np.sum((svm_classifer + 1)/2)
    accuracy = right_classication/x_batch.shape[0]
    return accuracy, svm_classifer 



#############################################################################
#                       Main Function
#############################################################################


# Derive the alpha based on svm_dual function
alpha = svm_dual(data,label,200)

# Guarantee the value of alpha being greater than or equal to zero based on the KKT conditons (3.2 Slides, Page 4)
sv = alpha > 1e-5
alpha_nz = alpha[sv]
data_nz = data[sv]
label_nz = label[sv]

# Derive the weights of the SVM classifer based on SVM Wolfe dual (3.2 Slides, Page 4)
n_features = data_nz.shape[1]
n_samples = data_nz.shape[0]
beta = np.zeros(n_features)
print("%d support vectors out of %d points" % (len(alpha_nz), n_samples))
for i in range(n_samples):
    beta += alpha_nz[i] * label_nz[i] * data_nz[i]
#print(beta)

# Print the SVM classifcation accurarcy and results.
svm_accuracy,svm_results = classifcation_ratio(beta,data,label)
#print(svm_results)
print('the svm classifcation accuracy is %.2f%%'%(svm_accuracy*100))


#############################################################################
# Visualizing the dataset and the learned SVM classifer
#############################################################################
Data_s = sorted(zip(x_batch1,y_batch1))
X_s, Y_s = zip(*Data_s)
plt.plot(X_s,Y_s,'ro')

Data_s = sorted(zip(x_batch2,y_batch2))
X_s, Y_s = zip(*Data_s)
plt.plot(X_s,Y_s,'b*')

plt.xlabel('X')
plt.ylabel('Y')

max_x = np.max(x_batch1)
min_x = np.min(x_batch1)
xx = np.linspace(min_x, max_x, n)

weight = -beta[0]/beta[1]
bias = -beta[2]/beta[1]

yy = weight * xx + bias
plt.plot(xx,yy,'g')
plt.title('The blue/red points are the dataset, the green line is the learned SVM classifer')

plt.show()