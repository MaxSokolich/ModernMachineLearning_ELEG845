
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import cvxopt
import cvxopt.solvers

np.random.seed(1)
#############################################################################
# Preparing the MNIST dataset for SVM classifcation
#############################################################################
train_digits = sio.loadmat('/Users/bizzarohd/Desktop/ModernMachineLearning_ELEG845/3-Mod 3-Support Vector Machines Slides_ELEG867/3-Computer Assignment Module 3(Python+Matlab)/Python files/data_minst.mat')
# The dataset has two compoents: train_data and train_labels
# Train_data has 5000 digits images, and the dimension of each image is 28 * 28 (784 * 1)
# Train_lables has 5000 labels, 
train_data = train_digits['train_feats']
train_data -= np.mean(train_data,axis=0)
train_data /= np.std(train_data)
# There are totally 5000 images and the dimension of each image is 784 * 1
num_train_images = train_data.shape[0]
dim_images = train_data.shape[1]

train_bias = np.ones([num_train_images,1])
data_x = np.concatenate((train_data, train_bias), axis=1)

# Generating the labels for the one-verus-all logistic regression
labels = train_digits['train_labels']
data_y = np.zeros([num_train_images,10])-1
for i in range(num_train_images):
	data_y[i,labels[i][0]-1] = 1


train_samples = int(round(data_x.shape[0]*0.95))
train_data_x = data_x[:train_samples]
train_data_y = data_y[:train_samples]

print('The dimension of train dataset is [%d, %d]' %(train_data_x.shape[0],train_data_x.shape[1]))
print('The dimension of train labels  is [%d, %d]' %(train_data_y.shape[0],train_data_y.shape[1]))

test_data_x = data_x[train_samples:]
test_data_y = data_y[train_samples:]
#print("\n", test_data_y)
#print(labels[train_samples:]-1)

print('The dimension of testing dataset is [%d, %d]' %(test_data_x.shape[0],test_data_x.shape[1]))
print('The dimension of testing labels  is [%d, %d]' %(test_data_y.shape[0],test_data_y.shape[1]))



#############################################################################
# The SVM Wolfe dual problem solution
# The input dataset: x_batch with dimesnion (n_samples, n_features) 
# The input dataset: y_batch with dimension (n_samples,1)
# The input value: C is the margin defined in 3.2 slides page 8 to 9
# The output vector: alpha with deimension (n_smaples,1) is defined in 3.2 slides page 4
# Note: the svm_dual function is implement based on the cone programming function in the cvxopt package (https://cvxopt.org/userguide/coneprog.html). 
# Note: comparing 3.2 slide page 4, you can understand how to solve SVM Wolfe dual based on the cone programming.
#############################################################################

def svm_dual(x_batch, y_batch, C):

    n_samples = x_batch.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = np.dot(x_batch[i,:2], x_batch[j,:2])
            #K[i,j] = np.dot(x_batch[i,:785], x_batch[j,:785])

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

def classifcation_ratio(beta,x_batch,y_batch):
    svm_classifer = y_batch * np.sign(np.matmul(x_batch,beta))
    right_classication = np.sum((svm_classifer + 1)/2)
    accuracy = right_classication/x_batch.shape[0]
    return accuracy, svm_classifer 

def classification_accuracy(beta, x_data, y_data):
    svm_classifier = np.sign(np.dot(x_data, beta))
    correct_classifications = np.sum(svm_classifier == y_data)
    accuracy = correct_classifications / len(y_data)
    return accuracy
#############################################################################
#                       Main Function
#############################################################################
# In the computer assignment, you should complete the main functino by yourself
# In the main function, you need to finish three tasks
# (1) use the one-vs-all method to train SVM models to get the corresponding weights.
# (2) use the one-vs-all method to test the learned SVM modesl to check the learned models.
# (3) obtain the training accuracy and testing accuracy during the above two tasks.
# Note:
# (0) you should install the cvxopt package into your environment, the command is 'conda install -c conda-forge cvxopt'
# (1) you can reuse the code in the svm_example.py file to solve Wolfe Dual problem for learning weights.
# (2) MNIST dataset is non-sperable case, you can refer 3-2 slides Page 7-8 to complete the classifcation.
# (3) you can refer 3-1 slides Page 5-8 to design classifcation accuracy function.
# (4) The training accuracy should be around 80% and the testing accuarcy should be around 60%.
# (5) you can resue the code from Model 2 to solve the one-vs-all problem.
# (6) you should use the pre-defined training dataset and testing dataset to train and to test SVM models



#############################################################################
# Training SVM models to get the weigts 


def train_one_vs_all_svm(train_data_x, train_data_y, C): 
    svm_weights = {}
    for i in range(10):
        print('Training SVM model for digit %d...' % i)
        y = train_data_y[:,i]#.reshape(-1,1)
        x = train_data_x

        alpha = svm_dual(x,y,C)
        # Guarantee the value of alpha being greater than or equal to zero based on the KKT conditons (3.2 Slides, Page 4)
        sv = alpha > 1e-5
        alpha_nz = alpha[sv]
        data_nz = x[sv]
        label_nz = y[sv]

        # Derive the weights of the SVM classifer based on SVM Wolfe dual (3.2 Slides, Page 4)
        n_features = data_nz.shape[1]
        n_samples = data_nz.shape[0]
        beta = np.zeros(n_features)

        print("%d support vectors out of %d points" % (len(alpha_nz), n_samples))
        
        for j in range(n_samples):
            beta += alpha_nz[j] * label_nz[j] * data_nz[j]
        
        acc,_ = classifcation_ratio(beta, x, y)
        print('the svm classifcation accuracy is {}'.format(acc*100))
        print(classification_accuracy(beta, x, y))

        svm_weights[str(i)] = beta
    
    return svm_weights
#############################################################################
# Testing SVM models

def test_one_vs_all_svm(test_data_x, test_data_y, svm_weights):
    #take average
    train_accuracy = 0
    test_accuracy = 0

    for i in range(10):
        train_acc, _ = classifcation_ratio(svm_weights[str(i)], train_data_x, train_data_y[:, i])
        train_accuracy += train_acc
        test_acc, _ = classifcation_ratio(svm_weights[str(i)], test_data_x, test_data_y[:, i])
        test_accuracy += test_acc

    train_accuracy /= 10
    test_accuracy /= 10

    return train_accuracy, test_accuracy 

#############################################################################
# Main loop throught different C
train_acc_C = []
test_acc_C = []
C_list = [0.00001, 0.05, 10] 
for c in C_list:
    print("#################  FOR C = {} ######################## ".format(c))
    svm_weights = train_one_vs_all_svm(train_data_x, train_data_y, c)
    train_accuracy, test_accuracy = test_one_vs_all_svm(test_data_x, test_data_y, svm_weights)
    train_accuracy *=100
    test_accuracy *= 100
    print('Training accuracy: %.2f%%' % (train_accuracy))
    print('Testing accuracy: %.2f%%' % (test_accuracy))
    train_acc_C.append(train_accuracy)
    test_acc_C.append(test_accuracy)
    
#plot
fig, ax = plt.subplots(1,2)
C = ["0.00001", "0.05", "10"]
ax[0].bar(C, train_acc_C)
ax[1].bar(C, test_acc_C)
ax[0].set_title("training accuracy for 3 Cs")
ax[1].set_title("test accuracy for 3 Cs")

ax[0].set_ylim([50,70])
ax[1].set_ylim([50,70])




#############################################################################
# Visualizing five images of the MNIST dataset
fig, _axs = plt.subplots(nrows=1, ncols=5)
axs = _axs.flatten()

for i in range(5):
	axs[i].grid(False)
	axs[i].set_xticks([])
	axs[i].set_yticks([])
	image = data_x[i*10,:784]
	image = np.reshape(image,(28,28))
	aa = axs[i].imshow(image,cmap=plt.get_cmap("gray"))

fig.tight_layout()
plt.show()

"""
C = 0.00005:
Training accuracy: 65.15%
Testing accuracy: 64.40%


C = 1:

58.91%
Testing accuracy: 57.88%


C = 1000:
Training accuracy: 58.82%
Testing accuracy: 57.64%


"""