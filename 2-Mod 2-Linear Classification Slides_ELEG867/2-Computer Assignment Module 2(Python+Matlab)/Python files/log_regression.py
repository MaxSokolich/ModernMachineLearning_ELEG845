

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#############################################################################
# The logistic function
def logistic(x,beta):
    y_pred = np.matmul(x.T,beta)
    logistic_prob = 1/(1 + np.exp(-y_pred))
    return logistic_prob

#############################################################################
# The function to generate the dataset
def generate_dataset(W,b,n):
    #x_batch = np.linspace(0, 2, n)
    x_batch = np.random.randn(n)
    y_batch = W * x_batch + b + np.random.randn(*x_batch.shape)
    #y_batch = W * x_batch + b
    return x_batch, y_batch

#############################################################################
# Preparing the dataset for logistic regression
# The dimension of the dataset
n = 1000
# The weight and the bias of the linear model1 y = Wx + b
Weight1 = 2
bias1 = -2
# The weight and the bias of the linear model2 y = Wx + b
Weight2 = 2
bias2 = 2


# Generating the dataset of two linear models
x_batch1, y_batch1 = generate_dataset(Weight1,bias1, n)
x_batch2, y_batch2 = generate_dataset(Weight2,bias2, n)

data_x = np.concatenate((x_batch1, x_batch2), axis=0)
data_y = np.concatenate((y_batch1, y_batch2), axis=0)

# Generating the training dataset based on the two linear modesl
data = np.append([data_x],[data_y],axis=0)
data = np.append(data,np.ones([1,data.shape[1]]),axis=0)
print('The dimension of dataset is [%d, %d]' %(data.shape[0],data.shape[1]))
# Generating the training labels 
label = np.zeros(data.shape[1])
label[0:x_batch1.shape[0]] = 1
print('The label fo the dataset is ')
print(label)


#############################################################################
# Hyper-parameters
training_epochs = 500
learning_rate = 0.0005            # The optimization initial learning rate

#############################################################################
# Here you create you own LOGISTIC REGRESSION algorith for the logistic regression model
# Hint: please check the slides of the logistic regression 
#############################################################################

def logistic_regression(beta, lr, x_batch, y_batch):


    return cost, beta_next

#############################################################################


#############################################################################
#                       Main Function
#############################################################################
beta = np.random.randn(3)
cost = 0

for epoch in range(training_epochs):
    cost, beta_next = logistic_regression(beta,learning_rate,data,label)
    print('Epoch %3d, cost %.3f, the learned weight and bias are (%.3f, %.3f)' % (epoch+1,cost,-beta_next[0]/beta_next[1],-beta_next[2]/beta_next[2]))
    beta = beta_next


# Visualizing the dataset and the learned linear model
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
plt.title('The blue/red points are the dataset, the green line is the learned linear classifer')

plt.show()