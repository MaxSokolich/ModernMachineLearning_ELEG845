

import numpy as np
import matplotlib.pyplot as plt

#############################################################################
# ***************************************************************************
# you should create you own GRADIENT DESCENT algorithm in the location specified below
# ***************************************************************************
#############################################################################

#############################################################################
# The function to generate the dataset
def generate_dataset(W,b,n):
    #x_batch = np.linspace(0, 2, n)
    x_batch = np.random.randn(n)
    y_batch = W * x_batch + b + np.random.randn(*x_batch.shape)
    return x_batch, y_batch

#############################################################################
# Preparing the dataset for linear regression
# The dimension of the dataset
n = 1000
# The weight of the linear model y = Wx + b
Weight = 2
# The bias of the linear model y = Wx + b
bias = -3

x_batch, y_batch = generate_dataset(Weight,bias, n)
print(type(x_batch))
print(y_batch.shape)

#############################################################################
# Hyper-parameters
training_epochs = 1000
learning_rate = 0.0001            # The optimization initial learning rate

#############################################################################
# Here you create you own GRADIENT DESCENT algorith for the linear regression model
# Hint: please check the slides of the gradient descent (1.3, the 9th to 10th page)
#############################################################################
"""
notes:
p=2
y size = n x 1 = [1000 x 1]
X size = n x (p+1) = [1000 x 2] ?
beta size = (p+1) x 1 = [2 x 1]

basically our output hb(x) = beta[0] + xbatch * beta[1]
"""

def gradient_descent(beta, lr, x_batch, y_batch):
    alpha = lr

    slope = beta[0]
    intercept = beta[1]

    hb = slope * x_batch + intercept                            #define output
    cost = (1/2) * np.sum([val**2 for val in (hb - y_batch)])   # define cost function aka J(beta)
  
    dslope = np.sum((hb - y_batch) * x_batch)                   # differentiate with respect to the weght component beta[0] aka slope
    dintercept = np.sum((hb - y_batch))                         # differentiate with respect to the bias component beta[1] aka intercept

    slope_next = slope - alpha * dslope                         # put all the terms together to weight/slope for the next slope term
    intercept_next = intercept - alpha * dintercept             # put all the terms together to bias/intercept for the next slope term


    weight_next = slope_next
    bias_next =   intercept_next 
    beta_next = [weight_next,bias_next]

    return cost, beta_next

#############################################################################


#############################################################################
#                       Main Function
#############################################################################
beta = np.random.randn(2)
cost = 0

for epoch in range(training_epochs):
    cost, beta_next = gradient_descent(beta,learning_rate,x_batch,y_batch)
    print('Epoch %3d, cost %.3f, weight %.3f bias is %.3f' % (epoch+1,cost,beta_next[0],beta_next[1]))
    beta = beta_next

#############################################################################
# Visualizing the dataset and the learned linear model
Data_s = sorted(zip(x_batch,y_batch))
X_s, Y_s = zip(*Data_s)
plt.plot(X_s,Y_s,'o')
plt.ylabel('Y')
plt.ylabel('Y')
plt.title('The blue points are the dataset, the red line is the learned linear model')
#plt.hold(True)

max_x = np.max(x_batch)
min_x = np.min(x_batch)
xx = np.linspace(min_x, max_x, n)

weight = beta[0]
bias = beta[1]

yy = weight * xx + bias
plt.plot(xx,yy,'r')
plt.show()



