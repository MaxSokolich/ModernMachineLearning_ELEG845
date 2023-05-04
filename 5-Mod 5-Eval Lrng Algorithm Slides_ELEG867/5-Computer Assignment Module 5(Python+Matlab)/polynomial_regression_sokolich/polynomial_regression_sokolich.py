

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)


#############################################################################
# The GRADIENT DESCENT algorith to learn the polynomial regression model
#############################################################################
def gradient_descent(beta, lr, x_batch, y_batch):
    # Prediction
    y_pred = x_batch.dot(beta)
    # The error between the prediction and the true y values
    error = y_pred - y_batch
    # Least Squre Error Cost Function 
    cost = np.sum(error**2) * 0.5
    # Calculate the gradient
    gradient_weight = np.transpose(x_batch).dot(error)
    # Beta update based on gradient descent
    beta_next = beta - lr*gradient_weight

    return cost, beta_next

#############################################################################

#############################################################################
# The function to generate the polynomial dataset
def generate_polynomial_dataset(W,n,degree):
    #x_batch = np.linspace(0, 2, n)
    x_batch = np.random.randn(n)
    y_batch = 0.4 * x_batch + 0.3
    p_batch = np.zeros(np.shape(x_batch))
    for ii in range(degree+1):
        p_batch += W[ii] * y_batch ** (ii)
    p_batch += np.random.randn(n)*0.5

    return y_batch, p_batch



#############################################################################
# Preparing the dataset for polynomial regression
# The dimension of the dataset

#############################################################################
# Preparing the dataset for polynomial regression
# The dimension of the dataset
n = 5000
# degree
degree = 5
# The weight of the polynomial model with degree 5: 
# y = weight[0] + weight[1]*x  + weight[2]*(x**2)  + weight[3]*(x**3)  + weight[4]*(x**4)  + weight[5]*(x**5)
Weight = [-2, 1, 10, -5, -8, 4]
# Generating the dataset for the above polynomial model with degree 5
x_batch, y_batch = generate_polynomial_dataset(Weight, n,degree)






degrees = [1,2,3,4,5,6,7]
fig, axs = plt.subplots(nrows=7, ncols=2)
fig, ax_vt = plt.subplots(nrows=2, ncols=1)
#axs = _axs.flatten()


JvalB = []
JtrainB = []
for deg in range(len(degrees)):
    degree = degrees[deg]

    x_batch_ext = np.ones((n,degree+1))
    for ii in range(1,degree+1):
        x_batch_ext[:,ii] = x_batch ** ii


    #############################################################################
    # Seperating the dataset into the training, validating, and testing dataset
    n_train = int(0.6*n)
    n_val = int(0.2*n)
    n_test = n - n_train - n_val
    # The training dataset
    X_train = x_batch_ext[:n_train]
    Y_train = y_batch[:n_train]
    # The validating dataset
    X_val = x_batch_ext[n_train:n_train+n_val]
    Y_val = y_batch[n_train:n_train+n_val]
    # The testing dataset
    X_test = x_batch_ext[n_train+n_val:]
    Y_test = y_batch[n_train+n_val:]

    print(X_train.shape)
        


    #############################################################################
    #                       Main Function
    #############################################################################
    # Hyper-parameters
    training_epochs = 2000
    learning_rate = 0.0001            # The optimization initial learning rate

    beta = np.random.randn(degree+1)
    cost_train_vector = np.zeros(training_epochs)
    cost_val_vector = np.zeros(training_epochs)
    cost_test_vector = np.zeros(training_epochs)

    # Training the polynomial model
    for epoch in range(training_epochs):
        cost_train_vector[epoch], beta_next = gradient_descent(beta,learning_rate,X_train,Y_train)
        cost_val_vector[epoch] = np.sum((X_val.dot(beta) - Y_val)**2) * 0.5
        cost_test_vector[epoch] = np.sum((X_test.dot(beta) - Y_test)**2) * 0.5
        #print('Epoch %3d, cost_train %.3f, cost_val %.3f,cost_test %.3f' % (epoch+1,cost_train_vector[epoch],cost_val_vector[epoch],cost_test_vector[epoch]))
        #print('Epoch %3d, weigts are [%.3f,%.3f,%.3f,%.3f,%.3f,%.3f]' % (epoch+1,beta_next[0],beta_next[1],beta_next[2],beta_next[3],beta_next[4],beta_next[5]))
        beta = beta_next
    print('Epoch %3d, cost_train %.3f, cost_val %.3f,cost_test %.3f' % (epoch+1,cost_train_vector[epoch],cost_val_vector[epoch],cost_test_vector[epoch]))
    print(beta)

    JvalB.append(cost_test_vector[-1])
    JtrainB.append(cost_train_vector[-1])

    #return beta, X_train,Y_train, X_test
    max_x = np.max(X_train[:,1])
    min_x = np.min(X_train[:,1])
    
    
    xx = np.linspace(min_x, max_x, n)
    yy = np.zeros(np.shape(x_batch))
    print("                            degreee                              "+ str(degree))
    for iii in range(len(beta)):
        yy += beta[iii] * xx ** (iii)
    axs[deg,0].plot(X_train[:,1],Y_train,'bo')
    axs[deg,0].plot(xx,yy,'r')
    
    
    
    axs[deg,0].set_xlabel('X_train')
    axs[deg,0].set_ylabel('Y_train')
    axs[deg,0].set_title('The red line is the polynomial model for degree {}'.format(degree))

    axs[deg,1].grid(True)
    l01,=axs[deg,1].plot(cost_train_vector,'r')
    l02,=axs[deg,1].plot(cost_val_vector,'g')
    #l03,=axs[deg,1].plot(cost_test_vector,'b')
    axs[deg,1].legend(handles = [l01, l02], labels = ['Train_error', 'Val_error'],fontsize=12)
    axs[deg,1].set_title('The variation of erros')
    axs[deg,1].set_xlabel('Epoch')
    axs[deg,1].set_ylabel('Error')
    axs[deg,1].set_ylim([0,2000])



    ax_vt[0].plot(cost_val_vector, label = "deg {}".format(degree))
    ax_vt[0].set_xlabel('Epoch')
    ax_vt[0].set_ylabel('Error')
    ax_vt[0].set_ylim([0,2000])
    ax_vt[0].set_title('validation error')
    ax_vt[0].legend()


    ax_vt[1].plot(cost_train_vector, label = "deg {}".format(degree))
    ax_vt[1].set_xlabel('Epoch')
    ax_vt[1].set_ylabel('Error')
    ax_vt[1].set_ylim([0,2000])
    ax_vt[1].set_title('training error error')
    ax_vt[1].legend()


fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(degrees, JtrainB, label = "J_train(B)", color = "green")
ax.plot(degrees, JvalB, label = "J_val(B)", color = "red")
ax.set_xlabel('Degree of polynomial p')
ax.set_ylabel('Error')
ax.legend()
    
plt.show()