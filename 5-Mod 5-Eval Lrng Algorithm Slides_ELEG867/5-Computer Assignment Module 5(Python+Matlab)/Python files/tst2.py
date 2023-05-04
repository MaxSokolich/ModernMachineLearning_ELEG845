

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
n = 1000

degrees = [1,2,3,4,5,6,7]
fig, axs = plt.subplots(nrows=7, ncols=2)
fig, ax_vt = plt.subplots(nrows=2, ncols=1)

#axs = _axs.flatten()

JvalB = []
JtrainB = []
for deg in range(len(degrees)):
    degree = degrees[deg]
    Weight = np.random.randint(-10, 10, size=degree+1)
    x_batch, y_batch = generate_polynomial_dataset(Weight, n, degree)

    
    # Extending the dataset to matrix with dimension n * degree + 1,
    # For a single x_batch data, x_batch_ext has the form [1,x_batch^1,x_batch^2,x_batch^3,x_batch^4,x_batch^5]
    x_batch_ext = np.ones((n, degree+1))
    for ii in range(1,degree+1):
        x_batch_ext[:,ii] = x_batch ** ii

    #############################################################################
    # Seperating the dataset into the training, validating, and testing dataset
    n_train = int(0.6*n)
    n_val = int(0.1*n)
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
    training_epochs = 10000
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
    JvalB.append(cost_val_vector[-1])
    JtrainB.append(cost_train_vector[-1])


    #return beta, X_train,Y_train, X_test
    max_x = np.max(X_train[:,1])
    min_x = np.min(X_train[:,1])
    
    
    xx = np.linspace(min_x, max_x, n)
    
    #yy = beta[0] + beta[1:] @ xx  #beta[0] + beta[1] * xx**1 + beta[2] * xx**2 + beta[3] * xx**3 + beta[4] * xx**4 + beta[5] * xx**5
    
    yy = np.zeros(np.shape(x_batch))
    print("                            degreee                              "+ str(degree))
    for ii in range(len(beta)):
        yy += beta[ii] * xx ** (ii)


    axs[deg,0].set_title("Degree {}".format(degree))
    axs[deg,0].plot(X_train[:,1],Y_train,'bo')
    
    axs[deg,0].plot(xx,yy,'r')
    
    
    axs[deg,0].set_xlabel('X_train')
    axs[deg,0].set_ylabel('Y_train')
    axs[deg,0].set_title('The red line is the polynomial model for deg {}'.format(degree))

    axs[deg,1].grid(True)
    l01,=axs[deg,1].plot(cost_train_vector,'r')
    l02,=axs[deg,1].plot(cost_val_vector,'g')
    l03,=axs[deg,1].plot(cost_test_vector,'b')
    axs[deg,1].legend(handles = [l01, l02, l03], labels = ['Train_error', 'Val_error', 'test_error'],fontsize=12)
    axs[deg,1].set_title('The variation of erros')
    axs[deg,1].set_xlabel('Epoch')
    axs[deg,1].set_ylabel('Error')
    axs[deg,1].set_ylim([0,2000])



    ax_vt[0].plot(cost_val_vector, label = "deg {}".format(degree))
    ax_vt[0].set_xlabel('Epoch')
    ax_vt[0].set_ylabel('Error')
    ax_vt[0].set_ylim([0,2000])
    ax_vt[0].legend()


    ax_vt[1].plot(cost_train_vector, label = "deg {}".format(degree))
    ax_vt[1].set_xlabel('Epoch')
    ax_vt[1].set_ylabel('Error')
    ax_vt[1].set_ylim([0,2000])
    ax_vt[1].legend()



    
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(degrees, JtrainB, label = "J_train(B)")
ax.plot(degrees, JvalB, label = "J_val(B)")
ax.legend()
    
plt.show()