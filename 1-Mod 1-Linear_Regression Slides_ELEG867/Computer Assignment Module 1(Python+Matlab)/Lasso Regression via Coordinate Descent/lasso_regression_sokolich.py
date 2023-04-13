
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
#############################################################################
# ***************************************************************************
# you should create you own LASSO algorithm in the location specified below
# ***************************************************************************
#############################################################################

#############################################################################
# The function to generate the dataset
def generate_dataset(n,W,b=0):
    #x_batch = np.linspace(0, 2, n)
    x_batch = np.random.randn(n,W.shape[0])
    y_batch =  x_batch.dot(W) + b + np.random.randn(n)*2
    return x_batch, y_batch

#############################################################################
# Preparing the dataset for linear regression
# The dimension of the dataset
n = 30
# The weight of the linear model y = Bx
Beta_true = np.array([10, -2, 0, -3, 0, 5, 0, 0])
x_batch, y_batch = generate_dataset(n,Beta_true)

#############################################################################
# Here you create you own LASSO algorithm for the linear regression model
# Hint: please check the slides of the gradient descent (1.5, the 7th to 14th page)
#############################################################################
"""
notes:
p=7
n= 30
y size = n x 1 = [30 x 1]
X size = n x (p+1) = [30 x 8] ?
beta size = (p+1) x 1 = [8 x 1]
"""

def lasso_regression(parmsl, lamda, x_batch, y_batch):

    betas = parmsl
    def soft_threshold(a, b):
        #could be g from slide 14
        #or beta from slide 15
        if a > b:
            return a - b
        elif a < -b:
            return a + b
        else:
            return 0
  
    #define cost function with L1 regulariation
    cost_b4_l1 = (1/2) *  np.sum(   (y_batch - (np.sum(np.dot(x_batch , betas)))) **2   )
    L1_Regularization = lamda * np.sum(np.abs(betas))
    cost = cost_b4_l1  +  L1_Regularization

   
    new_betas = []
    for i in range(len(betas)):
        XiT = x_batch[:,i].T
        X_i = np.delete(x_batch,i,1)
        B_i = np.delete(betas,i)
        y = y_batch
        Xi = x_batch[:,i]


        gi =  np.dot(XiT , (y - np.dot(X_i , B_i) )) / np.dot(XiT , Xi) 


        z = lamda / np.linalg.norm(Xi)**2

        new_beta = soft_threshold(gi, z)
        new_betas.append(new_beta)


    weights = np.array(new_betas)
    weight_next = weights

    return cost, weight_next
    
    


    #define prediction
    y_pred = np.dot(x_batch, betas)

    #define cost function with L1 regulariation
    cost_b4_l1 = (1/2) *  np.sum(   (y_batch - (np.sum(x_batch * betas))) **2   )
    L1_Regularization = lamda * np.sum(np.abs(betas))
    cost = cost_b4_l1  +  L1_Regularization

    #define derivtive
    dcost = np.dot(x_batch.T, y_pred - y_batch) + lamda * np.sign(betas)

    #calcualte new betas
    new_betas = betas - 0.0001  * dcost
    
    weights = np.array(new_betas)
    weight_next = weights

    #return cost, weight_next




#############################################################################


#############################################################################
#                       Main Function
#############################################################################
# Hyper-parameters
training_epochs = 1000
weight_var = 1
#beta = np.random.randn(*Beta_true.shape)*weight_var
beta = np.zeros(*Beta_true.shape)


cost1 = 0
cost2 = 0
cost3 = 0

#lamda = 0.0001
lamda1 = 30
beta_show1 = np.zeros((training_epochs,Beta_true.shape[0]))
lamda2 = 3
beta_show2 = np.zeros((training_epochs,Beta_true.shape[0]))
lamda3 = 0.0001
beta_show3 = np.zeros((training_epochs,Beta_true.shape[0]))

#############################################################################
# Learning three lasso models given different lambda values

beta1 = beta
for epoch in range(training_epochs):   
    beta_show1[epoch] = beta1
    cost, beta_next = lasso_regression(beta1,lamda1,x_batch,y_batch)
    #print('Epoch %3d, cost %.3f, beta[0] %.3f beta[1] is %.3f' % (epoch+1,cost,beta1[0],beta1[1]))
    beta1 = beta_next
cost1 = cost
print(beta1, " ")


beta2 = beta
print(beta)
for epoch in range(training_epochs):   
    beta_show2[epoch] = beta2
    cost, beta_next = lasso_regression(beta2,lamda2,x_batch,y_batch)
    #print('Epoch %3d, cost %.3f, beta[0] %.3f beta[1] is %.3f' % (epoch+1,cost,beta2[0],beta2[1]))
    beta2 = beta_next
cost2 = cost
print(beta2, " ")


beta3 = beta
for epoch in range(training_epochs):   
    beta_show3[epoch] = beta3
    cost, beta_next = lasso_regression(beta3,lamda3,x_batch,y_batch)
    #print('Epoch %3d, cost %.3f, beta[0] %.3f beta[1] is %.3f' % (epoch+1,cost,beta3[0],beta3[1]))
    beta3 = beta_next
cost3 = cost
print(beta3, "  ")


#############################################################################
# Visualizing the dataset and the learned lasso model

fig, _axs = plt.subplots(nrows=1, ncols=3)
axs = _axs.flatten()

title_str = r'$\lambda$ %.2f, cost %.2f' % (lamda1,cost1)
axs[0].grid(True)
axs[0].set_title(title_str)
axs[0].semilogx(beta_show1)
axs[0].legend(('Beta0', 'Beta1', 'Beta2','Beta3', 'Beta4', 'Beta5','Beta6', 'Beta7'),
           loc='upper center')

title_str = r'$\lambda$ %.2f, cost %.2f' % (lamda2,cost2)
axs[1].grid(True)
axs[1].set_title(title_str)
axs[1].semilogx(beta_show2)

title_str = r'$\lambda$ %.4f, cost %.2f' % (lamda3,cost3)
axs[2].grid(True)
axs[2].set_title(title_str)
axs[2].semilogx(beta_show3)

plt.show()



