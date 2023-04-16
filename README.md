# ModernMachineLearning_ELEG845
7 modules worth of material including my python computer assignments and homework assignments

Nice


# below were my struggles from module 2

still having trouble figuring out how best to convert the math notation into code. For loops are most intuitive for me but result in algorithms taking way to long to converge. For example lets consider the derivitive of the negative log liklihood function.  The goal is to determine beta that maximizes this function. We do this by setting the derivitve equal to 0 and using this "slope" or gradient to guide our function to its local minimum. This will result in the optimal betas that fit our model. 

For example, consider the simple 2D case where we have an output y, and an input x. This algorithm aims to find the betas, in this case B_0 and B_1 that result in predicting y given x. Its the common y = mx+b or in this case y = B_1 x + B_0.

The derivitive equation with respect to beta below is as follows:  it is the negative sum of our output (y) minus the probability of achiving our a predicted y given an x value. This is mulitple by our input x value. 


<img width="1011" alt="Screenshot 2023-04-16 at 1 23 06 PM" src="https://user-images.githubusercontent.com/50302377/232329774-4dcc291a-3ab5-4bc4-ac84-c5d22ec8aedd.png">


Thus, the question becomes how to convert this "math" into code.  The examples below assume x is a  matrix of length 5000 x 2 and y is a vector of length 5000 x 1.  In other words we have 5000 observations of coordinates (x_1, x_2) and 5000 outputs that dsecribe whether a coordinate is above or below a boundary line. In addition we have 3 betas. A beta_0 which is the intercept, a beta_1 for x_1 and a beta_2 for x_2. One approach is the for loop approach which is essentially:

'''
#for loop way
dcost = 0
n = len(x_batch)
for i in range(n):
      z = np.dot(beta.T, x_batch[:,i])  
      dcost -= np.dot(x[:,i] , (y[i] - sigmoid(z)) ) 
return dcost

beta_next = beta_current - alpha * dcost
'''


where the sigmoid function is the logistic function:

'''
def sigmoid(z):
       return 1 / (1 + np.exp(-z))
'''

Or we could do it the numpy matrix way:

'''
z = np.dot(beta.T, x_batch)
dcost = -np.dot(x , (y - sigmoid(z)))
beta_next = beta - alpha * dcost
'''
      

The result of this algorithm is the opitmal beta values that segrate our data set. That is, a line as shown below in green. Now, our model is able to classify whether an incoming coordinate (x_1, x_2) is part of blue color or red color. 

<img width="635" alt="Screenshot 2023-04-16 at 1 37 46 PM" src="https://user-images.githubusercontent.com/50302377/232330543-dde71dc3-45ab-46b6-9336-8904831959df.png">
