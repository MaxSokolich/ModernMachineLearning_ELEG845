import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

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
data = np.transpose(data)
print(data.shape)
print('The dimension of dataset is (%d, %d)' %(data.shape[0],data.shape[1]))
# Generating the training labels 
label = np.zeros([data.shape[0],1])
label[0:x_batch1.shape[0],:] = 1
print('The label fo the dataset is ')
print(label.shape)


#############################################################################
# Hyper-parameters
training_epochs = 1000
learning_rate = 0.0001               # The optimization initial learning rate

# Generate a linear model(graph)
g = tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32, shape=(2*n,3), name='x')
    y = tf.placeholder(tf.float32, shape=(2*n,1), name='y')
    W = tf.Variable(np.random.randn(3,1), dtype=tf.float32, name = "W") 
    # Linear Hypothesis
    y_pred = tf.matmul(x, W)
    # The logistic regression Cost Function
    left = tf.multiply(y, tf.log(tf.nn.sigmoid(y_pred)))
    right = tf.multiply(1-y, tf.log(tf.nn.sigmoid(1-y_pred)))
    cost = -tf.reduce_sum(tf.add(left,right))
  
    # Gradient Descent Optimizer 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 

    # create saver object
    saver = tf.train.Saver()

# Training the linear model by gradient descent
# Starting the Tensorflow Session 
with tf.Session(graph=g) as sess: 
      
    # Initializing the Variables 
    sess.run(tf.global_variables_initializer()) 
      
    # Iterating through all the epochs 
    for epoch in range(training_epochs): 
          
        # Feeding each data point into the optimizer using Feed Dictionary 
        _, yy,c,weights = sess.run([optimizer,y_pred,cost,W], feed_dict = {x: data, y: label}) 
        # Calculating the cost a every epoch 
        print('Epoch %3d, cost %.3f, the learned weight and bias are (%.3f, %.3f)' % (epoch+1,c,-weights[0]/weights[1],-weights[2]/weights[2]))
      
    # Storing necessary values to be used outside the Session 
    training_cost = sess.run(cost, feed_dict = {x: data, y: label}) 

#############################################################################
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
print(weights)
weight = -weights[0]/weights[1]
bias = -weights[2]/weights[1]

yy = weight * xx + bias

plt.plot(xx,yy,'g')
plt.title('The blue/red points are the dataset, the green line is the learned linear classifer')



beta = np.array([bias, weight])

x = np.array([0,0])  # MY NEW GUESS

Pr = np.dot(beta.T, x)

if Pr > 0:
    plt.plot(x[0], x[1], color = "y", marker = ".", markersize = 20)
else:
    plt.plot(x[0], x[1], color = "k", marker = "*", markersize = 20)


plt.show()





