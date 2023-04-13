

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() 
import numpy as np
import matplotlib.pyplot as plt

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


#############################################################################
# Hyper-parameters
training_epochs = 500
learning_rate = 0.1                # The optimization initial learning rate

# Generate a linear model(graph)
g = tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32, shape=(None, ), name='x')
    y = tf.placeholder(tf.float32, shape=(None, ), name='y')

    W = tf.Variable(np.random.randn(), name = "W") 
    b = tf.Variable(np.random.randn(), name = "b")


    # Linear Hypothesis
    y_pred = tf.add(tf.multiply(x, W), b) 
    # Mean Squared Error Cost Function 
    cost = tf.reduce_sum(tf.pow(y_pred-y, 2)) / (2 * n) 

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
        sess.run(optimizer, feed_dict = {x: x_batch, y: y_batch}) 
          
        # Displaying the result after every 50 epochs 
        if (epoch + 1) % 2 == 0: 
            # Calculating the cost a every epoch 
            c = sess.run(cost, feed_dict = {x: x_batch, y: y_batch}) 
            print('Epoch %3d, cost %.3f, weight %.3f bias is %.3f' % (epoch+1,c,sess.run(W),sess.run(b)))
      
    # Storing necessary values to be used outside the Session 
    training_cost = sess.run(cost, feed_dict = {x: x_batch, y: y_batch}) 
    weight = sess.run(W) 
    bias = sess.run(b) 

###############################################################################
# Visualizing the dataset and the learned linear model
Data_s = sorted(zip(x_batch,y_batch))
X_s, Y_s = zip(*Data_s)
plt.plot(X_s,Y_s,'o')
plt.ylabel('Y')
plt.ylabel('Y')
plt.title('The blue points are the dataset, the red line is the learned linear model')

max_x = np.max(x_batch)
min_x = np.min(x_batch)
xx = np.linspace(min_x, max_x, n)
yy = weight * xx + bias
plt.plot(xx,yy,'r')
plt.show()


