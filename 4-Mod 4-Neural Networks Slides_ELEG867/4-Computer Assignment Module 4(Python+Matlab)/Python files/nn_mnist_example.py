import tensorflow.compat.v1 as tf
import numpy as np
import numpy.matlib
import numpy as np
import matplotlib.pyplot as plt

import input_data



##########################
### Loading MNIST Dataset
##########################
path = "/Users/bizzarohd/Desktop/ModernMachineLearning_ELEG845/4-Mod 4-Neural Networks Slides_ELEG867/4-Computer Assignment Module 4(Python+Matlab)/Python files"
mnist = input_data.read_data_sets(path, one_hot=True)


##########################
### Hpyer parameters
##########################

# Hyperparameters
training_epochs = 100            
learning_rate = 0.1
batch_size = 100

img_h = img_w = 28              # MNIST images are 28x28
n_input = 784                   # 28x28 = 784
n_hidden_1 = 32
n_hidden_2 = 16
n_classes = 10


##########################
### Neural Network DEFINITION
##########################

g = tf.Graph()
with g.as_default():
    # Input data 
    tf_x = tf.placeholder(tf.float32, [None, n_input], name='features')
    # Input labels 
    tf_y = tf.placeholder(tf.float32, [None, n_classes], name='targets')

    # The first hidden layer
    fc1 = tf.layers.dense(inputs=tf_x, units=n_hidden_1, activation=tf.nn.sigmoid, name='fc1')
    # The second hidden layer
    fc2 = tf.layers.dense(inputs=fc1, units=n_hidden_2, activation=tf.nn.sigmoid, name='fc2')
    # The third hidden layer
    out_layer = tf.layers.dense(inputs=fc2, units=n_classes, activation=None, name='out_layer')

    # Cost and optimizer
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=tf_y)
    cost = tf.reduce_mean(loss, name='cost')
    # Training method is gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cost, name='train')

    # Prediction
    correct_prediction = tf.equal(tf.argmax(tf_y, 1), tf.argmax(out_layer, 1))
    # Prediction accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    # create saver object
    saver = tf.train.Saver()

    
##########################
### TRAINING & EVALUATION
##########################
cost_vector = np.zeros(training_epochs)
train_acc_vector = np.zeros(training_epochs)
test_acc_vector = np.zeros(training_epochs)

print(mnist.train.images.shape)
print(mnist.validation.images.shape)
print(mnist.test.images.shape)


with tf.Session(graph=g) as sess:

    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = mnist.train.num_examples // batch_size


        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            
            _, c = sess.run(['train', 'cost:0'], feed_dict={'features:0': batch_x,
                                                            'targets:0': batch_y})
            avg_cost += c
 
        train_acc_vector[epoch] = sess.run('accuracy:0', feed_dict={'features:0': mnist.train.images,
                                                      'targets:0': mnist.train.labels})
        test_acc_vector[epoch] = sess.run('accuracy:0', feed_dict={'features:0': mnist.validation.images,
                                                      'targets:0': mnist.validation.labels})
        cost_vector[epoch] = avg_cost
     
        print("Epoch: %03d | AvgCost: %.3f Train/Test ACC: %.3f/%.3f" % (epoch + 1, avg_cost / (i + 1),train_acc_vector[epoch], test_acc_vector[epoch]))


##########################
### Visualizing The Result
##########################

Fontsize = 12
fig, _axs = plt.subplots(nrows=1, ncols=2)
axs = _axs.flatten()

l01, = axs[0].plot(range(training_epochs), cost_vector,'g')
axs[0].set_xlabel('Epoch',fontsize=Fontsize)
axs[0].set_ylabel('Cost',fontsize=Fontsize)
axs[0].grid(True)

l11, = axs[1].plot(range(training_epochs), 1-train_acc_vector,'b')
l12, = axs[1].plot(range(training_epochs), 1-test_acc_vector,'r')
axs[1].set_xlabel('Epoch',fontsize=Fontsize)
axs[1].set_ylabel('Error',fontsize=Fontsize)
axs[1].grid(True)
axs[1].legend(handles = [l11, l12], labels = ['train error', 'test error'],loc = 'upper right', fontsize=Fontsize)

plt.show()
