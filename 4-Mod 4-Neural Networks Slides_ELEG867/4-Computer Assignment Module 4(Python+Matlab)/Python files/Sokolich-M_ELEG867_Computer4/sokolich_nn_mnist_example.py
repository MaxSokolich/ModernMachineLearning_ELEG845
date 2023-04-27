import tensorflow.compat.v1 as tf

import numpy as np
import numpy.matlib
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical

##########################
### Loading CIFAR-10 Dataset
##########################

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1, reshape
train_images, test_images = [train_images / 255.0, test_images / 255.0]

train_images = np.array(random.sample(list(train_images),5000)) 
train_labels = np.array(random.sample(list(train_labels),5000)) 

test_images = np.array(random.sample(list(test_images),1000)) 
test_labels = np.array(random.sample(list(test_labels),1000)) 

#test_images = train_images[:10]
#test_labels = train_labels[:10] 

#train_images = train_images[:100]
#train_labels = train_labels[:100]



class_names = ['airplane', 'automobile', 'bird', 'cat',
                'deer','dog', 'frog', 'horse',
                'ship', 'truck']


train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)
print(train_labels[0])

print(train_images.shape)
print(train_labels.shape)

print(test_images.shape)
print(test_labels.shape)

##########################
### Hpyer parameters
##########################

# Hyperparameters
training_epochs = 500            
learning_rate = 0.1
batch_size = 100

img_h = img_w = 32              # CIFAR-10 images are 32x32
n_input = 32, 32, 3                   # 1024 x 1


n_hidden_1 = 64
n_hidden_2 = 32
n_classes = 10

##########################
### Neural Network DEFINITION
##########################

g = tf.Graph()
with g.as_default():
    # Input data 
    tf_x = tf.placeholder(tf.float32, [None,32,32,3], name='features')
    # Input labels 
    tf_y = tf.placeholder(tf.float32, [None,n_classes], name='targets')


    #figurd out need to flatten this for some reason other wise the logits/out_layer will be 1024 x 10 instead of batch_size x 10
    flattened = tf.layers.flatten(tf_x, name='flatten')
    # The first hidden layer
    fc1 = tf.layers.dense(inputs=flattened, units=n_hidden_1, activation=tf.nn.sigmoid, name='fc1')
    # The second hidden layer
    fc2 = tf.layers.dense(inputs=fc1, units=n_hidden_2, activation=tf.nn.sigmoid, name='fc2')
# The third hidden layerqqqq                                                                                                                Q                                   q                                                                                           Q                                                                                               q                                   Q                                                                                           Q   qq                                                                                                                          QQ              q                                           q                                       q                                                                           q                                                                                                                                                                                                                   Q           QQ                                                                                                                                                                                                                                                                      Q                                                   q                                                                                   Q                                                                                           Q                                                                                                                   Q   Q                                                   QQ  Q                                                                   Q                                                                                                                                                                                                                                                                                                                   q       Q                                                                                                                                                                                                                       Q       q                                                                   q                                                                                                           Q                                                                                                       Q                                                                                                                                                                                                                   Q                                                                                                                                                                                                   q                                                                           q                                                                                                                                                                   q                                                                                       q                                                                                                                                                                   Q                                                                                   q                                                                           q                                                                                                   q                                                                                       QQ                          q                                                                                                                                                                                                                           q                                                                                   qq                                                                  q                                           q                                                       q                                               Q               q                                                                                                   q                                           q                                                               q                                               Q   qq          Q   q                                   q                   Q               q                       Q   q                       Q           q                                   Q               q                                                           q                           Q           q                                                                                                                                                                                               q                                                                                                           q                                                                                                               q                                                                                                                                                                                                                                                                                                                                                           q                                                                               Q                                                                                                                                                                                                                                               q                                                                               Q                                                               Q                                                                                               Q                                                                                                                                                                                                                                                                                                                                                                                                                                   q                                                                                                                                                                                                                                                                                                                                                                                                                               q                                                                                   q                                                                               q                                                                                                                                                                                                       q           
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




with tf.Session(graph=g) as sess:

    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = train_images.shape[0] // batch_size
        print(total_batch)
        for i in range(total_batch):
       
            bx = train_images[i*batch_size:(i+1)*batch_size]
            by = train_labels[i*batch_size:(i+1)*batch_size]
        
            _, c = sess.run(['train', 'cost:0'], feed_dict={'features:0': bx, 'targets:0': by})
            avg_cost += c
        
        train_acc_vector[epoch] = sess.run('accuracy:0', feed_dict={'features:0': train_images,
                                                      'targets:0': train_labels})
        
        test_acc_vector[epoch] = sess.run('accuracy:0', feed_dict={'features:0': test_images,
                                                      'targets:0': test_labels})
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
