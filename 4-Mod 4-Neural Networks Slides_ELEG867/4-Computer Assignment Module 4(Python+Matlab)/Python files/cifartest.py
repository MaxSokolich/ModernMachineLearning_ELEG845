import tensorflow.compat.v1 as tf
import numpy as np
import numpy.matlib
import numpy as np
import matplotlib.pyplot as plt

tf.disable_v2_behavior()

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convert labels to one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# Define hyperparameters
learning_rate = 0.01
batch_size = 128
num_epochs = 100

# Define network architecture
graph = tf.Graph()

with graph.as_default():
    # Define placeholders for input and output
    X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y = tf.placeholder(tf.float32, shape=[None, 10])



    # Define layers
    conv1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten the output from the convolutional layers
    flat = tf.reshape(pool2, [-1, 8 * 8 * 64])

    # Add two fully connected layers
    fc1 = tf.layers.dense(inputs=flat, units=256, activation=tf.nn.relu)
    fc2 = tf.layers.dense(inputs=fc1, units=128, activation=tf.nn.relu)

    # Output layer
    logits = tf.layers.dense(inputs=fc2, units=10)

    # Define loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

    # Define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # Define accuracy metric
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


cost_vector = np.zeros(num_epochs)
train_acc_vector = np.zeros(num_epochs)
test_acc_vector = np.zeros(num_epochs)


# Train the model
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
   
    for epoch in range(num_epochs):
        avg_cost = 0.
        for i in range(train_images.shape[0] // batch_size):
            batch_x = train_images[i*batch_size:(i+1)*batch_size]
            batch_y = train_labels[i*batch_size:(i+1)*batch_size]

            _, cost = sess.run([optimizer, loss], feed_dict={X: batch_x, y: batch_y})
            avg_cost += c

        acc = sess.run(accuracy, feed_dict={X: test_images, y: test_labels})
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(cost), "accuracy =", "{:.3f}".format(acc))
        train_acc_vector[epoch] = acc
        cost_vector[epoch] = avg_cost
    print("Training complete!")



Fontsize = 12
fig, _axs = plt.subplots(nrows=1, ncols=2)
axs = _axs.flatten()

l01, = axs[0].plot(range(num_epochs), cost_vector,'g')
axs[0].set_xlabel('Epoch',fontsize=Fontsize)
axs[0].set_ylabel('Cost',fontsize=Fontsize)
axs[0].grid(True)

l11, = axs[1].plot(range(num_epochs), 1-train_acc_vector,'b')
l12, = axs[1].plot(range(num_epochs), 1-test_acc_vector,'r')
axs[1].set_xlabel('Epoch',fontsize=Fontsize)
axs[1].set_ylabel('Error',fontsize=Fontsize)
axs[1].grid(True)
axs[1].legend(handles = [l11, l12], labels = ['train error', 'test error'],loc = 'upper right', fontsize=Fontsize)

plt.show()