import tensorflow.compat.v1 as tf
import scipy.io as sio
import numpy as np
import numpy.matlib
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras import datasets
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models

##########################
### Loading mindst Dataset
##########################

train_digits = sio.loadmat('/Users/bizzarohd/Desktop/ModernMachineLearning_ELEG845/3-Mod 3-Support Vector Machines Slides_ELEG867/3-Computer Assignment Module 3(Python+Matlab)/Python files/data_minst.mat')
# The dataset has two compoents: train_data and train_labels
# Train_data has 5000 digits images, and the dimension of each image is 28 * 28 (784 * 1)
# Train_lables has 5000 labels, 
train_data = train_digits['train_feats']
train_data -= np.mean(train_data,axis=0)
train_data /= np.std(train_data)
# There are totally 5000 images and the dimension of each image is 784 * 1
num_train_images = train_data.shape[0]
dim_images = train_data.shape[1]

train_bias = np.ones([num_train_images,1])
data_x = np.concatenate((train_data, train_bias), axis=1)

# Generating the labels for the one-verus-all logistic regression
labels = train_digits['train_labels']

data_y = np.zeros([num_train_images,10])
for i in range(num_train_images):
	data_y[i,labels[i][0]-1] = 1


train_samples = int(round(data_x.shape[0]*0.95))
train_images = data_x[:train_samples]
train_labels = data_y[:train_samples]

print('The dimension of train dataset is [%d, %d]' %(train_images.shape[0],train_images.shape[1]))
print('The dimension of train labels  is [%d, %d]' %(train_labels.shape[0],train_labels.shape[1]))

test_images = data_x[train_samples:]
test_labels = data_y[train_samples:]
#print("\n", test_data_y)
#print(labels[train_samples:]-1)

print('The dimension of testing dataset is [%d, %d]' %(test_images.shape[0],test_images.shape[1]))
print('The dimension of testing labels  is [%d, %d]' %(test_labels.shape[0],test_labels.shape[1]))
print(test_labels[0])

##########################
### Hpyer parameters
##########################

# Hyperparameters
training_epochs = 10            
learning_rate = 0.1
batch_size = 100
##########################
### Neural Network DEFINITION
##########################


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))



print(model)
#fig, ax = plt.subplots()

#image = train_images[1,:784]
#image = np.reshape(image,(28,28,1))
#pred = model.predict(image)
#print(pred)

#predict

#ax.imshow(image,cmap=plt.get_cmap("gray"))

#fig.tight_layout()
#plt.show()

