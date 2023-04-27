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


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = [train_images / 255.0, test_images / 255.0]

train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3)))
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

model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
model.save("/Users/bizzarohd/Desktop/cifarmodel.h5")


fig, ax = plt.subplots()

image = train_images[4,:,:,:]
ax.imshow(image,cmap=plt.get_cmap("gray"))


image = np.expand_dims(image, axis = 0)

pred = model.predict(image)

class_names = ['airplane', 'automobile', 'bird', 'cat',
                'deer','dog', 'frog', 'horse',
                'ship', 'truck']
index = np.argmax(pred)
print(class_names[index])



#fig.tight_layout()
plt.show()
