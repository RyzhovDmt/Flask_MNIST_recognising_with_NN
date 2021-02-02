
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras import backend as K
import distutils
import cv2


batch_size = 128
num_classes = 10
epochs = 12
height, width = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
def create_model_maxpool_dropout():
  inputs = keras.Input(shape=(height, width, 1, ), name='X')
  conv_layer = layers.Conv2D(32, (3,3), strides=(1,1), padding="same",   activation='relu',  name='conv_1')(inputs)
  pool = layers.MaxPool2D((2,2), (2,2))(conv_layer)
  drop = layers.Dropout(0.5)(pool)
  conv_layer = layers.Conv2D(32, (3,3), strides=(1,1), padding="same",   activation='relu',  name='conv_2')(drop)
  pool = layers.MaxPool2D((2,2), (2,2))(conv_layer)
  flat = layers.Flatten()(pool)
  outputs = tf.keras.layers.Dense(10, activation="softmax")(flat)
  return keras.Model(inputs=inputs, outputs=outputs)


model = create_model_maxpool_dropout()
model.compile(optimizer=keras.optimizers.Adam(),  loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('model.h5')
#model_json = model.to_json()

#with open("model.json", "w") as json_file:
  #json_file.write(model_json)

#model.save_weights("model.h5")