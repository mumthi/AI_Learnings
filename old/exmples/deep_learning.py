# Larger CNN for the MNIST Dataset
import os
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

from keras import backend as K
import numpy as np
# load data
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# define the larger model
def larger_model():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu', name='layer_1'))
    model.add(MaxPooling2D(name='layer_2'))
    model.add(Conv2D(15, (3, 3), activation='relu', name='layer_3'))
    model.add(MaxPooling2D( name='layer_4'))
    model.add(Dropout(0.2, name='layer_5'))
    model.add(Flatten( name='layer_6'))
    model.add(Dense(128, activation='relu', name='layer_7'))
    model.add(Dense(50, activation='relu', name='layer_8'))
    model.add(Dense(num_classes, activation='softmax', name='layer_9'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_layer_outputs(model, layer_name, input_data, learning_phase=1):
    outputs   = [layer.output for layer in model.layers if layer_name in layer.name]
    layers_fn = K.function([model.layers[0].input], [model.layers[1].output])
    return layers_fn(input_data)


model_path = 'Resources/deep_forward'
if not os.path.exists(model_path):
    # build the model
    model = larger_model()
    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Large CNN Error: %.2f%%" % (100 - scores[1] * 100))
    model.save(model_path)
else:
    model = load_model(model_path)
x = X_test[0]
inp = model.input
layer_outs = get_layer_outputs(model, 'layer_2', x)
plt.imshow(layer_outs)
plt.show()
# Testing


