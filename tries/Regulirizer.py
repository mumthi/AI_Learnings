import os
import tensorflow as tf
from keras.models import Model
from keras.models import Sequential, load_model, save_model
from keras import Input, regularizers
from keras.layers import Dense, Flatten
import numpy as np
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from tensorflow import keras
# To stop potential randomness
(mnist_data_X_train, mnist_data_y_train), (mnist_data_X_test, mnist_data_y_test) = keras.datasets.mnist.load_data()


class Regularize:

    def __init__(self, X_train, y_train, is_save_model):
        self.data_set = None
        self.model = None
        self.X_train, self.y_train, = X_train, y_train
        self.X_train = self.X_train.reshape(60000, 784).astype("float32") / 255

        self.model_path = "Resources/Regularizer_1"
        self.define_model(is_save_model)

    def define_model(self, is_save_model):
        if is_save_model or not os.path.exists(self.model_path):
            self.model = Sequential(name="DFF-Model")  # Model
            self.model.add(Input(shape=(784), name='Input-Layer'))
            self.model.add(Dense(500, activation='relu', name='Hidden-Layer-1',
                                 kernel_initializer='HeNormal',
                                 kernel_regularizer=regularizers.l2(0.0001)))  # Hidden Layer, relu(x) = max(x, 0)
            self.model.add(Dense(500, activation='relu', name='Hidden-Layer-2',
                                 kernel_initializer='HeNormal',
                                 kernel_regularizer=regularizers.l2(0.0001)))  # Hidden Layer, relu(x) = max(x, 0)
            self.model.add(Dense(500, activation='relu', name='Hidden-Layer-3',
                                 kernel_initializer='HeNormal',
                                 kernel_regularizer=regularizers.l2(0.0001)))  # Hidden Layer, relu(x) = max(x, 0)
            self.model.add(Dense(500, activation='relu', name='Hidden-Layer-4',
                                 kernel_initializer='HeNormal',
                                 kernel_regularizer=regularizers.l2(0.0001)))  # Hidden Layer, relu(x) = max(x, 0)
            self.model.add(Dense(500, activation='relu', name='Hidden-Layer-5',
                                 kernel_initializer='HeNormal',
                                 kernel_regularizer=regularizers.l2(0.0001)))  # Hidden Layer, relu(x) = max(x, 0)
            self.model.add(Dense(10, activation='softmax',
                                 name='Output-Layer'))  # Output Layer, softmax(x) = exp(x) / tf.reduce_sum(exp(x))

            self.model.compile(optimizer='adam',  # default='rmsprop', an algorithm to be used in backpropagation
                               loss='SparseCategoricalCrossentropy',
                               # Loss function to be optimized. A string (name of loss function),
                               # or a tf.keras.losses.Loss instance.
                               metrics=['Accuracy'],
                               # List of metrics to be evaluated by the model during training and testing. Each of
                               # this can be a string (name of a built-in function), function or a
                               # tf.keras.metrics.Metric instance.
                               loss_weights=None,
                               # default=None, Optional list or dictionary specifying scalar coefficients (Python
                               # floats) to weight the loss contributions of different model outputs.
                               weighted_metrics=None,
                               # default=None, List of metrics to be evaluated and weighted by sample_weight or
                               # class_weight during training and testing.
                               run_eagerly=None,
                               # Defaults to False. If True, this Model's logic will not be wrapped in a tf.function.
                               # Recommended to leave this as None unless your Model cannot be run inside a tf.function.
                               steps_per_execution=None
                               # Defaults to 1. The number of batches to run during each tf.function call. Running
                               # multiple batches inside a single tf.function call can greatly improve performance on
                               # TPUs or small models with a large Python overhead.
                               )
            self.model.fit(
                self.X_train,  # input data
                self.y_train,  # target data
                batch_size=250,
                # Number of samples per gradient update. If unspecified, batch_size will default to 32.
                epochs=5,
                # default=1, Number of epochs to train the model. An epoch is an iteration over the entire x and y
                # data provided
                verbose='auto',
                # default='auto', ('auto', 0, 1, or 2). Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per
                # epoch. 'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
                callbacks=None,  # default=None, list of callbacks to apply during training. See tf.keras.callbacks
                validation_split=0.2,
                # default=0.0, Fraction of the training data to be used as validation data. The model will set apart
                # this fraction of the training data, will not train on it, and will evaluate the loss and any model
                # metrics on this data at the end of each epoch.
                # validation_data=(X_test, y_test), # default=None, Data on which to evaluate the loss and any model
                # metrics at the end of each epoch.
                shuffle=True,
                # default=True, Boolean (whether to shuffle the training data before each epoch) or str (for 'batch').
                class_weight=None,
                # default=None, Optional dictionary mapping class indices (integers) to a weight (float) value, used
                # for weighting the loss function (during training only). This can be useful to tell the model to "
                # pay more attention" to samples from an under-represented class.
                sample_weight=None,
                # default=None, Optional Numpy array of weights for the training samples, used for weighting the
                # loss function (during training only).
                initial_epoch=0,
                # Integer, default=0, Epoch at which to start training (useful for resuming a previous training run).
                steps_per_epoch=None,
                # Integer or None, default=None, Total number of steps (batches of samples) before declaring one
                # epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data
                # tensors, the default None is equal to the number of samples in your dataset divided by the batch
                # size, or 1 if that cannot be determined.
                validation_steps=None,
                # Only relevant if validation_data is provided and is a tf.data dataset. Total number of steps
                # (batches of samples) to draw before stopping when performing validation at the end of every epoch.
                validation_batch_size=None,
                # Integer or None, default=None, Number of samples per validation batch. If unspecified, will default
                # to batch_size.
                validation_freq=5,
                # default=1, Only relevant if validation data is provided. If an integer, specifies how many training
                # epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation
                # every 2 epochs.
                max_queue_size=10,
                # default=10, Used for generator or keras.utils.Sequence input only. Maximum size for the generator
                # queue. If unspecified, max_queue_size will default to 10.
                workers=1,
                # default=1, Used for generator or keras.utils.Sequence input only. Maximum number of processes to
                # spin up when using process-based threading. If unspecified, workers will default to 1.
                use_multiprocessing=False,
                # default=False, Used for generator or keras.utils.Sequence input only. If True, use process-based
                # threading. If unspecified, use_multiprocessing will default to False.
            )
            save_model(self.model, self.model_path)
        else:
            self.model = load_model(self.model_path)

    def start(self, data):
        X_test, y_test = data
        X_test = X_test.reshape(10000, 784).astype("float32") / 255
        pred_labels_tr = np.array(tf.math.argmax(self.model.predict(self.X_train), axis=1))
        # Predict class labels on a test data
        pred_labels_te = np.array(tf.math.argmax(self.model.predict(X_test), axis=1))

        print("")
        print('-------------------- Model Summary --------------------')
        self.model.summary()  # print model summary
        print("")

        # I am not printing the parameters since my Deep Feed Forward Neural Network contains more than 100K of them
        # print('-------------------- Weights and Biases --------------------')
        # for layer in model_d1.layers:
        # print("Layer: ", layer.name) # print layer name
        # print("  --Kernels (Weights): ", layer.get_weights()[0]) # kernels (weights)
        # print("  --Biases: ", layer.get_weights()[1]) # biases

        print("")
        print('---------- Evaluation on Training Data ----------')
        print(classification_report(self.y_train, pred_labels_tr))
        print("")

        print('---------- Evaluation on Test Data ----------')
        print(classification_report(y_test, pred_labels_te))
        print("")


r = Regularize(mnist_data_X_train, mnist_data_y_train, True)
r.start([mnist_data_X_test, mnist_data_y_test])
