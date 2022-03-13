import csv
import os
import re
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow
from cv2 import cv2
from keras_preprocessing import image
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Flatten
from tensorflow.keras.models import load_model

import seaborn as sns


class CNNLayers:
    def __init__(self):
        self.num_epochs = 100
        self.batch_size = 32
        self.callbacks_list = []
        test_data_folder = r'images'
        test_data_files = self.recursive_file_filter(test_data_folder,
                                                     file_ext_filter=['.png'],
                                                     number_of_files=0)
        labels = LabelEncoder().fit_transform(self.categorise_directory(test_data_files))
        X_train, X_test, self.y_train, y_test = train_test_split(test_data_files, labels, test_size=0.33,
                                                            random_state=42)
        X_test, X_validation, self.y_test, self.y_validation = train_test_split(X_test, y_test, test_size=0.33,
                                                                                random_state=42)
        self.x_train = self.get_data_from_image(X_train)
        self.x_test = self.get_data_from_image(X_test)
        self.x_validation = self.get_data_from_image(X_validation)

        self.model_path = 'model'
        self.sizes = [9, 9]

    def get_data_from_image(self, all_file_path):
        data_read = np.array([])
        for each_file_path in all_file_path:
            intensity = np.array(cv2.imread(each_file_path, cv2.IMREAD_ANYDEPTH))
            if data_read.shape[0] == 0:
                data_read = np.array([intensity])
            else:
                data_read = np.append(data_read, [intensity], axis=0)

        return data_read

    def categorise_directory(self, files):
        file_categories = []
        for file in files:
            directory_file, file_name = os.path.split(file)
            directory_file = directory_file.replace("\\", "/")
            split_directories = directory_file.split("/")
            file_categories.append(split_directories[-1])
        return file_categories

    def recursive_file_filter(self, path, file_ext_filter=None, file_name_filter=None, file_dir_filter=None,
                              number_of_files=None):
        all_files = []
        for root, dirs, files in os.walk(path):
            if len(files) > 0:
                if file_ext_filter is not None:
                    files = [f for f in files if os.path.splitext(f)[1] in file_ext_filter]

                if file_name_filter is not None:
                    files = [f for f in files if re.search(file_name_filter, os.path.splitext(f)[0])]

                if file_dir_filter is not None:
                    files = [f for f in files if re.search(file_name_filter, os.path.splitext(f)[0])]
                if number_of_files > 0:
                    files = files[0:number_of_files]
            files = [os.path.join(root, p) for p in files]
            all_files.extend(files)
        return all_files

    def define_model(self, sizes):
        self.classifier = Sequential()
        self.classifier.add(Conv2D(32, (3, 3), input_shape=(sizes[0], sizes[1], 1), name='c2d_1'))
        self.classifier.add(AveragePooling2D((2, 2), name='avg_1'))
        # self.classifier.add(Conv2D(64, (3, 3), activation='relu', name='c2d_2'))
        # self.classifier.add(AveragePooling2D((2, 2), name='avg_2'))
        # self.classifier.add(Conv2D(128, (3, 3), activation='relu', name='c2d_3'))
        # self.classifier.add(AveragePooling2D((2, 2), name='avg_3'))
        # # self.classifier.add(MaxPooling2D((2, 2), name='max_1'))  # if stride not given it equal to pool filter size
        # self.classifier.add(Conv2D(64, (3, 3), activation='relu', name='c2d_4'))
        # self.classifier.add(AveragePooling2D((2, 2), name='avg_4'))

        self.classifier.add(Flatten())
        self.classifier.add(Dense(units=32, activation='relu', name='dns_1'))
        self.classifier.add(Dense(units=16, activation='relu', name='dns_2'))
        self.classifier.add(Dense(units=1, activation='sigmoid', name='dns_3'))
        adam = tensorflow.optimizers.Adam(lr=0.0001, beta_1=0.99, beta_2=0.999, epsilon=None, decay=0.0,
                                          amsgrad=False, clipnorm=1)
        self.classifier.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    def run_algorithm(self, load_model):
        if load_model:
            self.load_and_process(self.model_path)
        else:
            self.define_model(self.sizes)
            self.algorithm_train_n_save(self.x_train, self.y_train, self.x_validation, self.y_validation,
                                        self.model_path, True)

        self.algorithm_predict(self.x_test, self.y_test, ["Bad", "Good"])

    def algorithm_train_n_save(self, train_x, train_y, validation_x, validation_y, model_path, save_model=False):

        self.classifier.fit(train_x, train_y, epochs=self.num_epochs, batch_size=self.batch_size,
                            callbacks=self.callbacks_list, validation_data=(validation_x, validation_y))
        x1 = self.classifier.evaluate(train_x, train_y)
        x2 = self.classifier.evaluate(validation_x, validation_y)
        print('Training Accuracy  : %1.2f%%     Training loss  : %1.6f' % (x1[1] * 100, x1[0]))
        print('Validation Accuracy: %1.2f%%     Validation loss: %1.6f' % (x2[1] * 100, x2[0]))
        if save_model:
            self.classifier.save(model_path)

    def test_set_verifier(self):
        ytesthat = self.classifier.predict((self.x_test, self.y_test))
        df = pd.DataFrame({
            'filename': self.file_name_val,
            'predict': ytesthat[:, 0],
            'y': self.y_val
        })
        pd.set_option('display.float_format', lambda x: '%.5f' % x)
        df['y_pred'] = df['predict'] > 0.5
        df.y_pred = df.y_pred.astype(int)
        df.head(10)
        misclassified = df[df['y'] != df['y_pred']]
        print('Total misclassified image from 5000 Validation images : %d' % misclassified['y'].count())
        conf_matrix = confusion_matrix(df.y, df.y_pred)
        sns.heatmap(conf_matrix, cmap="YlGnBu", annot=True, fmt='g');
        plt.xlabel('predicted value')
        plt.ylabel('true value')

    def layer_output_visualization(self):
        self.classifier.summary()
        # Input Image for Layer visualization
        # img1 = image.load_img(r'images\good\13.png')
        img = self.get_data_from_image([r'images\good\13.png'])
        # plt.imshow(img)
        # preprocess image

        model_layers = [layer.name for layer in self.classifier.layers]
        print('layer name : ', model_layers)
        conv2d_1_output = Model(inputs=self.classifier.input, outputs=self.classifier.get_layer('c2d_1').output)
        avg_1_output = Model(inputs=self.classifier.input, outputs=self.classifier.get_layer('avg_1').output)
        # dense_1_output = Model(inputs=self.classifier.input, outputs=self.classifier.get_layer('c2d_3').output)
        # dense_2_output = Model(inputs=self.classifier.input, outputs=self.classifier.get_layer('c2d_4').output)
        conv2d_1_features = conv2d_1_output.predict(img)
        avg_1_features = avg_1_output.predict(img)
        print('First conv layer feature output shape : ', conv2d_1_features.shape)
        print('First conv layer feature output shape : ', avg_1_features.shape)
        # plt.imshow(conv2d_1_features[0, :, :, 4], cmap='gray')
        fig = plt.figure(figsize=(14, 7))
        columns = 8
        rows = 8
        for i in range(32):
            # img = mpimg.imread()
            fig.add_subplot(rows, columns, i + 1)
            plt.axis('off')
            plt.title('filter' + str(i))
            plt.imshow(conv2d_1_features[0, :, :, i], cmap='gray')
        # plt.show()
        # fig = plt.figure(figsize=(14, 7))

        for i in range(32, 64):
            # img = mpimg.imread()
            fig.add_subplot(rows, columns, i + 1)
            i = i- 32
            plt.axis('off')
            plt.title('filter' + str(i))
            plt.imshow(avg_1_features[0, :, :, i], cmap='gray')
        plt.show()

    def algorithm_predict(self, test_x, test_y, labels, test_data_files=None):
        prediction = self.classifier.predict(test_x)
        threshold = 0.5
        # self.csv_write(prediction, test_data_files)/
        predictions = [value > threshold for value in prediction]
        accuracy = accuracy_score(test_y, predictions)
        print(classification_report(test_y, predictions, target_names=labels))
        conf_matrix = confusion_matrix(y_true=test_y, y_pred=predictions)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        fig = plt.gcf()
        fig.set_size_inches((8.5, 11), forward=False)
        fig.savefig(r"Results/feed_forward_" + datetime.now().strftime("%d%m%y%H%M%S") + ".png", dpi=800)
        plt.close()

    def prediction_of_single_image(self, image_path):
        img1 = image.load_img(image_path, target_size=(64, 64))
        img = image.img_to_array(img1)
        img = img / 255
        # create a batch of size 1 [N,H,W,C]
        img = np.expand_dims(img, axis=0)
        prediction = self.classifier.predict(img, batch_size=None, steps=1)  # gives all class prob.
        if (prediction[:, :] > 0.5):
            value = 'Dog :%1.2f' % (prediction[0, 0])
            plt.text(20, 62, value, color='red', fontsize=18, bbox=dict(facecolor='white', alpha=0.8))
        else:
            value = 'Cat :%1.2f' % (1.0 - prediction[0, 0])
            plt.text(20, 62, value, color='red', fontsize=18, bbox=dict(facecolor='white', alpha=0.8))

        plt.imshow(img1)
        plt.show()

    def load_and_process(self, model_path):
        self.classifier = load_model(model_path)

    def csv_write(self, prediction, test_data_files):
        fn = os.path.join(os.getcwd(),
                          "Results/Prediction_feed_forward_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".csv")
        with open(os.path.join(os.getcwd(), fn), "w", newline='') as f:
            csv_h = csv.writer(f)
            for pr, fn in zip(prediction, test_data_files):
                p_r = 'Positive' if fn.count('Positive') > 0 else 'Negative'
                row = [fn, p_r, pr]
                csv_h.writerow(row)


if __name__ == '__main__':
    c = CNNLayers()
    c.run_algorithm(False)
    c.layer_output_visualization()