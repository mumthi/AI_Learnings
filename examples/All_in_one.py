from examples.Neral_networks.Deep_feed_forward import DeepFeedForward
from examples.Perceptron.Perceptron import Perceptron
from examples.Regression.LinearRegression import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras

plt.rcParams['figure.figsize'] = [8, 5]
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.style.use('seaborn-whitegrid')


# Load a CSV file


def load_csv(filename):
    data_set = pd.read_csv(filename)
    data_set = data_set.values.tolist()
    return data_set


# Convert string column to float
def str_column_to_float(data_set, column):
    for row in data_set:
        row[column] = float(row[column])


# Convert string column to integer
def str_column_to_int(data_set, column):
    class_values = [row[column] for row in data_set]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in data_set:
        row[column] = lookup[row[column]]
    return lookup


(mnist_data_X_train, mnist_data_y_train), (mnist_data_X_test, mnist_data_y_test) = keras.datasets.mnist.load_data()
sonar_all = r'Data\sonar.all-data'
sonar_all_data_set = load_csv(sonar_all)

for i in range(len(sonar_all_data_set[0]) - 1):
    str_column_to_float(sonar_all_data_set, i)
# convert string class to integers
str_column_to_int(sonar_all_data_set, len(sonar_all_data_set[0]) - 1)

options = {
    "Perceptron": [Perceptron(sonar_all_data_set), True],
    "DeepForward": [DeepFeedForward(mnist_data_X_train, mnist_data_y_train), True],
    "Linear_regression": [LinearRegression(), False]
}

for val in options.values():
    if val[1]:
        val[0].start(val[2])
