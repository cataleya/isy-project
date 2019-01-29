from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from utils import plot_utils
import matplotlib.pyplot as plt
import numpy as np

# models from "Deep Big Simple Neural Nets Excel on Hand- written Digit Recognition"
# https://arxiv.org/pdf/1003.0358.pdf

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#preprocessing training data
row, col = 28, 28
X_train = X_train.reshape(60000, row * col)
X_test = X_test.reshape(10000, row * col)
print(X_train.shape)
print(X_test.shape)

classes = 10

nn1 = Sequential()
nn1.add(Dense(1000, activation='relu', input_shape=(784,)))
nn1.add(Dense(500, activation ='relu'))
nn1.add(Dense(classes, activation='softmax'))
nn1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

nn2 = Sequential()
nn2.add(Dense(1500, activation='relu', input_shape=(784,)))
nn2.add(Dense(1000, activation ='relu'))
nn2.add(Dense(500, activation ='relu'))
nn2.add(Dense(classes, activation='softmax'))
nn2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

nn3 = Sequential()
nn3.add(Dense(2000, activation='relu', input_shape=(784,)))
nn3.add(Dense(1500, activation ='relu'))
nn3.add(Dense(1000, activation ='relu'))
nn3.add(Dense(500, activation ='relu'))
nn3.add(Dense(classes, activation='softmax'))
nn3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

nn4 = Sequential()
nn4.add(Dense(2500, activation='relu', input_shape=(784,)))
nn4.add(Dense(2000, activation ='relu'))
nn4.add(Dense(1500, activation ='relu'))
nn4.add(Dense(1000, activation ='relu'))
nn4.add(Dense(500, activation ='relu'))
nn4.add(Dense(classes, activation='softmax'))
nn4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

nn5 = Sequential()
nn5.add(Dense(1000, activation='relu', input_shape=(784,)))
nn5.add(Dense(1000, activation ='relu'))
nn5.add(Dense(1000, activation ='relu'))
nn5.add(Dense(1000, activation ='relu'))
nn5.add(Dense(1000, activation ='relu'))
nn5.add(Dense(1000, activation ='relu'))
nn5.add(Dense(1000, activation ='relu'))
nn5.add(Dense(1000, activation ='relu'))
nn5.add(Dense(1000, activation ='relu'))
nn5.add(Dense(classes, activation='softmax'))
nn5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
