from keras.datasets import mnist
from keras.preprocessing.image import load_img, array_to_img
from keras.utils import to_categorical
from keras.models import Sequential
#fully connected layer Dense
from keras.layers import Dense 

import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()
#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)
print(X_train[0])

plt.imshow(X_train[0])
plt.colorbar()

#preprocessing training data
row, col = 28, 28
X_train = X_train.reshape(60000, row * col)
X_test = X_test.reshape(10000, row * col)
print(X_train.shape)
print(X_test.shape)

#values from range 0-255 change to range 0-1
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train[0])

#preprocessing test data
classes = 10
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

#building model

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(512, activation ='relu'))
model.add(Dense(classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.summary()

#train the model
hist = model.fit(X_train, y_train, epochs = 1, validation_data=(X_test, y_test))
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.show()

#evaluate

score = model.evaluate(X_test, y_test)

plot_utils.plot_model_history(hist)
plot_utils.plot_result_examples(model, X_test, y_test, rows, cols)

