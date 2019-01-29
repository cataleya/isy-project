from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# loading images
X_train_all, y_train_all = mnist_reader.load_mnist('../MNISTdataset', kind='train')
X_test_all, y_testa_all = mnist_reader.load_mnist('../MNISTdataset', kind='t10k')

X_train = X_train_all[0:6000]
y_train = y_train_all[0:6000]
X_test = X_test_all[0:1000]
y_test = y_test_all[0:1000]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255.0
X_test_test = X_test / 255.0

classes = 10

cnnmodel = Sequential()
cnnmodel.add(Convolution2D(32, (3, 3), padding='same', activation='relu',
                                input_shape=(28, 28, 1)))
cnnmodel.add(MaxPooling2D(pool_size=(2, 2)))
cnnmodel.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
cnnmodel.add(MaxPooling2D(pool_size=(2, 2)))
cnnmodel.add(MaxPooling2D(pool_size=(2, 2)))
cnnmodel.add(Flatten())
cnnmodel.add(Dense(units=1024, activation='relu'))
cnnmodel.add(Dense(units=classes, activation='softmax'))

cnnmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(cnnmodel.summary())

hist = model.fit(X_train, y_train, epochs = 1, validation_data=(X_test, y_test))
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.show()

#evaluate

score = model.evaluate(X_test, y_test)

