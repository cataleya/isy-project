import matplotlib.pyplot as plt
from keras.layers import Convolution2D, MaxPooling2D, Dense
from keras.models import Sequential
from keras.datasets import mnist
from utils import mnist_reader
from keras.utils import to_categorical
import numpy as np

# loading images
X_train_all, y_train_all = mnist_reader.load_mnist('../MNISTdataset', kind='train')
X_test_all, y_test_all = mnist_reader.load_mnist('../MNISTdataset', kind='t10k')

X_train = X_train_all[0:600]
y_train = y_train_all[0:600]
X_test = X_test_all[0:100]
y_test = y_test_all[0:100]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255.0
X_test_test = X_test / 255.0

#preprocessing training data
row, col = 28, 28
classes = 10

X_train = X_train.reshape(X_train.shape[0], row * col)
X_test = X_test.reshape(X_test.shape[0], row * col)
print(X_train.shape)
print(X_test.shape)

y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)
print(y_train.shape)
print(y_test.shape)
# VGG19 Architecture



vgg19model = Sequential()
vgg19model.add(Convolution2D(64, (3, 3), padding='same', activation='relu',
                                input_shape=(28, 28, 1)))
vgg19model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
vgg19model.add(MaxPooling2D(pool_size=(2, 2)))

vgg19model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
vgg19model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
vgg19model.add(MaxPooling2D(pool_size=(2, 2)))

vgg19model.add(Convolution2D(256, (3, 3), padding='same', activation='relu'))
vgg19model.add(Convolution2D(256, (3, 3), padding='same', activation='relu'))
vgg19model.add(Convolution2D(256, (3, 3), padding='same', activation='relu'))
vgg19model.add(Convolution2D(256, (3, 3), padding='same', activation='relu'))
vgg19model.add(MaxPooling2D(pool_size=(2, 2)))

vgg19model.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
vgg19model.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
vgg19model.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
vgg19model.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
#vgg19model.add(MaxPooling2D(pool_size=(2, 2)))

vgg19model.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
vgg19model.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
vgg19model.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
vgg19model.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
vgg19model.add(MaxPooling2D(pool_size=(2, 2)))

vgg19model.add(Dense(units=4096, activation='relu'))
vgg19model.add(Dense(units=4096, activation='relu'))
vgg19model.add(Dense(units=classes, activation='softmax'))

vgg19model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 128
epochs = 30
csv_logger = CSVLogger('training.log', separator=',', append=False)

#print(vgg19model.summary())
hist = vgg19model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_train, y_train), callbacks=[csv_logger])
'''
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.show()
'''
#evaluate

score = model.evaluate(X_test, y_test)

print(score)