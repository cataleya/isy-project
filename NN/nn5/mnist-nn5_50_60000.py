from keras.datasets import mnist
from keras.preprocessing.image import load_img, array_to_img
from keras.utils import to_categorical
from keras.models import Sequential
#fully connected layer Dense
from keras.layers import Dense 
#from keras.utils import plot_model
from keras.callbacks import CSVLogger

from matplotlib import pyplot as plt
import numpy as np

def plot_model_history(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

(X_train_all, y_train_all), (X_test_all, y_test_all) = mnist.load_data()
#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)
#print(X_train[0])

X_train = X_train_all
y_train = y_train_all
X_test = X_test_all
y_test = y_test_all

#preprocessing training data
row, col = 28, 28
X_train = X_train.reshape(3000, row * col)
X_test = X_test.reshape(1000, row * col)
print(X_train.shape)
print(X_test.shape)

#values from range 0-255 change to range 0-1
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
#print(X_train[0], )

#preprocessing test data
classes = 10
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

#building model

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

print(nn5.summary())

csv_logger = CSVLogger('training2.log', separator=',', append=False)

nn5_history = nn5.fit(X_train, y_train, epochs = 50, verbose=1, validation_data=(X_test, y_test), callbacks=[csv_logger])
nn5_score = nn5.evaluate(X_test, y_test)


print('Test score:', nn5_score[0])
print('Test accuracy:', nn5_score[1])

plot_model_history(nn5_history)
