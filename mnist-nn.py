from keras.datasets import mnist
from keras.preprocessing.image import load_img, array_to_img
from keras.utils import to_categorical
from keras.models import Sequential
#fully connected layer Dense
from keras.layers import Dense 
#from keras.utils import plot_model
from keras.callbacks import CSVLogger

import numpy as np

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
    plt.savefig('acc.png')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('loss.png')

def plot_result_examples(model, X_test, y_test, img_rows, img_cols):
    """
    The predict_classes function outputs the highest probability class
    according to the trained classifier for each input example.
    :return:
    """
    predicted_classes = model.predict_classes(X_test)

    # Check which items we got right / wrong
    correct_indices = np.nonzero(predicted_classes == y_test)[0]
    incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

    plt.figure()
    for i, correct in enumerate(correct_indices[:9]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[correct].reshape(img_rows, img_cols), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
    plt.savefig('correct.png')

    plt.figure()
    for i, incorrect in enumerate(incorrect_indices[:9]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[incorrect].reshape(img_rows, img_cols), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))
    plt.savefig('incorrect.png')

    plt.show()


(X_train, y_train), (X_test, y_test) = mnist.load_data()
#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)
#print(X_train[0])

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
#print(X_train[0], )

#preprocessing test data
classes = 10
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

#building model
'''
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(512, activation ='relu'))
model.add(Dense(classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#train the model
hist = model.fit(X_train, y_train, epochs = 1, validation_data=(X_test, y_test))
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.show()

#evaluate
score = model.evaluate(X_test, y_test)

'''
nn1 = Sequential()
nn1.add(Dense(1000, activation='relu', input_shape=(784,)))
nn1.add(Dense(500, activation ='relu'))
nn1.add(Dense(classes, activation='softmax'))
nn1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

nn1.summary()

csv_logger = CSVLogger('training.log', separator=',', append=False)

nn1_history = nn1.fit(X_train, y_train, epochs = 30, verbose=1, validation_data=(X_test, y_test), callbacks=[csv_logger])
nn1_score = nn1.evaluate(X_test, y_test)
#plot_model(nn1, to_file='nn1-model.png')

plot_model_history(nn1_history)
plot_result_examples(nn1, X_test, y_test, row, col)
