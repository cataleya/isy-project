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

    plt.figure()
    for i, incorrect in enumerate(incorrect_indices[:9]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[incorrect].reshape(img_rows, img_cols), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))

    plt.show()


(X_train_all, y_train_all), (X_test_all, y_test_all) = mnist.load_data()
#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)
#print(X_train[0])

X_train = X_train_all[0:3000]
y_train = y_train_all[0:3000]
X_test = X_test_all[0:1000]
y_test = y_test_all[0:1000]
X_test2 = X_test_all[1000:2000]
y_test2 = y_test_all[1000:2000]
#preprocessing training data
row, col = 28, 28
X_train = X_train.reshape(3000, row * col)
X_test = X_test.reshape(1000, row * col)
X_test2 = X_test.reshape(1000, row * col)
print(X_train.shape)
print(X_test.shape)

#values from range 0-255 change to range 0-1
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_test2 = X_test.astype('float32')
X_train /= 255
X_test /= 255
X_test2 /= 255
#print(X_train[0], )

#preprocessing test data
classes = 10
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)
y_test2 = to_categorical(y_test2, classes)


#building model

nn3 = Sequential()
nn3.add(Dense(2000, activation='relu', input_shape=(784,)))
nn3.add(Dense(1500, activation ='relu'))
nn3.add(Dense(1000, activation ='relu'))
nn3.add(Dense(500, activation ='relu'))
nn3.add(Dense(classes, activation='softmax'))
nn3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(nn3.summary())

csv_logger = CSVLogger('training2.log', separator=',', append=False)

nn3_history = nn3.fit(X_train, y_train, epochs = 50, verbose=1, validation_data=(X_test, y_test), callbacks=[csv_logger])
nn3_score = nn3.evaluate(X_test, y_test)
nn3_score2 = nn3.evaluate(X_test2, y_test2)

print('Test score:', nn3_score[0])
print('Test accuracy:', nn3_score[1])

print('Test score2:', nn3_score2[0])
print('Test accuracy2:', nn3_score2[1])

plot_model_history(nn3_history)
#plot_result_examples(nn2, X_test, y_test, row, col)
