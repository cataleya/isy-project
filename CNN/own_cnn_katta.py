import matplotlib.pyplot as plt
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Sequential
import numpy as np
from keras.callbacks import CSVLogger

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






def own_cnn(X_train_all, y_train_all, X_test_all, y_test_all, X_train, y_train, X_test, y_test, classes=10,
            batch_size=128, epochs=50, num_conv_layer_per_pooling=2, num_of_poolings=3,
            pool_size=2, kernel_size=3, padding='same', activation='relu',
            list_of_kernel_numbers=[64, 128, 256, 512], dense_layers=2, neurons_in_dense_layer=4096, dropout=0):
    """sklearn_svm(...)
    Writes the results into the log file.
    """

    # VGG19 Architecture

    model = Sequential()

    for ii in np.arange(0, num_of_poolings):
        for jj in np.arange(0, num_conv_layer_per_pooling):
            if ii == 0 and jj == 0:
                model.add(Convolution2D(list_of_kernel_numbers[ii], (kernel_size, kernel_size),
                        padding=padding, activation=activation, input_shape=(28, 28, 1)))
            else:
                model.add(Convolution2D(list_of_kernel_numbers[ii], (kernel_size, kernel_size),
                    padding=padding, activation=activation))

        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

        if dropout == 1:
            model.add(Dropout(rate=0.25))

    model.add(Flatten())
    for nn in np.arange(0, dense_layers):
        model.add(Dense(units=neurons_in_dense_layer, activation=activation))

    model.add(Dense(units=classes, activation='softmax'))





    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    log_datei = 'training_katta.log'
    csv_logger = CSVLogger(log_datei, separator=',', append=True)

    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
            verbose=1, validation_data=(X_test, y_test), callbacks=[csv_logger])


    #evaluate
    score = model.evaluate(X_test, y_test, batch_size)
    model.summary() # prints Summary on console
    print('Score:')
    print(model.metrics_names[0], model.metrics_names[1])
    print(score[0], score[1])

    f = open(log_datei, 'a')
    f.write('\n')
    f.close()

