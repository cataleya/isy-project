import matplotlib.pyplot as plt
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Sequential
import numpy as np

def own_cnn(X_train_all, y_train_all, X_test_all, y_test_all, X_train, y_train, X_test, y_test, classes=10,
            batch_size=128, epochs=50, num_conv_layer_per_pooling=2, num_of_poolings=4,
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

    csv_logger = CSVLogger('training.log', separator=',', append=True)

    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
            verbose=1, validation_data=(X_train, y_train), callbacks=[csv_logger])
    '''
    plt.plot(hist.history['val_acc'])
    plt.plot(hist.history['loss'])
    plt.show()
    '''

    #evaluate
    score = model.evaluate(X_test, y_test, batch_size, )
    summary = model.summary()
    print(score)
    print(summary)