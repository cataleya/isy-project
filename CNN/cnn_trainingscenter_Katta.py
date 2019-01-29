from keras.datasets import mnist
from own_cnn_katta import own_cnn
from utils import mnist_reader
from keras.utils import to_categorical
import numpy as np


# loading images
X_train_all, y_train_all = mnist_reader.load_mnist('../MNISTdataset', kind='train')
X_test_all, y_test_all = mnist_reader.load_mnist('../MNISTdataset', kind='t10k')

#preprocessing training data
row, col = 28, 28
X_train_all = X_train_all.reshape(X_train_all.shape[0], row, col, 1)
X_test_all = X_test_all.reshape(X_test_all.shape[0], row, col, 1)

X_train = X_train_all[0:2500]
y_train = y_train_all[0:2500]
X_test = X_test_all
y_test = y_test_all

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255.0
X_test = X_test / 255.0


classes = 10


print('X_train: ', X_train.shape)
print('X_test: ',X_test.shape)


y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)
print('y_train: ',y_train.shape)
print('X_test: ',y_test.shape)


############# TRAININGSCENTER ###################
log_datei = 'training_katta.log'

f = open(log_datei, 'a')
f.write('Parameter der nächsten 6 Trainings: list_of_kernel_numbers = [32, 32, 32, 32], [64, 64, 64, 64], [128, 128, 128, 128], [256, 128, 64, 32] mit Dropout ... = [32, 32, 32, 32] , [128, 128, 128, 128] ohne Dropout\n')
f.close()

for ii in [[32, 32, 32, 32], [64, 64, 64, 64], [128, 128, 128, 128], [256, 128, 64, 32]]:
    own_cnn(X_train_all, y_train_all, X_test_all, y_test_all, X_train, y_train, X_test, y_test, classes=10,
            batch_size=64, epochs=50, num_conv_layer_per_pooling=2, num_of_poolings=3,
            pool_size=2, kernel_size=3, padding='same', activation='relu',
            list_of_kernel_numbers=ii, dense_layers=4, neurons_in_dense_layer=1024, dropout=1)

own_cnn(X_train_all, y_train_all, X_test_all, y_test_all, X_train, y_train, X_test, y_test, classes=10,
            batch_size=64, epochs=50, num_conv_layer_per_pooling=2, num_of_poolings=3,
            pool_size=2, kernel_size=3, padding='same', activation='relu',
            list_of_kernel_numbers=[32, 32, 32, 32], dense_layers=4, neurons_in_dense_layer=1024, dropout=1)

own_cnn(X_train_all, y_train_all, X_test_all, y_test_all, X_train, y_train, X_test, y_test, classes=10,
            batch_size=64, epochs=50, num_conv_layer_per_pooling=2, num_of_poolings=3,
            pool_size=2, kernel_size=3, padding='same', activation='relu',
            list_of_kernel_numbers=[128, 128, 128, 128], dense_layers=4, neurons_in_dense_layer=1024, dropout=1)


f = open(log_datei, 'a')
f.write('Parameter der nächsten 4 Trainings: dense_layers = [1,2,4,8]\n')
f.close()
for ii in [1,2,4,8]:
    own_cnn(X_train_all, y_train_all, X_test_all, y_test_all, X_train, y_train, X_test, y_test, classes=10,
            batch_size=64, epochs=50, num_conv_layer_per_pooling=2, num_of_poolings=3,
            pool_size=2, kernel_size=3, padding='same', activation='relu',
            list_of_kernel_numbers=[32, 64, 128, 256], dense_layers=ii, neurons_in_dense_layer=1024, dropout=1)



f = open(log_datei, 'a')
f.write('Parameter der nächsten 6 Trainings: kernel_size = [1,6,15,28] mit Dropout ... = [1, 8] ohne Dropout\n')
f.close()
for ii in [1,6,15,28]:
    own_cnn(X_train_all, y_train_all, X_test_all, y_test_all, X_train, y_train, X_test, y_test, classes=10,
            batch_size=64, epochs=50, num_conv_layer_per_pooling=2, num_of_poolings=3,
            pool_size=2, kernel_size=ii, padding='same', activation='relu',
            list_of_kernel_numbers=[32, 64, 128, 256], dense_layers=4, neurons_in_dense_layer=1024, dropout=1)

own_cnn(X_train_all, y_train_all, X_test_all, y_test_all, X_train, y_train, X_test, y_test, classes=10,
            batch_size=64, epochs=50, num_conv_layer_per_pooling=2, num_of_poolings=3,
            pool_size=2, kernel_size=1, padding='same', activation='relu',
            list_of_kernel_numbers=[32, 64, 128, 256], dense_layers=4, neurons_in_dense_layer=1024, dropout=0)

own_cnn(X_train_all, y_train_all, X_test_all, y_test_all, X_train, y_train, X_test, y_test, classes=10,
            batch_size=64, epochs=50, num_conv_layer_per_pooling=2, num_of_poolings=3,
            pool_size=2, kernel_size=28, padding='same', activation='relu',
            list_of_kernel_numbers=[32, 64, 128, 256], dense_layers=4, neurons_in_dense_layer=1024, dropout=0)



f = open(log_datei, 'a')
f.write('Parameter der nächsten 6 Trainings: neurons_in_dense_layer=ii = [8, 16, 64, 256, 1024, 2048]\n')
f.close()
for ii in [8, 16, 64, 256, 1024, 2048]:
    own_cnn(X_train_all, y_train_all, X_test_all, y_test_all, X_train, y_train, X_test, y_test, classes=10,
            batch_size=64, epochs=50, num_conv_layer_per_pooling=2, num_of_poolings=3,
            pool_size=2, kernel_size=3, padding='same', activation='relu',
            list_of_kernel_numbers=[32, 64, 128, 256], dense_layers=4, neurons_in_dense_layer=ii, dropout=1)