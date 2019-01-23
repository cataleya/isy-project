from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("y_Test", y_test)