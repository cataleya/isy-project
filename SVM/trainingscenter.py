from utils import mnist_reader
import numpy as np
from sklearn_svm import sklearn_svm

##################################################################################################
#
#       Zum Trainieren und Testen von Machine-learning
#       Algorithmen mit unterschiedlichen Parametern
#
##################################################################################################

# loading and preprocessing images

X_train, y_train = mnist_reader.load_mnist('../MNISTdataset', kind='train')
X_test, y_test = mnist_reader.load_mnist('../MNISTdataset', kind='t10k')

X_train = X_train[0:18000]
y_train = y_train[0:18000]
X_test = X_test[0:3000]
y_test = y_test[0:3000]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255.0
X_test_test = X_test / 255.0

print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

# Trainiere und Teste SVM

# Kernel: linear
sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='linear', degree=2, gamma=1,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='linear', degree=2, gamma=1,
                 coef0=1.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='linear', degree=2, gamma=1,
                 coef0=10.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='linear', degree=2, gamma=1,
                 coef0=100.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='linear', degree=2, gamma=10,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-5, cache_size=4000)

# kernel: poly

# degree: 2
sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='poly', degree=2, gamma=1,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='poly', degree=2, gamma=1,
                 coef0=1.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='poly', degree=2, gamma=1,
                 coef0=10.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='poly', degree=2, gamma=1,
                 coef0=100.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='poly', degree=2, gamma=10,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

# degree: 4
sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='poly', degree=4, gamma=1,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='poly', degree=4, gamma=1,
                 coef0=1.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='poly', degree=4, gamma=1,
                 coef0=10.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='poly', degree=4, gamma=1,
                 coef0=100.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='poly', degree=4, gamma=10,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='poly', degree=4, gamma=0.01,
                 coef0=1.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='poly', degree=4, gamma=0.001,
                 coef0=1.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

# degree : 9
sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='poly', degree=9, gamma=1,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='poly', degree=9, gamma=10,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='poly', degree=9, gamma=0.01,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='poly', degree=9, gamma=0.001,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

#kernel: rbf
sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='rbf', degree=9, gamma=1,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='rbf', degree=9, gamma=10,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='rbf', degree=9, gamma=0.01,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='rbf', degree=9, gamma=0.001,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

