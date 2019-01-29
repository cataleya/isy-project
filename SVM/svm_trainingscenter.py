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

X_trainALL, y_trainALL = mnist_reader.load_mnist('../MNISTdataset', kind='train')
X_testALL, y_testALL = mnist_reader.load_mnist('../MNISTdataset', kind='t10k')

X_train = X_trainALL[0:18000]
y_train = y_trainALL[0:18000]
X_test = X_testALL[0:3000]
y_test = y_testALL[0:3000]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255.0
X_test_test = X_test / 255.0

print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

# Trainiere und Teste SVM

# kernel: poly

# degree: 2
for n in np.arange(1,51):
    sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='poly', degree=2, gamma=n*0.004,
                 coef0=60, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

for n in np.arange(1,51):
    sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='poly', degree=2, gamma=0.004,
                 coef0=n*4, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000)

for n in np.arange(1,101):
    X_train = X_trainALL[0:n*600]
    y_train = y_trainALL[0:n*600]
    X_test = X_testALL[0:n*100]
    y_test = y_testALL[0:n*100]

    sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='poly', degree=2, gamma=0.05,
                     coef0=0.0, shrinking=True, probability=False,
                     tol=1e-3, cache_size=4000)

