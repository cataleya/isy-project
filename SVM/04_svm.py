import numpy as np
import cv2
import glob
from utils import mnist_reader
from sklearn import svm
from sklearn import preprocessing


############################################################
#
#              Support Vector Machine
#              Image Classification
#
############################################################

# 1. Implement a SIFT feature extraction for a set of training images ./images/db/train/** (see 2.3 image retrieval)
# use 256x256 keypoints on each image with subwindow of 15x15px

w = 28
h = 28

# loading and preprocessing images

X_train, y_train = mnist_reader.load_mnist('../MNISTdataset', kind='train')
X_test, y_test = mnist_reader.load_mnist('../MNISTdataset', kind='t10k')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255.0
X_test_test = X_test / 255.0

print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)



# 3. We use scikit-learn to train a SVM classifier. Specifically we use a LinearSVC in our case. Have a look at the documentation.
# You will need .fit(X_train, y_train)

lsvm = svm.SVC(C=1.0, kernel='rbf', gamma='auto',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=1000)

# scaler = preprocessing.StandardScaler()
# scaler.fit(xTrain, yTrain)
print('hallo')
lsvm.fit(X_train, y_train)

print('hallo')


numCorrectlyPredicted = 0
for i in range(0, len(X_test)):
    if y_train[i] == lsvm.predict(X_test[i]):
        numCorrectlyPredicted = numCorrectlyPredicted + 1;

detection_rate = numCorrectlyPredicted / X_test.shape[0]
print('Number of correctly predicted classes: ', detection_rate)




# 5. output the class + corresponding name


