import numpy as np
import cv2
import glob
from sklearn import svm
from sklearn import preprocessing


############################################################
#
#              Support Vector Machine
#              Image Classification
#
############################################################

def create_keypoints(w, h):
    keypoints = []
    keypointSize = 15
    keypoints = [cv2.KeyPoint(x=x, y=y, _size=keypointSize) for x in np.arange(0, 256, 10)
        for y in np.arange(0, 256, 10)]
    return keypoints

# 1. Implement a SIFT feature extraction for a set of training images ./images/db/train/** (see 2.3 image retrieval)
# use 256x256 keypoints on each image with subwindow of 15x15px

# loading and preprocessing images
imageNamesCars = glob.glob('./images/db/train/cars/*.jpg')
imageNamesFaces = glob.glob('./images/db/train/faces/*.jpg')
imageNamesFlowers = glob.glob('./images/db/train/flowers/*.jpg')
imageNames = np.concatenate((imageNamesCars, imageNamesFaces, imageNamesFlowers), axis=0)
imgs = np.array([cv2.imread(name, cv2.IMREAD_COLOR) for name in imageNames])
grayImgs = np.array([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in imgs])

# create keypoints and based on these create feature vectors
kps = create_keypoints(256, 256)
sift = cv2.xfeatures2d.SIFT_create() # create SIFT-object
features = np.array([sift.compute(image, kps)[1] for image in grayImgs])

# flatten matrix to get the following shape: (num_train_images, num_keypoints*num_entry_per_keypoint)
xTrain = np.array([features[i].flatten() for i, featureVector in enumerate(features)])

# create yTrain vector containing the labels encoded as integers
yTrain = np.array([0,0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,2])

# 3. We use scikit-learn to train a SVM classifier. Specifically we use a LinearSVC in our case. Have a look at the documentation.
# You will need .fit(X_train, y_train)

lsvm = svm.LinearSVC(random_state=0, tol=1e-5)
# scaler = preprocessing.StandardScaler()
# scaler.fit(xTrain, yTrain)

lsvm.fit(xTrain, yTrain)

# test the lsvm with test images
testImageNames = glob.glob('./images/db/test/*')
testImgs = np.array([cv2.imread(name, cv2.IMREAD_COLOR) for name in testImageNames])
testGrayImgs = np.array([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in testImgs])
testImageFeatures = np.array([sift.compute(image, kps)[1] for image in testGrayImgs])
xTest = np.array([testImageFeatures[i].flatten() for i, featureVector in enumerate(testImageFeatures)])

print('0 : car')
print('1 : face')
print('2 : flower')

for i in range(0, len(testImageNames)):
    print(testImageNames[i], 'is predicted as', lsvm.predict(xTest[i]))




# 5. output the class + corresponding name


