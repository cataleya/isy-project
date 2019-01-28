import numpy as np
from sklearn import svm

############################################################
#
#              Support Vector Machine
#              Image Classification
#
############################################################

def sklearn_svm(X_train, y_train, X_test, y_test, C=1.0, kernel='linear', degree=2, gamma='auto',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=4000):
    """sklearn_svm(...)
    Returns the detection rate of the SVM with the given Parameters.
    """

    # 1. We use scikit-learn to train a SVM classifier. Specifically we use a LinearSVC in our case. Have a look at the documentation.
    # You will need .fit(X_train, y_train)

    lsvm = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma,
                     coef0=coef0, shrinking=shrinking, probability=probability,
                     tol=tol, cache_size=cache_size)

    lsvm.fit(X_train, y_train)

    y_predict = lsvm.predict(X_test)
    numCorrectlyPredicted = 0
    for i in np.arange(len(y_predict)):
        if y_predict[i] == y_test[i]:
            numCorrectlyPredicted = numCorrectlyPredicted + 1
    detection_rate = numCorrectlyPredicted / X_test.shape[0]

    # Terminalausgabe:

    print('Teache SVM mit', X_train.shape[0], 'Trainingsdaten:')
    print('SVM-Parameter:')
    print('C=',C, ',kernel=', kernel, ', degree=', degree, ', gamma=', gamma,
        ', coef=', coef0, ', shrinking=', shrinking, ', probability=', probability,
        ', tol=',tol, ', cache_size=', cache_size)
    print('Teste SVM mit:', X_test.shape[0], 'Testdaten:')
    print('detection rate=', detection_rate)
    print('\n')

    # schreibe Trainingslog in Datei:

    f = open("svm_ergebnisse.txt", "a")
    f.write(repr(X_train.shape[0]) + ' ' + repr(X_test.shape[0]) + ' ')
    f.write(repr(C) + ' ' + repr(kernel) + ' ' + repr(degree) + ' ' + repr(gamma) + ' ' + repr(coef0) + ' ' + repr(shrinking)
            + ' ' + repr(probability) + ' ' + repr(tol) + ' ' + repr(cache_size) + ' ')
    f.write(repr(detection_rate))
    f.write('\n')
    f.close()

    return detection_rate

