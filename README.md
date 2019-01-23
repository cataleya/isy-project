# Projektdokumentation Interactive Systems WS 18 / 19
## Projektmitglieder 
- Andreas Hallmann
- Katharina Krebs
## Projektbeschreibung
Ziel des Projekts ist die Auseinandersetzung mit verschiedenen Methoden des maschinellen Lernens anhand des MNIST-Datensatzes.
// Beschreibung

## Stand der Technik - Erkennung von handgeschriebenen Ziffern

## Implementierung
Hier folgt die Beschreibung der Umsetzung
### Datensatz 
// Katharina
Bilddaten von handgeschriebenen Ziffern von 0–9:
- Graustufen (0-255): ursprünglich black and white 20 x 20 pixel box, aspect ratio wurde eingehalten
- Vereinheitlichte, konstante Bildgröße (28 x 28 Pixel)
- Zentrierung der Ziffern auf dem Bild: entered in a 28x28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field

- 60.000 Training Samples 
- 10.000 Test Samples

(https://github.com/cataleya/isy-project/blob/master/img/documentation/Example-images-from-the-MNIST-dataset.png)

4 Datensets:
Training set images
Training set labels
Test set images
Test set labels

siehe auch: https://github.com/datapythonista/mnist

Das Keras Framework stellt den Import des MNIST-Datensatzes bereit. 
Rückgabewert: 
2 tuples:
x_train, x_test: uint8 array of grayscale image data with shape (num_samples, 28, 28).
y_train, y_test: uint8 array of digit labels (integers in range 0-9) with shape (num_samples,).

siehe (https://keras.io/datasets/)

### Support Vector Machine
// Andreas
Hyperparameter: Anzahl der Keypoints, Größe der Features
Classifiers (C, ... gamma ..) -> Korrelation der Werte
Toleranz (epsilon) -> for regression only
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm
Architekturen mit Ergebnissen (Paper): http://yann.lecun.com/exdb/mnist/
SVM und Hyperparameter: https://stats.stackexchange.com/questions/290642/how-to-choose-a-support-vector-machine-classifier-and-tune-the-hyperparameters

### Neuronales Netz 
// Katharina
 
**Input Layer** 
Welche Merkmale / features eignen sich als Input?
- Pixelwerte
- Gradienten
- Globale Bilddeskriptoren
- …

### CNN


## getestete Hyperparameter


## Plots
Loss function, accuracy, Hyperparameter (heatmap)

### Simulated Annealing
// Andreas 
wichtig: nicht alle Parameter sind unabhängig voneinander. Nur die unabhängigen Parameter verändern, die abhängigen initial wählen und dann konstant lassen

### Preprocessing der Daten
**Test** 
// Katharina
– Verringerung von Kontrast und Helligkeit des Bilddatensatzes:
- Verändert sich die Erkennungsrate und wenn ja, wie / wie stark?
- Wie beeinflusst die Anpassung des Bias-Wertes die eventuelle Veränderung? 

## Ergebnisse
Hier folgt die Ergebnisdiskussion

Vergleich von Erkennungsraten und Rechenaufwand von SVM und NN / CNN

> With some classification methods (particuarly template-based methods, such as SVM and K-nearest neighbors), the error rate improves when the digits are centered by bounding box rather than center of mass. If you do this kind of pre-processing, you should report it in your publications.

### Quellen
Dataset: http://yann.lecun.com/exdb/mnist/
https://pypi.org/project/mnist/#description
