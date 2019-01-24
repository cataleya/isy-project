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
Hyperparameter: Anzahl der Keypoints, Größe der Features,
Classifiers (C, ... gamma ..) -> Korrelation der Werte
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm
Architekturen mit Ergebnissen (Papers): http://yann.lecun.com/exdb/mnist/
SVM und Hyperparameter: https://stats.stackexchange.com/questions/290642/how-to-choose-a-support-vector-machine-classifier-and-tune-the-hyperparameters
//
SVMs können zu zweierlei Zwecke eingesetzt werden. Einerseits dienen sie als Regressor. Für unser Projekt ist aber nur die zweite Funktion -- die Klassifikation -- interessant.

TODO:
The advantages of support vector machines are:
- Effective in high dimensional spaces
- Still effective in cases where number of dimensions is greater than the number of samples
- Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient
- Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels

The disadvantages of support vector machines include:
If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).
Quelle: https://scikit-learn.org/stable/modules/svm.html


Die Trainingsobjekte müssen gelabelt und als Vektoren vorliegen. Ziel des Trainings einer SVM ist es, in den Vektorraum *X*, in dem die Vektoren leben Trennflächen -- sog. *Hyperflächen* -- einzupassen, die die Trainingsobjekte in Klassen unterteilen. Der Abstand der nächsten Nachbarn zu diesen *Hyperflächen* wird dabei maximiert. Dieser breite Rand soll garantieren, dass später die Testobjekte richtig klassifiziert werden. Beim Berechnen der *Hyperflächen* spielen die weiter von ihr entfernten Trainingsvektoren keine Rolle. Die Vektoren, welche zur Berechnung herangezogen werden, werden gemäß ihrer Funktion auch *Stützvektoren* genannt.

Für den Fall, dass die Trennflächen *Hyperebenen* sind, nennt man die Objekte linear trennbar. Diese Eigenschaft erfüllen die meisten Objektmengen jedoch nicht. Um nichtlineare Klassengrenzen zu berechnen, wird der sogenannte *Kernel-Trick* benutzt.
Die Idee hinter dem *Kernel-Trick* ist, die Trainingsvektoren aus dem Raum *X* in einen höherdimensionalen Raum *F* zu überführen, in dem sie dann linear trennbar sind. Es werden die Trennebenen berechnet und diese anschließend in den Raum *X* zurücktransformiert.

Sogenannte *Schlupfvariablen* machen SVMs flexibler. Mit ihnen lassen sich sogenannte *Soft Margin Hyperflächen* berechnen. Diese lassen es zu, dass Ausreißer in den Trainngsdaten weniger Beachtung finden. Dadurch wird *overfitting* vermieden und es werden weniger *Stützvektoren* benötigt.




Quelle: https://de.wikipedia.org/wiki/Support_Vector_Machine


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
