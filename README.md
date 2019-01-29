# Projektdokumentation Interactive Systems WS 18 / 19
## Projektmitglieder 
- Andreas Hallmann
- Katharina Krebs
## Projektbeschreibung
Im Rahmen des Kurses Interactive Systems an der Beuth Hochschule werden verschiedene Aspekte des maschinellen Lernens beleuchtet. Ziel dieses Projekts ist eine weitere Auseinandersetzung mit verschiedenen Methoden, wie Support Vector Machines und neuronale Netze. Für dieses Projekt wird der MNIST-Datensatzes verwendet.
## Stand der Technik - Erkennung von handgeschriebenen Ziffern
Die Erkennung von handgeschriebenen Ziffern ist gut erforscht, dieses Beispiel wird in Verbindung mit dem MNIST-Datensatz zum Trainieren und Testen von Modellen im Bereich des maschinellen Lernens verwendet.
### Datensatz 
Der MNIST-Datensatz enthält Bilddaten von handgeschiebenen Ziffen von 0 bis 9 in Graustufen (Pixelwerte von 0 bis 255).
Er basiert auf dem ursprünglichen NIST-Datensatz (Quelle: https://www.nist.gov/sites/default/files/documents/srd/nistsd19.pdf), der für die bessere Nutzung normalisiert und optimiert wurde. Die Bildgröße wurde von 20px * 20px auf 28px * 28px gebracht, wobei das Verhältnis der Bilder beibehalten wurde. Die schwarz-weiss-Bilder wurden in Graustufen umgerechnet. Bei der Optimierung wurden weiterhin die Ziffen auf den Bildern zentriert (center of mass Berechnung).

Der Datensatz ist wiefolgt aufgebaut:
- 60.000 Training Samples 
- 10.000 Test Samples

4 Datensets:
- Training set images 
- Training set labels
- Test set images
- Test set labels

Das Keras Framework stellt den Import des MNIST-Datensatzes bereit. 
Rückgabewert:
2 tuples:
X_train, X_test: uint8 array of grayscale image data with shape (num_samples, 28, 28).
y_train, y_test: uint8 array of digit labels (integers in range 0-9) with shape (num_samples,).

siehe (https://keras.io/datasets/)

Beispielbilder des Datensatzes:
![](https://github.com/cataleya/isy-project/blob/master/img/documentation/Example-images-from-the-MNIST-dataset.png)

## Implementierung
Folgend werden die verschiedenen Implementierungen und Test des MNIST-Datensatzes beschrieben.

### Support Vector Machine
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

Zu implementierende Modelle + Hyperparameter:
 linear kernel: <x, x'>
 rbf kernel: exp(-gamma*||x-x'||^2)
 polynomial: (gamma*<x, x'> + r)^d ... d->degree , r->coef0

SVC(...):
 C=1.0 
 kernel=’rbf’, ‘linear’, ‘poly’
 degree=2, 4, 9
 gamma=’auto_deprecated’ (=1/n_features) ODER: Probiere mit gamma= 1/n_features, 100/n_features, 0.01/n_features
 coef0=0.0 ... habe keine Ahnung, welche Werte hier sinnvoll wären
 shrinking=True, 
 probability=False 
 tol=0.001 
 cache_size=2000
 class_weight=None
 verbose=False
 max_iter=-1 ... no limit for number of iterations
 decision_function_shape=’ovr’
 random_state=None)


Auswertung:

Bisherige Virtual-SVMs haben eine Test-Fehlerrate von 0,56% erreicht (*Virtual SVM deg-9 poly*).
Beste Fehlerrate mit *Reduced Set SVM deg 5 polynomial*: 1,0%
Quelle: DeCoste and Scholkopf (2002)

Y. LeCun, L. Bottou, Y. Bengio and P. Haffner: Gradient-Based Learning Applied to Document Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998, \cite{lecun-98}. 

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
