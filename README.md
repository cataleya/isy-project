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
Er basiert auf dem ursprünglichen NIST-Datensatz (Quelle: https://www.nist.gov/sites/default/files/documents/srd/nistsd19.pdf), der für die bessere Nutzung normalisiert und optimiert wurde. Die Bildgröße wurde von 20px * 20px auf 28px * 28px gebracht, wobei das Verhältnis der Bilder beibehalten wurde. Die schwarz-weiss-Bilder wurden in Graustufen umgerechnet. Bei der Optimierung wurden weiterhin die Ziffern auf den Bildern zentriert (center of mass Berechnung).

(Dataset: http://yann.lecun.com/exdb/mnist/)

Der Datensatz ist wiefolgt aufgebaut:
- 60.000 Training Samples 
- 10.000 Test Samples

4 Datensets:
- Training set images 
- Training set labels
- Test set images
- Test set labels

Das Keras Framework stellt den Import des MNIST-Datensatzes bereit. 

Rückgabewert sind 2 Tupel:
- X_train, X_test: uint8 array of grayscale image data with shape (num_samples, 28, 28).
- y_train, y_test: uint8 array of digit labels (integers in range 0-9) with shape (num_samples,).
(Quelle: https://keras.io/datasets/)

Für die Berechnung der Modelle wurde die Anzahl der Trainings- und Testsamples, also die Anzahl der für das Training und das Testen des Modells verwendeten Bilder, variiert. Das Verkleinern der Trainingsmenge geschah vor allem aus Mangel der geeigneten Hardware um die Modelle schneller trainieren zu lassen.

Beispielbilder des Datensatzes:

![](https://github.com/cataleya/isy-project/blob/master/img/documentation/Example-images-from-the-MNIST-dataset.png)

## Implementierung
Folgend werden die verschiedenen Implementierungen und Test des MNIST-Datensatzes beschrieben.

Die Berechnungen erfolgten auf folgenden Computern:
- MacBook Pro (13-inch, Late 2011), Graphics: Intel HD Graphics 3000 512 MB

### Support Vector Machine
*Support Vector Machines* (SVMs) sind hilfreich in zwei denkbaren Anwendungsfällen. Einerseits dienen sie als Regressor. Wir setzen sie jedoch als Klassifikatoren ein.

Dabei haben sie gegenüber den mehrlagigen Perzeptronen folgende Vorteile:
- klassifizieren effizient hochdimensionalen Daten
- liefern auch noch gute Ergebnisse, wenn die Anzahl der Dimensionen größer als die Anzahl der Samples ist
- benutzt nur eine Untermenge der Trainingssamples (die sogenannten *Support Vectors*) für die Vorhersagefunktion

Zu den Nachteilen gehören u.a. folgende Punkte:
- falls die Dimension der Inputdaten viel größer als die Anzahl der Samples ist, kann man *overfitting* durch Regularisierungsterme und die Wahl einer geeigneten Kernelfunktion verhindern
- SVMs führen nicht direkt Wahrscheinlichkeitsschätzungen durch. Diese können aber durch eine rechenlastige fünffache cross-validation berechnet werden
- SVMs do not directly provide probability estimates. these are calculated using an expensive five-fold cross-validation
(https://scikit-learn.org/stable/modules/svm.html)

SVMs gehören zu den supervised machine learning Algorithmen. Ziel des Trainings einer SVM ist es, in den Vektorraum *X*, in dem die Input-Daten leben Trennflächen -- sog. *Hyperflächen* -- einzupassen, die die Trainingsobjekte in Klassen unterteilen. Der Abstand der nächsten Nachbarn zu diesen *Hyperflächen* wird dabei maximiert. Dieser breite Rand soll garantieren, dass später die Testobjekte richtig klassifiziert werden. Beim Berechnen der *Hyperflächen* spielen die weiter von ihr entfernten Trainingsvektoren keine Rolle. Die Vektoren, welche zur Berechnung herangezogen werden, werden gemäß ihrer Funktion auch *Stützvektoren (support vectors)* genannt.

Für den Fall, dass die Trennflächen *Hyperebenen* sind, nennt man die Input-Daten linear trennbar. Diese Eigenschaft erfüllen die meisten Objektmengen jedoch nicht. Um nichtlineare Klassengrenzen zu berechnen, wird der sogenannte *Kernel-Trick* benutzt.
Die Idee hinter dem *Kernel-Trick* ist, die Trainingsvektoren aus dem Raum *X* in einen höherdimensionalen Raum *F* zu überführen, in dem sie dann linear trennbar sind. Es werden die Trennebenen berechnet und diese anschließend in den Raum *X* zurücktransformiert. Diese Rechenoperationen sind effizient mit den sogenannten Kernels durchführbar.
(https://de.wikipedia.org/wiki/Support_Vector_Machine)

![](https://github.com/cataleya/isy-project/blob/master/img/documentation/svmwiki.png)

Wir haben verschiedene SVMs auf ihre erreichte Erkennungsrate hin untersucht. Die SVMs unterscheiden sich in der verwendeten Kernelfunktion und ihren Funktionsparametern. Als Kernelfunktionen haben wir lineare Funktionen und Polynome 2.,4. und 9. Grades benutzt.
Die Funktion des linearen Kernels sieht dabei folgendermaßen aus: $<x, x'>$. 
Die Polynom-Kernels haben folgende Gestalt: $(\gamma\cdot \langle x, x'\rangle + r)^d$

Die mit linearem Kernel trainierte SVM erreichte eine Erkennungsrate beim Test mit den 10.000 Testdaten. von 84$\,$%.

Mit den Polynom-Kernels haben wir jeweils verschiedene Kombinationen der Kernelfunktionsparameter $r$ und $\gamma$ durchprobiert.
Die Ergebnisse sind in den folgenden Diagrammen und Tabellen dargestellt.

![](https://github.com/cataleya/isy-project/blob/master/img/documentation/poly2_variation_von_gamma.jpg)
![](https://github.com/cataleya/isy-project/blob/master/img/documentation/poly2_variation_von_r.jpg)
![](https://github.com/cataleya/isy-project/blob/master/img/documentation/poly2_trainings_samples_serie.jpg)
![](https://github.com/cataleya/isy-project/blob/master/img/documentation/svmpoly4,9.png)


Bisherige Virtual-SVMs haben eine Test-Fehlerrate von 0,56% erreicht (*Virtual SVM deg-9 poly*).
Beste Fehlerrate mit *Reduced Set SVM deg 5 polynomial*: 1,0%
Quelle: DeCoste and Scholkopf (2002)

Y. LeCun, L. Bottou, Y. Bengio and P. Haffner: Gradient-Based Learning Applied to Document Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998, \cite{lecun-98}. 

### Neuronale Netze
Neuronale Netze ahmen die Strukturen des menschlichen Gehirns nach. Die Grundidee dabei ist, dass Neuronen angesteuert und je nach Eingabewert eine Aktivierung des Neurons erfolgt oder nicht, was eine weitere Aktion auslöst, respektive nicht auslöst.
Ziel des Einsatzes des neuronalen Netzes ist die Klassifizierung der Eingabewerte.

Üblicherweise sind neuronale Netzwerke folgendermaßen aufgebaut:
1. Input-Layer (In diesem Projekt die Daten aus MNIST)
2. Hidden Layer (variabel in Anzahl der Layer und Anzahl der Neuronen)
3. Output-Layer (In diesem Projekt 10 Klassen – die Ziffern 0 bis 9)

Beispielsweise wird ein Bild der Ziffer 3 eingegeben, die Bilddaten werden gewichtet, durchlaufen verschiedene Layer und schließlich wird das Bild mit einer gewissen Wahrscheinlichkeit einer gewissen Klasse zugewiesen, zum Beispiel der Klasse Ziffer 3. Dies entspricht dann einer richtigen Klassifizierung.
 
 (http://pages.cs.wisc.edu/~bolo/shipyard/neural/local.html)

Bei der Erstellung von neuronalen Netzen sind vor allem die Anzahl der hidden Layer und Anzahl der Neuronen die maßgebenden Stellschrauben.

In diesem Projekt wurden zwei verschiedene Strukturen getestet, die auch im Paper [“Deep Big Simple Neural Nets Excel on Handwritten Digit Recognition“](https://arxiv.org/pdf/1003.0358.pdf) von Ciresan, et al. getestet wurden:

![](https://github.com/cataleya/isy-project/blob/master/img/documentation/netze_paper.png)

Wie in der Tabelle zu sehen ist, ähneln sich die ersten 4 Netzstrukturen, es wird jeweils ein Layer mit 500 Neuronen vorgeschaltet. Die 5. Architektur hingegen besteht aus 9 Layern mit jeweils 1000 Neuronen + 10 (Outputlayer mit 10 Klassen).
Die Ergebnisse zeigen, dass mit steigender Layer- und Neuronenanzahl auch die Erkennungsrate steigt, beziehungsweise der Test Error sinkt. 

Beispielhaft sieht man hier die Strukturen für Architekturen mit ID 1 und ID 5:

**Struktur 1: 1000, 500, 10**
```
nn1 = Sequential()
nn1.add(Dense(1000, activation='relu', input_shape=(784,)))
nn1.add(Dense(500, activation ='relu'))
nn1.add(Dense(classes, activation='softmax'))
nn1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
**Struktur 2: 9 x 1000, 10**
```
nn5 = Sequential()
nn5.add(Dense(1000, activation='relu', input_shape=(784,)))
nn5.add(Dense(1000, activation ='relu'))
nn5.add(Dense(1000, activation ='relu'))
nn5.add(Dense(1000, activation ='relu'))
nn5.add(Dense(1000, activation ='relu'))
nn5.add(Dense(1000, activation ='relu'))
nn5.add(Dense(1000, activation ='relu'))
nn5.add(Dense(1000, activation ='relu'))
nn5.add(Dense(1000, activation ='relu'))
nn5.add(Dense(classes, activation='softmax'))
nn5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

Es wurden für alle 5 Netzstrukturen eine Berechnung mit 3000 Trainings- und 1000 Testsamples durchgeführt.
Die Modelle wurden zum einen über 30 Epochen, zum zweiten über 50 Epochen trainiert.

Folgend sieht man die dazugehörigen Diagramme im Bezug Erkennungsrate zu Epoche:

![](https://github.com/cataleya/isy-project/blob/master/img/documentation/nn3000_30_50.jpg)

Man erkennt, dass sich die Daten für Architektur 1 bis 5 ähnlich sind und die Graphen ähnlich verlaufen. Die Architektur mit ID 5 sieht man immer wieder Ausreißer nach unten. 

Für die Netzarchitekturen mit ID 1 und ID 5 (siehe Tabelle) wurden die Modelle weiterhin mit dem vollständigen unveränderten MNIST-Datensatz über 30 und 50 Epochen trainiert und getestet um zu sehen, ob sich die Erkennungsraten weiter steigern lassen, wenn die Trainingssample-Anzahl erhöht wird. 

![](https://github.com/cataleya/isy-project/blob/master/img/documentation/nn1_nn5_30_50.jpg)

Man erkennt, dass die Erkennungsrate von ca. 93 % auf ca. 98 % steigt. Man kann außerdem beobachten, dass sich die Ergebnisse durch Erhöhung der Epochen nicht wesentlich steigern lassen.

In der folgenden Abbildung sind die Diagramme für Erkennungsraten und Lossfunktionen für die Architekturen 1 und 5 gegenübergestellt, beide Architekturen mit dem vollständigen Datensatz und über 50 Eprochen trainiert:

![](https://github.com/cataleya/isy-project/blob/master/img/documentation/nn1nn5_60000_50.jpeg)

Auch hier erkennt man bei Architektur 5 einen starken Ausreißer bei 32 Epochen. Generell kommen aber beide Testläufe auf sehr gute Erkennungsraten und niedrige Loss-Werte.

**Übersicht der Erkennungsraten**
Die folgende Tabelle zeigt die Übersicht der erreichten Erkennungsraten.

![](https://github.com/cataleya/isy-project/blob/master/img/documentation/table.png)

Man sieht, dass bei 3000 Trainingssamples und 1000 Testsamples die Erkennungsraten schlechter sind als bei 60000 Trainingssamples und 10000 Testsamples (ca. 5 %-Punkte). Die Unterschiede zwischen den Architekturen der Modelle beeinflussen nicht wesentlich die Erkennungsraten, also verbessern sich nicht wie zu erwarten wäre, wenn sich Layer- und Neuronenanzahl erhöhen. Auch durch die Erhöhung der Epochenanzahl werden die Erkennungsraten nicht gesteigert.

Im Vergleich zu den Erkennungsraten bzw. Test Errors des oben erwähnten Papers [“Deep Big Simple Neural Nets Excel on Handwritten Digit Recognition“](https://arxiv.org/pdf/1003.0358.pdf) konnten die Ergebnisse nicht reproduziert werden.
Dies kann damit erklärt werden, dass das Paper zusätzlich zum originalen MNIST-Datensatz die Trainings- und Testdaten vor dem Training durch Drehung, Skalierung, etc. erweitert haben und somit die Modelle noch besser trainieren konnten. Dies bestätigt weiterhin, dass die verwendeten Trainigs- und Testdaten essentiell für den Erfolg neuronaler Netzwerke sind.

### CNN

Neben *SVMs* und *NNs* haben wir auch *Convolutional Neural Nets* (*CNNs*) für die Ziffernerkennung benutzt. Ein *CNN* ist aus drei verschiedenen Arten von Layers aufgebaut. Die Struktur beginnt grundsätzlich mit einer beliebigen Anzahl und Abfolge von *Convolutional Layers* und *Pooling Layers*. Dem Ganzen ist eine beliebige Zahl *Fully Connected Layers* wie beim mehrlagigen Perzeptron verbunden. Eine besondere Aufgabe kommt dabei den *Convolutional Layers* zu, welche die Aktivierung der nachfolgenden Neuronen über eine diskrete Faltung berechnen. Veranschaulicht findet das Netzwerk über die Optimierung der faltenden Kerneleinträge geeignete Features, die die Klassen für das nachgeschaltete Perzeptron besser unterscheidbar machen.
Man muss sich deshalb keine Gedanken machen, mit welchen Featurevektoren man das *CNN* speist, da das netzwerk diese Aufgabe übernimmt.
Als Input verwenden wir deshalb die 28x28 Graustufenpixelwerte der Bilder.

In der folgenden Abbildung ist die Netzwerkarchitektur gezeigt, welche uns als Basis für weitere Variationen dient.

![](https://github.com/cataleya/isy-project/blob/master/img/documentation/CNN_Basic_Architecture.png)

Als weiteren Hyperparameter haben wir die Batchsize zur Verfügung, welche wir -- abgesehen von der Batch-Size-Serie -- zu 128 wählen. Die folgenden Architekturen wurden alle mit 2500 Trainingsdaten trainiert und jede Epoche mit 10000 Testdaten validiert.

Als erstes haben wir untersucht, welchen Einfluss die Tiefe des *CNN's* hat. Dazu haben wir die Anzahl an hintereinanderfolgenden Conv-Conv-Pooling-Dropout-Schichten variiert. Die Ergebnisse sind in der folgenden Abbildung zu sehen, wobei wir 20 Epochen trainiert haben.

![](https://github.com/cataleya/isy-project/blob/master/img/documentation/CNN_pooling_layer_serie.jpg)

Wie subjektiv erkannt werden kann, schneidet die Architektur, bei der wir die Dropout-Layers weggelassen haben, um ca. einen halben Prozent schlechter ab, als die Architekturen mit 1 bzw. 2 Schichten und mit Dropout-Layer.

Als nächstes haben wir die Anzahl der conv-Layers pro Schicht variiert. Wie in der folgenden Abbildung zu sehen haben wir 1, 3, 4 und 5 conv-Layers ausprobiert.

![](https://github.com/cataleya/isy-project/blob/master/img/documentation/CNN_conv_layers_serie.jpg)

Dabei zeigt sich, dass die Netze mit 1 und 3 conv-Layers pro Schicht einen ähnlichen Trainingserfolg von 96 bis 98$\,$% aufweisen. Bei den Netzen mit 4 und 5 conv-Layers ist augenscheinlich ein Fehler aufgetreten, da sich die Erkennungsraten um 10$\,$% aufhalten, was einem zufälligen Raten gleichkommt. 

Aus den ersten beiden Testreihen entnehmen wir, dass die Veränderung der Layeranzahl nahezu keinen Einfluss auf die Qualität des trainierten Netzes hat.

Dagegen ist bei der Variation der im Folgenden betrachteten Parameter jeweils eine klare Auswirkung beobachtbar.

Als nächstes haben wir die Poolgröße verändert.

![](https://github.com/cataleya/isy-project/blob/master/img/documentation/CNN_pool_size_serie.jpg)

Mit einer Poolgröße von 3x3 Pixel benötigt das Netz weniger Epochen zum Training, als bei der Verwendung der 5x5 Poolgröße. Außerdem 
sieht man, dass die Verwendung von Dropout layers die benötigten Epochen zusätzlich reduziert.
Allerdings kommen alle Architekturen nach 20 Epochen auf etwa dieselbe Erkennungsrate zwischen 96 und 97$\,$%.

Einen noch stärker erkennbaren Einfluss auf das Training hat die Wahl der batch size, wie im folgenden Diagramm zu sehen.

![](https://github.com/cataleya/isy-project/blob/master/img/documentation/CNN_pool_size_serie.jpg)

Bei der Verwendung einer zu großen batch size finden pro Epoche weniger back propagations statt. Dadurch läuft das Training schneller, jedoch benötigt man wie zu sehen auch mehr Epochen. Die Ergebnisse liegen zwischen 95 und 97$\,$%.

Als letztes haben wir noch den Einfluss der Neuronenzahl in den dense layers variiert. Das Ergebnis ist im nachfolgenden Diagramm zu sehen.

![](https://github.com/cataleya/isy-project/blob/master/img/documentation/CNN_neuronenzahl_serie.jpg)

Klar ersichtlich ist, dass das Netz mit nur 8 Neuronen pro dense layer viel mehr Epochen benötigt und auch nach vielen Epochen noch keine gute Erkennungsrate erzielt. Mit steigender Anzahl der Neuronen steigt auch der Trainingserfolg pro Epoche und auch die letztendlich erreichte Erkennungsrate nach 20 Epochen an.


## Ergebnisse
Generell kann festgehalten werden, dass die Convolutional Neural Networks im Vergleich zu den Neural Networks schneller, zu besseren Ergebnissen führen. Dies ist darin begründet, dass die NN als klassischer Ansatz für die Klassifizierung von verschiednen Datenstrukturen verwendet werden kann, CNNs aber besonders geeignet für die Klassifizierung von Bilddaten sind. Der Erfolg der Support Vector Machines hingegen ist stark von den verwendeten Kernelparametern abhängig, SVM kann im Vergleich zu den Netzstrukturen schnell berechnet werden und kommt mit wenig Hyperparametern aus.

**Beste Erkennungsraten der getesteten Modelle** 
- SVM: 98,1 % (Polynom 2. Grades, 60000 Trainingsdaten) 
- NN: 98,2 % (NN1, 50 Epochen, 60000 Trainingsdaten) 
- CNN: 97,4 % (20 Epochen, 2500 Trainingsdaten)

In Vergleich dazu können die Ergebnisse aus verschiedenen Veröffentlichungen (Quelle: http://yann.lecun.com/exdb/mnist/) gesetzt werden: 

- SVM: 99,4 % (Virtual SVM, deg-9 poly, 2-pixel jittered) 
- NN: 99,65 % (6-layer, 784-2500-2000-1500-1000-500-10 (on GPU) [elastic distortions]) 
- CNN: 99,7 % (35 conv. net, 1-20-P-40-P-150-10 [elastic distortions]) 

Man sieht, dass die von uns erzeugten Ergebnisse um 1.3 % bei SVM, 1.45 % bei NN und 2.3 % bei CNN von den zuvor erzielten Werten abweichen. Die relativ starke Abweichung bei CNN lässt sich dadurch erklären, dass die im Paper verwendete Architektur wesentlich größer gewählt wurde, zusätzlich wurden auch wie bei NN die Daten vor dem Trainingsbeginn durch eine elastic distortion erweitert, was ebenfalls zu besseren Erkennungsraten führt.
