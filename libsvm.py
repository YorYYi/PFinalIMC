#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:14:36 2017

@author: pedroa
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# Cargar el dataset
data = pd.read_csv('dataset3.csv',header=None)
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

#División del dataset en train y test de manera estratificada
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.25, train_size=0.75)

for train_indexes, test_indexes in sss.split(X,y):
    #Split devuelve los indices de los patrones que van a pertecener a train y a test
    X_train, X_test = X[train_indexes], X[test_indexes]
    y_train, y_test = y[train_indexes], y[test_indexes]

    #Estandarización del conjunto de train y test
    scaler = StandardScaler()
    #Se calcula la media y desviación típica del conjunto de train y se realiza la transformación
    X_train = scaler.fit_transform(X_train)
    #Se aplica la transformación con la media y desviación tipica del conjunto de train
    X_test = scaler.transform(X_test)


    # Entrenar el modelo SVM
    svm_model = svm.SVC(kernel='rbf',C=2, gamma=200)
    #Se entrena el modelo con el conjunto de entrenamiento
    svm_model.fit(X_train, y_train)


    #Obtener las predicciones del modelo
    predict_test = svm_model.predict(X_test)
    print("Predicciones del modelo:")
    print(predict_test)
    print("Etiquetas reales:")
    print(y_test)

    #Porcentaje de buena clasificacion

    precisionTrain = svm_model.score(X_train,y_train) 
    precisionTest = svm_model.score(X_test,y_test) 

    print("CCR train = %.2f%%, \tCCR test = %.2f%%" % (precisionTrain*100, precisionTest*100))


    # Representar los puntos
    plt.figure(1)
    plt.clf()
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, zorder=10, cmap=plt.cm.Paired)

    # Representar el hiperplano separador
    plt.axis('tight')
    # Extraer límites
    x_min = X_test[:, 0].min()
    x_max = X_test[:, 0].max()
    y_min = X_test[:, 1].min()
    y_max = X_test[:, 1].max()

    # Crear un grid con todos los puntos y obtener el valor Z devuelto por la SVM
    XX, YY = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]
    Z = svm_model.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Hacer un plot a color con los resultados
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                    levels=[-.5, 0, .5])

    plt.show()