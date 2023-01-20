#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 12:40:03 2022

@author: benlvr

MNIST Classifier
Question 3 in from Chapter 3 of Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow by Aurelien Geron
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl

from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef

print("Loading the data... \n")
mnist = fetch_openml('mnist_784', version=1)
X,y = mnist["data"], mnist['target']

#cast labels to integrs
y = y.astype(np.uint8)

#MNIST data set is already split into training and test sets
print("Creating training and test sets... \n")
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#K neighbors classifier
knn_clf = KNeighborsClassifier()
print("Training a K neighbors classifier... \n")
param_grid = [{'n_neighbors': [3, 5, 7, 9]}]
#We do a grid search over different values of k neighbors.
#We set scoring to accuracy because the original question asks for accuracy
#We set return_train_score=False because this saves some computation time
#We set verbose=1 to minimize the message outputs
grid_search = GridSearchCV(knn_clf, param_grid, cv=2, scoring='accuracy', return_train_score=False,verbose=1)
grid_search.fit(X_train, y_train)


#results
print("Best model parameters are: ", grid_search.best_params_)
    

print("Evaluating model on test set... \n")
final_model = grid_search.best_estimator_
y_pred = final_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("The accuracy is ", acc, "\n")

f1 = f1_score(y_test, y_pred,average='weighted')
print("The F1 score is ", f1, "\n")

mcc = matthews_corrcoef(y_test, y_pred)
print("The Matthews correlation coefficient is ", mcc)

