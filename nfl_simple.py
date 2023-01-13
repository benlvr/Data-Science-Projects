#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 21:28:02 2023

@author: benlvr
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl

from sklearn import linear_model
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score


#load the data... define a function to help with that
def load_nfl_data(nfl_path):
    csv_path = os.path.join(nfl_path,"train_labels.csv")
    return pd.read_csv(csv_path)

print("Loading the dataset... ")
nfl_path = "/Users/benlvr/Documents/Python-Projects/Kaggle/nfl-player-contact-detection"
nfl = load_nfl_data(nfl_path)

#drop some unwanted or redundant info
nfl = nfl.drop("contact_id",axis=1)
nfl = nfl.drop("datetime",axis=1)
nfl = nfl.drop("game_play",axis=1) #want to keep this eventually

#remove interactions with ground (player id is G)... also want to keep this eventually
nfl = nfl[nfl.nfl_player_id_1 != 'G']
nfl = nfl[nfl.nfl_player_id_2 != 'G']

#separate out labels
nfl_labels = nfl["contact"].copy()
nfl = nfl.drop("contact",axis=1)

#train model
print("Training our model... ")
#sgd_clf = SGDClassifier(loss='log_loss',random_state=42)
#sgd_clf.fit(nfl, nfl_labels)

#logistic regression
# log_reg = linear_model.LogisticRegression()
# log_reg.fit(nfl,nfl_labels)

#Random forest
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(nfl,nfl_labels)

#see how he did on training set
# nfl_pred = log_reg.predict(nfl)
nfl_pred = forest_clf.predict(nfl)


mcc = matthews_corrcoef(nfl_labels, nfl_pred)
con_mat = confusion_matrix(nfl_labels, nfl_pred)
print("Matthews correlation coefficient is ", mcc)
 




