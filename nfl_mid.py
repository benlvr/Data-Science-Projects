#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 22:05:05 2023

@author: benlvr
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupKFold

# import xgboost as xgb... I don't think I have this, might have it in TF or Anaconda somewhere

print("Loading the dataset... ")
nfl_path = "/Users/benlvr/Documents/Python-Projects/Kaggle/nfl-player-contact-detection"
train = pd.read_csv(os.path.join(nfl_path, 'train_labels.csv'), parse_dates=["datetime"])
tr_tracking = pd.read_csv(os.path.join(nfl_path, 'train_player_tracking.csv'), parse_dates=["datetime"])


#taken from XGB Baseline
def create_features(df, tr_tracking, merge_col="step", use_cols=["x_position", "y_position"]):
    output_cols = []
    df_combo = (
        df.astype({"nfl_player_id_1": "str"})
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id",] + use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_1"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .rename(columns={c: c+"_1" for c in use_cols})
        .drop("nfl_player_id", axis=1)
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id"] + use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_2"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .drop("nfl_player_id", axis=1)
        .rename(columns={c: c+"_2" for c in use_cols})
        .sort_values(["game_play", merge_col, "nfl_player_id_1", "nfl_player_id_2"])
        .reset_index(drop=True)
    )
    output_cols += [c+"_1" for c in use_cols]
    output_cols += [c+"_2" for c in use_cols]
    
    if ("x_position" in use_cols) & ("y_position" in use_cols):
        index = df_combo['x_position_2'].notnull()
        # if torch.cuda.is_available():
        #     index = index.to_array()
        distance_arr = np.full(len(index), np.nan)
        tmp_distance_arr = np.sqrt(
            np.square(df_combo.loc[index, "x_position_1"] - df_combo.loc[index, "x_position_2"])
            + np.square(df_combo.loc[index, "y_position_1"]- df_combo.loc[index, "y_position_2"])
        )
        # if torch.cuda.is_available():
        #     tmp_distance_arr = tmp_distance_arr.to_array()
        distance_arr[index] = tmp_distance_arr
        df_combo['distance'] = distance_arr
        output_cols += ["distance"]
        
    df_combo['G_flug'] = (df_combo['nfl_player_id_2']=="G")
    output_cols += ["G_flug"]
    return df_combo, output_cols


use_cols = [
    'x_position', 'y_position', 'speed', 'distance',
    'direction', 'orientation', 'acceleration', 'sa'
]
train, feature_cols = create_features(train, tr_tracking, use_cols=use_cols)
# test, feature_cols = create_features(test, te_tracking, use_cols=use_cols)
# if torch.cuda.is_available():
#     train = train.to_pandas()
#     test = test.to_pandas()

#drop some unwanted or redundant info
train = train.drop("contact_id",axis=1)
train = train.drop("game_play",axis=1) #want to keep this eventually
train = train.drop("datetime",axis=1)
train = train.drop("G_flug",axis=1) #want to keep this eventually

#remove interactions with ground (player id is G)... also want to keep this eventually
train = train[train.nfl_player_id_1 != 'G']
train = train[train.nfl_player_id_2 != 'G']

#split into training and test set
train, test = skl.model_selection.train_test_split(train, test_size=0.2, random_state=42)

#separate out labels
train_labels = train["contact"].copy()
train        = train.drop("contact",axis=1)
test_labels  = test["contact"].copy()
test         = test.drop("contact",axis=1)

#train Random forest model
print("Training our model... ")
forest_clf = RandomForestClassifier(class_weight="balanced",random_state=42)
forest_clf.fit(train,train_labels)

print("Making predictions... ")
pred      = forest_clf.predict(train)
test_pred = forest_clf.predict(test)

mcc = matthews_corrcoef(train_labels,pred)
mcc_test = matthews_corrcoef(test_labels, test_pred)
print("Matthews correlation coefficient on the training set is ", mcc)
print("Matthews correlation coefficient on the test set is ", mcc_test)




