#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 19:01:43 2018

@author: tunkie
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    return X_train, X_test, y_train, y_test

def preProcess(X_train, X_test):
    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train)
    X_test_std = std_scaler.transform(X_test)
    return X_train_std, X_test_std

def getData(fin):
    df = pd.read_csv(fin)
    df_train = df.drop(['formation_energy_ev_natom', 'bandgap_energy_ev', 'Hf_log', 'band_log'], axis=1)
    X_train = df_train.values
    y_Hf = df['formation_energy_ev_natom'].values
    y_band = df['bandgap_energy_ev'].values
    return X_train, y_band, y_Hf

def rmsle(real, predicted):
    sum=0.0
    for x in range(len(predicted)):
        if predicted[x]<0 or real[x]<0: #check for negative values
            continue
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5

def randomForest(X_train, X_test, y_band_train, y_band_test, y_Hf_train, y_Hf_test):
    pipe = make_pipeline(
           RandomForestRegressor(random_state=42))
    band_model = pipe.fit(X_train, y_band_train)
    y_band_pred = band_model.predict(X_test)
    Hf_model = pipe.fit(X_train, y_Hf_train)
    y_Hf_pred = Hf_model.predict(X_test)
    band_mae = mean_absolute_error(y_band_test, y_band_pred)
    Hf_mae = mean_absolute_error(y_Hf_test, y_Hf_pred)
    print('band mae:', band_mae)
    print('hf mae:', Hf_mae)
    band_rmsle = rmsle(y_band_test, y_band_pred)
    Hf_rmsle = rmsle(y_Hf_test, y_Hf_pred)
    print('band rmsle:', band_rmsle)
    print('hf rmsle:', Hf_rmsle)
    return y_band_pred, y_Hf_pred

def extraForest(X_train, y_band, y_Hf):
    pipe = make_pipeline(
           ExtraTreesRegressor())
    band_model = pipe.fit(X_train, y_band)
    Hf_model = pipe.fit(X_train, y_Hf)
    band_scores = cross_val_score(band_model, X_train, y_band, cv=5)
    Hf_scores = cross_val_score(Hf_model, X_train, y_Hf, cv=5)
    print('et band: ', band_scores)
    print('Hf: ', Hf_scores)
    return band_scores, Hf_scores

def lasso(X_train, y_band, y_Hf):
    pipe = make_pipeline(
           Lasso())
    band_model = pipe.fit(X_train, y_band)
    Hf_model = pipe.fit(X_train, y_Hf)
    band_scores = cross_val_score(band_model, X_train, y_band, cv=5)
    Hf_scores = cross_val_score(Hf_model, X_train, y_Hf, cv=5)
    print('lasso band: ', band_scores)
    print('Hf: ', Hf_scores)
    return band_scores, Hf_scores

def nn(X_train, y_band, y_Hf):
    pipe = make_pipeline(
           MLPRegressor())
    band_model = pipe.fit(X_train, y_band)
    Hf_model = pipe.fit(X_train, y_Hf)
    band_scores = cross_val_score(band_model, X_train, y_band, cv=5)
    Hf_scores = cross_val_score(Hf_model, X_train, y_Hf, cv=5)
    print('nn band: ', band_scores)
    print('Hf: ', Hf_scores)
    return band_scores, Hf_scores

def scoring(y_act, y_pred):
    mae = mean_absolute_error(y_act, y_pred)
    print("MAE (test): %.4f" % mae)
    return mae

def importances(X, y):
    forest = ExtraTreesRegressor(n_estimators=250,
                                  random_state=42)
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    
    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()
    return indices

def sampleSub(y_band_pred, y_Hf_pred):
    pass
    

def main():
    SGs = ['SG_33_data.csv', 'SG_194_data.csv', 'SG_227_data.csv', 'SG_167_data.csv', 'SG_12_data.csv', 'SG_206_data.csv']
    for SG in SGs:
        fin = 'SG_12_data.csv'
        X, y_band, y_Hf = getData(fin)
        X_train, X_test, y_band_train, y_band_test = split(X, y_band)
        X_train, X_test, y_Hf_train, y_Hf_test = split(X, y_Hf)
        X_train, X_test = preProcess(X_train, X_test)
        randomForest(X_train, X_test, y_band_train, y_band_test, y_Hf_train, y_Hf_test)
    return 0

if __name__ == '__main__':
    out = main()