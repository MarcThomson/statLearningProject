#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 19:38:41 2018

@author: tunkie
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import matplotlib.pyplot as plt

def preProcess(X_train, X_test):
    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train)
    X_test_std = std_scaler.transform(X_test)
    return X_train_std, X_test_std

def getData(fin_train, fin_test):
    df = pd.read_csv(fin_train)
    df_train = df.drop(['formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
    X_train = df_train.values
    y_Hf = df['formation_energy_ev_natom'].values
    y_band = df['bandgap_energy_ev'].values
    df_test = pd.read_csv(fin_test)
    X_test = df_test.values
    return X_train, X_test, y_band, y_Hf

def rmsle(real, predicted):
    sum=0.0
    for x in range(len(predicted)):
        if predicted[x]<0 or real[x]<0: #check for negative values
            continue
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5

def randomForest(X_train, X_test, y_band, y_Hf):
    pipe = make_pipeline(
           RandomForestRegressor(random_state=42))
    band_model = pipe.fit(X_train, y_band)
    y_band_pred = band_model.predict(X_test)
    Hf_model = pipe.fit(X_train, y_Hf)
    y_Hf_pred = Hf_model.predict(X_test)
    #band_mae = mean_absolute_error(y_band, y_band_pred)
    #Hf_mae = mean_absolute_error(y_Hf, y_Hf_pred)
    #print('band mae:', band_mae)
    #print('hf mae:', Hf_mae)
    #band_rmsle = rmsle(y_band, y_band_pred)
    #Hf_rmsle = rmsle(y_Hf, y_Hf_pred)
    #print('band rmsle:', band_rmsle)
    #print('hf rmsle:', Hf_rmsle)
    return y_band_pred, y_Hf_pred

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

def main():
    fin_train = 'train_w_feats.csv'
    fin_test = 'test_w_feats.csv'
    X_train, X_test, y_band, y_Hf = getData(fin_train, fin_test)
    X_train, X_test = preProcess(X_train, X_test)
    y_band, y_Hf = randomForest(X_train, X_test, y_band, y_Hf)
    np.savetxt('band_pred.csv', y_band, delimiter=",")
    np.savetxt('Hf_pred.csv', y_Hf, delimiter=",")
    return 0

if __name__ == '__main__':
    main()