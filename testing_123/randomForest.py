#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 10:40:16 2018

@author: tunkie
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import train_test_split


def preProcess(X_train, y):
    """Scale training and testing data
    """
    std_scaler = StandardScaler()
    X_sub_train, X_sub_test, y_sub_train, y_sub_test = train_test_split(X_train, y, test_size=0.33, random_state=42)
    X_sub_train_std = std_scaler.fit_transform(X_sub_train)
    X_sub_test_std = std_scaler.transform(X_sub_test)
    return X_sub_train_std, X_sub_test_std, y_sub_train, y_sub_test

def getData(fin_train, fin_test):
    """Convert training and test data to arrays
    """
    df = pd.read_csv(fin_train)
    df_train = df.drop(['formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
    X_train = df_train.values
    y_Hf = df['formation_energy_ev_natom'].values
    y_band = df['bandgap_energy_ev'].values
    df_test = pd.read_csv(fin_test)
    X_test = df_test.values
    return X_train, X_test, y_band, y_Hf

def rmsle(real, predicted):
    """RMSLE scoring metric for assessing model performance
    """
    sum=0.0
    for x in range(len(predicted)):
        if predicted[x]<0 or real[x]<0: #check for negative values
            continue
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5

def report(results, n_top=3):
    """
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def randomForest(X_train, X_test, y, search=False):
    """
    """
    if search == True:
        rfr = RandomForestRegressor(random_state=42)
        param_dist = {"n_estimators" : [175, 200],
                      "max_depth": [15, 20],
                      "max_features": [20, 25],
                      "min_samples_split": [2, 3],
                      "min_samples_leaf": [2, 3],
                      "bootstrap": [True]}
        grid_search = GridSearchCV(rfr, param_grid=param_dist)
        grid_search.fit(X_train, y)
        report(grid_search.cv_results_)
    else:
        rfr = RandomForestRegressor(n_estimators = 175,
                                    max_depth = 20,
                                    max_features = 25)
        band_model = rfr.fit(X_train, y)
        y_pred = band_model.predict(X_test)
        print('rmsle: ', rmsle(y, y_pred))
        return y_pred


def main():
    fin_train = 'train_w_feats.csv'
    fin_test = 'test_w_feats.csv'
    X_train, X_test, y_band, y_Hf = getData(fin_train, fin_test)
    X_sub_train_std, X_sub_test_std, y_sub_train, y_sub_test = preProcess(X_train, y_band)
    #y_pred = randomForest(X_sub_train_std, X_sub_test_std, y_sub_train)
    return 0

if __name__ == '__main__':
    main()