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
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def getData(fin_train, fin_test):
    """Read in training and testing features and targets
    Args:
        fin_train (str) - training data filename
        fin_test (str) - testing data filename
    Return:
        X_train (array) - training feature dataset
        X_test (array) - testing feature dataset
        y_band (array) - training band targets
        y_Hf (array) - training Hf targets
    """
    df = pd.read_csv(fin_train)
    df_train = df.drop(['formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
    X_train = df_train.values
    y_Hf = df['formation_energy_ev_natom'].values
    y_band = df['bandgap_energy_ev'].values
    df_test = pd.read_csv(fin_test)
    X_test = df_test.values
    return X_train, X_test, y_band, y_Hf

def preProcess(X_train, X_test, y_band, y_Hf):
    """Scale training and testing data
    Args:
        X_train (array) - training feature dataset
        X_test (array) -  testing feature dataset
        protocol (str) - specify analysis
    Returns:
        X_train_std (array) - training feature dataset scaled
        X_test_std (array) - testing feature dataset scaled
    """
    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train)
    X_test_std = std_scaler.transform(X_test)
    return X_train_std, X_test_std


def rmsle(real, predicted):
    """Compute the rmsle score
    Args:
        real (array) - targets
        predicted (array) - predictions
    Returns rmlse (float) - rmsle score
    """
    sum=0.0
    for x in range(len(predicted)):
        if predicted[x] < 0 or real[x] < 0: #check for negative values
            continue
        p = np.log(predicted[x] + 1)
        r = np.log(real[x] + 1)
        sum = sum + (p - r)**2
    rmsle = (sum / len(predicted))**0.5
    return rmsle

def randomForest(X_train, X_test, y_band, y_Hf):
    """
    Args:
        X_train (array) - training feature dataset
        X_test (array) - testing feature dataset
        y_band (array) - training bandgap targets
        y_Hf (array) - training Hf targets
    Returns:
        y_band_pred (array) - predicted bandgap energy
        y_Hf_pred (array) - predicted enthalpy of formation
    """
    pipe = make_pipeline(RandomForestRegressor(random_state=42))
    band_model = pipe.fit(X_train, y_band)
    y_band_pred = band_model.predict(X_test)
    Hf_model = pipe.fit(X_train, y_Hf)
    y_Hf_pred = Hf_model.predict(X_test)
    return y_band_pred, y_Hf_pred

def importances(X, y):
    """Rank and plot features based on Random Forest Importances
    Args:
        X (array) - features
        y (array) - targets
    Returns:
        indices (list) - list correspoding to ranked feature order
    """
    forest = ExtraTreesRegressor(n_estimators=250,
                                 random_state=42)
    forest.fit(X, y)
    importances = forest.feature_importances_
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
    # dataset filenames
    fin_train = 'train_w_feats.csv'
    fin_test = 'test_w_feats.csv'
    
    # get datasets
    X_train, X_test, y_band, y_Hf = getData(fin_train, fin_test)
    
    # proceed with analysis
    X_train, X_test = preProcess(X_train, X_test, y_band, y_Hf)
    y_band, y_Hf = randomForest(X_train, X_test, y_band, y_Hf)
    
    id = [i for i in range(1, 601)]
    df_sub = pd.DataFrame({'id' : id, 'bandgap_energy_ev' : y_band, 'formation_energy_ev_natom' : y_Hf})
    df_sub.to_csv(path_or_buf='GOD2.csv', columns=['id', 'formation_energy_ev_natom', 'bandgap_energy_ev'], index=False)
    
#    np.savetxt('band_pred.csv', y_band, delimiter=",")
#    np.savetxt('Hf_pred.csv', y_Hf, delimiter=",")
    return X_train, X_test, y_band, y_Hf

if __name__ == '__main__':
    X_train, X_test, y_band, y_Hf = main()