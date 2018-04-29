#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:04:06 2018

@author: tunkie
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

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

#fin_train = 'train_w_feats.csv'
#df = pd.read_csv(fin_train)
#df_train = df.drop(['formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
#X_train = df_train.values
#y_Hf = df['formation_energy_ev_natom'].values
##y_band = df['bandgap_energy_ev'].values
#
#std_scaler = StandardScaler()
#X_train, X_test, y_train, y_test = train_test_split(X_train, y_Hf, test_size=0.05)
#X_train_std = std_scaler.fit_transform(X_train)
#X_test_std = std_scaler.transform(X_test)
#
#rfr = RandomForestRegressor()
#extra = ExtraTreesRegressor()
#rf_model = rfr.fit(X_train, y_train)
#ex_model = extra.fit(X_train, y_train)
#y_pred_rf = rf_model.predict(X_test)
#y_pred_ex = ex_model.predict(X_test)
#print('RF rmsle: ', rmsle(y_test, y_pred_rf))
#print('RF mae: ', mean_absolute_error(y_test, y_pred_rf))
#print('ET rmsle: ', rmsle(y_test, y_pred_ex))
#print('ET mae: ', mean_absolute_error(y_test, y_pred_ex))


fin_train = 'train_w_feats.csv'
fin_test = 'test_w_feats.csv'
df_train = pd.read_csv(fin_train)
df_test = pd.read_csv(fin_test)
y_Hf = df_train['formation_energy_ev_natom'].values
y_band = df_train['bandgap_energy_ev'].values
df_train = df_train.drop(['formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
X_train = df_train.values
X_test = df_test.values

std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)

extra1 = RandomForestRegressor()
extra2 = RandomForestRegressor()
#extra1 = ExtraTreesRegressor()
#extra2 = ExtraTreesRegressor()
band_model = extra1.fit(X_train, y_band)
Hf_model = extra2.fit(X_train, y_Hf)
y_pred_band = band_model.predict(X_test)
y_pred_Hf= Hf_model.predict(X_test)

id = [i for i in range(1, 601)]
df_sub = pd.DataFrame({'id' : id, 'bandgap_energy_ev' : y_pred_band, 'formation_energy_ev_natom' : y_pred_Hf})
df_sub.to_csv(path_or_buf='GOD.csv', columns=['id', 'formation_energy_ev_natom', 'bandgap_energy_ev'], index=False)