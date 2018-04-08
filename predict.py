#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 19:01:43 2018

@author: tunkie
"""

import pandas as pd

def getData(fin):
    df = pd.read_csv(fin)
    df_train = df.drop(['formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
    X_train = df_train.values
    y_band = df['formation_energy_ev_natom'].values
    y_Hf = df['bandgap_energy_ev'].values
    return X_train, y_band, y_Hf

def main():
    fin = 'train_no_xyz.csv'
    X_train, y_band, y_Hf = getData(fin)
    return X_train, y_band, y_Hf

if __name__ == '__main__':
    X_train, y_band, y_Hf = main()