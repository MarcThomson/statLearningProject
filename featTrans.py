#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 11:04:07 2018

@author: tunkie
"""
import pandas as pd
import numpy as np

def getData(fin):
    df = pd.read_csv(fin)
    df_train = df.drop(['formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
    X_train = df_train.values
    y_band = df['formation_energy_ev_natom'].values
    y_Hf = df['bandgap_energy_ev'].values
    return X_train, y_band, y_Hf

def logTrans(df):
    df['Hf_log'] = df['formation_energy_ev_natom'].apply(lambda row: np.log(row))
    df['band_log'] = df['bandgap_energy_ev'].apply(lambda row: np.log(row))
    return df

def amIn(df):
    df['Al?'] = (df['percent_atom_al'] > 0) * 1
    df['Ga?'] = (df['percent_atom_ga'] > 0) * 1
    df['In?'] = (df['percent_atom_in'] > 0) * 1
    return df

def spaceGrouper(df):
    df_33 = df[df['SG_33'] == 1]
    df_33 = df_33.drop(['SG_33', 'SG_194', 'SG_227', 'SG_167', 'SG_12', 'SG_206'], axis=1)
    df_194 = df[df['SG_194'] == 1]
    df_194 = df_194.drop(['SG_33', 'SG_194', 'SG_227', 'SG_167', 'SG_12', 'SG_206'], axis=1)
    df_227 = df[df['SG_227'] == 1]
    df_227 = df_227.drop(['SG_33', 'SG_194', 'SG_227', 'SG_167', 'SG_12', 'SG_206'], axis=1)
    df_167 = df[df['SG_167'] == 1]
    df_167 = df_167.drop(['SG_33', 'SG_194', 'SG_227', 'SG_167', 'SG_12', 'SG_206'], axis=1)
    df_12 = df[df['SG_12'] == 1]
    df_12 = df_12.drop(['SG_33', 'SG_194', 'SG_227', 'SG_167', 'SG_12', 'SG_206'], axis=1)
    df_206 = df[df['SG_206'] == 1]
    df_206 = df_206.drop(['SG_33', 'SG_194', 'SG_227', 'SG_167', 'SG_12', 'SG_206'], axis=1)
    return df_33, df_194, df_227, df_167, df_12, df_206

def main():
    fin = 'feats.csv'
    df = pd.read_csv(fin)
    df = logTrans(df)
    df = amIn(df)
    df_33, df_194, df_227, df_167, df_12, df_206 = spaceGrouper(df)
    df.to_csv('feats_trans.csv', index=False)
    df_33.to_csv('SG_33_data.csv')
    df_194.to_csv('SG_194_data.csv')
    df_227.to_csv('SG_227_data.csv')
    df_167.to_csv('SG_167_data.csv')
    df_12.to_csv('SG_12_data.csv')
    df_206.to_csv('SG_206_data.csv')
    return df_33

if __name__ == '__main__':
    df_33 = main()