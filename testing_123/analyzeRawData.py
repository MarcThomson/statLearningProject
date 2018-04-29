#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 10:39:10 2018

@author: tunkie

Script to analyze raw dataset...
"""
import pandas as pd
from pandas.plotting import scatter_matrix
from pandas.plotting import radviz

def funky(df):
    df = df.drop(['id'], axis=1)
    feats = df.drop(['formation_energy_ev_natom', 'bandgap_energy_ev']).columns
    scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
    #radviz(df, 'spacegroup')

def main():
    fin = 'train.csv'
    df = pd.read_csv(fin)
    funky(df)
    return df

if __name__ == '__main__':
    df = main()