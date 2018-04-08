#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 10:20:22 2018

@author: tunkie
"""

import pandas as pd

def combineData(fin_1, fin_2):
    df_1 = pd.read_csv(fin_1)
    df_2 = pd.read_csv(fin_2)
    df_1 = pd.concat([df_1, df_2],axis = 1)
    return df_1

def main():
    fin_1 = 'train_no_xyz.csv'
    fin_2 = 'xyz_feats.csv'
    df = combineData(fin_1, fin_2)
    fout = 'feats.csv'
    df.to_csv(fout, index=False)

if __name__ == '__main__':
    main()