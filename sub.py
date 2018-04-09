#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 13:41:43 2018

@author: tunkie
"""

def main():
    SGs = ['SG_33_data.csv', 'SG_194_data.csv', 'SG_227_data.csv', 'SG_167_data.csv', 'SG_12_data.csv', 'SG_206_data.csv']
    for SG in SGs:
        X, y_band, y_Hf = getData(SG)
        X_train, X_test, y_band_train, y_band_test = split(X, y_band)
        X_train, X_test, y_Hf_train, y_Hf_test = split(X, y_Hf)
        X_train, X_test = preProcess(X_train, X_test)
        randomForest(X_train, X_test, y_band_train, y_band_test, y_Hf_train, y_Hf_test)
    return 0

if __name__ == '__main__':
    out = main()