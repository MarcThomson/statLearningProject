# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 15:00:18 2018

@author: Marc Thomson
"""

import pandas as pd
DF = pd.read_csv('train.csv')
spacegroup = DF['spacegroup']

def spaceGroup(dfSpaceGroup):
    uniques = list(set(dfSpaceGroup))
    dfOut = pd.DataFrame()
    for i in range(len(uniques)):
        newData = pd.DataFrame({str(uniques[i]):spacegroup==uniques[i]})
        dfOut = pd.concat([dfOut, newData],axis = 1)
    dfOut = 1*dfOut
    return dfOut,uniques
