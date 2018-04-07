#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 20:57:00 2018

@author: tunkie

Feature Generation
"""

import pandas as pd
import numpy as np
import os

def getTrain(fin):
    """
    Args:
        fin (str) - training data filename
    Returns:
        DataFrame of training data
    """
    df = pd.read_csv(fin)
    return df

def get_prop_list():
    """
    Args:
        path_to_element_data (str) - path to folder of elemental property files
    Returns:
        list of elemental properties (str) which have corresponding .csv files
    Source:
        https://www.kaggle.com/cbartel/random-forest-using-elemental-properties
    """
    path_to_element_data = os.path.join('elemental-properties')
    return [f[:-4] for f in os.listdir(path_to_element_data)]

def getProp(prop):
    """
    Args:
        prop (str) - name of elemental property
        path_to_element_data (str) - path to folder of elemental property files
    Returns:
        dictionary of {element (str) : property value (float)}
    Source:
        https://www.kaggle.com/cbartel/random-forest-using-elemental-properties
    """
    fin = os.path.join('elemental-properties', prop + '.csv')
    with open(fin) as f:
        all_els = {line.split(',')[0] : float(line.split(',')[1][:-1]) for line in f}
        my_els = ['Al', 'Ga', 'In']
        return {el : all_els[el] for el in all_els if el in my_els}

def avgProp(x_Al, x_Ga, x_In, prop, prop_dict):
    """
    Args:
        x_Al (float or DataFrame series) - concentration of Al
        x_Ga (float or DataFrame series) - concentration of Ga
        x_In (float or DataFrame series) - concentration of In
        prop (str) - name of elemental property
    Returns:
        average property for the compound (float or DataFrame series), 
        weighted by the elemental concentrations
    Source:
        https://www.kaggle.com/cbartel/random-forest-using-elemental-properties
    """
    els = ['Al', 'Ga', 'In']
    concentration_dict = dict(zip(els, [x_Al, x_Ga, x_In]))
    return np.sum(prop_dict[prop][el] * concentration_dict[el] for el in els)

def getVol(a, b, c, alpha, beta, gamma):
    """
    Args:
        a (float) - lattice vector 1
        b (float) - lattice vector 2
        c (float) - lattice vector 3
        alpha (float) - lattice angle 1 [radians]
        beta (float) - lattice angle 2 [radians]
        gamma (float) - lattice angle 3 [radians]
    Returns:
        volume (float) of the parallelepiped unit cell
    Source:
        https://www.kaggle.com/cbartel/random-forest-using-elemental-properties
    """
    return a*b*c*np.sqrt(1 + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma)
                           - np.cos(alpha)**2
                           - np.cos(beta)**2
                           - np.cos(gamma)**2)

def main():
    fin = 'train.csv'
    df_train = pd.read_csv(fin)
    df_train.head()
    properties = get_prop_list()
    
    # make nested dictionary which maps {property (str) : {element (str) : property value (float)}}
    prop_dict = {prop : getProp(prop) for prop in properties}

    for prop in properties:
        df_train['_'.join(['avg', prop])] = avgProp(df_train['percent_atom_al'], 
                                                 df_train['percent_atom_ga'],
                                                 df_train['percent_atom_in'],
                                                 prop,
                                                 prop_dict)
        
    # convert lattice angles from degrees to radians for volume calculation
    lattice_angles = ['lattice_angle_alpha_degree',
                      'lattice_angle_beta_degree',
                      'lattice_angle_gamma_degree']
    
    for lang in lattice_angles:
        df_train['_'.join([lang, 'r'])] = np.pi * df_train[lang] / 180
        
    # compute the cell volumes 
    df_train['vol'] = getVol(df_train['lattice_vector_1_ang'], 
                              df_train['lattice_vector_2_ang'],
                              df_train['lattice_vector_3_ang'],
                              df_train['lattice_angle_alpha_degree_r'],
                              df_train['lattice_angle_beta_degree_r'],
                              df_train['lattice_angle_gamma_degree_r'])
    
    df_train['atomic_density'] = df_train['number_of_total_atoms'] / df_train['vol'] 
    return df_train;

if __name__ == '__main__':
    out = main()