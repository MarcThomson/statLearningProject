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
from numpy.linalg import inv
import itertools
from numba import jit

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
def spaceGroup(dfSpaceGroup):
    uniques = list(set(dfSpaceGroup))
    dfOut = pd.DataFrame()
    for i in range(len(uniques)):
        newData = pd.DataFrame({'SG_'+str(uniques[i]):dfSpaceGroup==uniques[i]})
        dfOut = pd.concat([dfOut, newData],axis = 1)
    dfOut = 1*dfOut
    return dfOut

def length(v):
    """
    Source:
        https://www.kaggle.com/tonyyy/how-to-get-atomic-coordinates
    """
    return np.linalg.norm(v)

def get_xyz_data(filename):
    """
    Source:
        https://www.kaggle.com/tonyyy/how-to-get-atomic-coordinates
    """
    pos_data = []
    lat_data = []
    with open(filename) as f:
        for line in f.readlines():
            x = line.split()
            if x[0] == 'atom':
                pos_data.append([np.array(x[1:4], dtype=np.float),x[4]])
            elif x[0] == 'lattice_vector':
                lat_data.append(np.array(x[1:4], dtype=np.float))
    return pos_data, np.array(lat_data)

def get_shortest_distances(reduced_coords, amat):
    """
    Source:
        https://www.kaggle.com/tonyyy/how-to-get-atomic-coordinates
    """
    natom = len(reduced_coords)
    dists = np.zeros((natom, natom))
    Rij_min = np.zeros((natom, natom, 3))

    for i in range(natom):
        for j in range(i):
            rij = reduced_coords[i][0] - reduced_coords[j][0]
            d_min = np.inf
            R_min = np.zeros(3)
            for l in range(-1, 2):
                for m in range(-1, 2):
                    for n in range(-1, 2):
                        r = rij + np.array([l, m, n])
                        R = np.matmul(amat, r)
                        d = length(R)
                        if d < d_min:
                            d_min = d
                            R_min = R
            dists[i, j] = d_min
            dists[j, i] = dists[i, j]
            Rij_min[i, j] = R_min
            Rij_min[j, i] = -Rij_min[i, j]
    return dists, Rij_min

def get_min_length(distances, A_atoms, B_atoms):
    """
    Source:
        https://www.kaggle.com/tonyyy/how-to-get-atomic-coordinates
    """
    A_B_length = np.inf
    for i in A_atoms:
        for j in B_atoms:
            d = distances[i, j]
            if d > 1e-8 and d < A_B_length:
                A_B_length = d
    
    return A_B_length   

def xyzFeats(df, dataset):
    xyz_feats_dict = {}
    for idx in range(1,len(df.index)+1):
        fn = "{}/{}/geometry.xyz".format(dataset, idx)
        crystal_xyz, crystal_lat = get_xyz_data(fn)
        A = np.transpose(crystal_lat)
        R = crystal_xyz[0][0]
        B = inv(A)
        r = np.matmul(B, R)
        crystal_red = [[np.matmul(B, R), symbol] for (R, symbol) in crystal_xyz]
        crystal_dist, crystal_Rij = get_shortest_distances(crystal_red, A)
        natom = len(crystal_red)
        al_atoms = [i for i in range(natom) if crystal_red[i][1] == 'Al']
        ga_atoms = [i for i in range(natom) if crystal_red[i][1] == 'Ga']
        in_atoms = [i for i in range(natom) if crystal_red[i][1] == 'In']
        o_atoms = [i for i in range(natom) if crystal_red[i][1] == 'O']
        al_length, ga_length, in_length = 0, 0, 0
        al_o_dist = np.zeros([len(al_atoms), len(o_atoms)])
        ga_o_dist = np.zeros([len(ga_atoms), len(o_atoms)])
        in_o_dist = np.zeros([len(in_atoms), len(o_atoms)])
        al_coord = 0
        al_o_mean = 0
        ga_coord = 0
        ga_o_mean = 0
        in_coord = 0
        in_o_mean = 0
        if len(al_atoms):
            al_length = 1.30 * get_min_length(crystal_dist, al_atoms, o_atoms)
            for i in range(len(al_atoms)):
                for j in range(len(o_atoms)):
                    al_o_dist[i,j] = crystal_dist[al_atoms[i], o_atoms[j]]
            al_o_dist = np.select([al_o_dist < al_length], [al_o_dist])
            al_o_dist = al_o_dist.flatten()
            al_o_dist = al_o_dist[np.nonzero(al_o_dist)]
            al_coord = len(al_o_dist) / len(al_atoms)
            al_o_mean = np.mean(al_o_dist)
        if len(ga_atoms):
            ga_length = 1.30 * get_min_length(crystal_dist, ga_atoms, o_atoms)
            for i in range(len(ga_atoms)):
                for j in range(len(o_atoms)):
                    ga_o_dist[i,j] = crystal_dist[ga_atoms[i], o_atoms[j]]
            ga_o_dist = np.select([ga_o_dist < ga_length], [ga_o_dist])
            ga_o_dist = ga_o_dist.flatten()
            ga_o_dist = ga_o_dist[np.nonzero(ga_o_dist)]
            ga_coord = len(ga_o_dist) / len(ga_atoms)
            ga_o_mean = np.mean(ga_o_dist)
        if len(in_atoms):
            in_length = 1.30 * get_min_length(crystal_dist, in_atoms, o_atoms)
            for i in range(len(in_atoms)):
                for j in range(len(o_atoms)):
                    in_o_dist[i,j] = crystal_dist[in_atoms[i], o_atoms[j]]
            in_o_dist = np.select([in_o_dist < in_length], [in_o_dist])
            in_o_dist = in_o_dist.flatten()
            in_o_dist = in_o_dist[np.nonzero(in_o_dist)]
            in_coord = len(in_o_dist) / len(in_atoms)
            in_o_mean = np.mean(in_o_dist)
        o_coord = (len(al_o_dist) + len(ga_o_dist) + len(in_o_dist)) / len(o_atoms)
        xyz_feats_dict[idx] = {"Al_coord" : al_coord,
                               "Al_O_mean" : al_o_mean,
                               "Ga_coord" : ga_coord,
                               "Ga_O_mean" : ga_o_mean,
                               "In_coord" : in_coord,
                               "In_O_mean" : in_o_mean,
                               "O_coord" : o_coord}
        
    fout = 'xyz_feats.csv'
    with open(fout, 'w') as f:
        f.write('Al_coord, Al_O_mean, Ga_coord, Ga_O_mean, In_coord, In_O_mean, O_coord\n')
        for idx in xyz_feats_dict:
            d = xyz_feats_dict[idx]
            seq = [d['Al_coord'], d['Al_O_mean'], d['Ga_coord'], d['Ga_O_mean'], d['In_coord'], d['In_O_mean'], d['O_coord']]
            seq = [str(val) for val in seq]
            f.write(','.join(seq)+'\n')

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
    
    df_SG = spaceGroup(df_train['spacegroup'])
    df_train = pd.concat([df_train, df_SG], axis=1)
    df_train = df_train.drop(['id', 'spacegroup'], axis=1)
    
    dataset = 'train'
    #
   xyzFeats(df_train, dataset)
 
    return df_train

if __name__ == '__main__':
    df_train = main()
    df_train.to_csv('train_no_xyz.csv', index=False)