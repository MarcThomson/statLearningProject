#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 20:57:00 2018

@author: tunkie

Script to generate features
"""

import pandas as pd
import numpy as np
import os
from numpy.linalg import inv
import os.path


def getPropList():
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

def getStdProps(df):
    """
    Args:
        stuf
    Returns:
        df (DataFrame) - data with property features appended
    """
    properties = getPropList()
    # make nested dictionary which maps {property (str) : {element (str) : property value (float)}}
    prop_dict = {prop : getProp(prop) for prop in properties}
    for prop in properties:
        df['_'.join(['avg', prop])] = avgProp(df['percent_atom_al'], 
                                              df['percent_atom_ga'],
                                              df['percent_atom_in'],
                                              prop,
                                              prop_dict)
    return df

def deg2Rad(df):
    """
    Args:
        df (DataFrame) - dataset
    Returns:
        df_out (DataFrame) - dataset with degrees converted to radians
    """
    lattice_angles = ['lattice_angle_alpha_degree',
                      'lattice_angle_beta_degree',
                      'lattice_angle_gamma_degree']
    for lang in lattice_angles:
        df['_'.join([lang, 'r'])] = np.pi * df[lang] / 180
    df_out = df.drop(['lattice_angle_alpha_degree',
                  'lattice_angle_beta_degree',
                  'lattice_angle_gamma_degree'], axis=1)
    return df_out

def getVol(df):
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
    a = df['lattice_vector_1_ang']
    b = df['lattice_vector_2_ang']
    c = df['lattice_vector_3_ang']
    alpha = df['lattice_angle_alpha_degree_r']
    beta = df['lattice_angle_beta_degree_r']
    gamma = df['lattice_angle_gamma_degree_r']
    vol =  a*b*c*np.sqrt(1 + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma)
                           - np.cos(alpha)**2
                           - np.cos(beta)**2
                           - np.cos(gamma)**2)
    df['vol'] = vol
    return df

def getAtomDens(df):
    """
    Args:
        df (DataFrame) - dataset
    Returns:
        df (DataFrame) - dataset with atomic density feature appended
    """
    df['atomic_density'] = df['number_of_total_atoms'] / df['vol']
    return df

def spaceGroup(df):
    """
    Args:
        df (DataFrame) - training or testing data
    Returns:
        df (DataFrame) - training or testing data with spacegroup features appended
    """
    uniques = list(set(df['spacegroup']))
    df_out = pd.DataFrame()
    for i in range(len(uniques)):
        newData = pd.DataFrame({'SG_' + str(uniques[i]) : df['spacegroup'] == uniques[i]})
        df_out = pd.concat([df_out, newData],axis = 1)
    df_out = 1 * df_out
    df_out = pd.concat([df, df_out], axis=1)
    df_out = df_out.drop(['id', 'spacegroup'], axis=1)
    return df_out

def length(v):
    """
    Args:
        v (vector)
    Returns:
        d (float) - norm of vector
    Source:
        https://www.kaggle.com/tonyyy/how-to-get-atomic-coordinates
    """
    d = np.linalg.norm(v)
    return d

def getXyzData(fin):
    """
    Args:
        fin (str) - file contraining lattice data
    Returns:
        pos_data () - atomic coordinates
        lat_data (matrix) - lattice vectors
    Source:
        https://www.kaggle.com/tonyyy/how-to-get-atomic-coordinates
    """
    pos_data = []
    lat_data = []
    with open(fin) as f:
        for line in f.readlines():
            x = line.split()
            if x[0] == 'atom':
                pos_data.append([np.array(x[1:4], dtype=np.float),x[4]])
            elif x[0] == 'lattice_vector':
                lat_data.append(np.array(x[1:4], dtype=np.float))
    return pos_data, np.array(lat_data)

def getShortestDistances(reduced_coords, amat):
    """
    Args:
        reduced_coords () -  ?
        amat (matrix) - ?
    Returns:
        dists () - ?
        Rij_min () - ?
    Source:
        https://www.kaggle.com/tonyyy/how-to-get-atomic-coordinates
    """
    natom = len(reduced_coords)
    dists = np.zeros((natom, natom))
    multMat = np.zeros((natom, natom))
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
                        if abs(d_min-d)/d < 0.05:
                            d_min = (d_min*mult + d)/(mult+1)
                            mult = mult + 1
                        elif d < d_min:
                            d_min = d
                            R_min = R
                            mult = 1
            dists[i, j] = d_min
            dists[j, i] = dists[i, j]
            multMat[i,j] = mult
            multMat[j,i] = mult
            Rij_min[i, j] = R_min
            Rij_min[j, i] = -Rij_min[i, j]
    return dists, Rij_min, multMat

def getMinLength(distances, A_atoms, B_atoms):
    """
    Args:
        distances () - ?
        A_atoms () - ?
        B_atoms () - ?
    Returns:
        A_B_length () - ?
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
    """
    Args:
        df (DataFrame) - dataset
        dataset (str) - file type: either train or test
    """
    xyz_feats_dict = {}
    for idx in range(1,len(df.index)+1):
        fn = "{}/{}/geometry.xyz".format(dataset, idx)
        crystal_xyz, crystal_lat = getXyzData(fn)
        A = np.transpose(crystal_lat)
        R = crystal_xyz[0][0]
        B = inv(A)
        crystal_red = [[np.matmul(B, R), symbol] for (R, symbol) in crystal_xyz]
        crystal_dist, crystal_Rij,crystal_mult = getShortestDistances(crystal_red, A)
        natom = len(crystal_red)
        al_atoms = [i for i in range(natom) if crystal_red[i][1] == 'Al']
        ga_atoms = [i for i in range(natom) if crystal_red[i][1] == 'Ga']
        in_atoms = [i for i in range(natom) if crystal_red[i][1] == 'In']
        o_atoms = [i for i in range(natom) if crystal_red[i][1] == 'O']
        al_length, ga_length, in_length = 0, 0, 0
        al_o_dist = np.zeros([9*len(al_atoms)*len(o_atoms),1])
        ga_o_dist = np.zeros([9*len(ga_atoms)*len(o_atoms),1])
        in_o_dist = np.zeros([9*len(in_atoms)*len(o_atoms),1])
        al_coord = 0
        al_o_mean = 0
        al_o_4 = 0
        al_o_2 = 0
        ga_coord = 0
        ga_o_mean = 0
        ga_o_4 = 0
        ga_o_2 = 0
        in_coord = 0
        in_o_mean = 0
        in_o_4 = 0
        in_o_2 = 0
        n_atoms = len(al_atoms) + len(ga_atoms) + len(in_atoms)
        if len(al_atoms):
            #al_length = 1.30 * getMinLength(crystal_dist, al_atoms, o_atoms)
            p = 0
            for i in range(len(al_atoms)):
                for j in range(len(o_atoms)):
                    for k in range(int(crystal_mult[al_atoms[i],o_atoms[j]])):
                        al_o_dist[p] = crystal_dist[al_atoms[i], o_atoms[j]]
                        p = p + 1
            al_o_dist = al_o_dist[np.nonzero(al_o_dist)]
            al_length  = 1.3*min(al_o_dist)
            al_o_dist = np.select([al_o_dist < al_length], [al_o_dist])
            al_o_dist = al_o_dist[np.nonzero(al_o_dist)]
            al_coord = len(al_o_dist) / len(al_atoms)
            al_o_mean = np.mean(al_o_dist)
            al_o_4 = np.sum(1 / al_o_dist ** 4) / n_atoms
            al_o_2 = np.sum(1 / al_o_dist ** 2) / n_atoms
        if len(ga_atoms):
            p = 0
            #ga_length = 1.30 * getMinLength(crystal_dist, ga_atoms, o_atoms)
            for i in range(len(ga_atoms)):
                for j in range(len(o_atoms)):
                    for k in range(int(crystal_mult[ga_atoms[i],o_atoms[j]])):
                        ga_o_dist[p] = crystal_dist[ga_atoms[i], o_atoms[j]]
                        p = p + 1
            ga_o_dist = ga_o_dist[np.nonzero(ga_o_dist)]
            ga_length  = 1.3*min(ga_o_dist)
            ga_o_dist = np.select([ga_o_dist < ga_length], [ga_o_dist])
            ga_o_dist = ga_o_dist[np.nonzero(ga_o_dist)]
            ga_coord = len(ga_o_dist) / len(ga_atoms)
            ga_o_mean = np.mean(ga_o_dist)
            ga_o_4 = np.sum(1 / ga_o_dist ** 4) / n_atoms
            ga_o_2 = np.sum(1 / ga_o_dist ** 2) / n_atoms
        if len(in_atoms):
            p = 0
            #in_length = 1.30 * getMinLength(crystal_dist, in_atoms, o_atoms)
            for i in range(len(in_atoms)):
                for j in range(len(o_atoms)):
                    for k in range(int(crystal_mult[in_atoms[i],o_atoms[j]])):
                        in_o_dist[p] = crystal_dist[in_atoms[i], o_atoms[j]]
                        p = p + 1
            in_o_dist = in_o_dist[np.nonzero(in_o_dist)]
            in_length  = 1.3*min(in_o_dist)
            in_o_dist = np.select([in_o_dist < in_length], [in_o_dist])
            in_o_dist = in_o_dist[np.nonzero(in_o_dist)]
            in_coord = len(in_o_dist) / len(in_atoms)
            in_o_mean = np.mean(in_o_dist)
            in_o_4 = np.sum(1 / in_o_dist ** 4) / n_atoms
            in_o_2 = np.sum(1 / in_o_dist ** 2) / n_atoms
        o_coord = (len(al_o_dist) + len(ga_o_dist) + len(in_o_dist)) / len(o_atoms)
        xyz_feats_dict[idx] = {"Al_coord" : al_coord,
                               "Al_O_mean" : al_o_mean,
                               "Al_O_4" : al_o_4,
                               "Al_O_2" : al_o_2,
                               "Ga_coord" : ga_coord,
                               "Ga_O_mean" : ga_o_mean,
                               "Ga_O_4" : ga_o_4,
                               "Ga_O_2" : ga_o_2,
                               "In_coord" : in_coord,
                               "In_O_mean" : in_o_mean,
                               "In_O_4" : in_o_4,
                               "In_O_2" : in_o_2,
                               "O_coord" : o_coord}
        
    fout = '{}_xyz_feats.csv'.format(dataset)
    with open(fout, 'w') as f:
        f.write('Al_coord, Al_O_mean, Al_O_4, Al_O_2, Ga_coord, Ga_O_mean, Ga_O_4, Ga_O_2, In_coord, In_O_mean, In_O_4, In_O_2, O_coord\n')
        for idx in xyz_feats_dict:
            d = xyz_feats_dict[idx]
            seq = [d['Al_coord'], d['Al_O_mean'], d['Al_O_4'], d['Al_O_2'], d['Ga_coord'], d['Ga_O_mean'], d['Ga_O_4'], d['Ga_O_2'], d['In_coord'], d['In_O_mean'], d['In_O_4'], d['In_O_2'], d['O_coord']]
            seq = [str(val) for val in seq]
            f.write(','.join(seq)+'\n')

def combineData(df_1, df_2):
    df_1 = pd.concat([df_1, df_2],axis = 1)
    return df_1

def main():
    fins = ['train.csv','test.csv']
    for fin in fins:
        df = pd.read_csv(fin)
        df = getStdProps(df)
        df = deg2Rad(df) 
        df = getVol(df)
        df = getAtomDens(df)
        df = spaceGroup(df)
        if 'train' in fin:
            dataset = 'train'
            fout = 'train_w_feats.csv'
        elif 'test' in fin:
            dataset = 'test'
            fout = 'test_w_feats.csv'
        else:
            print('Error!')
        
        f_xyz = dataset + '_xyz_feats.csv'
        if os.path.isfile(f_xyz):
            print('file already exists')
            df_xyz = pd.read_csv(f_xyz)
            df = combineData(df, df_xyz)
        else:
            xyzFeats(df, dataset)
            df_xyz = pd.read_csv(f_xyz)
            df = combineData(df, df_xyz)
            
        df.to_csv(fout, index=False)
 
    return 0

if __name__ == '__main__':
    main()