#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:37:39 2023

@author: laserglaciers
"""

import scipy.io as sio
import numpy as np
from scipy.interpolate import interp1d, interp2d

ctd_path = '/media/laserglaciers/upernavik/iceberg_py/infiles/ctdSFjord.mat'
adcp_path = '/media/laserglaciers/upernavik/iceberg_py/infiles/ADCP_cosine_BeccaSummer.mat'
salt_path = '/media/laserglaciers/upernavik/iceberg_py/infiles/salt.mat'

ctd = sio.loadmat(ctd_path)
depth = ctd['serm']['mar2010'][0][0][0][0][0]
temp = ctd['serm']['mar2010'][0][0][0][0][1]
salt = ctd['serm']['mar2010'][0][0][0][0][2]


# salt_rs = salt.reshape(len(salt),1)
salt = np.nanmean(salt,axis=1).reshape(len(salt),1)
salt_mat = sio.loadmat(salt_path)
salt_mat = salt_mat['salt']

#check if they are the same
salt_check = salt == salt_mat
# print(f'salt == salt_mat: {salt == salt_mat}')
ctdz = depth
ctdz_flat = depth.flatten()

S_far_func = interp1d(ctdz_flat,salt.flatten()) # interp1(ctdz,salt,Z(k));
S_far = S_far_func(5) # giving slightly different result than matlab
