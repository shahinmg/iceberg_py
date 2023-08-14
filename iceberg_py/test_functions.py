#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 11:54:53 2023

@author: laserglaciers
"""
import melt_functions as ice_melt
import numpy as np
import xarray as xr
import scipy.io as sio

L = np.arange(50,1050,50)
dz = 5

#Test first L

# # ice = ice_melt.init_iceberg_size(L[0],dz=dz)
# ice_init = []
# for length in L:
#     print(f'processing {length}')
#     ice_init.append(ice_melt.init_iceberg_size(length,dz=dz))
#     print(f'finished with {length}')

ctd_path = '/media/laserglaciers/upernavik/iceberg_py/infiles/ctdSFjord.mat'
adcp_path = '/media/laserglaciers/upernavik/iceberg_py/infiles/ADCP_cosine_BeccaSummer.mat'

ctd = sio.loadmat(ctd_path)

depth = ctd['serm']['mar2010'][0][0][0][0][0]
temp = ctd['serm']['mar2010'][0][0][0][0][1]
salt = ctd['serm']['mar2010'][0][0][0][0][2]

ctd_ds = xr.Dataset({'depth':(['Z','X'], depth),
                     'temp': (['tZ','tX'], temp),
                     'salt': (['tZ','tX'], salt)
                     }
    )


adcp = sio.loadmat(adcp_path)

adcp_ds = xr.Dataset({'zadcp': (['adcpX','adcpY'],adcp['zadcp']),
                      'vadcp': (['adcpX','adcpZ'], adcp['vadcp']),
                      'tadcp': (['adcpY','adcpZ'], adcp['tadcp'])
    })