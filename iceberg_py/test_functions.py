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
from plot_icebergshape import plot_icebergshape

L = np.arange(50,1050,50)
# L = [400]
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
                      'tadcp': (['adcpY','adcpZ'], adcp['tadcp']),
                      'wvel':  (['adcpY'], np.array([0.05]))
    })

Tair = 6.5 # air temp in C
SWflx = 306 # W/m2 of shortwave flux
Winds = 2.3 # wind speed in m/s
IceC = 0.36 # sea ice conc 0 - 1 (0 - 100%)
ni = len(L)
timespan = 86400.0 * 30.0 # 1 month

mberg_dict = {}
for length in L:
    print(f'Processing Length {length}')
    mberg = ice_melt.iceberg_melt(length, dz, timespan, ctd_ds, IceC, Winds, Tair, SWflx, adcp_ds)
    mberg_dict[length] = mberg


berg = mberg_dict[350]
plot_icebergshape(berg)


l_heat = 3.34e5
k=mberg_dict[1000].KEEL.sel(time=86400*2)
ul = mberg_dict[1000].UWL.sel(Z=slice(150,k.data[0]),time=86400*2)
uw = mberg_dict[1000].UWL.sel(Z=slice(150,k.data[0]),time=86400*2)
A = uw * ul

mfw = mberg_dict[1000].i_mfreew.sel(Z=slice(150,k.data[0]), time=86400*2)
mtw = mberg_dict[1000].i_mturbw.sel(Z=slice(150,k.data[0]), time=86400*2)

# not_hf = np.mean(mtw + mfw,axis=1) * l_heat  * A * 1000
