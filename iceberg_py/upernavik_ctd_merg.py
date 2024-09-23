#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:37:54 2023

@author: laserglaciers
"""

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

ctd_path = '/media/laserglaciers/upernavik/iceberg_py/infiles/upernavik/'
ctd_list = [ctd for ctd in os.listdir(ctd_path) if ctd.endswith('nc')]
os.chdir(ctd_path)

# ds = xr.open_dataset(ctd_list[1])

fig, ax = plt.subplots()
ds_list = []
for ctd in ctd_list:
    
    ds = xr.open_dataset(ctd)
    ds_list.append(ds.drop_vars(['lat','lon']))
    
    ax.plot(ds.temp.data,ds.depth.data)
    
ax.set_ylim(850,0)
merge_ds = xr.merge(ds_list,compat="override")
fill_arr = np.empty((merge_ds.Z.shape[0],len(ctd_list))) * np.nan

temp = fill_arr.copy()
salt = fill_arr.copy()
depth = fill_arr.copy()
for i,ds in enumerate(ds_list):
    
    temp[:ds.temp.shape[0],i] = ds.temp.data.flatten()
    salt[:ds.salt.shape[0],i] = ds.salt.data.flatten()
    depth[:ds.depth.shape[0],i] = ds.depth.data.flatten()
    
    
final_ds = xr.Dataset(
    {'depth':(["Z","X"],depth),
      'salt':(["Z","X"],salt),
      'temp':(["Z","X"],temp),
        },
    coords = {
        "Z":("Z",merge_ds.Z.data),
        "X":("X",np.arange(10))
        }
    )
        
# op = '/media/laserglaciers/upernavik/iceberg_py/infiles/upernavik/'
# final_ds.to_netcdf(f'{op}20160917_bulk_ctd.nc')
