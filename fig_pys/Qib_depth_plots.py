#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:34:01 2024

@author: laserglaciers
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import cm,colors
import geopandas as gpd

mbergs_dict_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/berg_model/2024-08-22_urel0.05_TF6.67_bergs_coeff4_v2.pkl'
with open(mbergs_dict_path, 'rb') as src:
    mberg_dict = pickle.load(src)



gdf_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/iceberg_geoms/dim_with_bin/2024-08-22-icebergs_helheim_keel_depth.gpkg'


icebergs_gdf = gpd.read_file(gdf_path)

vc = icebergs_gdf['binned'].value_counts()
fig3, ax3 = plt.subplots()     
icebergs_gdf['binned'].value_counts().sort_index().plot(kind='bar',logy=True,ax=ax3,
                                                        edgecolor = 'k')



L = np.arange(50,1450,50)
l_heat = 3.34e5
Aww_depth = 150
Z = np.arange(5,500,5)

Qib_arr = np.empty((len(Z),len(L)))
Qib_arr[:] = np.nan
# Heat flux figure per layer per size of iceberg
Qib_dict = {}
Qib_list = []
for i,length in enumerate(L):
    berg = mberg_dict[length]
    k = berg.KEEL.sel(time=86400*2)
    # if k >= Aww_depth:
    Mfreew = berg.Mfreew.sel(Z=slice(None,k.data[0]), time=86400*2)
    Mturbw = berg.Mturbw.sel(Z=slice(None,k.data[0]), time=86400*2)
    
    total_iceberg_melt = np.mean(Mfreew + Mturbw,axis=1)
    Qib = total_iceberg_melt * l_heat * 1000 # iceberg heatflux per z layer
    Qib.name = str(length)

    Qib_arr[:Qib.data.shape[0], i] = Qib.data
    
    
    
    Qib_dict[length] = Qib
    
    Qib_list.append(Qib)
    

Qib_merge = xr.merge(Qib_list)




Qib_xr_data_arr = xr.DataArray(data=Qib_arr, name='Qib', coords = {"Z":Z, "length":L},  
             dims=["Z","length"], attrs={'Description':"Iceberg Heat Flux", 'Units': 'W'})
                                                                                                            






# # ax3.hist(icebergs_gdf['max_dim'].values, bins=np.arange(0,1050,50),
# #          edgecolor = "black")
# ax3.set_ylabel('Count')
# ax3.set_xlabel('Iceberg surface length (m)')


Qib_totals = {}
Qib_sums = {}
for length in L:
    
    if np.isin(length,vc.index):
        count = vc[length]
        Qib_sum = np.nansum(Qib_dict[length].sel(Z=slice(Aww_depth,None)))
        Qib_depths = Qib_dict[length].sel(Z=slice(Aww_depth,None))

        
        Qib_totals[length] = Qib_depths * count
        Qib_sums[length] = Qib_sum
        print(f'{length}: {Qib_sum*count}')







# depth_mean = Qib_xr_data_arr.mean(dim='length')

# plt.plot(depth_mean.data, depth_mean.Z)
# plt.ylim(600,0)

# colors_blues = cm.Blues(np.linspace(0,1,len(Qib_dict)))

# cmap = plt.get_cmap('Blues').copy()
# divider = make_axes_locatable(ax2)
# cax = divider.append_axes("right", size="5%", pad=0.1)

# for i,length in enumerate(Qib_dict):

#     x = Qib_dict[length].data #/ 1e9 #gigawatts
#     y = Qib_dict[length].Z.data
#     ax2.plot(x, y, c='black', lw=5)
#     ax2.plot(x,y,color=colors_blues[i],lw=3)
#     ax2.set_ylim(600,y.min())
# ax2.axhline(y=Aww_depth,linewidth=3, color='#d62728')

# l_min = L.min()
# l_max = L.max()
# cbar = fig2.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin = l_min,vmax = l_max), cmap=cmap),
#               cax=cax,label='Iceberg Length (m)')
# cbar.ax.tick_params(labelsize=20)
# cbar.set_label(label='Iceberg Length (m)',fontsize=20)
# ax2.tick_params(axis='both', which='major', labelsize=20)
# ax2.set_ylabel('Depth', size=20)
# ax2.set_xlabel('Q$_{ib}$ (GW)',size=20)

