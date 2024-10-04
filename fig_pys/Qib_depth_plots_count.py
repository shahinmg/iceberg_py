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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
import types
import string

mbergs_dict_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/berg_model/2024-08-22_urel0.05_TF6.67_bergs_coeff4_v2.pkl'
with open(mbergs_dict_path, 'rb') as src:
    mberg_dict = pickle.load(src)



gdf_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/iceberg_geoms/dim_with_bin/2024-08-22-icebergs_helheim_keel_depth.gpkg'


icebergs_gdf = gpd.read_file(gdf_path)

vc = icebergs_gdf['binned'].value_counts()
# fig3, ax3 = plt.subplots()     
# icebergs_gdf['binned'].value_counts().sort_index().plot(kind='bar',logy=True,ax=ax3,
#                                                         edgecolor = 'k')



L = np.arange(50,1450,50)
l_heat = 3.34e5
Aww_depth = 150
Z = np.arange(5,500,5)
Z_sa = np.arange(5,605,5)
sec2day = 86400

Qib_arr = np.empty((len(Z),len(L)))
Qib_arr[:] = np.nan


SA_arr = np.empty((len(Z_sa),len(L)))
SA_arr[:] = np.nan

# Mfreew[k,i,j] =  2 * (mb[k,i,j] * dz * uwL[k][0]) + 2 *(mb[k,i,j] * dz * uwW[k])

# Heat flux figure per layer per size of iceberg
Qib_dict = {}
Qib_list = []

Qib_totals = {}
Qib_sums = {}

for i,length in enumerate(L):
    berg = mberg_dict[length]
    k = berg.KEEL.sel(time=86400*2)
    # if k >= Aww_depth:
    Mfreew = berg.Mfreew.sel(Z=slice(None,k.data[0]), time=86400*2)
    Mturbw = berg.Mturbw.sel(Z=slice(None,k.data[0]), time=86400*2)
    
    total_iceberg_melt = np.mean(Mfreew + Mturbw,axis=1)
    Qib = total_iceberg_melt * l_heat * 1000 # iceberg heatflux per z layer
    Qib.name = str(length)

    uw_length = berg.uwL
    uw_width = berg.uwW
    dz = berg.dz.data
    
    uw_sa = 2 * (dz * uw_length) +  2 *(dz * uw_width)  # underwater surface area
    
    if np.isin(length,vc.index):
        count = vc[length]
        print(length)
        Qib_depths = Qib.sel(Z=slice(None,None))

        
        Qib_totals = Qib_depths * count
        # print(f'{length}: {Qib_totalsz*count}')
        uw_sa_totals = uw_sa * count

        Qib_arr[:Qib_totals.data.shape[0], i] = Qib_totals.data
        SA_arr[:uw_sa_totals.data.shape[0], i] = uw_sa_totals.data.flatten()
    

Qib_xr_data_arr = xr.DataArray(data=Qib_arr, name='Qib', coords = {"Z":Z, "length":L},  
             dims=["Z","length"], attrs={'Description':"Iceberg Heat Flux", 'Units': 'W'})
                                                                                                            

uwSA_xr_data_arr = xr.DataArray(data=SA_arr, name='underwater_surface_area', coords = {"Z":Z_sa, "length":L},  
             dims=["Z","length"], attrs={'Description':"Estimated Underwater Surface Area", 'Units': 'm^2'})
                                                                                                




Qib_depth_mean = Qib_xr_data_arr.mean(dim='length')

Qib_depth_sum = Qib_xr_data_arr.sum(dim='length')
Qib_depth_cum_sum = Qib_depth_sum.sel(Z=slice(Aww_depth, None)).cumsum()


uwSA_depth_sum = uwSA_xr_data_arr.sum(dim='length')
uwSA_depth_sum_aw = uwSA_depth_sum.sel(Z=slice(None, 500))

# plt.plot(depth_mean.data, depth_mean.Z)


fig, ax = plt.subplots(1,4,sharey='row',figsize=(8, 6))
labelsize = 10


colors_blues = cm.Blues(np.linspace(0,1,len(L)))

cmap = plt.get_cmap('Blues').copy()
ax[0].plot(Qib_depth_mean.data, Qib_depth_mean.Z, c='black', lw=5)
ax[0].plot(Qib_depth_mean.data, Qib_depth_mean.Z, color='tab:blue',lw=3)

ax[0].axhspan(150,600,facecolor='0.6',alpha=0.3, zorder=1)

ax[0].set_ylim(600,0)

ax[0].set_ylabel('Depth' , size=12)
ax[0].set_xlabel('Q$_{ib}$ (W) Mean' ,size=labelsize)
ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)
#%%

ax[1].plot(Qib_depth_sum.data, Qib_depth_sum.Z, c='black', lw=5)
ax[1].plot(Qib_depth_sum.data, Qib_depth_sum.Z, color='tab:blue',lw=3)

ax[1].axhspan(150,600,facecolor='0.6',alpha=0.3, zorder=1)

ax[1].set_ylim(600,0)
ax[1].set_xlim(3e8,1.5e10)
ax[1].set_xscale('log')
ax[1].set_xlabel('Q$_{ib}$ (W) Sum' ,size=labelsize)
# ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)

#%%

ax[2].plot(Qib_depth_cum_sum.data, Qib_depth_cum_sum.Z, c='black', lw=5)
ax[2].plot(Qib_depth_cum_sum.data, Qib_depth_cum_sum.Z, color='tab:blue',lw=3)

ax[2].axhspan(150,600,facecolor='0.6',alpha=0.3, zorder=1)

ax[2].set_ylim(600,0)


ax[2].set_xlabel('Q$_{ib}$ (W) Cumulative Sum' ,size=labelsize)
ax[2].ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)



#%%
ax[3].plot(uwSA_depth_sum_aw.data, uwSA_depth_sum_aw.Z, c='black', lw=5)
ax[3].plot(uwSA_depth_sum_aw.data, uwSA_depth_sum_aw.Z, color='tab:blue',lw=3)

ax[3].axhspan(150,600,facecolor='0.6',alpha=0.3, zorder=1)
# ax[3].set_xlim(0,2e5)
ax[3].set_xscale('log')
ax[3].set_xlabel('Surface Area (m$^{2}$)' , size=labelsize)


# ax[3].ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)


pad = plt.rcParams["xtick.major.size"] + plt.rcParams["xtick.major.pad"]
def bottom_offset(self, bboxes, bboxes2):
    bottom = self.axes.bbox.ymin
    self.offsetText.set(va="top", ha="left") 
    oy = bottom - pad * self.figure.dpi / 1000
    self.offsetText.set_position((1, oy))


ax[0].xaxis._update_offset_text_position = types.MethodType(bottom_offset, ax[0].xaxis)
ax[1].xaxis._update_offset_text_position = types.MethodType(bottom_offset, ax[1].xaxis)
ax[2].xaxis._update_offset_text_position = types.MethodType(bottom_offset, ax[2].xaxis)
# ax[3].xaxis._update_offset_text_position = types.MethodType(bottom_offset, ax[2].xaxis)

text_dict = {'fontsize':18,
             'fontweight': 'bold'}
ax = ax.flatten()
alphabet = list(string.ascii_lowercase)
for i,axis in enumerate(ax):
    
    text_label = axis.text(.01, .99, alphabet[i], ha='left', va='top', transform=axis.transAxes, **text_dict)


# plt.subplots_adjust(right=0.9)
plt.tight_layout()


# 
op = '/media/laserglaciers/upernavik/iceberg_py/figs/'
fig.savefig(f'{op}2024-08-22_Qib_depth.png',dpi=300, bbox_inches='tight')





