#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:05:47 2023

@author: laserglaciers
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import geopandas as gpd
import matplotlib.pyplot as plt
import pickle
import melt_functions as ice_melt
import xarray as xr
from multiprocessing import Pool, cpu_count
import scipy.io as sio


ctd_path = '/media/laserglaciers/upernavik/iceberg_py/infiles/kanger/CTD_20160914_083150.nc'
kanger_dist_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/kanger/20170710T141009_icebergs_kanger.pkl'


kanger_dist = gpd.GeoDataFrame(pd.read_pickle(kanger_dist_path))
kanger_dist.sort_values(by=['binned'], inplace=True)
df_mean = np.mean(kanger_dist["binned"].dropna().values)
df_std = np.std(kanger_dist["binned"].dropna().values)



bins = np.arange(0,1450,50) # lengths for this example and bins
labels = np.arange(50,1450,50) # lengths for this example and 
n = len(kanger_dist.binned.dropna())

factor=4
adcp_ds = 0.03
Tair = 6.5 # air temp in C
SWflx = 306 # W/m2 of shortwave flux
Winds = 2.3 # wind speed in m/s
IceC = 1 # sea ice conc 0 - 1 (0 - 100%)
L = np.arange(50,1450,50)
ni = len(L)
timespan = 86400.0 * 2 # 2 days
l_heat = 3.34e5
dz = 5
Aww_depth = 150
do_constantUrel = True
use_constant_tf = True
u_rel = 0.05

spacing = 100
tf_range = np.linspace(1,8,spacing)
# input data paths
ctd_path = '/media/laserglaciers/upernavik/iceberg_py/infiles/ctdSFjord.mat'

ctd = sio.loadmat(ctd_path)

depth = ctd['serm']['mar2010'][0][0][0][0][0]
temp = ctd['serm']['mar2010'][0][0][0][0][1]
salt = ctd['serm']['mar2010'][0][0][0][0][2]

ctd_ds = xr.Dataset({'depth':(['Z','X'], depth),
                     'temp': (['tZ','tX'], temp),
                     'salt': (['tZ','tX'], salt)
                     }
    )




mberg_dict = {}
uwV_total_dict = {}
Q_ib_grid = np.ones((spacing,spacing))


        
def calc_row(TF_val, arr_index, initial_array = Q_ib_grid, uwV_total_dict = uwV_total_dict, 
             timespan = timespan, ctd_ds = ctd_ds,
             IceC = 1, Winds = Winds, Tair = Tair,
             SWflx = SWflx, u_rel = u_rel, do_constantUrel=True, 
             factor=4, use_constant_tf=True):
    
    
    # for i,temp in enumerate(tf_range): # get range for each temp
    print(f'Temp: {TF_val}')
    constant_tf = TF_val
        
    min_dist, max_dist = -40, 15  #-60, -30 2.5e9 - 100e9 Qib
    for j, mean_dist in enumerate(np.linspace(min_dist,max_dist,spacing)): # use sythetic iceberg distributions
        exp_test = np.random.exponential(scale=df_mean+mean_dist, size=n)
        exp_bins = pd.cut(exp_test,bins=bins,labels=labels)
        
        Qib_dict = {}
        L = np.arange(50,1450,50)
        for length in L:
            # print(f'Processing Length {length}')
            mberg = ice_melt.iceberg_melt(length, dz, timespan, ctd_ds, IceC, Winds, Tair, SWflx, u_rel, do_constantUrel=do_constantUrel, 
                              factor=factor, use_constant_tf=use_constant_tf, 
                              constant_tf = constant_tf)

            mberg_dict[length] = mberg
            
            berg = mberg_dict[length]
            k = berg.KEEL.sel(time=86400*2)
            # if k >= Aww_depth:
            Mfreew = berg.Mfreew.sel(Z=slice(None,k.data[0]), time=86400*2)
            Mturbw = berg.Mturbw.sel(Z=slice(None,k.data[0]), time=86400*2)
            
            total_iceberg_melt = np.mean(Mfreew + Mturbw,axis=1)
            Qib = total_iceberg_melt * l_heat * 1000 # iceberg heatflux per z layer
            Qib_dict[length] = Qib

        vc = exp_bins.value_counts()
        Qib_length_totals = {}
        # Qib_sums = {}
        uwV_dict = {}
        
        for length in L:
            count = vc[length]
            Qib_sum = np.nansum(Qib_dict[length].sel(Z=slice(Aww_depth,None)))
            uwV_sum = np.nansum(mberg_dict[length].uwV.sel(Z=slice(Aww_depth,None)))
            
            Qib_length_totals[length] = Qib_sum * count
            
            uwV_dict[length] = uwV_sum * count
            
        qib_total = np.nansum(list(Qib_length_totals.values()))
        uwV_total = np.nansum(list(uwV_dict.values()))
        uwV_total_dict[j] = uwV_total
        initial_array[arr_index,j] = qib_total
    
    return initial_array, uwV_total_dict



# final_array = calc_row(Q_ib_grid, tf_range, uwV_total_dict, 
#                        timespan, ctd_ds, IceC, Winds, Tair, SWflx, u_rel)


def create_Qib_array(TF_range):
    with Pool(processes=cpu_count()-4) as pool:
        qib_arr, vol_array = pool.starmap(calc_row, [(TF,i) for i,TF in enumerate(TF_range)])
        
    return  qib_arr, vol_array


final_qib, vols = create_Qib_array(tf_range)

        
# volarr = np.array(list(uwV_total_dict.values()))
# volarr_km = volarr/1e9    
# temps = np.linspace(start_temp, end_temp, total_num)
# # plt.pcolormesh(volarr_km, temps, Q_ib_grid)

# fig1, ax1 = plt.subplots()
# contour = ax1.imshow(Q_ib_grid,aspect='auto',origin='lower',
#            interpolation="none",extent=[volarr_km.min(), volarr_km.max(), temps.min(), temps.max()],
#            vmin=Q_ib_grid.min(),vmax=Q_ib_grid.max())
# ax1.set_xlabel('Ice Volume Below Aww (km$^{3}$)')
# ax1.set_ylabel('Average Aww Temperature (c)')
# fig1.colorbar(contour,label='Q$_{ib}$')
# # plt.savefig('./contour_100.png',dpi=300)
# # kenger_mbergs_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/kanger/berg_model/20170710T141009_bergs.pkl'
# # with open(kenger_mbergs_path, 'rb') as src:
# #     mbergs = pickle.load(src)

# # Heat flux figure per layer per size of iceberg
# fig2, axs = plt.subplots(2,1)
# kanger_dist.binned.value_counts().sort_index().plot(kind='bar',logy=True, edgecolor = 'k', ax=axs[0])
# vc.sort_index().plot(kind='bar',logy=True, edgecolor = 'k', ax=axs[1])
# axs[0].set_title('kanger dist')
# axs[1].set_title('exp dist')
# plt.tight_layout()
# # vc = bin_test.value_counts()

# Qib_op = '/media/laserglaciers/upernavik/iceberg_py/outfiles/Qib_grids/'
# vol_arr_op = '/media/laserglaciers/upernavik/iceberg_py/outfiles/vol_arr/'

# # #save Qib_grid
# np.save(f'{Qib_op}Qib_temp_{start_temp}-{end_temp}_{min_dist}-{max_dist}.npy',Q_ib_grid)
# np.save(f'{vol_arr_op}vol_arr_{min_dist}-{max_dist}.npy',volarr)

