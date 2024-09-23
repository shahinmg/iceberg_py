#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 10:16:23 2023

@author: laserglaciers
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import gaussian_filter
from matplotlib import colors
import matplotlib as mpl
import scipy.io as sio
import xarray as xr
import pickle

# Q_ib_grid = np.load('/media/laserglaciers/upernavik/iceberg_py/iceberg_py/Q_ib_grid.npy')
Q_ib_grid = np.load('/media/laserglaciers/upernavik/iceberg_py/outfiles/Qib_grids/Qib_temp_0.5-5.1_-40-15.npy')
vol_arr = np.load('/media/laserglaciers/upernavik/iceberg_py/outfiles/vol_arr/vol_arr_-40-15.npy')

Q_ib_grid = Q_ib_grid/1e9
Q_ib_grid_smooth = gaussian_filter(Q_ib_grid, 2)
vol_arr = vol_arr/1e9

cmap = plt.cm.viridis.copy()
bounds = np.arange(0,30,20)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

fig, axs = plt.subplots(2,1,figsize=(8,8))

contour = axs[0].imshow(Q_ib_grid, aspect='auto', origin='lower',
           interpolation="none", extent = [vol_arr.min(), vol_arr.max(), 0.5, 5.1],
           vmin=0, vmax=Q_ib_grid.max()-80
           )

axs[0].set_title('Not Smoothed')
axs[0].set_xlabel('Ice Volume Below Aww (km$^{3}$)')
axs[0].set_ylabel('Average Aww Temperature (c)')
# fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
#              ax=axs[0],label='Q$_{ib}$  (GW)')
fig.colorbar(contour,cmap='viridis',ax=axs[0])

contour_smooth = axs[1].imshow(Q_ib_grid_smooth, aspect='auto', origin='lower',
           interpolation="none", extent = [vol_arr.min(), vol_arr.max(), 0.5, 5.1],
           vmin=0, vmax=Q_ib_grid.max()-100
           )

axs[1].set_title('Smoothed')
axs[1].set_xlabel('Ice Volume Below Aww (km$^{3}$)')
axs[1].set_ylabel('Average Aww Temperature (c)')
# fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
#              ax=axs[1],label='Q$_{ib}$ (GW)')
fig.colorbar(contour_smooth,cmap='viridis',ax=axs[1])


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

hel_bergs_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/20230727T142031_icebergs_helheim.pkl'
with open(hel_bergs_path, 'rb') as src:
    hel_bergs_gdf = pickle.load(src)

Hel_qib_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/Qib/20230727T142031_high_helheim_coeff_4_test.nc'
Hel_qib = xr.open_dataset(Hel_qib_path)

Kanger_qib_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/kanger/Qib/20170710T141009_high_kanger_coeff_4.nc'
Kanger_qib = xr.open_dataset(Kanger_qib_path)

upernavik_ctd_path = '/media/laserglaciers/upernavik/iceberg_py/infiles/upernavik/20160917_bulk_ctd.nc'
upernavik_bergs_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/upernavik/20160606_icebergs_upernavik.pkl'
with open(upernavik_bergs_path, 'rb') as src:
    upernavik_bergs_gdf = pickle.load(src)

hel_mbergs_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/berg_model/20230727T142031_bergs_v2.pkl'
with open(hel_mbergs_path, 'rb') as src:
    hel_mbergs = pickle.load(src)


jkb_bergs_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/jkb/20200802T173803_icebergs_jkb.pkl'
with open(jkb_bergs_path, 'rb') as src:
    jkb_bergs_gdf = pickle.load(src)
jkb_ctd_path = '/media/laserglaciers/upernavik/iceberg_py/infiles/jkb/CTD_20200826_124816.nc'
jkb_ctd_ctd = xr.open_dataset(jkb_ctd_path)


vc = upernavik_bergs_gdf['binned'].value_counts()
vol_dict = {}
for l in hel_mbergs.keys():
    berg = hel_mbergs[l]
    vol = berg.uwV.sel(Z=slice(150,None))
    vol_sum = np.nansum(vol)
    vol_dict[l] = vol_sum * vc[l]


vc_jkb = jkb_bergs_gdf['binned'].value_counts()
vol_dict_jkb = {}
for l in hel_mbergs.keys():
    berg = hel_mbergs[l]
    vol_jkb = berg.uwV.sel(Z=slice(150,None))
    vol_jkb_sum = np.nansum(vol_jkb)
    vol_dict_jkb[l] = vol_jkb_sum * vc_jkb[l]

jkb_vol = np.sum(list(vol_dict_jkb.values()))/1e9
jkb_ctd_temp = np.nanmean(jkb_ctd_ctd.temp.sel(Z=slice(150,None)))

upernavik_vol = np.sum(list(vol_dict.values()))/1e9
upernavik_ctd = xr.open_dataset(upernavik_ctd_path)
upernavik_temp = np.nanmean(upernavik_ctd.temp.sel(Z=slice(150,None)))

hel_temp = np.mean(ctd_ds.temp.sel(tZ=slice(150,None))).data
hel_vol = Hel_qib.ice_vol.data/1e9
kanger_vol = Kanger_qib.ice_vol.data/1e9
kanger_temp = 1 #guess


# plot the data
plt_dict = {
'edgecolor':'k',
'lw':1,
's':80
}
labels = ['Helheim', 'Kangersertuaq', 'Upernavik', 'Sermeq Kujalleq']
data_list =  [(hel_vol,hel_temp, labels[0]),
            (kanger_vol, kanger_temp, labels[1]),
            (upernavik_vol, upernavik_temp, labels[2]),
            (jkb_vol, jkb_ctd_temp, labels[3])]

for vol, temp, label in data_list:
    
    axs[0].scatter(x=vol,y=temp,label=label, **plt_dict)
    axs[1].scatter(x=vol,y=temp,label=label, **plt_dict)


plt.tight_layout()