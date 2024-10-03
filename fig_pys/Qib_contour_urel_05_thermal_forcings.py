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
from matplotlib import colors, cm, ticker
import matplotlib as mpl
import scipy.io as sio
import xarray as xr
import pickle
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from matplotlib.ticker import ScalarFormatter
import xarray as xr
import os
from matplotlib.patches import Ellipse

# Q_ib_grid = np.load('/media/laserglaciers/upernavik/iceberg_py/iceberg_py/Q_ib_grid.npy')
Q_ib_grid = np.load('/media/laserglaciers/upernavik/iceberg_py/outfiles/Qib_grids/Qib_TF_1-8_-40-15_0.05.npy')
vol_arr = pd.read_pickle('/media/laserglaciers/upernavik/iceberg_py/outfiles/vol_arr/vol_series_-40_15.pkl')

#%%
Q_ib_grid_smooth = gaussian_filter(Q_ib_grid, 5)
vol_arr = vol_arr

cmap = plt.cm.viridis.copy()
vmin = 0
vmax = Q_ib_grid.max()

start_temp = 1
end_temp = 8
spacing = 100
tf_range = np.linspace(start_temp, end_temp, spacing)
vol_range = np.linspace(vol_arr.min(), vol_arr.max(), spacing)

#%%
levels = np.logspace(np.log10(Q_ib_grid_smooth.min()),np.log10(Q_ib_grid_smooth.max()), 20) #https://stackoverflow.com/questions/65823932/plt-contourf-with-given-number-of-levels-in-logscale



fig, axs = plt.subplots(figsize=(7,7))

contour_smooth = axs.contourf(vol_range, tf_range, Q_ib_grid_smooth, 
                              levels = levels, norm=colors.LogNorm())

axs.set_xlabel('Volume Below PW-AW Boundary (m$^{3}$)', size=15)

axs.set_ylabel('Ocean Thermal Forcing (C)', size=15)
axs.set_title('Iceberg Heatflux Q$_{ib}$', size=15)

axs.ticklabel_format(style='sci', axis='x',useMathText=True)


cbar = fig.colorbar(contour_smooth, ticks=levels[::2], 
                    format=ticker.FixedFormatter(levels[::2])
                    )


cbar.formatter = ScalarFormatter(useMathText=True)
cbar.formatter.set_scientific(True)

cbar.ax.yaxis.set_offset_position('left')
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Q$_{ib}$ (W)', size=14)
# cbar.ax.set_title('Your Label',fontsize=8)

#%%
# Get Volume from model run.

helheim_urel_05_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/Qib/u_rel_05_factor4_melt_rates/'

nc_list = [nc for nc in os.listdir(helheim_urel_05_path) if nc.endswith('nc')]
os.chdir(helheim_urel_05_path)
helheim_volume_list = []
for nc in nc_list:
    
    Qib_ds = xr.open_dataset(nc)
    aww_vol = Qib_ds.ice_vol.data
    helheim_volume_list.append(aww_vol)
    # axs.scatter(aww_vol, 6.67)

hel_vol_mean = np.mean(helheim_volume_list)
vol_range = np.ptp(helheim_volume_list)

temp_mean = 6.67
temp_std = 0.47
temp_min, temp_max = temp_mean - (2*temp_std), temp_mean + (2*temp_std)
temp_range = temp_max - temp_min

ellipse = Ellipse((hel_vol_mean, 6.67), vol_range, temp_range, fill=True, alpha=0.5)

ellipse2 = Ellipse((hel_vol_mean, 6.67), vol_range, temp_range, fill=False, 
                  edgecolor='tab:blue', linewidth=2, alpha=1)


axs.add_patch(ellipse)
axs.add_patch(ellipse2)

axs.text(hel_vol_mean-0.6e9, 6.67, 'Helheim Fjord')


#%%

jkb_urel_05_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/jkb/Qib/u_rel_05_factor4_melt_rates/'

nc_list = [nc for nc in os.listdir(jkb_urel_05_path) if nc.endswith('nc')]
os.chdir(jkb_urel_05_path)
jkb_volume_list = []
for nc in nc_list:
    
    Qib_ds = xr.open_dataset(nc)
    aww_vol = Qib_ds.ice_vol.data
    jkb_volume_list.append(aww_vol)
    # axs.scatter(aww_vol, 6.67)

jkb_vol_mean = np.mean(jkb_volume_list)
vol_range_jkb = np.ptp(jkb_volume_list)

temp_mean = 4.35
temp_std = 0.51
temp_min, temp_max = temp_mean - (2*temp_std), temp_mean + (2*temp_std)
temp_range = temp_max - temp_min


ellipse_jkb = Ellipse((jkb_vol_mean, temp_mean), vol_range_jkb, temp_range, fill=True, alpha=0.5,
                      color='tab:orange')

ellipse_jkb_outline = Ellipse((jkb_vol_mean, temp_mean), vol_range_jkb, temp_range, fill=False, 
                  edgecolor='tab:orange', linewidth=2, alpha=1)


axs.add_patch(ellipse_jkb)
axs.add_patch(ellipse_jkb_outline)

axs.text(jkb_vol_mean-0.6e9, temp_mean, 'Ilulissat Fjord')


#%%


kanger_urel_05_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/kanger/Qib/u_rel_05_factor4_melt_rates/'

nc_list = [nc for nc in os.listdir(kanger_urel_05_path) if nc.endswith('nc')]
os.chdir(kanger_urel_05_path)

kanger_volume_list = []
for nc in nc_list:
    
    Qib_ds = xr.open_dataset(nc)
    aww_vol = Qib_ds.ice_vol.data
    kanger_volume_list.append(aww_vol)
    # axs.scatter(aww_vol, 6.67)

kanger_vol_mean = np.mean(kanger_volume_list)
vol_range_kanger = np.ptp(kanger_volume_list)
temp_mean = 5.63
temp_std = 0.47
temp_min, temp_max = temp_mean - (2*temp_std), temp_mean + (2*temp_std)

temp_range = temp_max - temp_min


ellipse_kanger = Ellipse((kanger_vol_mean, temp_mean), vol_range_kanger, temp_range, fill=True, alpha=0.5,
                      color='tab:red')

ellipse_kanger_outline = Ellipse((kanger_vol_mean, temp_mean), vol_range_kanger, temp_range, fill=False, 
                  edgecolor='tab:red', linewidth=2, alpha=1)


axs.add_patch(ellipse_kanger)
axs.add_patch(ellipse_kanger_outline)

axs.text(kanger_vol_mean-0.6e9, temp_mean-0.05, 'Kangerlussuaq\nFjord')

#%%


upernavik_urel_05_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/upernavik/Qib/u_rel_05_factor4_melt_rates/'

nc_list = [nc for nc in os.listdir(upernavik_urel_05_path) if nc.endswith('nc')]
os.chdir(upernavik_urel_05_path)

upernavik_volume_list = []
for nc in nc_list:
    
    Qib_ds = xr.open_dataset(nc)
    aww_vol = Qib_ds.ice_vol.data
    upernavik_volume_list.append(aww_vol)
    # axs.scatter(aww_vol, 6.67)

upernavik_vol_mean = np.mean(upernavik_volume_list)
vol_range_upernavik = np.ptp(upernavik_volume_list)

temp_mean = 3.46
temp_std = 0.12

temp_min, temp_max = temp_mean - (2*temp_std), temp_mean + (2*temp_std)
temp_range = temp_max - temp_min

ellipse_upernavik = Ellipse((upernavik_vol_mean, temp_mean), vol_range_upernavik, temp_range, fill=True, alpha=0.5,
                      color='tab:olive')

ellipse_upernavik_outline = Ellipse((upernavik_vol_mean, temp_mean), vol_range_upernavik, temp_range, fill=False, 
                  edgecolor='tab:olive', linewidth=2, alpha=1)


axs.add_patch(ellipse_upernavik)
axs.add_patch(ellipse_upernavik_outline)

axs.text(upernavik_vol_mean-0.6e9, temp_mean-0.05, 'Upernavik Fjord')


plt.tight_layout()

op = '/media/laserglaciers/upernavik/iceberg_py/figs/'

fig.savefig(f'{op}Qib_u_rel_05.pdf', dpi=300, bbox_inches='tight', transparent=True)


