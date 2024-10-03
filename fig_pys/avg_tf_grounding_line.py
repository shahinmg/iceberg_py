#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 12:04:52 2023

@author: laserglaciers
"""

import scipy.io as sio
import geopandas as gpd
from ast import literal_eval
import matplotlib.pyplot as plt
import numpy as np

avg_TFGL_out_path = '/media/laserglaciers/upernavik/iceberg_py/avg_TFGL_out_v2.mat'

avg_TFGL = sio.loadmat(avg_TFGL_out_path)

morlighem_numbers = []
names = []
TF_GL = []
times = []
xcoords = []
ycoords = []

for row in avg_TFGL['avg_tf_arr'][0]:
    morlighem_numbers.append(int(row[0][0][0]))
    names.append(row[1][0])

    times.append(row[4][0])
    TF_GL.append(row[5][0])

    xcoords.append(row[6][0][0])
    ycoords.append(row[7][0][0])
    
    
points = gpd.points_from_xy(xcoords, ycoords,crs='EPSG:3413')

# # points = gpd.points_from_xy(xcoords, ycoords,crs='EPSG:4326')

gdf = gpd.GeoDataFrame(geometry=points,crs='EPSG:3413')

# gdf.to_crs('EPSG:3413', inplace=True)
gdf['morlighem_numbers'] = morlighem_numbers
gdf['names'] = names
gdf['TF_GL'] = TF_GL
gdf['times'] = times

mean_symbol = r'$\bar{X}$'
sigma = r'$\sigma$'


# make plot function 

def tf_plot(tf, times, name):
    
    fig, ax = plt.subplots(2,1)


    ax[0].plot(times, tf)
    
    ax[0].set_title(f'{name} Thermal Forcing Average')
    ax[0].set_ylabel(r'TF ($^{\circ}$C)')
    
    ax[1].set_title('Frequency of TF')
    ax[1].hist(tf, bins=20, edgecolor='black')
    ax[1].set_xlabel('TF ($^{\circ}$C)')
    
    ax[1].text(.01, .99, f'{mean_symbol} = {tf.mean():.2f}', ha='left', va='top', transform=ax[1].transAxes)
    ax[1].text(.01, .85, f'{sigma} = {tf.std():.2f}', ha='left', va='top', transform=ax[1].transAxes)
    
    ax[0].set_ylim(3,8.3)
    ax[1].set_ylim(0,80)
    
    plt.tight_layout()
    
    op = '/media/laserglaciers/upernavik/iceberg_py/figs/tf_timeseries/'
    fig.savefig(f'{op}{name}.pdf', dpi=300, )

# %%


helheim_tf = gdf['TF_GL'][2]
helheim_time = gdf['times'][2]
helheim_name = gdf['names'][2]

jkb_tf = gdf['TF_GL'][0]
jkb_times = gdf['times'][0]
jkb_name = gdf['names'][0]

kan_tf = gdf['TF_GL'][1]
kan_times = gdf['times'][1]
kan_name = gdf['names'][1]

upvk_tf = gdf['TF_GL'][15]
upvk_times = gdf['times'][15]
upvk_name = gdf['names'][15]


plot_tups = [(helheim_tf, helheim_time, helheim_name),
             (jkb_tf, jkb_times, jkb_name),
             (kan_tf, kan_times, kan_name),
             (upvk_tf, upvk_times, upvk_name)
    
    ]

for tf, time, name in plot_tups:
    
    tf_plot(tf, time, name)





#%%

# fig, ax = plt.subplots(2,1)
# helheim_tf = gdf['TF_GL'][2]
# helheim_time = gdf['times'][2]

# ax[0].plot(helheim_time, helheim_tf)

# ax[0].set_title('Helheim TF averge from Slater 2022')
# ax[0].set_ylabel('TF (C)')

# ax[1].set_title('Frequency of TF')
# ax[1].hist(helheim_tf, bins=20, edgecolor='black')

# ax[1].text(.01, .99, f'{mean_symbol} = {helheim_tf.mean():.2f}', ha='left', va='top', transform=ax[1].transAxes)
# ax[1].text(.01, .85, f'{sigma} = {helheim_tf.std():.2f}', ha='left', va='top', transform=ax[1].transAxes)


# plt.tight_layout()

# op = '/media/laserglaciers/upernavik/iceberg_py/figs/'
# fig.savefig(f'{op}Helheim_TF_slater2022.png')


# #%%

# figj, axj = plt.subplots(2,1)
# jkb_tf = gdf['TF_GL'][0]
# jkb_times = gdf['times'][0]

# axj[0].plot(jkb_times, jkb_tf)

# axj[0].set_title('Sermeq Kujalleq TF averge from Slater 2022')
# axj[0].set_ylabel('TF (C)')

# axj[1].set_title('Frequency of TF')
# axj[1].hist(jkb_tf, bins=20, edgecolor='black')

# axj[1].text(.01, .99, f'{mean_symbol} = {jkb_tf.mean():.2f}', ha='left', va='top', transform=axj[1].transAxes)
# axj[1].text(.01, .85, f'{sigma} = {jkb_tf.std():.2f}', ha='left', va='top', transform=axj[1].transAxes)


# plt.tight_layout()

# #%%

# figk, axk = plt.subplots(2,1)
# kan_tf = gdf['TF_GL'][1]
# kan_times = gdf['times'][1]

# axk[0].plot(kan_times, kan_tf)

# axk[0].set_title('Kangerlussuaq TF averge from Slater 2022')
# axk[0].set_ylabel('TF (C)')

# axk[1].set_title('Frequency of TF')
# axk[1].hist(kan_tf, bins=20, edgecolor='black')

# axk[1].text(.01, .99, f'{mean_symbol} = {kan_tf.mean():.2f}', ha='left', va='top', transform=axk[1].transAxes)
# axk[1].text(.01, .85, f'{sigma} = {kan_tf.std():.2f}', ha='left', va='top', transform=axk[1].transAxes)


# plt.tight_layout()

# #%%

# figu, axu = plt.subplots(2,1)
# upvk_tf = gdf['TF_GL'][15]
# upvk_times = gdf['times'][15]

# axu[0].plot(upvk_times, upvk_tf)

# axu[0].set_title('Upernavik Isstrøm C TF averge from Slater 2022')
# axu[0].set_ylabel('TF (C)')

# axu[1].set_title('Frequency of TF')
# axu[1].hist(upvk_tf, bins=20, edgecolor='black')

# axu[1].text(.01, .99, f'{mean_symbol} = {upvk_tf.mean():.2f}', ha='left', va='top', transform=axu[1].transAxes)
# axu[1].text(.01, .85, f'{sigma} = {upvk_tf.std():.2f}', ha='left', va='top', transform=axu[1].transAxes)



# plt.tight_layout()



# gdf['times'] = gdf['times'].apply(lambda x: str(x))
# gdf['avg_TFGL'] = gdf['avg_TFGL'].apply(lambda x: str(x))

# of = './tf_mean.gpkg'
# # https://stackoverflow.com/questions/70943128/can-i-save-a-geodataframe-that-contains-an-array-to-a-geopackage-file
# gdf.to_file(of, driver='GPKG')


# gpkg['array_col'] = gpkg['array_col'].apply(lambda x: np.array(literal_eval(x.replace(' ', ','))))

