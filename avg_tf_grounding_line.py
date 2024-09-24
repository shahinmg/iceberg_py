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


fig, ax = plt.subplots(2,1)
helheim_tf = gdf['TF_GL'][2]
helheim_time = gdf['times'][2]

ax[0].plot(helheim_time, helheim_tf)

ax[0].set_title('Helheim TF averge from Slater 2022')
ax[0].set_ylabel('TF (C)')

ax[1].set_title('Frequency of TF')
ax[1].hist(helheim_tf, bins=20, edgecolor='black')
plt.tight_layout()

op = '/media/laserglaciers/upernavik/iceberg_py/figs/'
fig.savefig(f'{op}Helheim_TF_slater2022.png')

# gdf['times'] = gdf['times'].apply(lambda x: str(x))
# gdf['avg_TFGL'] = gdf['avg_TFGL'].apply(lambda x: str(x))

# of = './tf_mean.gpkg'
# # https://stackoverflow.com/questions/70943128/can-i-save-a-geodataframe-that-contains-an-array-to-a-geopackage-file
# gdf.to_file(of, driver='GPKG')


# gpkg['array_col'] = gpkg['array_col'].apply(lambda x: np.array(literal_eval(x.replace(' ', ','))))

