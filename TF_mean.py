#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 12:04:52 2023

@author: laserglaciers
"""

import scipy.io as sio
import geopandas as gpd
from ast import literal_eval

EN4_TFGL_out_path = '/media/laserglaciers/upernavik/iceberg_py/TF_mean.mat'

EN4_TFGL = sio.loadmat(EN4_TFGL_out_path)

morlighem_numbers = []
names = []
xcoords = []
ycoords = []
tf_mean = []

for row in EN4_TFGL['TFmean'][0]:
    morlighem_numbers.append(int(row[0][0][0]))
    names.append(row[1][0])
    xcoords.append(row[2][0][0])
    ycoords.append(row[3][0][0])
    tf_mean.append(row[4][0][0])


points = gpd.points_from_xy(xcoords, ycoords,crs='EPSG:3413')

# points = gpd.points_from_xy(xcoords, ycoords,crs='EPSG:4326')

gdf = gpd.GeoDataFrame(geometry=points,crs='EPSG:3413')

gdf.to_crs('EPSG:3413', inplace=True)
gdf['morlighem_numbers'] = morlighem_numbers
gdf['names'] = names
gdf['tf_mean'] = tf_mean


# gdf['times'] = gdf['times'].apply(lambda x: str(x))
# gdf['TF_GL'] = gdf['TF_GL'].apply(lambda x: str(x))

of = './tf_mean.gpkg'
# https://stackoverflow.com/questions/70943128/can-i-save-a-geodataframe-that-contains-an-array-to-a-geopackage-file
gdf.to_file(of, driver='GPKG')


# gpkg['array_col'] = gpkg['array_col'].apply(lambda x: np.array(literal_eval(x.replace(' ', ','))))

