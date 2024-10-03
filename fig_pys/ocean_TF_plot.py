#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:31:40 2024

@author: laserglaciers
"""

import scipy.io as sio
import geopandas as gpd
from ast import literal_eval
import xarray as xr
import rasterio
import numpy as np

TF0 = sio.loadmat('/media/laserglaciers/upernavik/slater_2022_submelt/TF0_oceanTF.mat')
bmfile = '/media/laserglaciers/upernavik/iceberg_py/bed_for_slater/bed_v5_test.tif'


thermal_forcing = TF0['TF0']

# bed = xr.open_dataset(bmfile)
bed = rasterio.open(bmfile)


# transform = bed.bed.rio.transform()

# meta = {'affine':transform,
#         'count':1,
#         'crs':3413,
#         'width':bed.rio.width,
#         'height':bed.rio.height,
#         'nodata': np.nan,
#         'dtype':np.float64
#         }

with rasterio.open('ocean_tf.tif', mode='w', **bed.meta) as dst:
    
    dst.write(thermal_forcing,1)

bed.close()