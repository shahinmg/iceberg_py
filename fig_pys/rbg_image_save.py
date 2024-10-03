#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:43:43 2024

@author: laserglaciers
"""

import dask.distributed
import dask.utils
import numpy as np
import planetary_computer as pc
import xarray as xr
from IPython.display import display
from pystac_client import Client
import geopandas as gpd
import requests
from rasterio.plot import show
from odc.stac import configure_rio, stac_load

import matplotlib.pyplot as plt
import numpy as np

from contextlib import contextmanager  
import rasterio
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling
from rasterio.windows import Window
import cv2
from rasterio.plot import reshape_as_raster, reshape_as_image
import os
import pandas as pd

stac_path = '/media/laserglaciers/upernavik/iceberg_py/sam/stac_nc/upvk_netcdf.nc'

glacier = xr.open_dataset(stac_path)



#https://rasterio.groups.io/g/main/topic/memoryfile_workflow_should/32634761
@contextmanager
def mem_raster(data, **profile):
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset_writer:
            dataset_writer.write(data)
 
        with memfile.open() as dataset_reader:
            yield dataset_reader

profile_rgb = {'driver':'GTiff', 'count':3, 
                      'transform':glacier.rio.transform(), 'crs':glacier.odc.crs, 
                      'width':glacier.rio.shape[1], 'height': glacier.rio.shape[0], 
                      'dtype':np.float64
}

for time in glacier.time.data:
    
    rgb = np.dstack((glacier.red.sel(time=time).values,
                     glacier.green.sel(time=time).values,
                     glacier.blue.sel(time=time).values))
    
    rgb_raster = reshape_as_raster(rgb)
    date_str = str(pd.to_datetime(time).date())
    with mem_raster(rgb_raster, **profile_rgb) as ds:
        
        op = '/media/laserglaciers/upernavik/iceberg_py/sam/rbg_images/upernavik/'
        if not os.path.exists(op):
            os.makedirs(op)
            
        out_image = ds.read([1,2,3])
        if not os.path.exists(op):
            os.makedirs(op)
            
        with rasterio.open(f'{op}{date_str}.tif', mode='w', **profile_rgb) as dst:
            
            dst.write(out_image)
    
        
    


