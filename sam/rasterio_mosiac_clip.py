#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 17:11:02 2024

@author: laserglaciers
"""

import rasterio
from rasterio.merge import merge
import rasterio.mask
from rasterio.plot import show
import os
import geopandas as gpd


mosiacs_path = '/media/laserglaciers/upernavik/iceberg_py/sam/upvk/mosiacs/'
op = '/media/laserglaciers/upernavik/iceberg_py/sam/upvk/mosaics_clip/'
if not os.path.exists(op):
    os.makedirs(op)

mosiacs_list = [tif for tif in os.listdir(mosiacs_path) if tif.endswith('tif')]


poly_path = '/media/laserglaciers/upernavik/iceberg_py/geoms/upernavik/melange_outline.gpkg'
gdf = gpd.read_file(poly_path)
geom = gdf.geometry


for tif in mosiacs_list:

    os.chdir(mosiacs_path)

    with rasterio.open(f'{tif}') as src:
    
        out_image, out_transform = rasterio.mask.mask(src, geom, crop=True)
        out_meta = src.meta.copy()
        
        out_meta.update({"driver": "GTiff",
                          "height": out_image.shape[1],
                          "width": out_image.shape[2],
                         "transform": out_transform})
    
    
    with rasterio.open(f'{op}{tif}',mode='w', **out_meta) as dst:
        
        dst.write(out_image)
    
    
    
    
