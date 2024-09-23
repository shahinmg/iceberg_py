#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 17:11:02 2024

@author: laserglaciers
"""

import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import os


root_path = '/media/laserglaciers/upernavik/iceberg_py/ec2_transfer/helheim/'
dir_list = [d for d in os.listdir(root_path)][1:]

for folder in dir_list:
    dirpath = f'/media/laserglaciers/upernavik/iceberg_py/ec2_transfer/helheim/{folder}/'
    op = f'/media/laserglaciers/upernavik/iceberg_py/ec2_transfer/helheim/{folder}/'
    
    
    tifs = [tif for tif in os.listdir(dirpath) if tif.endswith('tif')]
    
    
    
    src_files_to_mosaic = []
    os.chdir(dirpath)
    for fp in tifs:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
        
    
    mosaic, out_trans = merge(src_files_to_mosaic)
    
    
    # show(mosaic)
    
    out_meta = src.meta.copy()
    
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     "crs": 32624
                     }
                )
    
    with rasterio.open(f'{op}{folder}_mosiac.tif',mode='w', **out_meta) as dst:
        
        dst.write(mosaic)
    
    
    
    
