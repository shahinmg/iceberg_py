#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:18:19 2024

@author: laserglaciers
"""

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import os

path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/upernavik/iceberg_geoms/clean/'

file_list = [file for file in os.listdir(path) if file.endswith('gpkg')]

os.chdir(path)
for file in file_list:
    
    gdf = gpd.read_file(file)
    
    gdf['envelope_box'] = gdf.envelope
    
    bounds = gdf['envelope_box'].bounds
    
    width = bounds['maxx'] - bounds['minx']
    length = bounds['maxy'] - bounds['miny']
    
    gdf['width'] = width
    gdf['length'] = length
    gdf.drop('envelope_box',axis=1, inplace=True)
    
    op = '/media/laserglaciers/upernavik/iceberg_py/outfiles/upernavik/iceberg_geoms/dim_with_bin/'
    
    if not os.path.exists(op):
        os.makedirs(op)
    
    
    gdf.to_file(f'{op}{file[:-5]}_dims.gpkg', driver='GPKG')