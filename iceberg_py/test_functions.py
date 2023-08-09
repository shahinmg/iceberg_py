#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 11:54:53 2023

@author: laserglaciers
"""
import melt_functions as ice_melt
import numpy as np
import xarray as xr

L = np.arange(50,1050,50)
dz = 5

#Test first L

ice = ice_melt.init_iceberg_size(L[0],dz=dz)