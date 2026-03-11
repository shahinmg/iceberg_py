#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 12:04:01 2026

@author: laserglaciers
"""

import xarray as xr
import numpy as np
import os


class_berg_path = '/home/laserglaciers/icebergPy/data/test_class_outputs/iceberg_model_output/helheim/avg/2016-04-24_helheim_coeff_1_CTD_constant_UREL_07.nc'

old_berg_path = '/home/laserglaciers/icebergPy/data/iceberg_model_output_bug_fix_ctd/helheim/avg/2016-04-24_helheim_coeff_1_CTD_constant_UREL_07.nc'


class_berg_ds = xr.open_dataset(class_berg_path)
old_berg_ds = xr.open_dataset(old_berg_path)
old_berg_ds = old_berg_ds.rename_vars({"melt_rate_intergrated": "melt_rate_integrated"})


for var in old_berg_ds.data_vars:
    
    diff = old_berg_ds[var] - class_berg_ds[var]
    diff_mean = np.nanmean(diff)
    
    print(f'{var} diff: {diff_mean}')

