#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:17:41 2024

@author: laserglaciers
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import geopandas as gpd
import matplotlib.pyplot as plt
import pickle
import melt_functions as ice_melt
import xarray as xr



volume_below_aww = np.linspace(0, 600e9,100)
TF_range = np.linspace(0.5, 8,100)

# dt = np.arange(20,365,1) #flushing rate
# # dt = np.logspace(1, 365, 1)
# TF = np.linspace(5, 8, dt.shape[0]) #thermal forcing from Slater histogram TF of Helheim

Vol_x, TF_y = np.meshgrid(volume_below_aww, TF_range)

# # Volume_test = 23e3 * 5e3 * 300 #km3 Helheim Fjord test
# Volume_test = 148461041 * 300 # area from Helheim fjord shapefile ~ 28 km length
l_heat = 3.34e5

p_fw = 1000


Q_ib = 