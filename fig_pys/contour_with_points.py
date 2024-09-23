#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:41:18 2023

@author: laserglaciers
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pickle
from scipy.stats import powerlaw
import scipy.stats

helheim_mbergs_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/berg_model/20230727T142031_bergs.pkl'
helheim_gdf_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/20230727T142031_icebergs_helheim.pkl'
kanger_mbergs_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/kanger/berg_model/20170710T141009_bergs.pkl'
kanger_gdf_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/kanger/20170710T141009_icebergs_kanger.pkl'

L = np.arange(50,1450,50)
l_heat = 3.34e5
Aww_depth = 150

with open(helheim_mbergs_path, 'rb') as hm_src:
    helheim_mbergs = pickle.load(hm_src)
    
with open(helheim_gdf_path, 'rb') as hg_src:
    helheim_gdf = pickle.load(hg_src)


with open(kanger_mbergs_path, 'rb') as km_src:
    kanger_mbergs = pickle.load(km_src)
    
with open(kanger_gdf_path, 'rb') as kg_src:
    kanger_gdf = pickle.load(kg_src)


kanger_count = kanger_gdf.binned.value_counts()
helheim_count = helheim_gdf.binned.value_counts()


Qib_dict = {}
for length in L:
    # print(f'Processing Length {length}')
    
    berg = kanger_mbergs[length]
    k = berg.KEEL.sel(time=86400*2)
    # if k >= Aww_depth:
    Mfreew = berg.Mfreew.sel(Z=slice(None,k.data[0]), time=86400*2)
    Mturbw = berg.Mturbw.sel(Z=slice(None,k.data[0]), time=86400*2)
    
    total_iceberg_melt = np.mean(Mfreew + Mturbw,axis=1)
    Qib = total_iceberg_melt * l_heat * 1000 # iceberg heatflux per z layer
    Qib_dict[length] = Qib


Qib_sums = {}
uwV_dict = {}

for length in L:
    count = kanger_count[length]
    Qib_sum = np.nansum(Qib_dict[length].sel(Z=slice(Aww_depth,None)))
    uwV_sum = np.nansum(kanger_mbergs[length].uwV.sel(Z=slice(Aww_depth,None)))
    
    # Qib_length_totals[length] = Qib_sum * count
    
    uwV_dict[length] = uwV_sum * count
    
uwV_total = np.nansum(list(uwV_dict.values()))



kanger_count.sort_index().plot(kind='bar',logy=True,edgecolor = 'k')
# x = list(kanger_count.sort_index().keys())
# size = len(kanger_gdf.binned.dropna())
# dist_name = 'expon'
# dist = getattr(scipy.stats, dist_name)
# params = dist.fit(kanger_gdf.binned.dropna())
# arg = params[:-2]
# loc = params[-2]
# scale = params[-1]
# pdf_fitted = dist.pdf(x, *arg, loc=loc, scale=scale) * size
# plt.plot(pdf_fitted, label = dist_name)


