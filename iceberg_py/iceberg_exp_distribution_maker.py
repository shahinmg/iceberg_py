#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:05:47 2023

@author: laserglaciers
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import geopandas as gpd
import matplotlib.pyplot as plt
import pickle


kanger_dist_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/kanger/20170710T141009_icebergs_kanger.pkl'
kanger_dist = gpd.GeoDataFrame(pd.read_pickle(kanger_dist_path))

kanger_dist.sort_values(by=['binned'], inplace=True)
df_mean = np.mean(kanger_dist["binned"].dropna().values)
df_std = np.std(kanger_dist["binned"].dropna().values)
pdf = stats.norm.pdf(kanger_dist["binned"].dropna().values, df_mean, df_std)


bins = np.arange(0,1450,50) # lengths for this example and bins
labels = np.arange(50,1450,50) # lengths for this example and 
n = len(kanger_dist.binned.dropna())
exp_test = np.random.exponential(scale=df_mean+50,size=n)
bin_test = pd.cut(exp_test,bins=bins,labels=labels)

fig,axs = plt.subplots(2,1)
bin_test.value_counts().sort_index().plot(kind='bar',logy=True,edgecolor = 'k',ax=axs[0])
axs[0].set_title('Random exp dist')

kanger_dist['binned'].value_counts().sort_index().plot(kind='bar',logy=True,
                                                        edgecolor = 'k', ax=axs[1])

axs[1].set_ylabel('Count')
axs[1].set_xlabel('Iceberg surface length (m)')
axs[1].set_title('kanger dist')
plt.tight_layout()

kenger_mbergs_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/kanger/berg_model/20170710T141009_bergs.pkl'
with open(kenger_mbergs_path, 'rb') as src:
    mbergs = pickle.load(src)

# Heat flux figure per layer per size of iceberg
Qib_dict = {}
Aww_depth = 150
L = np.arange(50,1450,50)
l_heat = 3.34e5
for length in L:
    berg = mbergs[length]
    k = berg.KEEL.sel(time=86400*2)
    # if k >= Aww_depth:
    Mfreew = berg.Mfreew.sel(Z=slice(None,k.data[0]), time=86400*2)
    Mturbw = berg.Mturbw.sel(Z=slice(None,k.data[0]), time=86400*2)
    
    total_iceberg_melt = np.mean(Mfreew + Mturbw,axis=1)
    Qib = total_iceberg_melt * l_heat * 1000 # iceberg heatflux per z layer
    Qib_dict[length] = Qib



gdf_pkl_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/kanger/20170710T141009_icebergs_kanger.pkl'
with open(gdf_pkl_path, 'rb') as src:
    icebergs_gdf = pickle.load(src)

# vc = bin_test.value_counts()
vc = icebergs_gdf['binned'].value_counts()
Qib_totals = {}
Qib_sums = {}
uwV_dict = {}
for length in L:
    count = vc[length]
    Qib_sum = np.nansum(Qib_dict[length].sel(Z=slice(Aww_depth,None)))
    uwV_sum = np.nansum(mbergs[length].uwV.sel(Z=slice(Aww_depth,None)))
    
    Qib_totals[length] = Qib_sum * count
    Qib_sums[length] = Qib_sum
    
    uwV_dict[length] = uwV_sum * count
    print(f'{length}: {Qib_sum*count}')
    
qib_total=np.nansum(list(Qib_totals.values()))
uwV_total=np.nansum(list(uwV_dict.values()))
