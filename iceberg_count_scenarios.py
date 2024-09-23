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

count_ranges = {}
# count_ranges[50] = (900,1200)
# count_ranges[100] = (800,1100)
# count_ranges[150] = (200,400)
# count_ranges[200] = (100,200)
# count_ranges[250] = (50,80)
# count_ranges[300] = (20,50)
# count_ranges[350] = (20,50)
# count_ranges[400] = (10,25)
# count_ranges[450] = (5,20)
# count_ranges[500] = (5,20)
# count_ranges[550] = (5,10)
# count_ranges[600] = (3,10)
# count_ranges[650] = (3,10)
# count_ranges[700] = (1,8)
# count_ranges[750] = (1,8)
# count_ranges[800] = (1,6)
# count_ranges[850] = (1,5)
# count_ranges[900] = (1,5)
# count_ranges[950] = (1,5)
# count_ranges[1000] = (1,5)
# count_ranges[1050] = (1,5)
# count_ranges[1100] = (1,5)
# count_ranges[1150] = (1,5)
# count_ranges[1200] = (1,5)
# count_ranges[1250] = (1,5)
# count_ranges[1300] = (1,5)
# count_ranges[1350] = (1,5)
# count_ranges[1400] = (1,5)

count_ranges[50] = (900,1500)
count_ranges[100] = (800,1400)
count_ranges[150] = (200,600)
count_ranges[200] = (100,400)
count_ranges[250] = (50,200)
count_ranges[300] = (20,100)
count_ranges[350] = (20,100)
count_ranges[400] = (10,75)
count_ranges[450] = (5,50)
count_ranges[500] = (5,50)
count_ranges[550] = (5,50)
count_ranges[600] = (3,50)
count_ranges[650] = (3,50)
count_ranges[700] = (1,20)
count_ranges[750] = (1,20)
count_ranges[800] = (1,20)
count_ranges[850] = (1,20)
count_ranges[900] = (1,20)
count_ranges[950] = (1,20)
count_ranges[1000] = (1,15)
count_ranges[1050] = (1,15)
count_ranges[1100] = (1,15)
count_ranges[1150] = (1,15)
count_ranges[1200] = (1,15)
count_ranges[1250] = (1,15)
count_ranges[1300] = (1,15)
count_ranges[1350] = (1,15)
count_ranges[1400] = (1,15)

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
# ax3.hist(icebergs_gdf['max_dim'].values, bins=np.arange(0,1050,50),
#          edgecolor = "black")
axs[1].set_ylabel('Count')
axs[1].set_xlabel('Iceberg surface length (m)')
axs[1].set_title('kanger dist')
plt.tight_layout()