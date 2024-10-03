#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 13:33:20 2023

@author: laserglaciers
"""

import rasterio
from rasterio.plot import show
from rasterio.plot import adjust_band
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors, cm
import geopandas as gpd
import numpy as np
import pickle
import os

s2_path = '/media/laserglaciers/upernavik/iceberg_py/sam/rbg_images/kanger/2020-09-12.tif'
# convex_hull_path = '/media/laserglaciers/upernavik/segment-anything/prediction_geoms/helheim/20230727T142031/20230727T142031_merged_v2.gpkg'
# bounding_box_path = '/media/laserglaciers/upernavik/segment-anything/prediction_geoms/helheim/20230727T142031/20230727T142031_merged_bounding_box.gpkg'

mbergs_dict = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/berg_model/20230727T142031_bergs_v2.pkl'
berg_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/kanger/iceberg_geoms/dim_with_bin/'
os.chdir(berg_path)
berg_list = [pkg for pkg in os.listdir(berg_path) if pkg.endswith('gpkg')]

for file in berg_list:

        
    date = file[:10]
    qgis_join_path = file
    
    
    convex_hull_df2 = gpd.read_file(qgis_join_path)
    convex_hull_df2['max_dim'] = np.maximum(convex_hull_df2['length'], convex_hull_df2['width'])
    # mask = convex_hull_df2.max_dim > 1400.0
    # convex_hull_df2['max_dim'].loc[mask] = 1400.0
    
    bins = np.arange(0,1450,50) # lengths for this example and bins
    labels = np.arange(50,1450,50) # lengths for this example and bins
    
    convex_hull_df2['binned'] = gpd.pd.cut(convex_hull_df2['max_dim'],bins=bins,labels=labels)
    
    with open(mbergs_dict, 'rb') as src:
        mbergs = pickle.load(src)
    
    keel_dict = {}
    for length in labels:
        berg = mbergs[length]
        k = berg.KEEL.sel(time=86400)
        keel_dict[length] = k.data[0]
    
    convex_hull_df2['keel_depth'] = convex_hull_df2['binned'].map(keel_dict)
    # convex_hull_df2.dropna(inplace=True)
    fig, ax = plt.subplots(figsize=(12,8))
    
    def normalize(array):
        """Normalizes numpy arrays into scale 0.0 - 1.0"""
        array_min, array_max = array.min(), array.max()
        return ((array - array_min)/(array_max - array_min))
    
    # kwargs = {'extent':(3e5, 3.35e5, -2.585e6, -2.570e6)}
    
    
    
    with rasterio.open(s2_path) as src:
    
        rgb_r = src.read([1,2,3])
        rgb_r = adjust_band(rgb_r)
        vmin, vmax = np.nanpercentile(rgb_r, (5,95))  # 5-95% stretch
        
        show(rgb_r, transform=src.transform,  vmin=vmin, vmax=vmax,ax=ax)
    
    
    cmap = plt.get_cmap('viridis').copy()
    convex_hull_df2.plot(ax=ax,column='keel_depth',cmap=cmap)
    
    
    # ax.set_xlim((3.0e5, 3.35e5))
    # ax.set_ylim((-2.59e6, -2.57e6))
    
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    
    k_min = convex_hull_df2['keel_depth'].min()
    k_max = convex_hull_df2['keel_depth'].max()
    cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin = k_min,vmax = k_max), cmap=cmap),
                  cax=cax,label='Keel Depth (m)')
    
    cbar.set_label(label='Keel Depth (m)',fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    cbar.ax.tick_params(labelsize=10)
    # ext = [src.bounds[0],src.bounds[2], src.bounds[1], src.bounds[3]]
    
    
    op = f'/media/laserglaciers/upernavik/iceberg_py/outfiles/kanger/iceberg_geoms/keel_depths/'
    if not os.path.exists(op):
        os.makedirs(op)
    
    op_df = f'{op}/{date}_icebergs_kanger_keel_depth.gpkg'
    # convex_hull_df_out = convex_hull_df2.drop(['binned'],axis=1)
    # convex_hull_df_out['keel_depth'] = convex_hull_df2.keel_depth.to_numpy()
    convex_hull_df2.to_file(op_df, driver='GPKG')