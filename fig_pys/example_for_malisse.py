#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:18:29 2024

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




with rasterio.open(s2_path) as src:

    rgb_r = src.read([1,2,3])
    rgb_r = adjust_band(rgb_r)
    vmin, vmax = np.nanpercentile(rgb_r, (5,95))  # 5-95% stretch
    
    show(rgb_r, transform=src.transform,  vmin=vmin, vmax=vmax,ax=ax)

cmap = plt.get_cmap('viridis').copy()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

k_min = convex_hull_df2['keel_depth'].min()
k_max = convex_hull_df2['keel_depth'].max()
cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin = k_min,vmax = k_max), cmap=cmap),
              cax=cax,label='Keel Depth (m)')

cbar.set_label(label='Keel Depth (m)',fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
cbar.ax.tick_params(labelsize=20)
ext = [src.bounds[0],src.bounds[2], src.bounds[1], src.bounds[3]]
convex_hull_df2.plot(ax=ax,column='keel_depth',cmap=cmap)
ax.set_ylim(src.bounds[1], src.bounds[3])