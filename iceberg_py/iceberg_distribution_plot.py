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

s2_path = '/media/laserglaciers/upernavik/ghawk_2023/20170625_s2_clip.tif'
convex_hull_path = '/media/laserglaciers/upernavik/ghawk_2023/iceberg_geoms/2017-06-16_2017-06-30_convex_hull_clip2.gpkg'
bounding_box_path = '/media/laserglaciers/upernavik/ghawk_2023/iceberg_geoms/2017-06-16_2017-06-30_bbox_clip2.gpkg'
mbergs_dict = '/media/laserglaciers/upernavik/iceberg_py/mbergs.pickle'
ctd_path = '/media/laserglaciers/upernavik/ghawk_2023/mar2010_ctd_geom_helheim_fjord_3413.gpkg'


convex_hull_df = gpd.read_file(convex_hull_path)
bounding_box_df = gpd.read_file(bounding_box_path)
ctd_df = gpd.read_file(ctd_path)

convex_hull_df2 = convex_hull_df.join(bounding_box_df[['height','width']],rsuffix='_bbox')
convex_hull_df2['max_dim'] = np.maximum(convex_hull_df2.height,convex_hull_df2.width)

bins = np.arange(0,1050,50) # lengths for this example and bins
labels = np.arange(50,1050,50) # lengths for this example and bins

convex_hull_df2['binned'] = gpd.pd.cut(convex_hull_df2['max_dim'],bins=bins,labels=labels)

with open(mbergs_dict, 'rb') as src:
    mbergs = pickle.load(src)

keel_dict = {}
for length in labels:
    berg = mbergs[length]
    k = berg.KEEL.sel(time=86400)
    keel_dict[length] = k.data[0]

convex_hull_df2['keel_depth'] = convex_hull_df2['binned'].map(keel_dict)

fig, ax = plt.subplots(figsize=(16.77,8.86))

def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))

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

ctd_df.plot(ax=ax,color='red',edgecolor='k',markersize=35)
plt.tight_layout()
op = f'/media/laserglaciers/upernavik/ghawk_2023/figs/iceberg_dist.png'
plt.savefig(op, dpi=300)


op = '/media/laserglaciers/upernavik/iceberg_py/convex_hull_icebergs.pkl'
with open(op,'wb') as handle:
    pickle.dump(convex_hull_df2, handle, protocol=pickle.HIGHEST_PROTOCOL)


