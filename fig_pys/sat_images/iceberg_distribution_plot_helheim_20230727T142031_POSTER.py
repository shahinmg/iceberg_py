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
import matplotlib.ticker as ticker
import geopandas as gpd
import numpy as np
import pickle

s2_path = '/media/laserglaciers/upernavik/sentinel_2/helheim/S2A_MSIL1C_20230727T142031_N0509_R096_T24WWU_20230727T193820.tif'
convex_hull_path = '/media/laserglaciers/upernavik/segment-anything/prediction_geoms/helheim/20230727T142031/20230727T142031_merged_v2.gpkg'
bounding_box_path = '/media/laserglaciers/upernavik/segment-anything/prediction_geoms/helheim/20230727T142031/20230727T142031_merged_bounding_box.gpkg'
mbergs_dict = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/berg_model/20230727T142031_bergs.pkl'
ctd_path = '/media/laserglaciers/upernavik/ghawk_2023/mar2010_ctd_geom_helheim_fjord_3413.gpkg'
qgis_join_path = '/media/laserglaciers/upernavik/segment-anything/prediction_geoms/helheim/20230727T142031/20230727T142031_merged_qgis_joined.gpkg'

# convex_hull_df = gpd.read_file(convex_hull_path)
# bounding_box_df = gpd.read_file(bounding_box_path)
# bounding_box_df.set_index(['id'])
# ctd_df = gpd.read_file(ctd_path)
convex_hull_df2 = gpd.read_file(qgis_join_path)

# convex_hull_df2 = convex_hull_df.join(bounding_box_df[['height','width']],rsuffix='_bbox')
convex_hull_df2['max_dim'] = np.maximum(convex_hull_df2.height,convex_hull_df2.width)



bins = np.arange(0,1450,50) # lengths for this example and bins
labels = np.arange(50,1450,50) # lengths for this example and bins
keel_labels = np.arange(20,520,20)
keel_bins = np.arange(0,520,20)

convex_hull_df2['binned'] = gpd.pd.cut(convex_hull_df2['max_dim'],bins=bins,labels=labels)



with open(mbergs_dict, 'rb') as src:
    # mbergs = pickle.load(src)
    mbergs = gpd.pd.read_pickle(src)

keel_dict = {}
for length in labels:
    berg = mbergs[length]
    k = berg.KEEL.sel(time=86400)
    keel_dict[length] = k.data[0]

convex_hull_df2['keel_depth'] = convex_hull_df2['binned'].map(keel_dict)
convex_hull_df2['keel_binned'] = gpd.pd.cut(convex_hull_df2['keel_depth'],bins=keel_bins,labels=keel_labels)




# convex_hull_df2.dropna(inplace=True)
fig, ax = plt.subplots(figsize=(16.77,8.86))


kwargs = {'extent':(3e5, 3.35e5, -2.585e6, -2.570e6)}

def stretch_to_min_max(img):
    min_percent = 2   # Low percentile
    max_percent = 98  # High percentile
    lo, hi = np.nanpercentile(img, (min_percent, max_percent))

    res_img = (img.astype(float) - lo) / (hi-lo)
    
    return np.maximum(np.minimum(res_img*255, 255), 0).astype(np.uint8)

with rasterio.open(s2_path) as src:

    rgb_r = src.read([1,2,3])
    rgb_r = stretch_to_min_max(rgb_r)
    rgb_r = adjust_band(rgb_r)
    
    show(rgb_r, transform=src.transform, ax=ax)
    

ax.set_xlim((3.0e5, 3.35e5))
ax.set_ylim((-2.59e6, -2.57e6))

cmap = plt.get_cmap('viridis').copy()
divider = make_axes_locatable(ax) #ax for distribution plot and cbar

cax = divider.append_axes("right", size="2%", pad=0.1)
axright = divider.append_axes("right", size=1.2, pad=1.2) 

k_min = convex_hull_df2['keel_depth'].min()
k_max = convex_hull_df2['keel_depth'].max()

cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin = k_min,vmax = k_max), cmap=cmap),
              cax=cax,label='Keel Depth (m)')

cbar.set_label(label='Keel Depth (m)',fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20,pad=15)
cbar.ax.tick_params(labelsize=20)

# ext = [src.bounds[0],src.bounds[2], src.bounds[1], src.bounds[3]]
convex_hull_df2.plot(ax=ax,column='keel_depth',cmap=cmap)
convex_hull_df2['keel_binned'].value_counts().sort_index().plot(kind='barh',logx=True,ax=axright,
                                                           edgecolor = 'k',zorder=2)

axright.yaxis.tick_right()
# # axright.yaxis.set_ticks(np.arange(50, 1450, 200))
ticks = axright.yaxis.get_ticklocs()
ticklabels = [l.get_text() for l in axright.yaxis.get_ticklabels()]
axright.set_yticks(ticks[1::2])
axright.yaxis.set_ticklabels(ticklabels[1::2])
axright.tick_params(axis='both', which='major', labelsize=20)
axright.axhspan(5.5,28,facecolor='0.6',alpha=0.3, zorder=1)

op = '/media/laserglaciers/upernavik/agu_2023/figs/'
# fig.savefig(f'{op}helheim_20230727T193820.png',dpi=300,transparent=False)