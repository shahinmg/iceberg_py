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
from color_operations import operations, parse_operations


s2_path = '/media/laserglaciers/upernavik/iceberg_py/sam/rbg_images/jkb/2020-08-16.tif'

joined_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/jkb/iceberg_geoms/dim_with_bin/2020-08-16_icebergs_jkb_keel_depth.gpkg'
s2_path_2 = '/media/laserglaciers/upernavik/sentinel_2/jkb/S2A_MSIL1C_20200802T152911_N0209_R111_T22WEB_20200802T173803_clip.tif'


mbergs_dict = '/media/laserglaciers/upernavik/iceberg_py/outfiles/kanger/berg_model/20170710T141009_bergs.pkl'
# ctd_path = '/media/laserglaciers/upernavik/ghawk_2023/mar2010_ctd_geom_helheim_fjord_3413.gpkg'


# convex_hull_df = gpd.read_file(convex_hull_path)
# bounding_box_df = gpd.read_file(bounding_box_path)
# bounding_box_df.set_index(['id'])
# ctd_df = gpd.read_file(ctd_path)
# convex_hull_df2 = gpd.read_file(convex_hull_path)

convex_hull_df2 = gpd.read_file(joined_path)
convex_hull_df2['max_dim'] = np.maximum(convex_hull_df2['length'], convex_hull_df2['width'])
mask = convex_hull_df2.max_dim > 1400.0
convex_hull_df2['max_dim'].loc[mask] = 1400.0




m2km = lambda x, _: f'{x/1000:g}'

bins = np.arange(0,1450,50) # lengths for this example and bins
labels = np.arange(50,1450,50) # lengths for this example and bins
keel_labels = np.arange(20,520,20)
keel_bins = np.arange(0,520,20)

convex_hull_df2['binned'] = gpd.pd.cut(convex_hull_df2['max_dim'],bins=bins,labels=labels)

with open(mbergs_dict, 'rb') as src:
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

def stretch_to_min_max(img):
    min_percent = 2   # Low percentile
    max_percent = 98  # High percentile
    lo, hi = np.nanpercentile(img, (min_percent, max_percent))

    res_img = (img.astype(float) - lo) / (hi-lo)
    
    return np.maximum(np.minimum(res_img*255, 255), 0).astype(np.uint8)

with rasterio.open(s2_path) as src:

    rgb_r = src.read([1,2,3])
    # rgb_r = stretch_to_min_max(rgb_r)
    rgb_r = adjust_band(rgb_r)
    ops = "gamma b 1.7, gamma rg 1.5, sigmoidal rgb 3 0.13, saturation 1.0"
    # ops =  "sigmoidal rgb 5 0.25, saturation 1.5"

    # rgb_sig = operations.sigmoidal(rgb_r, 6, 0.5)
    rgb_sat = operations.saturation(rgb_r, 5)
    for func in parse_operations(ops):
        rgb_r = func(rgb_r)
    
    show(rgb_r, transform=src.transform, ax=ax)
    

ax.yaxis.set_major_locator(ticker.MultipleLocator(10e3))
ax.xaxis.set_major_formatter(m2km)
ax.yaxis.set_major_formatter(m2km)



cmap = plt.get_cmap('viridis').copy()
divider = make_axes_locatable(ax) #ax for distribution plot and cbar

cax = divider.append_axes("right", size="2%", pad=0.1)
axright = divider.append_axes("right", size=1.2, pad=1.2) 

k_min = convex_hull_df2['keel_depth'].min()
k_min = 46.79093160562594
k_max = 485.76918830202516
cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin = k_min,vmax = k_max), cmap=cmap),
              cax=cax,label='Keel Depth (m)')

cbar.set_label(label='Keel Depth (m)',fontsize=20, labelpad=15)
ax.tick_params(axis='both', which='major', labelsize=20,pad=15)
cbar.ax.tick_params(labelsize=20)
cbar.locator = ticker.MultipleLocator(50)

# ext = [src.bounds[0],src.bounds[2], src.bounds[1], src.bounds[3]]
convex_hull_df2.plot(ax=ax,column='keel_depth',cmap=cmap)
convex_hull_df2['keel_binned'].value_counts().sort_index().plot(kind='barh',logx=True,ax=axright,
                                                           edgecolor = 'k',zorder=2)


ax.yaxis.set_major_locator(ticker.MultipleLocator(5e3))
# ax.set_ylim(7591e3, 7617e3)
# ax.set_xlim(500e3, 535e3)
ax.set_ylabel('Northing (km)', size=20)
ax.set_xlabel('Easting (km)', size=20)
ax.set_title('Sermeq Kujalleq 2020-08-16', size=20)

axright.set(ylabel=None)
axright.yaxis.tick_right()
# axright.yaxis.set_ticks(np.arange(50, 1450, 200))

ticks = axright.yaxis.get_ticklocs()
ticklabels = [l.get_text() for l in axright.yaxis.get_ticklabels()]
axright.set_yticks(ticks[1::2])
axright.yaxis.set_ticklabels(ticklabels[1::2])
axright.tick_params(axis='both', which='major', labelsize=20)
axright.set_xlim(0, 1000)

ticks_x = axright.xaxis.get_ticklocs()
ticklabels_x = [l.get_text() for l in axright.xaxis.get_ticklabels()]
axright.set_xticks(ticks_x[2::3])
axright.xaxis.set_ticklabels(ticklabels_x[2::3])


axright.axhspan(4,28,facecolor='0.6',alpha=0.3, zorder=1)


text_dict = {'fontsize':20,
             'fontweight': 'bold'}
text_label = ax.text(.01, .99, 'b', ha='left', va='top', transform=ax.transAxes, **text_dict)

text_label.set_bbox(dict(facecolor='white', alpha=0.6, linewidth=0))


op = '/media/laserglaciers/upernavik/iceberg_py/figs/'
# fig.savefig(f'{op}kanger_20170710T141009.png',dpi=300,transparent=False)
# fig.savefig(f'{op}jkb_2020-08-02.pdf',dpi=300,transparent=False, bbox_inches='tight')