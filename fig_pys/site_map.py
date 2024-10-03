#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 15:58:17 2024

@author: laserglaciers
"""

import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
import geopandas as gpd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker 

ocean_tf_path = '/media/laserglaciers/upernavik/iceberg_py/fig_pys/ocean_tf.tif'
basins_path = '/media/laserglaciers/upernavik/iceberg_py/geoms/basins/mel_heatsink_basins.gpkg'
bedmachine_mask = '/media/laserglaciers/upernavik/iceberg_py/bed_mask/bedmachine_mask.tif'
gl_coast_path = '/media/laserglaciers/upernavik/iceberg_py/geoms/gl_canada_coast.shp'

labelsize = 23
m2km = lambda x, _: f'{x/1000:g}'
fig, ax = plt.subplots(figsize=(15,10))


# Define Colors 
white = '#ffffff'
brown = '#8e5216'
gray = '#bcbcbc'

c_list = [brown, white, white, gray]

cmap_name = 'gl_mask'
cm_gl = LinearSegmentedColormap.from_list(
        cmap_name, c_list, N=len(c_list))

bed_mask = rasterio.open(bedmachine_mask)
ocean_tf = rasterio.open(ocean_tf_path)
gl_coast = gpd.read_file(gl_coast_path)


bed_mask.meta['nodata'] = 0
bed_mask_1 = bed_mask.read(1)
bed_mask_1 = np.where(bed_mask_1==0, np.nan, bed_mask_1)
basins = gpd.read_file(basins_path)


vmin, vmax = 0.5, 8
cmap = plt.get_cmap('magma').copy()

basins.plot(ax=ax,facecolor='none', edgecolor='k',linestyle='--',linewidths=0.5)
gl_coast.plot(ax=ax,facecolor='none', edgecolor='k', linewidths=0.5)

tf_plot = show(ocean_tf, transform=ocean_tf.transform, cmap='magma', ax=ax, alpha=0.9, vmin=vmin, vmax=vmax)
mask_plot = show(bed_mask_1, transform=bed_mask.transform, cmap=cm_gl, ax=ax)

divider = make_axes_locatable(ax)

# Create an axes for the colorbar on top
cax = divider.append_axes("bottom", size="5%", pad=0.4)

cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin = vmin,vmax = vmax), cmap=cmap),
              cax=cax, orientation="horizontal")

celcius = r'$^{\circ}$C'
cbar.set_label(label=f'Ocean Thermal Forcing ({celcius})', fontsize = 18)


ax.xaxis.set_major_locator(ticker.MultipleLocator(500e3)) 
ax.yaxis.set_major_locator(ticker.MultipleLocator(1000e3)) 

ax.xaxis.set_major_formatter(m2km)
ax.yaxis.set_major_formatter(m2km)
ax.xaxis.label.set_size(labelsize)
ax.yaxis.label.set_size(labelsize)
# ax.set_yticklabels([])
# ax.set_xticklabels([])

# plt.rcParams["figure.figsize"] = [2.5 * i for i in plt.rcParams["figure.figsize"]] #https://stackoverflow.com/questions/28617263/matplotlib-set-width-or-height-of-figure-without-changing-aspect-ratio

text_dict = {'fontsize':20,
             'fontweight': 'bold'}
text_label = ax.text(.01, .99, 'c', ha='left', va='top', transform=ax.transAxes, **text_dict)

text_label.set_bbox(dict(facecolor='white', alpha=0.6, linewidth=0))

bed_mask.close()
ocean_tf.close()

op = '/media/laserglaciers/upernavik/iceberg_py/figs/'
fig.savefig(f'{op}site_map_ocean_thermal_forcing.pdf', dpi=300, transparent=False, bbox_inches='tight')

