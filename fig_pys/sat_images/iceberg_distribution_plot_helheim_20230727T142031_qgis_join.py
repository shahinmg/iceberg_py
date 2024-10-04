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

s2_path = '/media/laserglaciers/upernavik/sentinel_2/helheim/S2A_MSIL1C_20230727T142031_N0509_R096_T24WWU_20230727T193820.tif'
convex_hull_path = '/media/laserglaciers/upernavik/segment-anything/prediction_geoms/helheim/20230727T142031/20230727T142031_merged_v2.gpkg'
bounding_box_path = '/media/laserglaciers/upernavik/segment-anything/prediction_geoms/helheim/20230727T142031/20230727T142031_merged_bounding_box.gpkg'
mbergs_dict = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/berg_model/20230727T142031_bergs_v2.pkl'
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
fig, ax = plt.subplots(figsize=(16.77,8.86))

def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))

kwargs = {'extent':(3e5, 3.35e5, -2.585e6, -2.570e6)}

with rasterio.open(s2_path) as src:

    rgb_r = src.read([1,2,3])
    rgb_r = adjust_band(rgb_r)
    vmin, vmax = np.nanpercentile(rgb_r, (5,95))  # 5-95% stretch
    
    show(rgb_r, transform=src.transform,  vmin=vmin, vmax=vmax,ax=ax)

ax.set_xlim((3.0e5, 3.35e5))
ax.set_ylim((-2.59e6, -2.57e6))

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

# ctd_df.plot(ax=ax,color='red',edgecolor='k',markersize=35)
# plt.tight_layout()
# op = f'/media/laserglaciers/upernavik/ghawk_2023/figs/iceberg_dist.png'
# plt.savefig(op, dpi=300)


# op = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/20230727T142031_icebergs_helheim.pkl'
# with open(op,'wb') as handle:
#     pickle.dump(convex_hull_df2, handle, protocol=pickle.HIGHEST_PROTOCOL)

op_df = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/2023-07-27-icebergs_helheim_keel_depth.gpkg'
convex_hull_df_out = convex_hull_df2
convex_hull_df_out['keel_depth'] = convex_hull_df_out.keel_depth.to_numpy()
convex_hull_df_out.to_file(op_df, driver='GPKG')