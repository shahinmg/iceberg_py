#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:30:59 2023

@author: laserglaciers
"""

import matplotlib.pyplot as plt
from pywaffle import Waffle
import xarray as xr
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib import cm
import numpy as np


kanger_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/kanger/Qib/20170710T141009_high_kanger_coeff_4.nc'
helheim_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/Qib/20230727T142031_high_helheim_coeff_4_test.nc'


kanger_ds = xr.open_dataset(kanger_path)
helheim_ds = xr.open_dataset(helheim_path)

black = 'k'
ice_blue = '#baf2ef'
light_red = '#FF7276'
colors = cm.coolwarm(np.linspace(0, 1, 4))


data = pd.DataFrame(
    {
        'labels': ['Q$_{ib}$ lower bound', 'Q$_{ib}$ upper bound','Remaining Q$_{aww}$'],
        'Kanger': [1, kanger_ds.Qib.data/1e10 - 1, (kanger_ds.Qaww_high.data/1e10)-(kanger_ds.Qib.data/1e10)-1],
        'Kanger_perc': [5, 21, '79 - 95'],
        'Helheim': [1, helheim_ds.Qib.data/1e10 , (helheim_ds.Qaww_high.data/1e10)-(helheim_ds.Qib.data/1e10)-1]
        },
).set_index('labels')


fig = plt.figure(
    FigureClass=Waffle,
    plots={
        312: {
            'values': data['Kanger'],
            # 'labels': [f"{k}" for k, v in data['Kanger_perc'].items()],
            # 'legend': {'loc': 'upper left', 'bbox_to_anchor': (1, 1),
            #            "ncol":1, 'markerscale':50, 'fontsize': 20,
            #            'labelspacing':1, },
            'title': {'label': 'Kangersertuaq Heat Budget', 'loc': 'left', 'fontsize': 30,'pad':20}
        },
    },
    rows=4,  # Outside parameter applied to all subplots, same as below
    # cmap_name="Accent",  # Change color with cmap
    colors=[colors[0], colors[1], light_red],
    rounding_rule='ceil',  # Change rounding rule, so value less than 1000 will still have at least 1 block
    figsize=(20, 15)
)
# fig.legendHandles[0]._legmarker.set_markersize(15)
# fig.supxlabel('1 block = 10 GW', fontsize=20, x=0.14,y=.33)


op = '/media/laserglaciers/upernavik/melange_heatsink_manuscript/figures/kanger_melange_heatsink_waffle.png'

# plt.savefig(op, dpi=300, transparent=False)

