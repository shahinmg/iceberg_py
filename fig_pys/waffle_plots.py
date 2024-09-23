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

kanger_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/kanger/Qib/20170710T141009_high_kanger_coeff_4.nc'
helheim_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/Qib/20230727T142031_high_helheim_coeff_4_test.nc'


kanger_ds = xr.open_dataset(kanger_path)
helheim_ds = xr.open_dataset(helheim_path)


ice_blue = '#baf2ef'
light_red = '#FF7276'

data = pd.DataFrame(
    {
        'labels': ['Q$_{ib}$', 'Available Q$_{aww}$'],
        'Kanger': [kanger_ds.Qib.data/1e10, (kanger_ds.Qaww_high.data/1e10)-(kanger_ds.Qib.data/1e10)],
        'Helheim': [helheim_ds.Qib.data/1e10, (helheim_ds.Qaww_high.data/1e10)-(helheim_ds.Qib.data/1e10)]
        },
).set_index('labels')



fig = plt.figure(
    FigureClass=Waffle,
    plots={
        311: {
            'values': data['Kanger'],  # Convert actual number to a reasonable block number
            'labels': [f"{k}" for k, v in data['Kanger'].items()],
            'legend': {'loc': 'upper left', 'bbox_to_anchor': (1.05, 1), 'fontsize': 10},
            'title': {'label': 'Kanger Heat Budget', 'loc': 'left', 'fontsize': 12}
        },
        312: {
            'values': data['Helheim'],
            # 'labels': [f"{k}" for k, v in data['Helheim'].items()],
            'legend': {'loc': 'upper left', 'bbox_to_anchor': (1.2, 1), 'fontsize': 8},
            'title': {'label': 'Helheim Heat Budget', 'loc': 'left', 'fontsize': 12}
        },
    },
    rows=4,  # Outside parameter applied to all subplots, same as below
    # cmap_name="Accent",  # Change color with cmap
    colors=[ice_blue, light_red],
    rounding_rule='ceil',  # Change rounding rule, so value less than 1000 will still have at least 1 block
    figsize=(6, 5)
)

fig.supxlabel('1 block = 10 GW', fontsize=8, x=0.14,y=.35)

op = '/media/laserglaciers/upernavik/iceberg_py/figs/melange_heatsink_waffle.png'

# plt.savefig(op, dpi=300)

