#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:36:55 2024

@author: laserglaciers
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import gaussian_filter
from matplotlib import colors, cm, ticker
import matplotlib as mpl
import scipy.io as sio
import xarray as xr
import pickle
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from matplotlib.ticker import ScalarFormatter
import xarray as xr
import os
from pywaffle import Waffle


helheim_urel_05_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/Qib/u_rel_05_factor4_melt_rates/'

nc_list = [nc for nc in os.listdir(helheim_urel_05_path) if nc.endswith('nc')]
os.chdir(helheim_urel_05_path)


dt = 50

psw = 1024 #kg m3
csw = 3974 #J kg-1 C-1
day2sec = 86400
Volume_test = 28e3 * 5e3 * 300 #km3 Helheim Fjord test
# constant_tf_55 = 5.73 # from Slater 2022 nature geoscience
constant_tf_67 = 6.67 # from Slater 2022 nature geoscience
# constant_tf_8 = 7.62 # from Slater 2022 nature geoscience

Qaw_HEL_67 = psw * csw * ( (Volume_test * constant_tf_67) / (dt * day2sec) )

Qib_array_c4 = np.ones(len(nc_list))
for i,nc in enumerate(nc_list):
    
    Qib_ds = xr.open_dataset(nc)
    Qib_values = Qib_ds.Qib.data
    
    Qib_array_c4[i] = Qib_values
    # axs.scatter(aww_vol, 6.67)

Qib_percs_c4 = Qib_array_c4/Qaw_HEL_67

#%%
helheim_urel_05_c1_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/Qib/u_rel_05_factor1_melt_rates/'

nc_list_c1 = [nc for nc in os.listdir(helheim_urel_05_c1_path) if nc.endswith('nc')]
os.chdir(helheim_urel_05_c1_path)


Qib_array_c1 = np.ones(len(nc_list_c1))
for i,nc in enumerate(nc_list_c1):
    
    Qib_ds = xr.open_dataset(nc)
    Qib_values = Qib_ds.Qib.data
    
    Qib_array_c1[i] = Qib_values
    # axs.scatter(aww_vol, 6.67)

Qib_percs_c1 = Qib_array_c1/Qaw_HEL_67

#%%

Qib_array_c1_mean = Qib_array_c1.mean()
Qib_array_c4_mean = Qib_array_c4.mean()

black = 'k'
ice_blue = '#baf2ef'
light_red = '#FF7276'
colors = cm.coolwarm(np.linspace(0, 1, 4))


total_heat = (Qaw_HEL_67/1e10)
Qib_coef_4_heat = (Qib_array_c4_mean/1e10)
Qib_coef_1_heat = (Qib_array_c1_mean/1e10)
remaining_heat = total_heat - Qib_coef_4_heat - Qib_coef_1_heat

data = pd.DataFrame(
    {
        'labels': [r'Q$_{ib}$ $\Gamma_{S,T}$ x1', 'Q$_{ib}$ $\Gamma_{S,T}$ x4','Remaining Q$_{aww}$'],
        'Helheim': [Qib_coef_1_heat, Qib_coef_4_heat, remaining_heat]
        },
).set_index('labels')

fig = plt.figure(
    FigureClass=Waffle,
    plots={
        312: {
            'values': data['Helheim'],
            'labels': [f"{k}" for k, v in data['Helheim'].items()],
            'legend': {'loc': 'lower left',  'bbox_to_anchor': (0, -0.4), 
                       'fontsize': 20,  'ncol': 3},
            'title': {'label': 'Helheim Fjord Heat Budget', 'loc': 'left', 'fontsize': 30,'pad':20}
        },
    },
    rows=4,  # Outside parameter applied to all subplots, same as below
    # cmap_name="Accent",  # Change color with cmap
    colors=[colors[0],colors[1], light_red],
    rounding_rule='ceil',  # Change rounding rule, so value less than 1000 will still have at least 1 block
    figsize=(20, 15)
)

fig.supxlabel('1 block = 10 GW', fontsize=20, x=0.07,y=.35)

op = '/media/laserglaciers/upernavik/melange_heatsink_manuscript/figures/helheim_melange_heatsink_waffle.pdf'
plt.tight_layout()
# plt.savefig(op, dpi=300, transparent=False,bbox_inches='tight')

