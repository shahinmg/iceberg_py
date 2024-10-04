#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 09:48:25 2024

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
import matplotlib
#%%
#figsize = 6 4
fig, ax = plt.subplots(1,2, figsize=(14,4))

#%%

helheim_urel_05_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/Qib_melt_flux_v2/coeff_4_urel05/'

nc_list = [nc for nc in os.listdir(helheim_urel_05_path) if nc.endswith('nc')]
os.chdir(helheim_urel_05_path)


dt = 50

psw = 1024 #kg m3
csw = 3974 #J kg-1 C-1
day2sec = 86400
# Volume_test = 28e3 * 5e3 * 300 #km3 Helheim Fjord test #148461041 area from polygon
Volume_test = 148461041 * 300 #km3 Helheim Fjord test #148461041 area from polygon


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
helheim_urel_05_c1_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/Qib_melt_flux_v2/coeff_1_urel05/'

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
colors_waffle = cm.coolwarm(np.linspace(0, 1, 4))


total_heat = (Qaw_HEL_67/1e10)
Qib_coef_1_heat = (Qib_array_c1_mean/1e10)
Qib_coef_4_heat = (Qib_array_c4_mean/1e10)

remaining_heat = total_heat - Qib_coef_4_heat

Qib_coef_1_perc = (Qib_coef_1_heat/total_heat) * 100
Qib_coef_4_perc = (Qib_coef_4_heat/total_heat) * 100

remaining_Qaw_1 = total_heat - Qib_coef_1_heat
remaining_Qaw_4 = total_heat - Qib_coef_4_heat


Qib_1_label = r'Q$_{ib}$ $\Gamma_{S,T}$ x1'
Qib_4_label = r'Q$_{ib}$ $\Gamma_{S,T}$ x4'

data = pd.DataFrame(
    {
        'labels': [f'{Qib_1_label} ({Qib_coef_1_perc:.1f} %)', f'{Qib_4_label} ({Qib_coef_4_perc:.1f} %)','Remaining Q$_{aw}$'],
        'Helheim': [Qib_coef_1_heat, Qib_coef_4_heat, remaining_heat]
        },
).set_index('labels')


# Waffle.make_waffle(
#     ax=ax[0],  # pass axis to make_waffle
#     rows=5, 
#     columns=10, 
#     values=[30, 16, 4], 
#     title={"label": "Waffle Title", "loc": "left"}
# )


Waffle.make_waffle(
    ax=ax[0],

    values = data['Helheim'],
    labels = [f"{k}" for k, v in data['Helheim'].items()],
    legend ={'loc': 'lower left',  'bbox_to_anchor': (0, -0.3), 
               'fontsize': 12,  'ncol': 2},
    title = {'label': 'Helheim Fjord Heat Budget', 'loc': 'left', 'fontsize': 18,'pad':10},

    rows=4,  # Outside parameter applied to all subplots, same as below
    # cmap_name="Accent",  # Change color with cmap
    colors=[colors_waffle[0],colors_waffle[1], light_red],
    rounding_rule='ceil',  # Change rounding rule, so value less than 1000 will still have at least 1 block
    figsize=(20, 15)
)


# fig.supxlabel('1 block = 10 GW', fontsize=20, x=0.07,y=.35)




#%%

psw = 1024 #kg m3
csw = 3974 #J kg-1 C-1
day2sec = 86400
# fixed volume vary flushing and thermal forcing

dt = np.arange(20,365,1) #flushing rate
# dt = np.logspace(1, 365, 1)
TF = np.linspace(5, 8, dt.shape[0]) #thermal forcing from Slater histogram TF of Helheim

dTFX, dtY = np.meshgrid(TF, dt)

# Volume_test = 23e3 * 5e3 * 300 #km3 Helheim Fjord test
Volume_test = 148461041 * 300 # area from Helheim fjord shapefile ~ 28 km length

dQ_dt = psw * csw * ( (Volume_test * dTFX) / (dtY * day2sec) )


# levels_log = np.logspace(np.log10(dQ_dt.min()),np.log10(dQ_dt.max()), 10) #https://stackoverflow.com/questions/65823932/plt-contourf-with-given-number-of-levels-in-logscale
# levels = np.arange(dQ_dt.min(), dQ_dt.max(), 10)
# levels = np.linspace(dQ_dt.min(), dQ_dt.max(), 20)
levels = np.logspace(np.log10(dQ_dt.min()),np.log10(dQ_dt.max()), 10) #https://stackoverflow.com/questions/65823932/plt-contourf-with-given-number-of-levels-in-logscale


# levels = [0.  , 0.04, 0.08, 0.12, 0.16, 0.2 , 0.24, 0.28, 0.32]
# levels = np.arange(2e4, 5e5, 0.2e5)


CS = ax[1].contourf(dTFX, dtY, dQ_dt, levels = levels, norm=colors.LogNorm(),
                 cmap='cividis', extend='both')

kwargs = {'format': '%.1f'}

# https://www.tutorialspoint.com/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
class ScalarFormatterClass(ScalarFormatter):
   def _set_format(self):
      self.format = "%1.2f"

yScalarFormatter = ScalarFormatterClass(useMathText=True)

cbar = fig.colorbar(CS, ticks=levels)
                    # format=ticker.FixedFormatter(levels)
                    # )
                    
# cbar.formatter = ScalarFormatter(useMathText=True)
# cbar.formatter.set_scientific(True)

cbar.ax.yaxis.set_major_formatter(yScalarFormatter)

cbar.ax.yaxis.set_offset_position('left')


cbar.ax.set_ylabel(r'Q$_{aw}$ (W)', fontsize=15)
cbar.ax.tick_params(labelsize=12) 

# ax.clabel(CS)
ax[1].set_xlabel('Ocean Thermal Forcing ($^{\circ}$C)', size=15)
ax[1].set_ylabel('Flushing Time (Days)', size=15)
ax[1].tick_params(axis='both', which='major', labelsize=12)
ax[1].set_title('Helheim Fjord Q$_{aw}$ Space', size=18)


# calc helheim 

constant_tf_55 = 5.5 # from Slater 2022 nature geoscience
constant_tf_67 = 6.67 # from Slater 2022 nature geoscience
constant_tf_8 = 7.62 # from Slater 2022 nature geoscience

dQ_dt_HEL_55 = psw * csw * ( (Volume_test * constant_tf_55) / (50 * day2sec) )
dQ_dt_HEL_67 = psw * csw * ( (Volume_test * constant_tf_67) / (50 * day2sec) )
dQ_dt_HEL_8 = psw * csw * ( (Volume_test * constant_tf_8) / (50 * day2sec) )


# 'bbox_to_anchor': (0, -0.3)
text_dict = {'fontsize':20,
             'fontweight': 'bold'}
text_label = ax[0].text(.93, -0.2, 'a', ha='left', va='bottom', 
                        transform=ax[0].transAxes, **text_dict)

text_label = ax[1].text(1.05, -0.2, 'b', ha='left', va='bottom', 
                        transform=ax[1].transAxes, **text_dict)

text_label.set_bbox(dict(facecolor='white', alpha=0.6, linewidth=0))

op = '/media/laserglaciers/upernavik/iceberg_py/figs/'
fig.savefig(f'{op}Helheim_fjord_waffle_plot_Qaw_parameter_space.pdf', dpi=300, transparent=True, bbox_inches='tight')
fig.savefig(f'{op}Helheim_fjord_waffle_plot_Qaw_parameter_space.png', dpi=300, transparent=False, bbox_inches='tight')





