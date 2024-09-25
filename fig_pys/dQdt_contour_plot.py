#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:49:43 2024

@author: laserglaciers
"""

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from matplotlib import cm, ticker
from matplotlib import ticker, colors
from matplotlib.ticker import ScalarFormatter
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

fig, ax = plt.subplots()
CS = ax.contourf(dTFX, dtY, dQ_dt, levels = levels, norm=colors.LogNorm(),
                 cmap='cividis', extend='both')


cbar = fig.colorbar(CS, ticks=levels, 
                    format=ticker.FixedFormatter(levels)
                    )

cbar.formatter = ScalarFormatter(useMathText=True)
cbar.formatter.set_scientific(True)
cbar.ax.yaxis.set_offset_position('left')

cbar.ax.set_ylabel(r'Q$_{aw}$ (W)')
# cbar.ax.set_ylim(levels[0],levels[-3])

# ax.clabel(CS)
ax.set_xlabel('Ocean Thermal Forcing (C)')
ax.set_ylabel('Flushing Time (Days)')

ax.set_title('Helheim Fjord')


# calc helheim 

constant_tf_55 = 5.5 # from Slater 2022 nature geoscience
constant_tf_67 = 6.67 # from Slater 2022 nature geoscience
constant_tf_8 = 7.62 # from Slater 2022 nature geoscience

dQ_dt_HEL_55 = psw * csw * ( (Volume_test * constant_tf_55) / (50 * day2sec) )
dQ_dt_HEL_67 = psw * csw * ( (Volume_test * constant_tf_67) / (50 * day2sec) )
dQ_dt_HEL_8 = psw * csw * ( (Volume_test * constant_tf_8) / (50 * day2sec) )


# ax.scatter(constant_tf_55, 50)
# ax.scatter(constant_tf_67, 50)
# ax.scatter(constant_tf_8, 50)


op = '/media/laserglaciers/upernavik/iceberg_py/figs/'
fig.savefig(f'{op}helheim_fjord_Qaw_contour.png')
