#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:49:43 2024

@author: laserglaciers
"""

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd


psw = 1.024 #kg km3
csw = 3974 #J kg-1 C-1
day2sec = 86400
# fixed volume vary flushing and thermal forcing

dt = np.arange(20,365,1) #flushing rate

TF = np.linspace(5.5, 8, dt.shape[0]) #thermal forcing

dTFX, dtY = np.meshgrid(TF, dt)

Volume_test = 20 * 5 * 0.3 #km3 Helheim Fjord test

dQ_dt = psw * csw * ( (Volume_test * dTFX) / (dtY * day2sec) )


# levels_log = np.logspace(np.log10(dQ_dt.min()),np.log10(dQ_dt.max()), 10) #https://stackoverflow.com/questions/65823932/plt-contourf-with-given-number-of-levels-in-logscale
# levels = np.arange(dQ_dt.min(), dQ_dt.max(), 5)
# levels = [0.  , 0.04, 0.08, 0.12, 0.16, 0.2 , 0.24, 0.28, 0.32]
# levels = np.arange(0, 0.5, 0.05)

fig, ax = plt.subplots()
CS = ax.contourf(dTFX, dtY, dQ_dt)#, levels=levels)


cbar = fig.colorbar(CS)
cbar.ax.set_ylabel('dQ/dt')

# ax.clabel(CS)
ax.set_xlabel('Ocean Thermal Forcing (C)')
ax.set_ylabel('Flushing Time (Days)')

ax.set_title('Helheim Fjord Volume Test')