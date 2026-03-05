#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 15:04:47 2025

@author: laserglaciers
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Ellipse
from matplotlib import cm,colors
import os
import pandas as pd

fig, ax1 = plt.subplots(1,1, figsize=(8,4))
# Data for the bar plot
categories = [r'$\Delta t$ 50 days', r'$\Delta t$ 150 days']
Qaw_150dt = [100, 110, 110, 110, 100] #from tables
Qaw_50dt = [300, 330, 320, 330, 310]

values = [np.mean(Qaw_50dt), np.mean(Qaw_150dt)] # 50 day dt, 150 day dt

# Create the bar plot with hollow bars
ax1.barh(categories, values, 
        edgecolor='tab:red', # Set the edge color to black (or any desired color)
        facecolor='none',  # Make the bars hollow by setting facecolor to 'none'
        linewidth=2, # Adjust the line width of the edges if needed
        zorder=2,
        label=r'$Q_{aw}$',
        height=0.8)      

values_qib = [25, 25]
# Optionally fill the bars with a color
hbars = ax1.barh(categories, values_qib, 
        color='tab:blue',  # Set the fill color 
        alpha=1, # Adjust transparency if needed
        zorder=1,
        label=r'$Q_{ib}$',
        height=0.8)       


# Custom labels 
df_50 = pd.read_csv('../data/csv/AVG_dt50.csv')
df_150 = pd.read_csv('../data/csv/AVG_dt150.csv')

df_50_perc = np.round(df_50['percentage'].astype(np.float64).mean())
df_150_perc = np.round(df_150['percentage'].astype(np.float64).mean())

perc_values = [f'{df_50_perc:.0f}', f'{df_150_perc:.0f}']
latex_label = r'$Q_{ib}$/$Q_{aw}$'

custom_labels = [f"{v}%" for v in perc_values]
ax1.bar_label(hbars, labels=custom_labels, label_type='edge', padding=3, size=14)
# plt.bar_label(hbars, fmt='%.2f')
# Add labels and title
ax1.set_xlabel(r'Atlantic Water Heat Flux $Q_{aw}$ (GW)', size=14)
# plt.ylabel('Values')
# ax1.set_title('Proportion of Atlantic Water Heat Extracted by Icebergs' , size=15)

# Hide the top and right spines (frame)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.tick_params(axis="x", direction='in', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)

ax1.legend(fontsize=16)

op = f'./figs/'
if not os.path.exists(op):
    os.makedirs(op)

plt.tight_layout()
plt.savefig(f'{op}figure_6.pdf', dpi=300, transparent=True)
