#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:07:00 2023

@author: laserglaciers
"""

import matplotlib.pyplot as plt
import os
import xarray as xr
import numpy as np


def plot_icebergshape(iceberg):
    
    fig, ax = plt.subplots(figsize=(5,3.8))
    
    dz = iceberg.dz
    # dz = 50
    keeli = iceberg.keeli
    dzk = iceberg.dzk # thickness of the last layer
    
    # plot above water geometry
    xx = [-iceberg.L/2, iceberg.L/2, iceberg.L/2, -iceberg.L/2, -iceberg.L/2]
    yy = [0, 0, iceberg.freeB, iceberg.freeB, 0]
    
    xk = min(xx).data * np.ones((2,1))
    yk = [max(yy), min(yy)]
    ax.plot(xx,yy,color='b',linewidth=1)
    for i in range(int(keeli)-1):
        
        dt = iceberg.uwL.isel(Z=i) / 2
        xx = np.array([float(-dt.data),float(dt.data),float(dt.data),
              float(-dt.data), float(-dt.data)])
        
        yy = -1*np.array([float(iceberg.depth.isel(Z=i)-dz), float(iceberg.depth.isel(Z=i)-dz), 
                 float(iceberg.depth.isel(Z=i)), float(iceberg.depth.isel(Z=i)),
                 float(iceberg.depth.isel(Z=i)-dz)])
        
        xk = [xk, np.min(xx) * np.ones((2,1))]
        yk = [yk, [np.max(yy), np.min(yy)]]
        

        ax.plot(xx,yy,color='b',linewidth=1)
        # if i%10 == 0:
        #     print(i)
        #     ax.plot(xx,yy,color='b',linewidth=1)
    
    dt = iceberg.uwL.isel(Z=int(keeli)) / 2
    xx = np.array([float(-dt.data),float(dt.data),float(dt.data),
          float(-dt.data), float(-dt.data)])
        
    yy = -1*np.array([float(iceberg.depth.isel(Z=i)-dz), float(iceberg.depth.isel(Z=i)-dz), 
             float(iceberg.depth.isel(Z=i)), float(iceberg.depth.isel(Z=i)),
             float(iceberg.depth.isel(Z=i)-dz)])
    
    xk = [xk, np.min(xx) * np.ones((2,1))]
    yk = [yk, [np.max(yy), np.min(yy)]]
    
    # shape = np.zeros(())
    
    ax.plot(xx,yy,color='b',linewidth=1)
    ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.set_title(f'Initial {int(iceberg.L)} m Length Geometry',fontsize=20)
    op = f'/media/laserglaciers/upernavik/ghawk_2023/figs/{int(iceberg.L)}_iceberg_geom_no_title.png'
    plt.tight_layout()
    plt.savefig(op, dpi=300)
    # ax.set_xlim(500, -500)
    # ax.set_ylim(-400,75)
        
    
    
    