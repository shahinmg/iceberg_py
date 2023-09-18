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
    
    fig, ax = plt.subplots()
    
    dz = iceberg.dz
    keeli = iceberg.keeli
    dzk = iceberg.dzk # thickness of the last layer
    
    # plot above water geometry
    xx = [-iceberg.L/2, iceberg.L/2, iceberg.L/2, -iceberg.L/2, -iceberg.L/2]
    yy = [0, 0, iceberg.freeB, iceberg.freeB, 0]
    
    xk = min(xx).data * np.ones((2,1))
    yk = [max(yy), min(yy)]
    ax.plot(xx,yy,color='b')
    for i in range(int(keeli)-1):
        
        dt = iceberg.uwL.isel(Z=i) / 2
        xx = np.array([float(-dt.data),float(dt.data),float(dt.data),
              float(-dt.data), float(-dt.data)])
        
        yy = -1*np.array([float(iceberg.depth.isel(Z=i)-dz), float(iceberg.depth.isel(Z=i)-dz), 
                 float(iceberg.depth.isel(Z=i)), float(iceberg.depth.isel(Z=i)),
                 float(iceberg.depth.isel(Z=i)-dz)])
        
        xk = [xk, np.min(xx) * np.ones((2,1))]
        yk = [yk, [np.max(yy), np.min(yy)]]
        
        ax.plot(xx,yy,color='b')
        
    
    dt = iceberg.uwL.isel(Z=int(keeli)) / 2
    xx = np.array([float(-dt.data),float(dt.data),float(dt.data),
          float(-dt.data), float(-dt.data)])
        
    yy = -1*np.array([float(iceberg.depth.isel(Z=i)-dz), float(iceberg.depth.isel(Z=i)-dz), 
             float(iceberg.depth.isel(Z=i)), float(iceberg.depth.isel(Z=i)),
             float(iceberg.depth.isel(Z=i)-dz)])
    
    xk = [xk, np.min(xx) * np.ones((2,1))]
    yk = [yk, [np.max(yy), np.min(yy)]]
    
    # shape = np.zeros(())
    
    ax.plot(xx,yy,color='b')
    
        
    
    
    