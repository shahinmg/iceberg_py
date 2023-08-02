#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:00:25 2023

@author: laserglaciers
"""

import numpy as np
import  scipy



def melt_wave(windu, sst, sea_ice_conc):
    
    sea_state = 1.5 * np.sqrt(windu) + 0.1 * windu
    IceTerm = 1 + np.cos(np.power(sea_ice_conc,3) * np.pi)
    
    melt = (1/12) * sea_state * IceTerm * (sst + 2) # m/day
    
    melt = melt / 86400 # m/second 
    
    return melt


def melt_solar(solar_rad):
    
    latent_heat = 3.33e5 #J/kg
    rho_i = 917 #kg/m3
    albedo = 0.7
    absorbed = 1 - albedo
    
    melt = absorbed * solar_rad / (rho_i * latent_heat)
    
    return melt
    
