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
    

def melt_forcedwater(temp_far, salinity_far, pressure_base, U_rel):
    
    # constants from Jenkins
    a = -5.73e-2 # Salinity contribution
    b = 8.32e-2 # constant
    c = -7.61e-4 # pressure contribution C/dbar
    
    GT = 1.1e-3 # 6e-4; %econd value in Silva, first in Jenkins model heat transfer coefficient
    GS = 3.1e-5 # 2.2e-5; %second value in Silva, first in Jenkins model % salt transfer coefficient
    L = 3.35e5  # latent heat of fusion of ice (in J/kg)
    cw = 3974;  # specific heat of water (J/kg/C)
    ci = 2009;  # " of ice
    DT = 15;    # temp difference between iceberg core and bottom surface (this is for Antarctica, maybe less for Greenland?)
    
    T_fp = a * salinity_far + b + c * pressure_base # Freezing point temperature
    T_sh = temp_far - T_fp # Temperature above freezing point
    
    # Quadtratic Terms
    A = (L + DT * ci) / (U_rel * GT * cw)
    B = -((L + DT * ci)) *GS/(GT * cw) -a*salinity_far - T_sh
    C = -U_rel * GS * T_sh
    
    ROOT = np.power(B,2) - 4*A*C
    ROOT = np.where(ROOT<0,np.nan,ROOT)
    
    # Find quadratic roots
    Mtemp1 = (-B + np.sqrt(ROOT)) / (2*A)
    Mtemp2 = (-B - np.sqrt(ROOT)) / (2*A)
    
    melt = np.minimum(Mtemp1,Mtemp2)
    
    # clean data to remove melt rates below freezing point
    mask = T_sh < 0
    melt[mask] = 0
    nan_mask = np.isnan(melt)
    melt[nan_mask] = 0
    
    
    return melt, T_sh, T_fp


def melt_forcedair(T_air, U_rel, L):
    
    T_ice = -4 # ice temperature
    Li = 3.33e5 # latent heat of fusion in ice J/kg
    rho_i = 900 # density of ice
    air_viscosity = 1.46e-5 # kinematic viscosity of air
    air_diffusivity = 2.16e-5 # thermal diffusivity of air
    air_conductivity = 0.0249 # thermal condictivity of air
    cold = T_air < 0 # freezing celsius
    
    Pr = air_viscosity / air_diffusivity # Prandtl number
    Re = np.abs(U_rel) * L / air_viscosity # Reynolds number based on relative air speed
    Nu = 0.058 * (np.power(Re,0.8)) / (np.power(Pr,0.4))
    HF = (1/L) * (Nu * air_conductivity * (T_air - T_ice)) # HEAT FLUX
    
    melt = HF / (rho_i * Li)
    melt[cold] = 0
    
    return melt


def melt_buoyantwater(T_w, S_w, method):
    
    Tf = -0.036 - (0.0499 * S_w) - (0.00011128 * np.power(S_w,2)) # freezing pt of seawater due to S changes
    Tfp = Tf * np.exp(-0.19 * (T_w - Tf)) # freezing point temperature
    
    if method == 'bigg':
        dT = T_w
        mday = 7.62e-3 * dT + 1.3e-3 * np.power(dT,2) 
        
    elif method == 'cis':
        dT = T_w - Tfp
        mday = 7.62e-3 * dT + 1.29e-3 * np.power(dT,2)
        
    melt = mday / 86400
    
    return melt

def keeldepth(L, method):
    
    L_10 = round(L/10) * 10 # might have issues with this
    
    barker_La = L_10 <= 160
    hotzel_La = L_10 > 160
    
    # if nargin==1
    #     method = 'auto' # idk what this is from
    
    if method == 'auto':
        # not sure about this
        keel_depth_h = 3.78  * np.power(hotzel_La,0.63) # hotzel
        keel_depth_b = 2.91 * np.power(barker_La,0.71) # barker
        
        return keel_depth_b,keel_depth_h
        
    elif method == 'barker':
        keel_depth = 2.91 * np.power(L_10,0.71)
        
        return keel_depth
    
    elif method == 'hotzel':
        keel_depth = 3.78 * np.power(L_10,0.63)
    
        return keel_depth
    
    elif method == 'constant':
        keel_depth = 0.7 * L_10
        
        return keel_depth
    
    elif method == 'mean':
        
        keel_arr = np.ones(len(L_10),4)

        keel_arr[barker_La,0] = 2.91 * np.power(barker_La,0.71) # barker # feel like these should just be ind columns?
        keel_arr[hotzel_La,0] = 3.78 * np.power(hotzel_La,0.63) # hotzel
        keel_arr[:,1] = 2.91 * np.power(L_10,0.71)
        keel_arr[:,2] = 3.78 * np.power(L_10,0.63)
        keel_arr[:,3] = 0.7 * L_10
        
        mean = np.mean(keel_arr, axis=1)
        
        keel_depth = mean
        
        return keel_depth


def barker_carea(L, keel_depth, dz, LWratio):
    
    
    
    return

def init_iceberg_size():
    
    
    
    return


"""
NEED TO CODE
keeldepth - done
init_iceberg_size
barker_carea
test
"""