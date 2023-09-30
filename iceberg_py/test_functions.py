#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 11:54:53 2023

@author: laserglaciers
"""
import melt_functions as ice_melt
import numpy as np
import xarray as xr
import scipy.io as sio
from plot_icebergshape import plot_icebergshape
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d

L = np.arange(50,1050,50)
# L = [400]
dz = 5

#Test first L

# # ice = ice_melt.init_iceberg_size(L[0],dz=dz)
# ice_init = []
# for length in L:
#     print(f'processing {length}')
#     ice_init.append(ice_melt.init_iceberg_size(length,dz=dz))
#     print(f'finished with {length}')

ctd_path = '/media/laserglaciers/upernavik/iceberg_py/infiles/ctdSFjord.mat'
adcp_path = '/media/laserglaciers/upernavik/iceberg_py/infiles/ADCP_cosine_BeccaSummer.mat'

ctd = sio.loadmat(ctd_path)

depth = ctd['serm']['mar2010'][0][0][0][0][0]
temp = ctd['serm']['mar2010'][0][0][0][0][1]
salt = ctd['serm']['mar2010'][0][0][0][0][2]

ctd_ds = xr.Dataset({'depth':(['Z','X'], depth),
                     'temp': (['tZ','tX'], temp),
                     'salt': (['tZ','tX'], salt)
                     }
    )


adcp = sio.loadmat(adcp_path)

adcp_ds = xr.Dataset({'zadcp': (['adcpX','adcpY'],adcp['zadcp']),
                      'vadcp': (['adcpX','adcpZ'], adcp['vadcp']),
                      'tadcp': (['adcpY','adcpZ'], adcp['tadcp']),
                      'wvel':  (['adcpY'], np.array([0.05]))
    })



# elif do_constantUrel == False: # load ADCP pulse events here, based on SF ADCP data
#     Urel = np.nan * np.ones((nz,ni,nt))
#     # kki = 1 #dsearchn(Urelative.zadcp(:),ceil(ice_init(1).K));
#     kdt = cKDTree(Urelative.zadcp[:]) # https://stackoverflow.com/questions/66494042/dsearchn-equivalent-in-python
#     pq = np.ceil(ice_init[0].keel)
#     kki = kdt.query(pq)[-1]
    
#     if IceConc == 1:
#         # if sea ice conc = 100%, assume we're talking about melange and don't take out mean horizontal flow
#         vmadcp = Urelative.vadcp
            
#     else:
#         # for drifting icebergs, take out mean horizontal flow

# vmadcp = adcp_ds.vadcp - np.matlib.repmat(np.nanmean(adcp_ds.vadcp[0:kki+1,:],axis=0),len(adcp_ds.zadcp),1)

# # make zero below keel depth to be certain
# vmadcp[kki+1:,:] = 0
# vmadcp = np.abs(vmadcp) # speed
# # add in vertical velocity if any (wvel in Urelative structure)

# vmadcp = vmadcp + Urelative.wvel.values[0] * np.ones(np.shape(vmadcp)) # (right now wvel constant in time/space)

# # interpolate to Urel
# # Urel[:,0,:] = interp2d(Urelative.tadcp, Urelative.zadcp, vmadcp, np.arange(0,nt), ice_init[0].Z) # double check length of nt #interp2d will be depreciated
# interp2d_func = interp2d(Urelative.tadcp.values.flatten(), Urelative.zadcp.values.flatten(), vmadcp)
# Urel[:,0,:] = interp2d_func(np.arange(1,nt+1),ice_init[0].Z.to_numpy()) # this interpolates the Urel at specific times and depths
# interp2d = RegularGridInterpolator((Urelative.tadcp, Urelative.zadcp, vmadcp))

# interp2d_func = interp2d(adcp_ds.tadcp.values.flatten(), adcp_ds.zadcp.values.flatten(), vmadcp)
# Urel[:,0,:] = interp2d_func(np.arange(1,nt+1),ice_init[0].Z.to_numpy()) 


Tair = 6.5 # air temp in C
SWflx = 306 # W/m2 of shortwave flux
Winds = 2.3 # wind speed in m/s
IceC = 0.36 # sea ice conc 0 - 1 (0 - 100%)
ni = len(L)
timespan = 86400.0 * 30.0 # 1 month

mberg_dict = {}
for length in L:
    print(f'Processing Length {length}')
    mberg = ice_melt.iceberg_melt(length, dz, timespan, ctd_ds, IceC, Winds, Tair, SWflx, adcp_ds)
    mberg_dict[length] = mberg

ctdz = ctd_ds.depth
ctdz_flat = ctdz.T.to_numpy().flatten()

t = ctd_ds.temp.data
s = ctd_ds.salt.data
salt = np.nanmean(s,axis=1)
temp = np.nanmean(t,axis=1)
# constants from Jenkins
a = -5.73e-2 # Salinity contribution
b = 8.32e-2 # constant
c = -7.61e-4 # pressure contribution C/dbar

T_fp = a * salt + b + c * ctdz_flat # Freezing point temperature
T_Tfp = temp - T_fp # temperature - freezing point in celsius
T_Tfp_k = T_Tfp + 273.15 # convert from celsius to kelvin

berg = mberg_dict[350]
plot_icebergshape(berg)

l_heat = 3.34e5
Aww_depth = 150


k=mberg_dict[1000].KEEL.sel(time=86400*2)
ul = mberg_dict[1000].UWL.sel(Z=slice(Aww_depth,k.data[0]),time=86400*2)
uw = mberg_dict[1000].UWL.sel(Z=slice(Aww_depth,k.data[0]),time=86400*2)
A = uw * ul


mfw = mberg_dict[1000].i_mfreew.sel(Z=slice(Aww_depth,k.data[0]), time=86400*2)
mtw = mberg_dict[1000].i_mturbw.sel(Z=slice(Aww_depth,k.data[0]), time=86400*2)
Mfreew = mberg_dict[1000].Mfreew.sel(Z=slice(Aww_depth,k.data[0]), time=86400*2)
Mturbw = mberg_dict[1000].Mturbw.sel(Z=slice(Aww_depth,k.data[0]), time=86400*2)
Urel = mberg_dict[1000].Urel.sel(Z=slice(Aww_depth,k.data[0]), time=86400*2)

total_iceberg_melt = np.mean(Mfreew + Mturbw,axis=1)
Aww_melt_rate = np.mean(mtw + mfw,axis=1) / 86400 # convrrt to meters/second

# not_hf = Aww_melt_rate * l_heat * 1000
Qib = total_iceberg_melt * l_heat * 1000 # iceberg heatflux per z layer
Qib_sum = np.sum(Qib)

# calculate heat flux of Aww
fjord_width = 5000
fjord_depth = ctdz_flat.max()
Cp = 3980 # specific heat capactiy J/kgK
p_sw = 1027 # kg/m3

# need to get the temperature and depths at the same spacing



fig, ax  = plt.subplots()
d = ctd_ds.depth.data
t = ctd_ds.temp.data
temp = np.nanmean(t,axis=1)
ax.plot(temp, d.flatten())
ax.set_ylim(d.max(),0)

