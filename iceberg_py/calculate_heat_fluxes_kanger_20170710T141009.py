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
from matplotlib import cm,colors
import pickle
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable


L = np.arange(50,1450,50)
# L = [400]
dz = 5

#Test first L

# # ice = ice_melt.init_iceberg_size(L[0],dz=dz)
# ice_init = []
# for length in L:
#     print(f'processing {length}')
#     ice_init.append(ice_melt.init_iceberg_size(length,dz=dz))
#     print(f'finished with {length}')

ctd_path = '/media/laserglaciers/upernavik/iceberg_py/infiles/kanger/CTD_20160914_083150.nc'
adcp_path = '/media/laserglaciers/upernavik/iceberg_py/infiles/ADCP_cosine_BeccaSummer.mat'


ctd_ds = xr.open_dataset(ctd_path)

# force temp to be constant
# avg_temp35 = np.ones(ctd_ds.temp.shape)*3.5
# ctd_ds.temp.data = avg_temp35


# adcp = sio.loadmat(adcp_path)

# adcp_ds = xr.Dataset({'zadcp': (['adcpX','adcpY'],adcp['zadcp']),
#                       'vadcp': (['adcpX','adcpZ'], adcp['vadcp']),
#                       'tadcp': (['adcpY','adcpZ'], adcp['tadcp']),
#                       'wvel':  (['adcpY'], np.array([0.05]))
#     })

adcp_ds = 0.05

Tair = 6.5 # air temp in C
SWflx = 306 # W/m2 of shortwave flux
Winds = 2.3 # wind speed in m/s
# IceC = 0.36 # sea ice conc 0 - 1 (0 - 100%)
IceC = 1 # sea ice conc 0 - 1 (0 - 100%)
ni = len(L)
timespan = 86400.0 * 30.0 # 1 month
factor = 1
mberg_dict = {}
for length in L:
    print(f'Processing Length {length}')
    mberg = ice_melt.iceberg_melt(length, dz, timespan, ctd_ds, IceC, Winds, Tair, SWflx, adcp_ds,factor=factor,
                                  do_constantUrel=True)
    mberg_dict[length] = mberg

op = '/media/laserglaciers/upernavik/iceberg_py/outfiles/kanger/berg_model/20170710T141009_bergs.pkl'
with open(op,'wb') as handle:
    pickle.dump(mberg_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


plot_icebergshape(mberg_dict[350])
plot_icebergshape(mberg_dict[1000])

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

ctdz = ctd_ds.depth
ctdz_flat = ctdz.T.to_numpy().flatten()
# calculate heat flux of Aww
fjord_width = 5000
fjord_depth = ctdz_flat.max()
Cp = 3980 # specific heat capactiy J/kgK
p_sw = 1027 # kg/m3

# need to get the temperature and depths at the same spacing
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

z_coord_flat = np.arange(dz,600+dz,dz) # deepest iceberg is defined here 
z_coord = z_coord_flat.reshape(len(z_coord_flat),1)
temp_func = interp1d(ctdz_flat, T_Tfp_k)
T_Tfp_k_Z = temp_func(mberg_dict[1000].Z.data).reshape(len(z_coord_flat),1)
T_Tfp_k_Z = xr.DataArray(data=T_Tfp_k_Z, name='T_Tfp_k', coords = {"Z":z_coord_flat},  dims=["Z","X"])

fjord_widths = xr.DataArray(data=np.array([fjord_width]*120).reshape(len(z_coord_flat),1),
                            name='fjord_widths', coords = {"Z":z_coord_flat},  dims=["Z","X"])
Urel2 = mberg_dict[1000].Urel.sel(time=86400*2)
integrand=T_Tfp_k_Z*Urel2*(fjord_widths*5)
integrand_sum=np.sum(integrand.sel(Z=slice(Aww_depth,None)))
Qaww = integrand_sum.data * (Cp*p_sw)


# Heat flux figure per layer per size of iceberg
Qib_dict = {}
for length in L:
    berg = mberg_dict[length]
    k = berg.KEEL.sel(time=86400*2)
    # if k >= Aww_depth:
    Mfreew = berg.Mfreew.sel(Z=slice(None,k.data[0]), time=86400*2)
    Mturbw = berg.Mturbw.sel(Z=slice(None,k.data[0]), time=86400*2)
    
    total_iceberg_melt = np.mean(Mfreew + Mturbw,axis=1)
    Qib = total_iceberg_melt * l_heat * 1000 # iceberg heatflux per z layer
    Qib_dict[length] = Qib



gdf_pkl_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/kanger/20170710T141009_icebergs_kanger.pkl'
with open(gdf_pkl_path, 'rb') as src:
    icebergs_gdf = pickle.load(src)


vc = icebergs_gdf['binned'].value_counts()
fig3, ax3 = plt.subplots()     
icebergs_gdf['binned'].value_counts().sort_index().plot(kind='bar',logy=True,ax=ax3,
                                                        edgecolor = 'k')
# ax3.hist(icebergs_gdf['max_dim'].values, bins=np.arange(0,1050,50),
#          edgecolor = "black")
ax3.set_ylabel('Count')
ax3.set_xlabel('Iceberg surface length (m)')


Qib_totals = {}
Qib_sums = {}
for length in L:
    
    count = vc[length]
    Qib_sum = np.nansum(Qib_dict[length].sel(Z=slice(Aww_depth,None)))
    Qib_totals[length] = Qib_sum * count
    Qib_sums[length] = Qib_sum
    print(f'{length}: {Qib_sum*count}')
    
qib_total = np.nansum(list(Qib_totals.values()))
    
    

vol_dict = {}
for l in mberg_dict.keys():
    berg = mberg_dict[l]
    vol = berg.uwV.sel(Z=slice(150,None))
    vol_sum = np.nansum(vol)
    vol_dict[l] = vol_sum * vc[l]

total_v = np.sum(list(vol_dict.values()))

vc_test = vc.copy()
for l in vc_test.keys():
    if vc_test[l] == 0:
        vc_test[l] = 1
        
# take -50%, -25%, +25%, +50%
# low_vc =  np.round(vc_test.values - (vc_test.values*0.5))
# med_low_vc = np.round(vc_test.values - (vc_test.values*0.25))
# med_high_vc = np.round(vc_test.values + (vc_test.values*0.25))
# high_vc = np.round(vc_test.values + (vc_test.values*0.5))
# very_high_vc = np.round(vc_test.values + (vc_test.values*1))
# very_very_high_vc = np.round(vc_test.values + (vc_test.values*2))
Qaww_high = 0.26e12  # from Inall 2014 10.1038/175238c0
# Qaww_low = 51e9 # from Sutherland and Straneo 2012 
Qib_percentage_high = qib_total/Qaww_high
aww_temp = np.mean(ctd_ds.temp.sel(Z=slice(150,None)).data)

ctd_time = gpd.pd.to_datetime('20160914T083150')
date = gpd.pd.to_datetime('20170710T141009')

Q_ib_ds = xr.Dataset({'Qib':(qib_total),
                      'iceberg_date':(date),
                      'iceberg_concentraion': ('high'),
                      'Qaww_high':(Qaww_high),
                      'ctd_time':(ctd_time),
                      'constant_vel':(adcp_ds),
                      'transfer_coeff_factor': (factor),
                      'Qib_percentage_high': (Qib_percentage_high),
                      'ice_vol': (total_v),
                      'average_aww_temp': (aww_temp)
                      }
    )
Q_ib_ds.Qib.attrs = {'units':'W'}
Q_ib_ds.iceberg_date.attrs = {'sensor':'S2B'}
Q_ib_ds.iceberg_concentraion.attrs = {'description':'Qualatative assesment'}
Q_ib_ds.Qaww_high.attrs = {'description':'heat flux from Inall 2014 10.1038/175238c0',
                           'units':'W'}
Q_ib_ds.constant_vel.attrs = {'description': ' constant velocity used instead of ADCP measurments',
                              'units':'m/s'}
Q_ib_ds.ice_vol.attrs = {'description': 'Ice Volume below the Aww Depth',
                         'Units': 'm^3'}
Q_ib_ds.average_aww_temp.attrs = {'description': 'Average water temp below the Aww boundary',
                         'Units': 'C'}


op = '/media/laserglaciers/upernavik/iceberg_py/outfiles/kanger/Qib/'
# Q_ib_ds.to_netcdf(f'{op}20170710T141009_high_kanger_coeff_{factor}.nc')



