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



Tair = 6.5 # air temp in C
SWflx = 306 # W/m2 of shortwave flux
Winds = 2.3 # wind speed in m/s
# IceC = 0.36 # sea ice conc 0 - 1 (0 - 100%)
IceC = 1 # sea ice conc 0 - 1 (0 - 100%)
ni = len(L)
timespan = 86400.0 * 30.0 # 1 month

mberg_dict = {}
for length in L:
    print(f'Processing Length {length}')
    mberg = ice_melt.iceberg_melt(length, dz, timespan, ctd_ds, IceC, Winds, Tair, SWflx, adcp_ds)
    mberg_dict[length] = mberg

# op = '/media/laserglaciers/upernavik/iceberg_py/mbergs.pickle'
# with open(op,'wb') as handle:
#     pickle.dump(mberg_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


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

# Plot CTD profile
fig, axs  = plt.subplots(1,2,sharey=True)
axs = axs.flatten()
d = ctd_ds.depth.data
t = ctd_ds.temp.data
temp = np.nanmean(t,axis=1)
axs[0].plot(temp, d.flatten())
axs[0].set_ylim(d.max(),0)
axs[0].set_ylabel('Depth (m)')
axs[0].set_xlabel('Temperature ($^{\circ}$C)')
axs[0].set_title('Temperature Profile')

urel_x = mberg_dict[1000].Urel.sel(time=86400*2).data.flatten()
urel_y = mberg_dict[1000].Urel.sel(time=86400*2).Z
axs[1].plot(urel_x, urel_y)
axs[1].set_xlabel('Along Fjord Velocity (m s$^{-1}$)')
axs[1].set_title('Velocity Profile')

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


fig2, ax2 = plt.subplots()     
ax2.set_ylabel('Depth (m)')    
ax2.set_xlabel('Iceberg Heatflux (W)')    
colors_blues = cm.Blues(np.linspace(0,1,len(Qib_dict)))

cmap = plt.get_cmap('Blues').copy()
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.1)

for i,length in enumerate(Qib_dict):

    x = Qib_dict[length].data
    y = Qib_dict[length].Z.data
    ax2.plot(x, y, c='black', lw=5)
    ax2.plot(x,y,color=colors_blues[i],lw=3)
    ax2.set_ylim(600,y.min())
ax2.axhline(y=Aww_depth,linewidth=3, color='#d62728')

l_min = L.min()
l_max = L.max()
fig2.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin = l_min,vmax = l_max), cmap=cmap),
              cax=cax,label='Iceberg Length (m)')


gdf_pkl_path = '/media/laserglaciers/upernavik/iceberg_py/convex_hull_icebergs.pkl'
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
for length in L:
    
    count = vc[length]
    Qib_sum = np.nansum(Qib_dict[length].sel(Z=slice(Aww_depth,None)))
    Qib_totals[length] = Qib_sum * count
    print(f'{length}: {Qib_sum*count}')
    
qib_total=np.nansum(list(Qib_totals.values()))
    
    
    
    
    

