#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:29:44 2023

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
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


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


L = np.arange(50,1450,50)
# L = [400]
dz = 5


mbergs_dict_path = '/media/laserglaciers/upernavik/iceberg_py/mbergs.pickle'
with open(mbergs_dict_path, 'rb') as src:
    mberg_dict = pickle.load(src)


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
fig, axs  = plt.subplots(1,2,sharey=True,figsize=(6.3,5.9))
axs = axs.flatten()
d = ctd_ds.depth.data
t = ctd_ds.temp.data
temp = np.nanmean(t,axis=1)
axs[0].plot(temp, d.flatten(), c='black', lw=5)
axs[0].plot(temp, d.flatten(),lw=3)
axs[0].set_ylim(d.max(),0)
axs[0].set_xlim(-3,5)
axs[0].xaxis.set_minor_locator(MultipleLocator(1))
# axs[0].set_ylabel('Depth (m)')
# axs[0].set_xlabel('Temperature ($^{\circ}$C)')
# axs[0].set_title('Temperature Profile')

urel_x = mberg_dict[1000].Urel.sel(time=86400*2).data.flatten() * 10
urel_y = mberg_dict[1000].Urel.sel(time=86400*2).Z
axs[1].plot(urel_x, urel_y, c='black', lw=5)
axs[1].plot(urel_x, urel_y, lw=3)
axs[1].plot(urel_x-0.2, urel_y, c='black', lw=5)
axs[1].plot(urel_x-0.2, urel_y, lw=3, c='tab:blue')



axs[1].set_xlim(0, 1)
axs[1].xaxis.set_minor_locator(MultipleLocator(0.1))
# axs[1].set_xlabel('Along Fjord Velocity (m s$^{-1}$)')
# axs[1].set_title('Velocity Profile')
axs[0].tick_params(axis='both', which='major', labelsize=20)
axs[1].tick_params(axis='both', which='major', labelsize=20)
axs[0].set_xlabel('temperature (C)', size=20)
axs[1].set_xlabel('velocity (cm s$^{-1}$)', size=20)
axs[0].set_ylabel('depth (m)', size=20)
fig.tight_layout()
profile_op = f'/media/laserglaciers/upernavik/ghawk_2023/figs/temp_vel_profiles.png'
fig.savefig(profile_op,dpi=300)

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

# plot heat flux
fig2, ax2 = plt.subplots(figsize=(8.15,5.15))     
# ax2.set_ylabel('Depth (m)')    
# ax2.set_xlabel('Iceberg Heatflux (W)')    
colors_blues = cm.Blues(np.linspace(0,1,len(Qib_dict)))

cmap = plt.get_cmap('Blues').copy()
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.1)

for i,length in enumerate(Qib_dict):

    x = Qib_dict[length].data / 1e9 #gigawatts
    y = Qib_dict[length].Z.data
    ax2.plot(x, y, c='black', lw=5)
    ax2.plot(x,y,color=colors_blues[i],lw=3)
    ax2.set_ylim(600,y.min())
ax2.axhline(y=Aww_depth,linewidth=3, color='#d62728')

l_min = L.min()
l_max = L.max()
cbar = fig2.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin = l_min,vmax = l_max), cmap=cmap),
              cax=cax,label='Iceberg Length (m)')
cbar.ax.tick_params(labelsize=20)
cbar.set_label(label='Iceberg Length (m)',fontsize=20)
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.set_ylabel('Depth', size=20)
ax2.set_xlabel('Q$_{ib}$ (GW)',size=20)

fig2.tight_layout()
qib_op = f'/media/laserglaciers/upernavik/ghawk_2023/figs/qib_poster.png'
# fig2.savefig(qib_op, dpi=300)

gdf_pkl_path = '/media/laserglaciers/upernavik/iceberg_py/convex_hull_icebergs.pkl'
with open(gdf_pkl_path, 'rb') as src:
    icebergs_gdf = pickle.load(src)


vc = icebergs_gdf['binned'].value_counts()
fig3, ax3 = plt.subplots(figsize=(5,3.8))     
icebergs_gdf['binned'].value_counts().sort_index().plot(kind='bar',logy=True,ax=ax3,
                                                        edgecolor = 'k')
# ax3.hist(icebergs_gdf['max_dim'].values, bins=np.arange(0,1050,50),
#          edgecolor = "black")
# ax3.set_ylabel('Count')
# ax3.set_xlabel('Iceberg surface length (m)')
ax3.tick_params(axis='both', which='major', labelsize=20)
for label in ax3.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)
bar_op = f'/media/laserglaciers/upernavik/ghawk_2023/figs/iceberg_count_bar.png'
fig3.tight_layout()
# fig3.savefig(bar_op, dpi=300)


Qib_totals = {}
for length in L:
    
    count = vc[length]
    Qib_sum = np.nansum(Qib_dict[length].sel(Z=slice(Aww_depth,None)))
    Qib_totals[length] = Qib_sum * count
    print(f'{length}: {Qib_sum*count}')
    
qib_total=np.nansum(list(Qib_totals.values()))


# Enderlin 2018 2f comparison
fixed_m = 4e-6
# fixed_m = 0.36
rho_i = 917
l_heat = 3.34e5
Qice_dict = {}
uwA_dict = {}
aww_depth = 150
for length in L:
    berg = mberg_dict[length]
    uwL = berg.uwL.sel(Z=slice(150,None)).data
    uwW = berg.uwW.sel(Z=slice(150,None)).data
    
    melt = (2 * (fixed_m * dz * uwL)) + (2 * (fixed_m * dz * uwW))
    surface_area = (2 * (dz * uwL)) + (2 * (dz * uwW))
    # uwA = uwL * uwW
    area_sum = np.nansum(surface_area)
    q_ice = (rho_i * l_heat * fixed_m) * area_sum
    Qice_dict[length] = q_ice
    uwA_dict[length] = area_sum
    
qice_series = gpd.pd.Series(Qice_dict)
uwA_series = gpd.pd.Series(uwA_dict)
qice_df = gpd.pd.DataFrame(qice_series,columns=['q_ice'])
qice_df['uwA'] = uwA_series

# subset_df = qice_df.loc[:300,:]
fig4, ax4 = plt.subplots()  

# ax4.scatter(subset_df['uwA'] * 1e-6, (subset_df['q_ice']*86400)/1e6)
ax4.scatter(qice_df['uwA'] * 1e-6, (qice_df['q_ice']),edgecolor='k')
ax4.set_xlabel('Aww Submerged Area (km$^{2}$)',fontsize=15)
ax4.set_ylabel('Q$_{ice}$',fontsize=15)


gdf_pkl_path = '/media/laserglaciers/upernavik/iceberg_py/convex_hull_icebergs.pkl'
with open(gdf_pkl_path, 'rb') as src:
    icebergs_gdf = pickle.load(src)
    
vc = icebergs_gdf['binned'].value_counts()

qice_df['count'] = gpd.pd.Series(vc)
qice_df['total'] = qice_df['q_ice'] * qice_df['count']
    
    
    


