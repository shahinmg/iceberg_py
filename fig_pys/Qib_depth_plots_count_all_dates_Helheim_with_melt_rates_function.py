#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:34:01 2024

@author: laserglaciers
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import cm,colors
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
import types
import os
import string

berg_model_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/berg_model/factor_4/median/'
berg_model_path_1 = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/berg_model/factor_1/median/'

iceberg_geom_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/iceberg_geoms/dim_with_bin/'

berg_model_list = sorted([pkl for pkl in os.listdir(berg_model_path) if pkl.endswith('pkl')])
berg_model_list_c1 = sorted([pkl for pkl in os.listdir(berg_model_path_1) if pkl.endswith('pkl')])




colors_viridis = cm.viridis(np.linspace(0,1,len(berg_model_list)))

def get_xr_das(model_list, chdir):
    
    os.chdir(chdir)
    fig, ax = plt.subplots(1,4,sharey='row',figsize=(8, 6))
    
    for time_idx, berg_model_file in enumerate(model_list):
        
        date = berg_model_file[:10]
    
        with open(berg_model_file, 'rb') as src:
            mberg_dict = pickle.load(src)
            
        iceberg_geom_file = f'{iceberg_geom_path}{date}-icebergs_helheim_keel_depth.gpkg'
        icebergs_gdf = gpd.read_file(iceberg_geom_file)
        vc = icebergs_gdf['binned'].value_counts()
        
        
        L = np.arange(50,1450,50)
        l_heat = 3.34e5
        Aww_depth = 150
        Z = np.arange(5,500,5)
        Z_sa = np.arange(5,605,5)
        sec2day = 86400
        
        Qib_arr = np.empty((len(Z),len(L)))
        Qib_arr[:] = np.nan
        
        total_melt_arr = np.empty((len(Z),len(L)))
        total_melt_arr[:] = np.nan
        
        
        SA_arr = np.empty((len(Z_sa),len(L)))
        SA_arr[:] = np.nan
        
        # Mfreew[k,i,j] =  2 * (mb[k,i,j] * dz * uwL[k][0]) + 2 *(mb[k,i,j] * dz * uwW[k])
        
        # Heat flux figure per layer per size of iceberg
        Qib_dict = {}
        Qib_list = []
        
        Qib_totals = {}
        Qib_sums = {}
        
        for i,length in enumerate(L):
            berg = mberg_dict[length]
            k = berg.KEEL.sel(time=86400*2)
            # if k >= Aww_depth:
            Mfreew = berg.Mfreew.sel(Z=slice(None,k.data[0]), time=86400*2) #just use first time step after initialization 
            Mturbw = berg.Mturbw.sel(Z=slice(None,k.data[0]), time=86400*2) 
            
            total_iceberg_melt = np.mean(Mfreew + Mturbw,axis=1)
            Qib = total_iceberg_melt * l_heat * 1000 # iceberg heatflux per z layer
            Qib.name = str(length)
        
            uw_length = berg.uwL
            uw_width = berg.uwW
            dz = berg.dz.data
            
            uw_sa = 2 * (dz * uw_length) +  2 *(dz * uw_width)  # underwater surface area
            
            if np.isin(length,vc.index):
                count = vc[length]
                # print(length)
                Qib_depths = Qib.sel(Z=slice(None,None))
        
                
                Qib_totals = Qib_depths * count
                # print(f'{length}: {Qib_totalsz*count}')
                uw_sa_totals = uw_sa * count
                total_melt_totals = total_iceberg_melt * count
                
                
                Qib_arr[:Qib_totals.data.shape[0], i] = Qib_totals.data
                SA_arr[:uw_sa_totals.data.shape[0], i] = uw_sa_totals.data.flatten()
                total_melt_arr[:total_melt_totals.data.shape[0], i] = total_melt_totals.data.flatten()
    
        
        Qib_xr_data_arr = xr.DataArray(data=Qib_arr, name='Qib', coords = {"Z":Z, "length":L},  
                     dims=["Z","length"], attrs={'Description':"Iceberg Heat Flux", 'Units': 'W'})
                                                                                                                    
        
        uwSA_xr_data_arr = xr.DataArray(data=SA_arr, name='underwater_surface_area', coords = {"Z":Z_sa, "length":L},  
                     dims=["Z","length"], attrs={'Description':"Estimated Underwater Surface Area", 'Units': 'm^2'})
                                                                                                        
        total_melt_data_arr = xr.DataArray(data=total_melt_arr, name='underwater_surface_area', coords = {"Z":Z, "length":L},  
                     dims=["Z","length"], attrs={'Description':"Total integrated Melt Rate", 'Units': 'm^3 s^-1'})
        
        
        
        Qib_depth_mean = Qib_xr_data_arr.mean(dim='length')
        
        Qib_depth_sum = Qib_xr_data_arr.sum(dim='length')
    
        uwSA_depth_sum = uwSA_xr_data_arr.sum(dim='length')
        uwSA_depth_sum_aw = uwSA_depth_sum.sel(Z=slice(None, 500))
        
        
        total_melt_depth_sum = total_melt_data_arr.sum(dim='length')
        total_melt_depth_sum.name = 'Total Iceberg Melt'
        # get melt rate in meters per day
        
        melt_per_sec = total_melt_data_arr/uwSA_xr_data_arr #meters s^-1
        melt_per_day = melt_per_sec*sec2day
        melt_per_day.name = 'melt rate'
        melt_per_day_sum = melt_per_day.sum(dim='length')
    
    

    


    
    
        labelsize = 12
        
        Qib_mask = Qib_depth_sum>0
        Qib_depth_sum_masked = Qib_depth_sum[Qib_mask]
        ax[0].plot(Qib_depth_sum_masked.data, Qib_depth_sum_masked.Z, c='black', lw=5)
        ax[0].plot(Qib_depth_sum_masked.data, Qib_depth_sum_masked.Z, color=colors_viridis[time_idx], lw=3, label=date)
        
        ax[0].axhspan(150,600,facecolor='0.6',alpha=0.3, zorder=1)
        
        ax[0].set_ylim(600,0)
        ax[0].set_xlim(5e7,5e10)
        ax[0].set_xscale('log')
        ax[0].set_xlabel('Q$_{ib}$ (W) per Unit Depth' ,size=labelsize)
        # ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)
        
        uwSA_mask = uwSA_depth_sum_aw>0
        uwSA_depth_sum_aw_masked = uwSA_depth_sum_aw[uwSA_mask]
        
        ax[1].plot(uwSA_depth_sum_aw_masked.data, uwSA_depth_sum_aw_masked.Z, c='black', lw=5)
        ax[1].plot(uwSA_depth_sum_aw_masked.data, uwSA_depth_sum_aw_masked.Z, color=colors_viridis[time_idx], lw=3)
        
        ax[1].axhspan(150,600,facecolor='0.6',alpha=0.3, zorder=1)
        ax[1].set_xlim(1e4,8e6)
        ax[1].set_xscale('log')
        ax[1].set_xlabel('Surface Area (m$^{2}$)' , size=labelsize)
        
        
        ax[2].plot(total_melt_depth_sum.data, total_melt_depth_sum.Z, c='black', lw=5, zorder=2)
        ax[2].plot(total_melt_depth_sum.data, total_melt_depth_sum.Z, color=colors_viridis[time_idx],
                   lw=3, zorder=3, label=date)
        
        ax[2].axhspan(150,600,facecolor='0.6',alpha=0.3, zorder=1)
        # ax[2].set_xlim(10e3,1e7)
        # ax[2].set_xscale('log')
        ax[2].set_xlabel('Total Melt (m$^{3}$ s$^{-1}$)' , size=labelsize)
        
        ax[2].xaxis.set_minor_locator(MultipleLocator(25))
    
        colors_viridis_length = cm.viridis(np.linspace(0,1,len(melt_per_day.length)))
    
        keel_melt_rates = []
        keel_depths = []
    
        Gamma4 = r'$\Gamma_{S,T}$ $\times$ 4'
        Gamma1 = r'$\Gamma_{S,T}$ $\times$ 1'
        
        keel_melt_rates = []
        keel_depths = []
        for i,length in enumerate(melt_per_day.length):
            
            ind_berg = melt_per_day.sel(length=length)
            get_non_nans = ~np.isnan(ind_berg)
            ind_berg = ind_berg[get_non_nans]
            if ind_berg.data.size > 0:
                
                keel_melt_rates.append(ind_berg.data[-1])
                keel_depths.append(ind_berg.Z[-1])
                
                # # ax2.plot(ind_berg.data, ind_berg.Z, color='tab:blue',zorder=1)
                # ax[1].scatter(ind_berg.data[-1], ind_berg.Z[-1], color=colors_viridis_length[i],
                #             zorder=3, edgecolor='k')
            
                # # ax2.plot(ind_berg.data/4, ind_berg.Z, color='tab:orange', zorder=1)
                # ax[1].scatter(ind_berg.data[-1]/4, ind_berg.Z[-1],color=colors_viridis_length[i], 
                #             zorder=3, edgecolor='k')
                
        keel_melt_rates = np.array(keel_melt_rates)   
        ax[3].plot(keel_melt_rates, keel_depths, color='tab:blue', marker='o', mfc = 'tab:purple', mec='k',
                   label=Gamma4 if time_idx == 0 else "", zorder=2)
        
        # ax[3].plot(keel_melt_rates/4, keel_depths, color='tab:green',  marker='o', mfc = 'tab:olive', mec='k',
        #            label=Gamma1 if time_idx == 0 else "", zorder=2) 
        
        ax[3].set_ylim(600,0)
        ax[3].set_xlim(0,2)
        ax[3].set_xlabel('Melt Rate (m d$^{-1}$)', size=labelsize)
        ax[3].axhspan(150,600,facecolor='0.6',alpha=0.3, zorder=1)
        ax[3].xaxis.set_minor_locator(MultipleLocator(0.5))
    
    
    
            # plt.subplots_adjust(right=0.9)
        plt.tight_layout()
        ax[2].legend(loc='lower center',ncol=1)
        
        ax[3].legend(loc='lower center')
        
        
        
        pad = plt.rcParams["xtick.major.size"] + plt.rcParams["xtick.major.pad"]
        def bottom_offset(self, bboxes, bboxes2):
            bottom = self.axes.bbox.ymin
            self.offsetText.set(va="top", ha="left") 
            oy = bottom - pad * self.figure.dpi / 1000
            self.offsetText.set_position((1, oy))
        
        
        ax[0].xaxis._update_offset_text_position = types.MethodType(bottom_offset, ax[0].xaxis)
        ax[1].xaxis._update_offset_text_position = types.MethodType(bottom_offset, ax[1].xaxis)
        ax[2].xaxis._update_offset_text_position = types.MethodType(bottom_offset, ax[2].xaxis)
        # ax[3].xaxis._update_offset_text_position = types.MethodType(bottom_offset, ax[2].xaxis)
        
        text_dict = {'fontsize':18,
                     'fontweight': 'bold'}
        ax = ax.flatten()
        alphabet = list(string.ascii_lowercase)
        for i,axis in enumerate(ax):
            
            text_label = axis.text(.01, .99, alphabet[i], ha='left', va='top', transform=axis.transAxes, **text_dict)


#%% plot melt rate per day figure per iceberg class 

get_xr_das(berg_model_list, berg_model_path)
get_xr_das(berg_model_list_c1, berg_model_path_1)
# 
# op = '/media/laserglaciers/upernavik/iceberg_py/figs/'
# fig.savefig(f'{op}2024-08-22_Qib_depth.png',dpi=300, bbox_inches='tight')





