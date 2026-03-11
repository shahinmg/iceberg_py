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
from matplotlib.ticker import MultipleLocator, MaxNLocator
import types
import os
import string

berg_model_path_1 = '../data/iceberg_classes_output_bug_fix/helheim/avg_013/'

iceberg_geom_path = '../data/iceberg_geoms/helheim/'

berg_model_list_c1 = sorted([pkl for pkl in os.listdir(berg_model_path_1) if pkl.endswith('pkl')])


colors_viridis = cm.viridis(np.linspace(0,1,len(berg_model_list_c1)))

def get_xr_das(model_list):
    
    # os.chdir(chdir)
    fig, ax = plt.subplots(1,3,sharey='row',figsize=(9, 7))
    
    total_melt_list = [40, 41, 48, 89, 70] # come from Table 1
    
    for time_idx, berg_model_file in enumerate(model_list):
        
        date = berg_model_file[:10]
        berg_model_file = f'{berg_model_path_1}{berg_model_file}'
        with open(berg_model_file, 'rb') as src:
            mberg_dict = pickle.load(src)
            
        # iceberg_geom_file = f'{iceberg_geom_path}{date}-icebergs_helheim_keel_depth.gpkg'
        iceberg_geom_file = f'{iceberg_geom_path}{date}_all_merged_dims.gpkg'

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
                                                                                                        
        total_melt_data_arr = xr.DataArray(data=total_melt_arr, name='melt rate', coords = {"Z":Z, "length":L},  
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
    
    
        labelsize = 11
        
 
        
        # ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)
        
        uwSA_mask = uwSA_depth_sum_aw>0
        uwSA_depth_sum_aw_masked = uwSA_depth_sum_aw[uwSA_mask]
        uwSA_depth_sum_aw_masked = uwSA_depth_sum_aw_masked
        ax[1].plot((uwSA_depth_sum_aw_masked.data/5) / 1e3, uwSA_depth_sum_aw_masked.Z, c='black', lw=5)
        ax[1].plot((uwSA_depth_sum_aw_masked.data/5) / 1e3, uwSA_depth_sum_aw_masked.Z, color=colors_viridis[time_idx], lw=3, label=date)
        
        ax[1].axhspan(0,150,facecolor='tab:blue', zorder=2, alpha=0.1)
        # ax[1].set_xlim(0,3)
        # ax[1].set_xscale('log')
        ax[1].set_xlabel('Ice Area per Unit Depth (km)' , size=labelsize)
        ax[1].set_xlim(0, 1800)
        
        
    
        Qib_mask = Qib_depth_sum>0
        Qib_depth_sum_masked = Qib_depth_sum[Qib_mask]
        Qib_depth_sum_masked = Qib_depth_sum_masked/1e9
        ax[2].plot(Qib_depth_sum_masked.data/5, Qib_depth_sum_masked.Z, c='black', lw=5)
        ax[2].plot(Qib_depth_sum_masked.data/5, Qib_depth_sum_masked.Z, color=colors_viridis[time_idx], 
                   lw=3,  label=total_melt_list[time_idx])
        
        ax[2].axhspan(0,150,facecolor='tab:blue', zorder=2, alpha=0.1)
        
        ax[2].set_ylim(600,0)
        # ax[2].set_xlim(0.05e7,5e10)
        # ax[2].set_xscale('log')
        
        ax[2].set_xlim(0, 0.8)
        ax[2].xaxis.set_major_locator(MaxNLocator(4))
        ax[2].set_xlabel('Q$_{ib}$ per Unit Depth (GW/m)', size=labelsize)
        
    
    
    
        # # Set scond x-axis one time 
        
        if time_idx == 0:
            ax2_dupe = ax[2].twiny()
            new_pos = [0.0, 0.2, 0.4, 0.6, 0.8]
            new_labels = [0.0, 0.56, 1.12, 1.68, 2.24] # come from the commented out ax[3]
        
            ax2_dupe.set_xticks(new_pos)
            ax2_dupe.set_xticklabels(new_labels)
    
            ax2_dupe.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
            ax2_dupe.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
            ax2_dupe.spines['bottom'].set_position(('outward', 36))
            ax2_dupe.set_xlabel('Melt per Unit Depth (m$^{2}$ s$^{-1}$)', size=labelsize)
            # ax2_dupe.set_xlabel('test', size=labelsize)
    
            ax2_dupe.set_xlim(ax[2].get_xlim())

    
    
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
        # ax[0].plot(keel_melt_rates, keel_depths, color='tab:blue', marker='o', mfc = 'tab:purple', mec='k',
        #            label=Gamma4 if time_idx == 0 else "", zorder=2)
        
        ax[0].plot(keel_melt_rates, keel_depths,  c='black', lw=5, zorder=1)
        ax[0].plot(keel_melt_rates, keel_depths,  c='maroon', lw=3, zorder=1)
        avg_melt_rate = keel_melt_rates[4:].mean()
        
        
        
        # ax[3].plot(keel_melt_rates/4, keel_depths, color='tab:green',  marker='o', mfc = 'tab:olive', mec='k',
        #            label=Gamma1 if time_idx == 0 else "", zorder=2) 
        
        ax[0].set_ylim(600,0)
        ax[0].set_xlim(0, 0.7)
        ax[0].set_xlabel('Melt Rate (m d$^{-1}$)', size=labelsize)
        ax[0].axhspan(0,150,facecolor='tab:blue', zorder=2, alpha=0.1)
        ax[0].xaxis.set_minor_locator(MultipleLocator(0.5))
        ax[0].set_ylabel('Depth (m)', size=labelsize)
    
    
        
    
    
    
    
        # ax[3].plot(total_melt_depth_sum.data/5, total_melt_depth_sum.Z, c='black', lw=5, zorder=1)
        # ax[3].plot(total_melt_depth_sum.data/5, total_melt_depth_sum.Z, color=colors_viridis[time_idx],
        #            lw=3, zorder=1, label=total_melt_list[time_idx])
        

        # ax[3].axhspan(0,150,facecolor='tab:blue', zorder=2, alpha=0.1)

        # ax[3].set_xlabel('Melt per Unit Depth (m$^{2}$ s$^{-1}$)', size=labelsize)
        # ax[3].set_xlim(0, 2.1)
        # ax[3].xaxis.set_major_locator(MaxNLocator(5))
    
    
    
            # plt.subplots_adjust(right=0.9)
        plt.tight_layout()
        ax[1].legend(loc='lower center', ncol=1, facecolor='none', frameon=False,
                     fontsize='small')
        ax[2].legend(loc='lower center', ncol=1, facecolor='none', frameon=False,
                     fontsize='small', title = r'total melt m$^{3}$ s$^{-1}$')

        
        text_dict = {'fontsize':18,
                     'fontweight': 'bold'}
        ax = ax.flatten()
        alphabet = list(string.ascii_uppercase)
        for i,axis in enumerate(ax):
            
            text_label = axis.text(.01, .99, alphabet[i], ha='left', va='top', transform=axis.transAxes, **text_dict)

        # fig.align_xlabels()
        # plt.tight_layout()
        op = f'./figs/'
        if not os.path.exists(op):
            os.makedirs(op)
        # plt.subplots_adjust(right=0.2)
        # fig.savefig(f'{op}Qib_depth_updated_3_panel.pdf', dpi=300, bbox_inches='tight')
        # fig.savefig(f'{op}Qib_depth_updated_3_panel.pdf', dpi=300, pad_inches=0.03)

    return fig, ax

#%% plot melt rate per day figure per iceberg class 

# get_xr_das(berg_model_list, berg_model_path)
out_fig, out_ax = get_xr_das(berg_model_list_c1)
# 





