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
import os
import pandas as pd

dt = 50

psw = 1024 #kg m3
csw = 3974 #J kg-1 C-1
day2sec = 86400
Volume_test = 28e3 * 5e3 * 300 #km3 Helheim Fjord test

coeff_1_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/Qib/coeff_1_melt_rates/'
coeff_4_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/Qib/coeff_4_melt_rates//'

coeff_1_list = sorted([nc for nc in os.listdir(coeff_1_path) if nc.endswith('nc')])
coeff_4_list = sorted([nc for nc in os.listdir(coeff_4_path) if nc.endswith('nc')])


constant_tf_55 = 5.73 # from Slater 2022 nature geoscience
constant_tf_67 = 6.67 # from Slater 2022 nature geoscience
constant_tf_8 = 7.62 # from Slater 2022 nature geoscience

dQ_dt_HEL_55 = psw * csw * ( (Volume_test * constant_tf_55) / (dt * day2sec) )
dQ_dt_HEL_67 = psw * csw * ( (Volume_test * constant_tf_67) / (dt * day2sec) )
dQ_dt_HEL_8 = psw * csw * ( (Volume_test * constant_tf_8) / (dt * day2sec) )

os.chdir(coeff_1_path)
dQ_dt_list = [dQ_dt_HEL_55, dQ_dt_HEL_67, dQ_dt_HEL_8]

tfs = [constant_tf_55, constant_tf_67, constant_tf_8]
urels = [0.01, 0.05, 0.10] #m s-1

cols = ['coeff', 'urel', 'tf', 'dt', 'Qib', 'Qaww', 'melt_rate_avg', 'melt_rate_mday', 'percentage']
coef_1_dict = {}

series_list1 = []
for i,nc in enumerate(coeff_1_list):
    
    Qib = xr.open_dataset(nc)
    Qib_val = Qib.Qib.data
    
    percentage = Qib_val/dQ_dt_list[i]
    print(f'coeff 1: {percentage:.2f}')
    coef_1_dict['coeff'] = 1
    coef_1_dict['urel'] = urels[i]
    coef_1_dict['tf'] = tfs[i]
    coef_1_dict['dt'] = 50
    coef_1_dict['Qib'] = f'{(Qib_val/1e11):.2f}'
    coef_1_dict['Qaww'] = f'{(dQ_dt_list[i]/1e11):.2f}'
    coef_1_dict['melt_rate_avg'] = f'{Qib.melt_rate_avg.data:.2f}'
    coef_1_dict['melt_rate_mday'] = f'{Qib.melt_rate_intergrated.data:.2f}'
    
    coef_1_dict['percentage'] = f'{(Qib_val/dQ_dt_list[i])*100:.2f}'
    
    series = pd.Series(coef_1_dict)
    series_list1.append(series)

df_1 = pd.DataFrame(series_list1, columns=cols)   
os.chdir(coeff_4_path)

coef_4_dict = {}
series_list = []
for i,nc in enumerate(coeff_4_list):
    
    Qib = xr.open_dataset(nc)
    Qib_val = Qib.Qib.data
    
    percentage = Qib_val/dQ_dt_list[i]
    print(f'coeff 4: {percentage:.2f}')
    
    coef_4_dict['coeff'] = 4
    coef_4_dict['urel'] = urels[i]
    coef_4_dict['tf'] = tfs[i]
    coef_4_dict['dt'] = 50
    coef_4_dict['Qib'] = f'{(Qib_val/1e11):.2f}'
    coef_4_dict['Qaww'] = f'{(dQ_dt_list[i]/1e11):.2f}'
    coef_4_dict['melt_rate_avg'] = f'{Qib.melt_rate_avg.data:.2f}'
    coef_4_dict['melt_rate_mday'] = f'{Qib.melt_rate_intergrated.data:.2f}'

    
    coef_4_dict['percentage'] = f'{(Qib_val/dQ_dt_list[i])*100:.2f}'
    
    series = pd.Series(coef_4_dict)
    series_list.append(series)
    


df_1 = pd.DataFrame(series_list1, columns=cols)
df_4 = pd.DataFrame(series_list, columns=cols)

df_concat = pd.concat([df_1, df_4])


