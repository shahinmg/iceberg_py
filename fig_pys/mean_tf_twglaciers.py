#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:26:28 2023

@author: laserglaciers
"""

import scipy.io as sio
import xarray as xr
import pandas as pd
import numpy as np
import mat73
from datetime import datetime, timedelta

en4_tf_path = '/media/laserglaciers/upernavik/slater_2022_submelt/EN4_TFGL_out.mat'
en4_tf = sio.loadmat(en4_tf_path)
# en4_tf = mat73.loadmat(en4_tf_path)


def dec_year_2_pd_dt(dec_year):
# https://stackoverflow.com/questions/20911015/decimal-years-to-datetime-in-python
    start = dec_year
    year = int(start)
    rem = start - year

    base = datetime(year, 1, 1)
    result = base + timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem)

    return pd.to_datetime(result)
    

morlighem_number = en4_tf['EN4_tf_arr'][0][0][0]
names = en4_tf['EN4_tf_arr'][0][0][1]
times = en4_tf['EN4_tf_arr'][0][0][4]
first_tf_gl = en4_tf['EN4_tf_arr'][0][0][5]
first_x = en4_tf['EN4_tf_arr'][0][0][2]
first_y = en4_tf['EN4_tf_arr'][0][0][3]

# make times pd datetimes
vfunc = np.vectorize(dec_year_2_pd_dt)
pd_datetimes = vfunc(times)

ds = xr.Dataset(
    data_vars=dict(
        TF_GL=(["time"], first_tf_gl.flatten()),
        xcoord=(first_x[0][0]),
        ycoord=(first_y[0][0]),
        name = names[0]
    ),
    coords=dict(
        time=pd_datetimes.flatten(),
        reference_time=times.flatten(),
    ),
    attrs=dict(description="test"),
)


# for row in en4_tf['EN4_tf_arr'][0]:
#     print(row[1][0])
    