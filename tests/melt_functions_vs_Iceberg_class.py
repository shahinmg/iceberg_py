import numpy as np
import numpy.matlib
from scipy.interpolate import interp1d, interp2d
from scipy.spatial import cKDTree, KDTree
import xarray as xr
from math import ceil
from sklearn.linear_model import LinearRegression

from melt_functions import *
from iceberg import Iceberg

length = 350
dz = 5
iceberg_class = Iceberg(length=350).init_iceberg_size()

icebeg_func = init_iceberg_size(length,dz=dz)


cross_area_diff = icebeg_func['cross_area'] - iceberg_class['cross_area']

totalV_diff =  icebeg_func['totalV'] - iceberg_class['totalV']



# print(f'cross area test: {cross_area_diff}')


for var in iceberg_class.data_vars:
    
    diff = iceberg_class[var] - icebeg_func[var]
    diff_mean = np.nanmean(diff)
    
    print(f'{var} diff: {diff_mean}')