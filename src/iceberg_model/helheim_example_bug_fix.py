import melt_functions as ice_melt
import numpy as np
import xarray as xr
import scipy.io as sio
import pickle
import geopandas as gpd
import os


# set up initial surface lengths and depth intervals
L = np.arange(50,1450,50)
dz = 5
FJORD = 'helheim'
run_type = 'avg' # min, avg, max

# input data paths
ctd_path = f'../../data/ctd_data/{run_type}_temp_sal_sermilik_fjord.csv'
adcp_path = '../../data/adcp_template/ADCP_template.mat'



gdf_pkl_path = f'../../data/iceberg_geoms/{FJORD}/'
gdf_list = sorted([gdf for gdf in os.listdir(gdf_pkl_path) if gdf.endswith('gpkg')])

out_dir = '../../data/'


vel_dict = {'min': 0.02,
            'avg': 0.07,
            'max':0.15} #m/s


for berg_file in gdf_list:
    
    date_str = berg_file[:10]
    
    
    ctd = gpd.pd.read_csv(ctd_path)
    
    depth = np.array( ctd['depth'].values).reshape(1, len(ctd['depth']) )
    temp = np.array( ctd['temp'].values ).reshape(len(ctd['temp']), 1 )
    salt = np.array( ctd['salt'].values ).reshape(len(ctd['salt']), 1 )
    
    ctd_ds = xr.Dataset({'depth':(['Z','X'], depth),
                         'temp': (['tZ','tX'], temp),
                         'salt': (['tZ','tX'], salt)
                         }
                        )
    
    # force temp to be constant
    # avg_temp35 = np.ones(ctd_ds.temp.shape)*3.5
    # ctd_ds.temp.data = avg_temp35
    
    
    adcp = sio.loadmat(adcp_path)
    
    
    Tair = 5.5 # air temp in C
    SWflx = 306 # W/m2 of shortwave flux
    Winds = 2.3 # wind speed in m/s
    # IceC = 0.36 # sea ice conc 0 - 1 (0 - 100%)
    IceC = 1 # sea ice conc 0 - 1 (0 - 100%)
    ni = len(L)
    timespan = 86400.0 * 30.0 # 1 month
    
    
    # u_rel_tests = [0.02, 0.07, 0.15] #slow, medium, fast from Jackson 2016 and Davison 2020
    # tf_test = [5.73, 6.67, 7.62] # from histogram of TF from Slater
    
    u_rel_tests = [vel_dict[run_type]] #slow, medium, fast inspired Jackson 2016 and Davison 2020  [0.02, 0.07, 0.15]
    
    
    for u_rel in u_rel_tests:
    
        factor = 1 # 4 is from Jackson et al 2020 to increase transfer coeffs
        use_constant_tf = False
        do_constantUrel = True
        constant_tf = None 

        adcp_ds = xr.Dataset({'zadcp': (['adcpX','adcpY'],adcp['zadcp']),
                              'vadcp': (['adcpX','adcpZ'], adcp['vadcp']),
                              'tadcp': (['adcpY','adcpZ'], adcp['tadcp']),
                              'wvel':  (['adcpY'], np.array([u_rel]))
                              })
        
        
        # run the model for each length class and store in dict
        mberg_dict = {}
        for length in L:
            print(f'Processing Length {length}')

            mberg = ice_melt.iceberg_melt(length, dz, timespan, ctd_ds, IceC, Winds, Tair, SWflx, u_rel, do_constantUrel=do_constantUrel, 
                                          factor=factor, use_constant_tf=use_constant_tf, 
                                          constant_tf = None)
            
            mberg_dict[length] = mberg
        
        
        l_heat = 3.34e5 #J kg
        Aww_depth = 150
        Cp = 3980 # specific heat capactiy J/kgK
        p_sw = 1027 # kg/m3
        p_fw = 1000 # freshwater density kg/m3
        
        # Heat flux figure per layer per size of iceberg
        Qib_dict = {}
        total_melt_dict = {}
        
        for length in L:
            berg = mberg_dict[length]
            k = berg.KEEL.sel(time=86400*2)
            # if k >= Aww_depth:
            Mfreew = berg.Mfreew.sel(Z=slice(None,k.data[0]), time=86400*2)
            Mturbw = berg.Mturbw.sel(Z=slice(None,k.data[0]), time=86400*2)
            
            total_iceberg_melt = np.mean(Mfreew + Mturbw,
                                         axis=1) # Not sure why I took mean here; Mfeew and Mturbw are integreated melt terms in m3/sec per layer face of iceberg
            
            Qib = total_iceberg_melt * l_heat * p_fw #iceberg heatflux per z layer; since these are integrated terms do not add area
            
            
            Qib_dict[length] = Qib
            total_melt_dict[length] = total_iceberg_melt
        

        op_berg_model = f'{out_dir}iceberg_classes_output_bug_fix_ctd/{FJORD}/{run_type}/'
        if not os.path.exists(op_berg_model):
            os.makedirs(op_berg_model)
            
        out_file = f'{op_berg_model}{date_str}_urel{u_rel}_ctd_data_bergs_coeff{factor}.pkl'

        with open(out_file,'wb') as handle:
            # pickle.dump(mberg_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(mberg_dict, handle)
        
        
        berg_path = f'{gdf_pkl_path}{berg_file}'
        icebergs_gdf = gpd.read_file(berg_path)
        # icebergs_gdf = gpd.read_file(berg_file)
        
        vc = icebergs_gdf['binned'].value_counts()
        
        Qib_totals = {}
        total_iceberg_melt_totals = {} #underwater melt
        for length in L:
            
            if np.isin(length,vc.index):
                count = vc[length]
                Qib_sum = np.nansum(Qib_dict[length].sel(Z=slice(Aww_depth,None)))
                melt_sum =  np.nansum(total_melt_dict[length].sel(Z=slice(Aww_depth,None))) # Total melt not just AW
                
                Qib_totals[length] = Qib_sum * count
                total_iceberg_melt_totals[length] = melt_sum * count
                
                # print(f'{length}: {Qib_sum*count}')
                
        qib_total = np.nansum(list(Qib_totals.values()))
        entire_total_melt = np.nansum(list(total_iceberg_melt_totals.values()))
            
        
        vol_dict = {}
        for l in mberg_dict.keys():
            
            if np.isin(l,vc.index):
                
                berg = mberg_dict[l]
                vol = berg.uwV.sel(Z=slice(150,None))
                vol_sum = np.nansum(vol)
                vol_dict[l] = vol_sum * vc[l]
            
        total_v = np.sum(list(vol_dict.values()))
        
        
        i_mtotalm_totals_dict = {}
        Mtotal_totals_dict = {}
    
        for length in L:
            
            if np.isin(length,vc.index):
                count = vc[length]
                
                berg = mberg_dict[length]
                i_mtotalm_totals_dict[length] = berg.i_mtotalm.data * count
                
                Mtotal_totals_dict[length] = berg.Mtotal.mean() * count
            
            # print(f'{length}: {Qib_sum*count}')
            
        i_mtotalm_total = np.nansum(list(i_mtotalm_totals_dict.values()))
        Mtotal_total = np.nansum(list(Mtotal_totals_dict.values()))
    
        
    
        date = gpd.pd.to_datetime(date_str)
        aww_temp = np.mean(ctd_ds.temp.sel(tZ=slice(Aww_depth,None))).data
        Q_ib_ds = xr.Dataset(
                            {'Qib':(qib_total),
                              'iceberg_date':(date),
                              'transfer_coeff_factor':(factor),
    
                              'ice_vol': (total_v),
                              'average_aww_temp': (aww_temp),
                              'melt_rate_avg': (i_mtotalm_total),
                              'melt_rate_intergrated': (entire_total_melt),
                              
                              }
            )
        
        
        Q_ib_ds.Qib.attrs = {'units':'W'}
        Q_ib_ds.ice_vol.attrs = {'description': 'Ice Volume below the Aww Depth',
                                 'Units': 'm^3'}
        Q_ib_ds.average_aww_temp.attrs = {'description': 'Average water temp below the Aww boundary',
                                 'Units': 'C'}
        
        Q_ib_ds.melt_rate_avg.attrs = {'description': 'mean over all time, depths, processes for all iceberg classes and all number of icebergs in given iceberg distribution',
                                 'Units': 'm/day'}
        Q_ib_ds.melt_rate_intergrated.attrs = {'description': ' average total volume FW for each time step for all iceberg classes and all number of icebergs in given iceberg distribution ',
                                 'Units': 'm^3/s'}
        
        urel_str = str(u_rel).split('.')[1]
        
        op = f'{out_dir}iceberg_model_output_bug_fix_ctd/{FJORD}/{run_type}/'
        if not os.path.exists(op):
            os.makedirs(op)
        
        Q_ib_ds.to_netcdf(f'{op}{date_str}_{FJORD}_coeff_{factor}_CTD_constant_UREL_{urel_str}.nc')
        