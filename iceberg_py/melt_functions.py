#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:00:25 2023

@author: laserglaciers
"""

import numpy as np
import  scipy
import xarray as xr
from math import ceil
def melt_wave(windu, sst, sea_ice_conc):
    
    sea_state = 1.5 * np.sqrt(windu) + 0.1 * windu
    IceTerm = 1 + np.cos(np.power(sea_ice_conc,3) * np.pi)
    
    melt = (1/12) * sea_state * IceTerm * (sst + 2) # m/day
    melt = melt / 86400 # m/second 
    
    return melt


def melt_solar(solar_rad):
    
    latent_heat = 3.33e5 #J/kg
    rho_i = 917 #kg/m3
    albedo = 0.7
    absorbed = 1 - albedo
    
    melt = absorbed * solar_rad / (rho_i * latent_heat)
    
    return melt
    

def melt_forcedwater(temp_far, salinity_far, pressure_base, U_rel):
    
    # constants from Jenkins
    a = -5.73e-2 # Salinity contribution
    b = 8.32e-2 # constant
    c = -7.61e-4 # pressure contribution C/dbar
    
    GT = 1.1e-3 # 6e-4; %econd value in Silva, first in Jenkins model heat transfer coefficient
    GS = 3.1e-5 # 2.2e-5; %second value in Silva, first in Jenkins model % salt transfer coefficient
    L = 3.35e5  # latent heat of fusion of ice (in J/kg)
    cw = 3974;  # specific heat of water (J/kg/C)
    ci = 2009;  # " of ice
    DT = 15;    # temp difference between iceberg core and bottom surface (this is for Antarctica, maybe less for Greenland?)
    
    T_fp = a * salinity_far + b + c * pressure_base # Freezing point temperature
    T_sh = temp_far - T_fp # Temperature above freezing point
    
    # Quadtratic Terms
    A = (L + DT * ci) / (U_rel * GT * cw)
    B = -((L + DT * ci)) *GS/(GT * cw) -a*salinity_far - T_sh
    C = -U_rel * GS * T_sh
    
    ROOT = np.power(B,2) - 4*A*C
    ROOT = np.where(ROOT<0,np.nan,ROOT)
    
    # Find quadratic roots
    Mtemp1 = (-B + np.sqrt(ROOT)) / (2*A)
    Mtemp2 = (-B - np.sqrt(ROOT)) / (2*A)
    
    melt = np.minimum(Mtemp1,Mtemp2)
    
    # clean data to remove melt rates below freezing point
    mask = T_sh < 0
    melt[mask] = 0
    nan_mask = np.isnan(melt)
    melt[nan_mask] = 0
    
    
    return melt, T_sh, T_fp


def melt_forcedair(T_air, U_rel, L):
    
    T_ice = -4 # ice temperature
    Li = 3.33e5 # latent heat of fusion in ice J/kg
    rho_i = 900 # density of ice
    air_viscosity = 1.46e-5 # kinematic viscosity of air
    air_diffusivity = 2.16e-5 # thermal diffusivity of air
    air_conductivity = 0.0249 # thermal condictivity of air
    cold = T_air < 0 # freezing celsius
    
    Pr = air_viscosity / air_diffusivity # Prandtl number
    Re = np.abs(U_rel) * L / air_viscosity # Reynolds number based on relative air speed
    Nu = 0.058 * (np.power(Re,0.8)) / (np.power(Pr,0.4))
    HF = (1/L) * (Nu * air_conductivity * (T_air - T_ice)) # HEAT FLUX
    
    melt = HF / (rho_i * Li)
    melt[cold] = 0
    
    return melt


def melt_buoyantwater(T_w, S_w, method):
    
    Tf = -0.036 - (0.0499 * S_w) - (0.00011128 * np.power(S_w,2)) # freezing pt of seawater due to S changes
    Tfp = Tf * np.exp(-0.19 * (T_w - Tf)) # freezing point temperature
    
    if method == 'bigg':
        dT = T_w
        mday = 7.62e-3 * dT + 1.3e-3 * np.power(dT,2) 
        
    elif method == 'cis':
        dT = T_w - Tfp
        mday = 7.62e-3 * dT + 1.29e-3 * np.power(dT,2)
        
    melt = mday / 86400
    
    return melt

def keeldepth(L, method):
    
    L_10 = np.round(L/10) * 10 # might have issues with this
    
    barker_La = L_10 <= 160
    hotzel_La = L_10 > 160
    
    # if nargin==1
    #     method = 'auto' # idk what this is from
    
    if method == 'auto':
        # not sure about this
        keel_depth_h = 3.78  * np.power(hotzel_La,0.63) # hotzel
        keel_depth_b = 2.91 * np.power(barker_La,0.71) # barker
        
        return keel_depth_b,keel_depth_h
        
    elif method == 'barker':
        keel_depth = 2.91 * np.power(L_10,0.71)
        
        return keel_depth
    
    elif method == 'hotzel':
        keel_depth = 3.78 * np.power(L_10,0.63)
    
        return keel_depth
    
    elif method == 'constant':
        keel_depth = 0.7 * L_10
        
        return keel_depth
    
    elif method == 'mean':
        
        keel_arr = np.ones(len(L_10),4)

        keel_arr[barker_La,0] = 2.91 * np.power(barker_La,0.71) # barker # feel like these should just be ind columns?
        keel_arr[hotzel_La,0] = 3.78 * np.power(hotzel_La,0.63) # hotzel
        keel_arr[:,1] = 2.91 * np.power(L_10,0.71)
        keel_arr[:,2] = 3.78 * np.power(L_10,0.63)
        keel_arr[:,3] = 0.7 * L_10
        
        mean = np.mean(keel_arr, axis=1)
        
        keel_depth = mean
        
        return keel_depth


def barker_carea(L, keel_depth, dz, LWratio=1.62, method='barker'):
    # #
    # # calculates underwater cross sectional areas using Barker et al. 2004,
    # # and converts to length underwater (for 10 m thicknesses, this is just CrossArea/10) and sail area for K<200, 
    # # for icebergs K>200, assumes tabular shape
    # #
    # # [CArea, UWlength, SailA] = barker_carea(L)
    # #
    # # L is vector of iceberg lengths, 
    # # K is keel depth
    # # (if narg<2), then it calculates keel depths from this using keeldepth.m
    # # dz = layer thickness to use
    # # LWratio: optional argument, default is 1.62:1 L:W, or specify
    # #
    # # all variables in structure "icebergs"
    # #   CA is cross sectional area of each 10 m layer underwater
    # #   uwL is length of this 10 m underwater layer
    # #   Z is depth of layer
    # #   uwW calculated from length using length to width ratio of 1.62:1
    # # 
    # #   also get volumes and masses
    # #
    # # first get keel depth
    
    keel_depth = np.array([keel_depth])
    L = np.array([L])
    if keel_depth == None:
        keel_depth = keeldepth(L,'barker') # K = keeldepth(L,'mean');
        dz = 10
        LWratio = 1.62
    
    # NEED TO FIGURE OUT WHAT THESE ARE AND WRITE IT BETTER! THIS IS NOT THAT GOOD
    if dz == 10: # originally for dz=10 m layers
        a = [9.51,11.17,12.48,13.6,14.3,13.7,13.5,15.8,14.7,11.8,11.4,10.9,10.5,10.1,9.7,9.3,8.96,8.6,8.3,7.95]
        a = np.array(a).reshape((len(a),1))
        
        b = [25.9,107.5,232,344.6,457,433,520,1112,1125,853,931,1007,1080,1149,1216,1281,1343,1403,1460,1515]
        b = -1 * (np.array(b).reshape((len(b),1)))
        
    elif dz == 5:
        # I NEED TO UNDERSTAND WHAT THIS IS DOING BETTER BC IT DOES NOT MAKE SENSE
        a = [9.51,11.17,12.48,13.6,14.3,13.7,13.5,15.8,14.7,11.8,11.4,10.9,10.5,10.1,9.7,9.3,8.96,8.6,8.3,7.95]
        a = np.array(a).reshape((len(a),1))
        
        b = [25.9,107.5,232,344.6,457,433,520,1112,1125,853,931,1007,1080,1149,1216,1281,1343,1403,1460,1515]
        b = -1 * (np.array(b).reshape((len(b),1)))
    
        aa = np.empty(a.T.shape)
        aa[0] = a[0]
        bb = np.empty(b.T.shape)
        bb[0] = b[0]
        
        for i in range(len(a)-1):
            aa[0,i+1] = np.nanmean(a[i:i+2,:])
            bb[0,i+1] = np.nanmean(b[i:i+2,:])
    
        newa = np.empty((40,1)) 
        newa[:] = np.nan
        newb = newa.copy()
        
        newa[::2] = aa.T
        newa[1::2] = a
        
        newb[::2] = bb.T
        newb[1::2] = b
        
        a = newa/2
        b = newb/2
    
    a_s = 28.194; # for sail area
    b_s = -1420.2;    
    
    # initialize arrays
    # icebergs.Z = dz:dz:500; icebergs.Z=icebergs.Z';
    # zlen = length(icebergs.Z);
    # temp = nan.*ones(zlen,length(L));  # 100 layers of 5-m each, so up to 500 m deep berg
    # temps = nan.*ones(1,length(L));  # sail area
    
    z_coord_flat = np.arange(dz,500+dz,dz)
    z_coord = z_coord_flat.reshape(len(z_coord_flat),1)
    depth_layers = xr.DataArray(data=z_coord, coords = {"Z":z_coord_flat},  dims=["Z","X"], name="Z")
    zlen = len(depth_layers.Z)
    # temp = nan.*ones(zlen,length(L))
    # need to make L an array
    
    temp = np.nan * np.ones((zlen, len(L)))
    temps = np.nan * np.ones((1, len(L)))
    
    
    # K_l200 = keel_depth[keel_depth<200] # might cause an issue?
    K_l200 = np.where(keel_depth<=200)[0] # get indices of keel_depth < 200
    # if(~isempty(ind))
    if K_l200.size != 0: # check if empty
        for i in range(len(K_l200)):
            
            kz = keel_depth[i] # keel depth
            # dz_np = np.array([dz],dtype=np.float64)
            kza = np.ceil(kz/dz) # layer index for keel depth
            # kza = ceil(kz,dz) # layer index for keel depth
            
            for nl in range(int(kza)):
                temp[nl,i] = a[nl] * L[K_l200[i]] + b[nl]
                
        temps[K_l200] = a_s * L[K_l200] + b_s
        
        if L < 65:
            temps[L<65] = 0.077 * np.power(L[L<65],2) # fix for L<65, barker 2004
    
    
    # then do icebergs D>200 for tabular
    K_g200 = np.where(keel_depth>200)[0]
    if K_g200.size != 0:
        for i in range(len(K_g200)):
            
            kz = keel_depth[i] # keel depth
            kza = np.ceil(kz/dz) # layer index for keel depth
            
            for nl in range(int(kza)):
                # temp[nl,i] = a[nl] * L[K_g200[i]] + b[nl]
                temp[nl,i] = L[K_g200[i]] * dz
        
        temps[K_g200] = 0.1211 * L[K_g200] * keel_depth[K_g200]
        
    
    cross_area = xr.DataArray(data=temp, coords = {"Z":z_coord_flat}, dims=["Z","X"], name="cross_area")
    # icebergs.uwL = temp./dz; 
    length_layers = xr.DataArray(data=temp/dz, coords = {"Z":z_coord_flat},  dims=["Z","X"], name="uwL")
    
    # now use L/W ratio of 1.62:1 (from Dowdeswell et al.) to get widths I wonder if I can just get widths from Sid's data??
    widths = length_layers.values / LWratio 
    width_layers = xr.DataArray(data = widths, coords = {"Z":z_coord_flat},  dims=["Z","X"], name="uwW")
    
    dznew = dz * np.ones(length_layers.values.shape);
    
    vol = dznew * length_layers.values * width_layers.values
    volume = xr.DataArray(data=vol, coords = {"Z":z_coord_flat},  dims=["Z","X"], name="uwV")
    
    # I am ASSUMING everything is the same size. NEED TO CHECK when I get things running
    icebergs = xr.Dataset(data_vars={'depth':depth_layers,
                                     'cross_area':cross_area,
                                     'uwL':length_layers,
                                     'uwW':width_layers,
                                     'uwV':volume},
                          coords = {'Z': z_coord_flat}
                          )

    
    return icebergs

def init_iceberg_size(L, dz=10, stability_method='equal'):
    # # initialize iceberg size and shapes, based on length
    # # 
    # # given L, outputs all other iceberg parameters
    # # dz : specify layer thickness desired, default is 10m
    # #
    # # Updated to make stable using Wagner et al. threshold, Sept 2017
    # # either load in lengths L or specify here
    # #L = [100:50:1000]';
    # stablility method 'keel' or 'equal' 
    # keel changes keel depth, equal makes width and length equal
    
    keel_depth = keeldepth(L, 'barker')
    
    # now get underwater shape, based on Barker for K<200, tabular for K>200, and 
    ice = barker_carea(L, keel_depth, dz) # LWratio = 1.62 this gives you uwL, uwW, uwV, uwM, and vector Z down to keel depth
    
    # from underwater volume, calculate above water volume
    rho_i = 917 #kg/m3
    rat_i = rho_i/1024 # ratio of ice density to water density
    
    total_volume = (1/rat_i) * np.nansum(ice.uwV,axis=0) #double check axis need rows, ~87% of ice underwater
    sail_volume = total_volume - np.nansum(ice.uwV,axis=0) # sail volume is above water volune
    
    waterline_width = L/1.62
    freeB = sail_volume / (L * waterline_width) # Freeboard height
    # length = L.copy()
    thickness = keel_depth + freeB # total thickness
    deepest_keel = np.ceil(keel_depth/dz) # index of deepest iceberg layer, % ice.keeli = round(K./dz)
    # dz = dzS
    dzk = -1*((deepest_keel - 1) * dz - keel_depth) #
    
    # check if stable
    stability_thresh = 0.92 # from Wagner et al. 2017, if W/H < 0.92 then unstable
    stable_check = waterline_width / thickness[0]
    
    if stable_check > stability_thresh:
            ice['totalV'] = xr.DataArray(data=total_volume[0],name='totalV')
            ice['sailV'] = xr.DataArray(data=sail_volume[0], name='sailV')
            ice['W'] = xr.DataArray(waterline_width, name='W')
            ice['freeB'] = xr.DataArray(freeB[0],name='freeB')
            ice['L'] = xr.DataArray(np.float64(L),name='L')
            ice['keel'] = xr.DataArray(data=keel_depth, name='keel')
            ice['TH'] = xr.DataArray(data=thickness[0], name='thickness')
            ice['keeli'] = xr.DataArray(data=deepest_keel, name='keeli')
            ice['dz'] = xr.DataArray(data=dz, name='dz')
            ice['dzk'] = xr.DataArray(data=dzk, name='dzk')
            
            return ice
    
    
    if stable_check < stability_thresh:
        # Not sure when to use either? MATLAB code has if(0) and if(1) for 'keel' and 'equal'
        if stability_method == 'keel':
            # change keeldepth to be shallower
            print(f'Fixing keel depth for L = {L} m size class')
            
            diff_thick_width = thickness - waterline_width # Get stable thickness
            keel_new = keel_depth - rat_i * diff_thick_width # change by percent of difference
            
            ice = barker_carea(L,keel_new,dz)
            total_volume = (1/rat_i) * np.nansum(ice.uwV,axis=0) #double check axis need rows, ~87% of ice underwater
            sail_volume = total_volume - np.nansum(ice.uwV,axis=0) # sail volume is above water volune
            waterline_width = L/1.62 
            freeB = sail_volume / (L * waterline_width) # Freeboard height
            # length = L.copy()
            thickness = keel_depth + freeB # total thickness
            deepest_keel = np.ceil(keel_depth/dz) # index of deepest iceberg layer, % ice.keeli = round(K./dz)
            # dz = dzS
            dzk = -1*((deepest_keel - 1) * dz - keel_depth) #
            stability = waterline_width/thickness
            
            ice['totalV'] = xr.DataArray(data=total_volume[0],name='totalV')
            ice['sailV'] = xr.DataArray(data=sail_volume[0], name='sailV')
            ice['W'] = xr.DataArray(waterline_width, name='W')
            ice['freeB'] = xr.DataArray(freeB[0],name='freeB')
            ice['L'] = xr.DataArray(np.float64(L),name='L')
            ice['keel'] = xr.DataArray(data=keel_depth, name='keel')
            ice['TH'] = xr.DataArray(data=thickness[0], name='thickness')
            ice['keeli'] = xr.DataArray(data=deepest_keel, name='keeli')
            ice['dz'] = xr.DataArray(data=dz, name='dz')
            ice['dzk'] = xr.DataArray(data=dzk, name='dzk')
            
            return ice
        
        elif stability_method == 'equal':
            # change W to equal L, recalculate volumes
            print(f'Fixing width to equal L, for L = {L} m size class')
            # use L:W ratio of to make stable, set so L:W makes EC=EC_thresh
            
            width_temporary = stability_thresh * thickness[0]
            lw_ratio = np.floor((100*L)/width_temporary)/100 # round down to hundredth place
            
            ice = barker_carea(L, keel_depth, dz, LWratio=lw_ratio)
            
            total_volume = (1/rat_i) * np.nansum(ice.uwV,axis=0) #double check axis need rows, ~87% of ice underwater
            sail_volume = total_volume - np.nansum(ice.uwV,axis=0) # sail volume is above water volune
            waterline_width = L / lw_ratio 
            freeB = sail_volume / (L * waterline_width) # Freeboard height
            # length = L.copy()
            thickness = keel_depth + freeB # total thickness
            deepest_keel = np.ceil(keel_depth/dz) # index of deepest iceberg layer, % ice.keeli = round(K./dz)
            # dz = dzS
            dzk = -1*((deepest_keel - 1) * dz - keel_depth) #

            
            ice['totalV'] = xr.DataArray(data=total_volume[0],name='totalV')
            ice['sailV'] = xr.DataArray(data=sail_volume[0], name='sailV')
            ice['W'] = xr.DataArray(waterline_width, name='W')
            ice['freeB'] = xr.DataArray(freeB[0],name='freeB')
            ice['L'] = xr.DataArray(np.float64(L),name='L')
            ice['keel'] = xr.DataArray(data=keel_depth, name='keel')
            ice['TH'] = xr.DataArray(data=thickness[0], name='thickness')
            ice['keeli'] = xr.DataArray(data=deepest_keel, name='keeli')
            ice['dz'] = xr.DataArray(data=dz, name='dz')
            ice['dzk'] = xr.DataArray(data=dzk, name='dzk')
            EC = ice.W/ice.TH
            
            if EC < stability_thresh:
                raise Exception("Still unstable, check W/H ratios")
            
            return ice
        




def iceberg_melt(L,dz,timespan,ctddata,IceConc,WindSpd,Tair,SWflx,Urelative,do_constantUrel=True
                 do_melt={'wave':True, 'turbw':True, 
                          'turba':True, 'freea':True, 'freew':True
                          }):

    # script to initialize icebergs and melt them through time, also roll them
    # if they fail weeks/mellor stability criterion
    #
    # INPUT: 
    # L = iceberg length(s) in m
    # dz = layer thickness (default is 10)
    # timespan = length to run code, in seconds
    # ctddata = T,S vs. depth in structure, called temp,salt,depth (nz vs. # of casts)
    # IceConc: magnitude of ice conc., 0-1; make empty [] if don't want wave melt on, if size>1 then time series
    # WindSpd: wind speed, either time series or magnitude
    # Tair: air temp; either time series or magnitude
    # SWflx: solar insolation or shortwave flux in W m-2, either time series or magnitude
    # Urelative: relative water velocity, either constant or structure of tadcp,vadcp,zadcp
    # 
    # do_melt = 0/1 to indicate which processes to include
    # [wave  turbw  turba   freea   freew]; if empty then turn all on
    #
    # OUTPUT:
    # out = structure containing all output--see cell below
    #
    # NOTE: some parts of code not working yet: rolling and slab breakoff
    
    # Idk the best way to go about creating the melt outputs. might put into list, dict, or pd dataframe?
    
    diagnostics = False
    
    ice_init = []
    for length in L:
        ice_init.append(init_iceberg_size(length,dz=dz))
    
    
    nz = len(ice_init[0].Z)
    dz = ice_init[0].dz # ice_init
    dt = 86400
    t = np.arange(dt,timespan+dt,dt)
    nt = len(t)
    
    if len(IceConc) == 1:
        sice =  IceConc * np.ones(t.shape) # if want varying need to make vector in time
    else:
        sice = IceConc
    
    if len(WindSpd) == 1:
        WindV =  WindSpd * np.ones(t.shape) # could make varying to include katabatics
    else:
        WindV = WindV
    
    if len(Tair) == 1:
        Ta =  Tair * np.ones(t.shape) # if want varying need to make vector in time
    else:
        Ta = Tair
    
    if len(SWflx) == 1:
        Srad =  SWflx * np.ones(t.shape) # if want varying need to make vector in time
    else:
        Srad = SWflx
    
    
    m, n = np.shape(ctddata.temp) # need to see what format CTD data will be provided in
    
    if n>1:
        temp = np.nanmean(ctddata.temp,axis=1) #double check axis
        salt = np.nanmean(ctddata.salt,axis=1)
    elif n == 1:
        temp = ctddata.temp
        salt = ctddata.salt
    
    ctdz = ctddata.salt
    
    # WATER VELOCITY, should be horizontal currents and vertical velocities (plumes)
    
    if do_constantUrel:
        Urel = Urelative * np.ones(nz,ni,nt)
    
    else: # load ADCP pulse events here, based on SF ADCP data
        Urel = np.nan * np.ones(nz,ni,nt)
        kki = pass #dsearchn(Urelative.zadcp(:),ceil(ice_init(1).K));
        
        if IceConc == 1:
            # if sea ice conc = 100%, assume we're talking about melange and don't take out mean horizontal flow
            vmadcp = Urelative.vadc
    # stopped on line 110
    return






"""
NEED TO CODE
keeldepth - done
init_iceberg_size - done
barker_carea - done
iceberg_melt
"""