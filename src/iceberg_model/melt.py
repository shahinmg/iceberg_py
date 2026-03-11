import numpy as np
import numpy.matlib
from scipy.interpolate import interp1d, interp2d
from scipy.spatial import cKDTree, KDTree
import xarray as xr
from math import ceil
from sklearn.linear_model import LinearRegression


# Import constants
from . import constants as const


def melt_wave(wind_speed, sea_surface_temp, sea_ice_concentration):
    """
    Calculate wave erosion melt rate using Silva et al. formulation.
    
    Parameters
    ----------
    wind_speed : float or ndarray
        Wind speed in m/s (relative to water, assumed Wind >> water speeds)
    sea_surface_temp : float or ndarray
        Surface water temperature in °C
    sea_ice_concentration : float or ndarray
        Sea ice concentration, fraction 0-1 (0 = no ice, 1 = full coverage)
    
    Returns
    -------
    melt_rate : float or ndarray
        Melt rate in m/s
    
    References
    ----------
    Silva et al., Bigg et al. (1997), Gladstone et al. (2001)
    
    """
    # Wave height estimate from wind speed
    sea_state = (const.WAVE_HEIGHT_WIND_COEFF_1 * np.sqrt(wind_speed) + 
                 const.WAVE_HEIGHT_WIND_COEFF_2 * wind_speed)
    
    # Ice concentration effect (corrected from paper typo)
    ice_term = 1 + np.cos(np.power(sea_ice_concentration, 3) * np.pi)
    
    # Melt rate: m/day
    melt_rate = ((1 / const.WAVE_MELT_DIVISOR) * sea_state * ice_term * 
                 (sea_surface_temp + const.WAVE_MELT_TEMP_OFFSET))
    
    # Convert to m/s
    melt_rate = melt_rate / const.SECONDS_PER_DAY
    
    return melt_rate


def melt_solar(solar_radiation):
    """
    Calculate melt from solar radiation (above water).
    
    Based on Condron's mitberg formulation. Affects thickness above water only.
    Assumes constant albedo.
    
    Parameters
    ----------
    solar_radiation : float or ndarray
        Solar radiation flux downward (SW and LW) in W/m²
    
    Returns
    -------
    melt_rate : float or ndarray
        Melt rate in m/s
    
    Notes
    -----
    Albedo assumed to be 0.7 for ice
    """
    # Energy absorbed = solar flux * (1 - albedo)
    absorbed_energy = const.SOLAR_ABSORPTION_FRACTION * solar_radiation
    
    # Convert energy to melt: Q = ρ * L * m  =>  m = Q / (ρ * L)
    # where Q is heat flux (W/m²), ρ is ice density (kg/m³), L is latent heat (J/kg)
    melt_rate = absorbed_energy / (const.RHO_ICE * const.LATENT_HEAT_FUSION)
    
    return melt_rate  # m/s


def melt_forcedwater(temp_far, salinity_far, pressure_base, velocity_relative, 
                     factor=None, use_constant_tf=False, constant_tf=None):
    """
    Calculate forced convection melt in water (turbulent melt).
    
    Uses Silva et al. equation with parameters from Holland and Jenkins.
    
    Parameters
    ----------
    temp_far : float or ndarray
        Far-field water temperature in °C
    salinity_far : float or ndarray
        Far-field salinity in PSU
    pressure_base : float or ndarray
        Pressure at base in dbar
    velocity_relative : float or ndarray
        Water speed relative to iceberg surface in m/s
    factor : float, optional
        Adjustment factor for transfer coefficients (Jackson et al. 2020).
        Default is 1.
    use_constant_tf : bool, optional
        If True, use constant thermal forcing instead of calculating from T/S
    constant_tf : float, optional
        Constant thermal forcing value if use_constant_tf is True
    
    Returns
    -------
    melt_rate : float or ndarray
        Melt rate in m/s (positive = melting)
    thermal_forcing : float or ndarray
        Temperature above freezing point in °C (or constant_tf if specified)
    freezing_point : float or ndarray or None
        Freezing point temperature in °C (None if using constant_tf)
    
    References
    ----------
    Holland & Jenkins (1999), Silva et al., Jackson et al. (2020)
    """
    # Use default factor if not provided
    if factor is None:
        factor = const.DEFAULT_TRANSFER_COEFFICIENT_FACTOR
    
    # Transfer coefficients (adjusted by factor)
    heat_transfer = const.HEAT_TRANSFER_COEFFICIENT_GT * factor
    salt_transfer = const.SALT_TRANSFER_COEFFICIENT_GS * factor
    
    # Thermal forcing calculation
    if use_constant_tf:
        thermal_forcing = constant_tf
        freezing_point = None
    else:
        # Freezing point: T_fp = a*S + b + c*P
        freezing_point = (const.FREEZING_POINT_SALINITY_COEFF * salinity_far + 
                         const.FREEZING_POINT_CONSTANT + 
                         const.FREEZING_POINT_PRESSURE_COEFF * pressure_base)
        thermal_forcing = temp_far - freezing_point
    
    # Quadratic formula coefficients: A*m² + B*m + C = 0
    A = ((const.LATENT_HEAT_FUSION + const.TEMPERATURE_DIFFERENCE_CORE_SURFACE * const.SPECIFIC_HEAT_ICE) / 
         (velocity_relative * heat_transfer * const.SPECIFIC_HEAT_WATER))
    
    B = -1 * (((const.LATENT_HEAT_FUSION + const.TEMPERATURE_DIFFERENCE_CORE_SURFACE * const.SPECIFIC_HEAT_ICE) * 
               salt_transfer / (heat_transfer * const.SPECIFIC_HEAT_WATER)) - 
              const.FREEZING_POINT_SALINITY_COEFF * salinity_far - thermal_forcing)
    
    C = -velocity_relative * salt_transfer * thermal_forcing
    
    # Solve quadratic equation
    discriminant = np.power(B, 2) - 4 * A * C
    discriminant = np.where(discriminant < 0, np.nan, discriminant)
    
    # Find quadratic roots
    root1 = (-B + np.sqrt(discriminant)) / (2 * A)
    root2 = (-B - np.sqrt(discriminant)) / (2 * A)
    
    melt_rate = np.minimum(root1, root2)
    
    # Clean data: remove melt rates when water is below freezing
    mask = thermal_forcing < 0
    melt_rate = np.where(mask, 0, melt_rate)
    melt_rate = np.where(np.isnan(melt_rate), 0, melt_rate)
    
    # Make melt rate positive (convention: positive = melting)
    melt_rate = -1 * melt_rate
    
    return melt_rate, thermal_forcing, freezing_point


def melt_forcedair(air_temp, air_velocity_relative, iceberg_length):
    """
    Calculate forced convection melt in air.
    
    Based on Condron's mitberg formulation using Nusselt number relationships.
    
    Parameters
    ----------
    air_temp : float or ndarray
        Air temperature in °C
    air_velocity_relative : float or ndarray
        Air speed relative to iceberg in m/s
    iceberg_length : float
        Characteristic length of iceberg in meters
    
    Returns
    -------
    melt_rate : float or ndarray
        Melt rate in m/s (0 if air_temp < 0°C)
    
    Notes
    -----
    Uses Nusselt-Reynolds-Prandtl relationship for forced convection
    """
    is_freezing = air_temp < 0
    
    # Dimensionless numbers
    prandtl_number = const.PRANDTL_NUMBER
    reynolds_number = np.abs(air_velocity_relative) * iceberg_length / const.AIR_KINEMATIC_VISCOSITY
    
    # Nusselt number: Nu = 0.058 * Re^0.8 / Pr^0.4
    nusselt_number = (const.NUSSELT_COEFF * 
                     np.power(reynolds_number, const.NUSSELT_REYNOLDS_EXPONENT) / 
                     np.power(prandtl_number, const.NUSSELT_PRANDTL_EXPONENT))
    
    # Heat flux: Q = (Nu * k / L) * (T_air - T_ice)
    heat_flux = ((nusselt_number * const.AIR_THERMAL_CONDUCTIVITY / iceberg_length) * 
                 (air_temp - const.ICE_SURFACE_TEMPERATURE))
    
    # Convert heat flux to melt rate: m = Q / (ρ * L)
    melt_rate = heat_flux / (const.RHO_ICE * const.LATENT_HEAT_FUSION)
    
    # No melting if air temperature is below freezing
    if np.isscalar(is_freezing):
        if is_freezing:
            melt_rate = 0
    else:
        melt_rate = np.where(is_freezing, 0, melt_rate)
    
    return melt_rate  # m/s


def melt_buoyantwater(water_temp, salinity, method='cis', 
                      use_constant_tf=False, constant_tf=None):
    """
    Calculate buoyant (free) convection melt along vertical walls.
    
    Uses Bigg et al. (1997) / El-Tahan formulation.
    
    Parameters
    ----------
    water_temp : float or ndarray
        Water temperature in °C
    salinity : float or ndarray
        Salinity in PSU
    method : {'bigg', 'cis'}, optional
        Which formulation to use. Default is 'cis'.
    use_constant_tf : bool, optional
        If True, use constant thermal forcing
    constant_tf : float, optional
        Constant thermal forcing value
    
    Returns
    -------
    melt_rate : float or ndarray
        Melt rate in m/s
    
    References
    ----------
    Bigg et al. (1997), El-Tahan formulation
    """
    # Calculate freezing point temperature
    # T_f = -0.036 - 0.0499*S - 0.0001128*S²
    temp_freeze = (const.BIGG_FP_COEFF_1 + 
                   const.BIGG_FP_COEFF_2 * salinity + 
                   const.BIGG_FP_COEFF_3 * np.power(salinity, 2))
    
    # Adjust freezing point: T_fp = T_f * exp(-0.19 * (T - T_f))
    freezing_point = temp_freeze * np.exp(const.BIGG_FP_EXPONENT_COEFF * (water_temp - temp_freeze))
    
    # Calculate thermal forcing
    if method == 'bigg':
        if use_constant_tf:
            thermal_forcing = constant_tf
        else:
            thermal_forcing = water_temp
        
        # Melt rate: m = 7.62e-3 * dT + 1.3e-3 * dT² (m/day)
        melt_rate_per_day = (const.BUOYANT_MELT_LINEAR_COEFF * thermal_forcing + 
                            const.BUOYANT_MELT_QUADRATIC_COEFF_BIGG * np.power(thermal_forcing, 2))
    
    elif method == 'cis':
        if use_constant_tf:
            thermal_forcing = constant_tf
            melt_rate_per_day = (const.BUOYANT_MELT_LINEAR_COEFF * thermal_forcing + 
                                const.BUOYANT_MELT_QUADRATIC_COEFF_CIS * np.power(thermal_forcing, 2))
        else:
            thermal_forcing = water_temp - freezing_point
            melt_rate_per_day = (const.BUOYANT_MELT_LINEAR_COEFF * thermal_forcing + 
                                const.BUOYANT_MELT_QUADRATIC_COEFF_CIS * np.power(thermal_forcing, 2))
    
    # Convert from m/day to m/s
    melt_rate = melt_rate_per_day / const.SECONDS_PER_DAY
    
    return melt_rate


# def iceberg_melt(L,dz,timespan,ctddata,IceConc,WindSpd,Tair,SWflx,Urelative, do_constantUrel=False, factor=1, quiet=True,
#                  do_roll = True, do_slab = True, use_constant_tf = False, constant_tf = None,
#                  do_melt={'wave':True, 'turbw':True, 
#                           'turba':True, 'freea':True, 'freew':True
#                           }):

#     # script to initialize icebergs and melt them through time, also roll them
#     # if they fail weeks/melt or stability criterion
#     #
#     # INPUT: 
#     # L = iceberg length(s) in m
#     # dz = layer thickness (default is 10)
#     # timespan = length to run code, in seconds
#     # ctddata = T,S vs. depth in structure, called temp,salt,depth (nz vs. # of casts)
#     # IceConc: magnitude of ice conc., 0-1; make empty [] if don't want wave melt on, if size>1 then time series
#     # WindSpd: wind speed, either time series or magnitude
#     # Tair: air temp; either time series or magnitude
#     # SWflx: solar insolation or shortwave flux in W m-2, either time series or magnitude
#     # Urelative: relative water velocity, either constant or structure of tadcp,vadcp,zadcp
#     # 
#     # do_melt = 0/1 to indicate which processes to include
#     # [wave  turbw  turba   freea   freew]; if empty then turn all on
#     #
#     # OUTPUT:
#     # out = structure containing all output--see cell below
#     #
#     # NOTE: some parts of code not working yet: slab breakoff
    
#     # Idk the best way to go about creating the melt outputs. might put into list, dict, or pd dataframe?
    
#     diagnostics = False
    
#     # make inputs arrays
#     L = np.array([L])
#     IceConc = np.array([IceConc])
#     WindSpd = np.array([WindSpd])
#     Tair = np.array([Tair])
#     SWflx = np.array([SWflx])
    
#     ice_init = []
#     for length in L:
#         ice_init.append(init_iceberg_size(length,dz=dz))
    
    
#     nz = len(ice_init[0].Z)
#     dz = ice_init[0].dz # ice_init
#     dt = 86400
#     t = np.arange(dt,timespan+dt,dt)
#     nt = len(t)
#     ni = len(L)
    
    
    
#     if len(IceConc) == 1:
#         sice =  IceConc * np.ones(t.shape) # if want varying need to make vector in time
#     else:
#         sice = IceConc
    
#     if len(WindSpd) == 1:
#         WindV =  WindSpd * np.ones(t.shape) # could make varying to include katabatics
#     else:
#         WindV = WindV
    
#     if len(Tair) == 1:
#         Ta =  Tair * np.ones(t.shape) # if want varying need to make vector in time
#     else:
#         Ta = Tair
    
#     if len(SWflx) == 1:
#         Srad =  SWflx * np.ones(t.shape) # if want varying need to make vector in time
#     else:
#         Srad = SWflx
    
    
#     m, n = np.shape(ctddata.temp) # need to see what format CTD data will be provided in
    
#     if n>1:
#         temp = np.nanmean(ctddata.temp,axis=1) #double check axis
#         salt = np.nanmean(ctddata.salt,axis=1)
#     elif n == 1:
#         temp = ctddata.temp.data.flatten()
#         salt = ctddata.salt.data.flatten()
    
#     ctdz = ctddata.depth
#     ctdz_flat = ctdz.T.to_numpy().flatten()
#     # WATER VELOCITY, should be horizontal currents and vertical velocities (plumes)
    
#     if do_constantUrel:
#         Urel = Urelative * np.ones((nz,ni,nt))
#         Urel_unadj = Urel.copy()
        
#     elif do_constantUrel == False: # load ADCP pulse events here, based on SF ADCP data
#         Urel = np.nan * np.ones((nz,ni,nt))
#         Urel_unadj = Urel.copy()
#         # kki = 1 #dsearchn(Urelative.zadcp(:),ceil(ice_init(1).K));
#         kdt = cKDTree(Urelative.zadcp[:]) # https://stackoverflow.com/questions/66494042/dsearchn-equivalent-in-python
#         pq = np.ceil(ice_init[0].keel)
#         kki = kdt.query(pq)[-1]
        
#         if IceConc == 1:
#             # if sea ice conc = 100%, assume we're talking about melange and don't take out mean horizontal flow
#             vmadcp = Urelative.vadcp
                
#         else:
#             # for drifting icebergs, take out mean horizontal flow
#             vmadcp = Urelative.vadcp - np.matlib.repmat(np.nanmean(Urelative.vadcp[0:kki+1,:],axis=0),len(Urelative.zadcp),1)
    
#         # make zero below keel depth to be certain
#         vmadcp_unadj = vmadcp.copy()
#         vmadcp[kki+1:,:] = 0
#         vmadcp = np.abs(vmadcp) # speed
#         # add in vertical velocity if any (wvel in Urelative structure)
        
#         vmadcp = vmadcp + Urelative.wvel.values[0] * np.ones(np.shape(vmadcp)) # (right now wvel constant in time/space)
#         # vmadcp_unadj = np.abs(vmadcp_unadj) + Urelative.wvel.values[0] * np.ones(np.shape(vmadcp_unadj))
#         vmadcp_unadj = vmadcp_unadj + Urelative.wvel.values[0] * np.ones(np.shape(vmadcp_unadj))
    
    
#         # interpolate to Urel
#         # Urel[:,0,:] = interp2d(Urelative.tadcp, Urelative.zadcp, vmadcp, np.arange(0,nt), ice_init[0].Z) # double check length of nt #interp2d will be depreciated
#         interp2d_func = interp2d(Urelative.tadcp.values.flatten(), Urelative.zadcp.values.flatten(), vmadcp)
#         interp2d_func_unadj = interp2d(Urelative.tadcp.values.flatten(), Urelative.zadcp.values.flatten(), vmadcp_unadj)
        
#         Urel[:,0,:] = interp2d_func(np.arange(1,nt+1),ice_init[0].Z.to_numpy()) # this interpolates the Urel at specific times and depths
#         Urel_unadj[:,0,:] = interp2d_func_unadj(np.arange(1,nt+1),ice_init[0].Z.to_numpy())
    
#     # interp2d = RegularGridInterpolator((Urelative.tadcp, Urelative.zadcp, vmadcp))
    
#     # set up melt volume arrays
#     Mwave = np.zeros((ni,nt)) # melt volume for waves, affects just top layer
#     mw = Mwave.copy() # need to find out what this is
#     ma = Mwave.copy() # need to find out what this is
#     ms = Mwave.copy() # need to find out what this is
#     wave_height = np.zeros((ni,nt))
    
#     Mturbw = np.zeros((nz,ni,nt)) # forced convection underwater, acts on side and base
#     Mturba = np.zeros((ni,nt)) # forced convection in air, acts on sides and top
#     Mfreea = np.zeros((ni,nt)) # melting in air, reduces thickness only
#     Mfreew = np.zeros((nz,ni,nt)) # buoyant convection, only on sides
    
#     mtw = Mturbw.copy()
#     mb = Mturbw.copy()
    
#     # set up time dependent iceberg arrays
#     VOL = np.nan * np.zeros((ni,nt)) # total iceberg volume
#     LEN = np.nan * np.zeros((ni,nt)) # iceberg length
#     WIDTH = np.nan * np.zeros((ni,nt)) # iceberg width
#     THICK = np.nan * np.zeros((ni,nt)) # iceberg thickness
#     FREEB = np.nan * np.zeros((ni,nt)) # iceberg freeboard
#     KEEL = np.nan * np.zeros((ni,nt)) # iceberg keel
#     SAILVOL = np.nan * np.zeros((ni,nt)) # iceberg above water volume
#     DZKt = np.nan * np.zeros((ni,nt))
#     UWVOL = np.nan * np.zeros((nz,ni,nt)) # underwater volume, depth dependent
#     UWL = np.nan * np.zeros((nz,ni,nt)) # underwater length, depth dependent
#     UWW = np.nan * np.zeros((nz,ni,nt)) # underwater width, depth dependent
    
#     # put in first values
#     for i,iceberg in enumerate(ice_init):
#         VOL[i,0] = iceberg.totalV
#         LEN[i,0] = iceberg.L
#         WIDTH[i,0] = iceberg.W
#         THICK[i,0] = iceberg.TH
#         FREEB[i,0] = iceberg.freeB
#         KEEL[i,0] = iceberg.keel
#         SAILVOL[i,0] = iceberg.sailV
#         UWVOL[:,i,0] = iceberg.uwV.to_numpy().flatten()
#         UWL[:,i,0] = iceberg.uwL.to_numpy().flatten()
#         UWW[:,i,0] = iceberg.uwW.to_numpy().flatten()
    
#     # Start melting
    
#     for i,iceberg in enumerate(ice_init):
#         # get iceberg
#         # add DZKt bc idk how else to do this
#         iceberg['dzkt'] = xr.DataArray(data=DZKt, name='DZKt', coords = {"time":t},  dims=["X","time"])
#         iceberg['dzkt'].values[i,0] = iceberg.dzk 
        
#         #use ice just initializes the ice_init values
#         depth = iceberg.depth.copy().values
#         uwL = iceberg.uwL.copy().values
#         uwW = iceberg.uwW.copy().values
#         uwV = iceberg.uwV.copy().values
#         totalV = iceberg.totalV.copy().values
#         sailV = iceberg.sailV.copy().values
#         W = iceberg.W.copy().values
#         freeB = iceberg.freeB.copy().values
#         L = iceberg.L.copy().values
#         keel = iceberg.keel.copy().values
#         TH = iceberg.TH.copy().values
#         keeli = iceberg.keeli.copy().values
#         dz = iceberg.dz.copy().values
#         dzk = iceberg.dzk.copy().values
#         dzkt = iceberg.dzkt.copy().values
        
#         for j in range(1,nt): # iterate over time
#             # start calculating melt, get melt rates for each process included, then update at end
#             keeli = int(np.ceil(iceberg.keel/dz))
#             if do_melt['wave']:
                
#                 # SST = np.nanmean(interp1d(ctdz, temp))
#                 # SST = nanmean(interp1(ctdz,temp,0:5)); %0-10 m temp
                
#                 SST_func = interp1d(ctdz_flat, temp) # 0-10 m temp
#                 SST = np.nanmean(SST_func(np.arange(1,6))) # 0 - 10 m temp
                
#                 wave_height[i,j] = 0.010125 * np.power((np.abs(WindV[j])),2) # assume wind >> ocean vel, this estimates wave height
#                 WH_depth = np.minimum(freeB, 5 * wave_height[i,j])
#                 # apply 1/2 mw to L and 1/2 mw to uwL(1,:)
#                 mw[i,j] = melt_wave(WindV[j], SST, sice[j]) # m/s though I need to check units of data source
#                 mw[i,j] = mw[i,j] * dt # m/day
                
#                 top_length = np.nanmean([float(L), uwL[0][0]]) # mean of length and first layer underwater
#                 Mwave[i,j] = 1 * (mw[i,j] * WH_depth * top_length) + 1 * (mw[i,j] *WH_depth * top_length) # 1 lengths 1 widths (coming at it obliquely) confused by this
                
#                 # base on wave height estimate, to get right volume taken off but doesn't do L right then! FIX (confused by this -MS)
#                 mwabove = WH_depth / freeB
#                 mwbelow = WH_depth / dz
                
#             else:
#                 mw[i,j] = 0
#                 mwabove = 0
#                 mwbelow = 0
                    
#             if do_melt['turbw']:
#                 # apply melt for each depth level of the iceberg
#                 for k in range(keeli-1):
#                     T_far_func = interp1d(ctdz_flat,temp) # interp1(ctdz,temp,Z(k));
#                     T_far = T_far_func(depth[k])
                    
#                     S_far_func = interp1d(ctdz_flat,salt) # interp1(ctdz,salt,Z(k));
#                     S_far = S_far_func(depth[k])
                    
#                     mtw[k,i,j], T_sh, T_fp = melt_forcedwater(T_far, S_far, depth[k],Urel[k,i,j],factor=factor,
#                                                               use_constant_tf = use_constant_tf, constant_tf=constant_tf)
                    
#                     mtw[k,i,j] = mtw[k,i,j] * dt
#                     Mturbw[k,i,j] = 2 * (mtw[k,i,j] * dz * uwL[k]) + 1 * (mtw[k,i,j] * dz * uwW[k])
                    
#                 mtw[keeli-1,i,j], T_sh, T_fp = melt_forcedwater(T_far, S_far, depth[keeli-1],Urel[keeli-1,i,j],factor=factor, 
#                                                                 use_constant_tf = use_constant_tf, constant_tf=constant_tf)
#                 mtw[keeli-1,i,j] = mtw[keeli-1,i,j] * dt # m/day
#                 dz_keel = -1*((keeli-1) * dz - keel) # final layer depth
#                 # Calculate melt at Keel layer
#                 Mturbw[keeli-1,i,j] = 2 * (mtw[keeli-1,i,j] * dz_keel * uwL[keeli-1]) + 1 * (mtw[keeli-1,i,j] * dz_keel * uwW[keeli-1]) 
                    
                 
#             else:
#                 mtw[:nz,i,j] = 0
            
#             if do_melt['turba']:
                
#                 ma[i,j] = melt_forcedair(Ta[j], WindV[j], L)
#                 ma[i,j] = ma[i,j] * dt # melt rate in m/s
#                 Mturba[i,j] = (2 * (ma[i,j] * dz * L)  # two lengths
#                  + 1 * (ma[i,j] * dz * W)  # once width, lee side does not count
#                     + 0.5 * (ma[i,j] * L * W)) # half of surface
                
#             else:
#                 ma[i,j] = 0
                
#             if do_melt['freea']:
#                 ms[i,j] = melt_solar(Srad[j])
#                 ms[i,j] = ms[i,j] * dt # melt rate m/s
#                 Mfreea[i,j] = (ms[i,j] * W * L) # only melts top surface area
                
#             else:
#                 ms[i,j] = 0
                
#             if do_melt['freew']:
                
#                 for k in range(keeli-1):
                    
#                    T_far_func = interp1d(ctdz_flat,temp) # interp1(ctdz,temp,Z(k));
#                    T_far = T_far_func(depth[k])
                   
#                    S_far_func = interp1d(ctdz_flat,salt) # interp1(ctdz,salt,Z(k));
#                    S_far = S_far_func(depth[k]) # giving slightly different result than matlab
                   
#                    mb[k,i,j] = melt_buoyantwater(T_far, S_far, 'cis', use_constant_tf = use_constant_tf, 
#                                                  constant_tf=constant_tf) # bigg method, then S doesn't matter
#                    mb[k,i,j] = mb[k,i,j] * dt
#                    # Mfreew[k,i,j] = 2 * (((mb[k,i,j]) * dz * uwL[k][0]) # 2 lenghts
#                    #                       + 2 *(mb[keeli,i,j]) * dz * uwW[k])
#                    Mfreew[k,i,j] =  2 * (mb[k,i,j] * dz * uwL[k][0]) + 2 *(mb[k,i,j] * dz * uwW[k])
                   
#                 # dz_keel
#                 dz_keel = -1 * ((keeli-1)) * dz - keel # not sure about keeli -1
#                 Mfreew[keeli-1,i,j] = 2 * ((mb[keeli-1,i,j] * dz_keel * uwL[keeli-1])
#                                          + 2 * (mb[keeli-1,i,j] * dz_keel * uwW[keeli-1]))
                
#             else:
#                 mb[:nz,i,j] = 0
                
#             freeB = freeB - ms[i,j] - ma[i,j]
#             keel = keel - mtw[keeli-1,i,j]
#             TH = keel + freeB
            
#             # reduce thickness on sides, do one L and update W's accordingly
            
#             mult = 2 # takes melt off each side of L; original paper had mult = 1
            
#             uwL[0] = uwL[0] - mult * mw[i,j] * (mwbelow/1)
#             # putting all mw at L means taking out too much Volume, b/c it is freeB high
#             L = L - mult * ma[i,j] - mult  * mw[i,j] * (mwabove/1) #/1 idk??
            
#             #this really slow
#             for k in range(0,keeli+1):
#                 uwL[k] = uwL[k] - mult * mtw[k,i,j] - mult * mb[k,i,j]
    
#             ## FIX ?? - idk what to fix. this is an original comment in the code - ms
#             uwW = uwL / 1.62 # update widths
#             W = (L/1.62)
        
#             rho_i = 917
#             ratio_i = rho_i/1024 # ratio of ice density to water density 
            
#             keel_index_new = int(np.ceil(keel/dz))
            
#             if keel_index_new < keeli:
#                 # if quiet == False:
#                 #     print(f'Removing keel layer at timestep {j}')
#                 uwL[keeli-1] = np.nan
#                 uwW[keeli-1] = np.nan
#                 uwV[keeli-1] = np.nan
#                 keeli = keel_index_new
            
    
#             #update values

#             uwV[:keeli-1] = dz * uwL[:keeli-1] * uwW[:keeli-1]
#             dzkt[i,j] = -1 * ((keeli-1) * dz - keel)
#             uwV[keeli-1] = dzkt[i,j] * uwL[keeli-1] * uwW[keeli-1]
#             sailV = freeB * L * W
#             totalV = np.nansum(uwV) + sailV
            
#             sailV = (1 - ratio_i) * totalV
#             freeB = sailV / (L * W)
#             keel = TH - freeB
        
#         # check stability, roll, and update 
#             if do_roll:
#                 width_stability = 0.7
#                 l_thick_ratio = L / TH
                
#                 if l_thick_ratio < width_stability:
#                     print('iceberg rolling')
                    
#                     TH = L 
#                     L = np.sqrt(totalV / (TH / 1.62))
#                     W = L / 1.62
#                     freeB = (1 - ratio_i) * TH
#                     totalV = (1 / ratio_i) * sailV
#                     keel = TH - freeB
#                     keeli = np.ceil(keel/dz)
#                     uwL[int(keeli+1):] = np.nan
#                     uwW[int(keeli+1):] = np.nan
#                     uwV = dz * uwL * uwW
                    
    
#             # output time dependent parameters
#             VOL[i,j] = totalV
#             LEN[i,j] = L
#             WIDTH[i,j] = W
#             THICK[i,j] = TH
#             FREEB[i,j] = freeB
#             KEEL[i,j] = keel
#             SAILVOL[i,j] = sailV
#             UWVOL[:,i,j] = uwV.flatten()
#             UWL[:,i,j] = uwL.flatten()
#             UWW[:,i,j] = uwW.flatten()

#             vol_diff = np.round(np.diff(VOL[i,j-1:j+1]))
#             # print(f'{vol_diff}')
#             if diagnostics:
#                 print(f'dt = {j}\nKeel depth = {keel:.2f}\nLength = {L:.2f}\n'+\
#                       f'Sail Volume = {sailV:8.0f} Free Board = {freeB:.2f}\n'+\
#                           f'Volume Difference = {vol_diff:8.0f} DZf = {dzk[i,j]:3.1f}')
                        
        
#     # convert meltwater volumes to liquid freshwater. Convert from timestep
#     # originally in units of m3/day 
#     # units of dt to m3/s
#     rho_i_fw_ratio = rho_i / 1000
#     Mwave = (rho_i_fw_ratio * Mwave) / dt
#     Mfreea = (rho_i_fw_ratio * Mfreea) / dt
#     Mturbw = (rho_i_fw_ratio * Mturbw) / dt
#     Mturba = (rho_i_fw_ratio * Mturba) / dt
#     Mfreew = (rho_i_fw_ratio * Mfreew) / dt
    
#     Mtotal = np.zeros((ni,nt))
    
#     # sum all the fresh water
#     for i in range(len([ni])):
#         Mtotal[i,:] = (Mwave[i,:] + Mfreea[i,:] + Mturba[i,:] + np.nansum(np.squeeze(Mturbw[:,i,:]),axis=0) + np.nansum(np.squeeze(Mfreew[:,i,:]),axis=0)).reshape((ni,nt))
                                                                        
    
#     i_mtotalm = np.nanmean(mw) + np.nanmean(mb) + np.nanmean(ms) + np.nanmean(ma) + np.nanmean(mtw) # mean over all time, depths, processes  in m/day
#     # i_mtotalsum = np.nansum(mw) + np.nansum(mb) + np.nansum(ms) + np.nansum(ma) + np.nansum(mtw) # sum over all time, depths, processes  in m/day

    
#     # set up output
    
#     # integrated melt terms
#     iceberg['Mwave'] = xr.DataArray(data=Mwave, name='Mwave', coords = {"time":t},  dims=["X","time"], attrs={'Description':'Integrated wave melt',
#                                                                                                               'Units': 'm3/s'})
    
#     iceberg['Mfreea'] = xr.DataArray(data=Mfreea, name='Mfreea', coords = {"time":t},  dims=["X","time"], attrs={'Description':"Integrated melt from solar radiation in air, based on Condron's mitberg formulation",
#                                                                                                                  'Units': 'm3/s'})
    
#     iceberg['Mturbw'] = xr.DataArray(data=Mturbw, name='Mturbw', coords = {"time":t,"Z":ice_init[0].Z.values},  dims=["Z","X","time"],attrs={'Description':"Integrated forced water melt",
#                                                                                                                  'Units': 'm3/s'})
    
#     iceberg['Mturba'] = xr.DataArray(data=Mturba, name='Mturba', coords = {"time":t},  dims=["X","time"],attrs={'Description':"Integrated Forced convection in air, based on Condron's mitberg formulation",
#                                                                                                                  'Units': 'm3/s'})
    
#     iceberg['Mfreew'] = xr.DataArray(data=Mturbw, name='Mfreew', coords = {"time":t,"Z":ice_init[0].Z.values},  dims=["Z","X","time"], attrs={'Description':"Integrated buoyant convection along sidewalls in water, based on bigg (condron)",
#                                                                                                                  'Units': 'm3/s'})
    
#     iceberg['Mtotal'] = xr.DataArray(data=Mtotal, name='Mtotal', coords = {"time":t},  dims=["X","time"], attrs={'Description':"total volume FW for each time step",
#                                                                                                                  'Units': 'm3/s'})
    
#     # melt terms in m/day 
    
#     iceberg['i_mwave'] = xr.DataArray(data=mw, name='i_mwave', coords = {"time":t},  dims=["X","time"], attrs={'Description':"wave melt",
#                                                                                                                  'Units': 'm/day'})
    
    
#     iceberg['i_mfreea'] = xr.DataArray(data=ms, name='i_mfreea', coords = {"time":t},  dims=["X","time"], attrs={'Description':"melt from solar radiation in air, based on Condron's mitberg formulation",
#                                                                                                                  'Units': 'm/day'})
    
    
#     iceberg['i_mturbw'] = xr.DataArray(data=mtw, name='i_mturbw', coords = {"time":t,"Z":ice_init[0].Z.values},  dims=["Z","X","time"], attrs={'Description':"forced water melt",
#                                                                                                                  'Units': 'm/day'})
    
    
#     iceberg['i_mfreew'] = xr.DataArray(data=mb, name='i_mfreew', coords = {"time":t,"Z":ice_init[0].Z.values},  dims=["Z","X","time"], attrs={'Description':"buoyant convection along sidewalls in water, based on bigg (condron)",
#                                                                                                                  'Units': 'm/day'})
    
    
#     iceberg['i_mturba'] = xr.DataArray(data=ma, name='i_mturba', coords = {"time":t},  dims=["X","time"], attrs={'Description':"Forced convection in air, based on Condron's mitberg formulation",
#                                                                                                                  'Units': 'm/day'})
    
    
#     iceberg['i_mtotalm'] = xr.DataArray(data=i_mtotalm, name='i_mtotalm', attrs={'Description':"mean over all time, depths, processes  in m/day",
#                                                                                                                  'Units': 'm/day'})
    
#     iceberg['VOL'] = xr.DataArray(data=VOL, name='VOL', coords = {"time":t},  dims=["X","time"])
#     iceberg['FREEB'] = xr.DataArray(data=FREEB, name='FREEB', coords = {"time":t},  dims=["X","time"])
#     iceberg['KEEL'] = xr.DataArray(data=KEEL, name='KEEL', coords = {"time":t},  dims=["X","time"])
#     iceberg['LEN'] = xr.DataArray(data=LEN, name='LEN', coords = {"time":t},  dims=["X","time"])
#     iceberg['SAILVOL'] = xr.DataArray(data=SAILVOL, name='SAILVOL', coords = {"time":t},  dims=["X","time"])
#     iceberg['THICK'] = xr.DataArray(data=THICK, name='THICK', coords = {"time":t},  dims=["X","time"])
#     iceberg['WIDTH'] = xr.DataArray(data=WIDTH, name='WIDTH', coords = {"time":t},  dims=["X","time"])
#     iceberg['UWL'] = xr.DataArray(data=UWL, name='UWL', coords = {"time":t,"Z":ice_init[0].Z.values},  dims=["Z","X","time"])
#     iceberg['UWVOL'] = xr.DataArray(data=UWVOL, name='UWVOL', coords = {"time":t,"Z":ice_init[0].Z.values},  dims=["Z","X","time"])
#     iceberg['UWW'] = xr.DataArray(data=UWW, name='UWW', coords = {"time":t,"Z":ice_init[0].Z.values},  dims=["Z","X","time"])
#     iceberg['Urel'] = xr.DataArray(data=Urel_unadj, name='Urel', coords = {"time":t,"Z":ice_init[0].Z.values},  dims=["Z","X","time"])

#     # coords need to be time step and Z and X?
#     # Mwave_da = xr.DataArray(data=Mwave,)
    
#     # iceberg.assign_attrs(Description='Iceberg depth independent melt model from Moon et al., 2018.')
    
#     return iceberg
























