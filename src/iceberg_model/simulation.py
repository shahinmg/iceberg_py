"""
Iceberg Melt Simulation Module

This module provides classes for running iceberg melt simulations over time,
integrating the Iceberg geometry class with melt rate calculations.

The simulation handles:
- Time evolution of iceberg geometry
- Multiple melt mechanisms (wave, forced convection, buoyant convection, solar)
- Iceberg stability and rolling
- Environmental forcing (CTD data, wind, air temperature, solar radiation)

Classes
-------
IcebergMeltSimulation
    Main simulation class that wraps the iceberg_melt function
EnvironmentalForcing
    Container for environmental forcing data
"""

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d, interp2d
from scipy.spatial import cKDTree
from dataclasses import dataclass
from typing import Optional, Dict, Union, Tuple

from .iceberg import Iceberg
from . import melt
from . import constants as const


@dataclass
class EnvironmentalForcing:
    """
    Container for environmental forcing data.
    
    Attributes
    ----------
    ctd_data : xr.Dataset
        CTD data with temperature, salinity, and depth
    ice_concentration : float or ndarray
        Sea ice concentration (0-1)
    wind_speed : float or ndarray
        Wind speed in m/s
    air_temperature : float or ndarray
        Air temperature in °C
    solar_flux : float or ndarray
        Solar radiation flux in W/m²
    water_velocity : float or xr.Dataset
        Relative water velocity (constant or ADCP structure)
    """
    ctd_data: xr.Dataset
    ice_concentration: Union[float, np.ndarray]
    wind_speed: Union[float, np.ndarray]
    air_temperature: Union[float, np.ndarray]
    solar_flux: Union[float, np.ndarray]
    water_velocity: Union[float, xr.Dataset]


class IcebergMeltSimulation:
    """
    Iceberg melt simulation manager.
    
    This class wraps the iceberg_melt function and provides a cleaner
    interface for running simulations with the Iceberg class.
    
    Parameters
    ----------
    iceberg : Iceberg
        Iceberg instance with initialized geometry
    forcing : EnvironmentalForcing
        Environmental forcing data
    
    Attributes
    ----------
    iceberg : Iceberg
        The iceberg being simulated
    forcing : EnvironmentalForcing
        Environmental forcing data
    results : xr.Dataset or None
        Simulation results after run() is called
    
    Examples
    --------
    >>> # Create iceberg and forcing
    >>> iceberg = Iceberg(length=200, dz=10)
    >>> forcing = EnvironmentalForcing(
    ...     ctd_data=ctd,
    ...     ice_concentration=0.2,
    ...     wind_speed=10.0,
    ...     air_temperature=5.0,
    ...     solar_flux=250.0,
    ...     water_velocity=0.3
    ... )
    >>> 
    >>> # Run simulation
    >>> sim = IcebergMeltSimulation(iceberg, forcing)
    >>> results = sim.run(timespan=30*86400)  # 30 days
    >>> 
    >>> # Access results
    >>> print(f"Final volume: {results.VOL.values[0, -1]:.1e} m³")
    >>> print(f"Total melt: {results.Mtotal.mean().values:.3e} m³/s")
    """
    
    def __init__(self, iceberg: Iceberg, forcing: EnvironmentalForcing):
        """
        Initialize simulation.
        
        Parameters
        ----------
        iceberg : Iceberg
            Iceberg instance with geometry
        forcing : EnvironmentalForcing
            Environmental forcing data
        """
        self.iceberg = iceberg
        self.forcing = forcing
        self.results = None
        
    def run(
        self,
        timespan: float,
        do_constant_velocity: bool = False,
        transfer_coeff_factor: float = 1.0,  # Changed from 4.0 to 1.0 (standard physics)
        do_roll: bool = True,
        do_slab: bool = True,
        use_constant_tf: bool = False,
        constant_tf: Optional[float] = None,
        melt_mechanisms: Optional[Dict[str, bool]] = None,
        quiet: bool = True
    ) -> xr.Dataset:
        """
        Run the melt simulation.
        
        This is a wrapper around the iceberg_melt function that uses
        the Iceberg instance and EnvironmentalForcing data.
        
        Parameters
        ----------
        timespan : float
            Duration of simulation in seconds
        do_constant_velocity : bool, optional
            If True, use constant water velocity. Default is False.
        transfer_coeff_factor : float, optional
            Adjustment factor for heat/salt transfer coefficients.
            Default is 1.0 (standard Holland & Jenkins 1999 values).
            Use 4.0 for Jackson et al. 2020 adjusted values.
        do_roll : bool, optional
            Enable iceberg rolling when unstable. Default is True.
        do_slab : bool, optional
            Enable slab breakoff (not yet implemented). Default is True.
        use_constant_tf : bool, optional
            Use constant thermal forcing instead of calculating. Default is False.
        constant_tf : float, optional
            Constant thermal forcing value if use_constant_tf is True
        melt_mechanisms : dict, optional
            Dict specifying which melt mechanisms to include:
            {'wave': True, 'turbw': True, 'turba': True, 
             'freea': True, 'freew': True}
            If None, all mechanisms are enabled.
        quiet : bool, optional
            Suppress progress messages. Default is True.
        
        Returns
        -------
        results : xr.Dataset
            Complete simulation results including:
            - Time-varying geometry (VOL, LEN, WIDTH, KEEL, etc.)
            - Melt rates by mechanism (Mwave, Mturbw, etc.)
            - Integrated melt fluxes
            - Underwater geometry evolution
        
        Examples
        --------
        >>> sim = IcebergMeltSimulation(iceberg, forcing)
        >>> results = sim.run(
        ...     timespan=30*86400,  # 30 days
        ...     transfer_coeff_factor=4.0,
        ...     melt_mechanisms={'wave': True, 'turbw': True, 
        ...                      'turba': False, 'freea': True, 'freew': True}
        ... )
        """
        # Set default melt mechanisms if not provided
        if melt_mechanisms is None:
            melt_mechanisms = const.DEFAULT_MELT_MECHANISMS.copy()
        
        # Call the iceberg_melt function
        self.results = iceberg_melt(
            L=self.iceberg.length,
            dz=self.iceberg.dz,
            timespan=timespan,
            ctddata=self.forcing.ctd_data,
            IceConc=self.forcing.ice_concentration,
            WindSpd=self.forcing.wind_speed,
            Tair=self.forcing.air_temperature,
            SWflx=self.forcing.solar_flux,
            Urelative=self.forcing.water_velocity,
            do_constantUrel=do_constant_velocity,
            factor=transfer_coeff_factor,
            quiet=quiet,
            do_roll=do_roll,
            do_slab=do_slab,
            use_constant_tf=use_constant_tf,
            constant_tf=constant_tf,
            do_melt=melt_mechanisms
        )
        
        return self.results
    
    def get_final_state(self) -> Dict[str, float]:
        """
        Get the final state of the iceberg.
        
        Returns
        -------
        state : dict
            Final iceberg state with keys:
            - length, width, thickness, keel, freeboard, volume
        
        Raises
        ------
        ValueError
            If simulation hasn't been run yet
        """
        if self.results is None:
            raise ValueError("Must run simulation first")
        
        return {
            'length': float(self.results.LEN.values[0, -1]),
            'width': float(self.results.WIDTH.values[0, -1]),
            'thickness': float(self.results.THICK.values[0, -1]),
            'keel': float(self.results.KEEL.values[0, -1]),
            'freeboard': float(self.results.FREEB.values[0, -1]),
            'volume': float(self.results.VOL.values[0, -1]),
        }
    
    def get_melt_summary(self) -> Dict[str, float]:
        """
        Get summary statistics of melt rates.
        
        Returns
        -------
        summary : dict
            Mean melt rates for each mechanism in m³/s
        
        Raises
        ------
        ValueError
            If simulation hasn't been run yet
        """
        if self.results is None:
            raise ValueError("Must run simulation first")
        
        return {
            'wave': float(self.results.Mwave.mean().values),
            'forced_water': float(self.results.Mturbw.mean().values),
            'forced_air': float(self.results.Mturba.mean().values),
            'solar': float(self.results.Mfreea.mean().values),
            'buoyant_water': float(self.results.Mfreew.mean().values),
            'total': float(self.results.Mtotal.mean().values),
        }
    
    def calculate_lifetime(self, volume_threshold: float = 1e3) -> Optional[float]:
        """
        Calculate iceberg lifetime (time until volume < threshold).
        
        Parameters
        ----------
        volume_threshold : float, optional
            Volume threshold in m³. Default is 1000 m³.
        
        Returns
        -------
        lifetime : float or None
            Time in days until volume drops below threshold,
            or None if threshold never reached
        """
        if self.results is None:
            raise ValueError("Must run simulation first")
        
        volume = self.results.VOL.values[0, :]
        time = self.results.time.values
        
        # Find when volume drops below threshold
        below_threshold = volume < volume_threshold
        
        if np.any(below_threshold):
            index = np.argmax(below_threshold)
            return time[index] / const.SECONDS_PER_DAY
        else:
            return None


# ==============================================================================
# HELPER FUNCTIONS (Refactored for better maintainability)
# ==============================================================================

def _prepare_forcing_timeseries(forcing_value: Union[float, np.ndarray], 
                                n_timesteps: int) -> np.ndarray:
    """
    Convert scalar or array forcing to time series.
    
    Parameters
    ----------
    forcing_value : float or ndarray
        Either a single value or array of values
    n_timesteps : int
        Number of timesteps in simulation
    
    Returns
    -------
    ndarray
        Time series of forcing values
    """
    forcing_array = np.atleast_1d(forcing_value)
    
    if len(forcing_array) == 1:
        return forcing_array[0] * np.ones(n_timesteps)
    else:
        return forcing_array


def _prepare_ctd_profiles(ctd_data: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract and average CTD temperature and salinity profiles.
    
    Parameters
    ----------
    ctd_data : xr.Dataset
        CTD data with temp, salt, depth fields
    
    Returns
    -------
    temperature : ndarray
        Temperature profile (depth-averaged if multiple casts)
    salinity : ndarray
        Salinity profile (depth-averaged if multiple casts)
    depths : ndarray
        Depth coordinates
    """
    m, n = np.shape(ctd_data.temp)
    
    if n > 1:
        temperature = np.nanmean(ctd_data.temp, axis=1)
        salinity = np.nanmean(ctd_data.salt, axis=1)
    else:
        temperature = ctd_data.temp.data.flatten()
        salinity = ctd_data.salt.data.flatten()
    
    depths = ctd_data.depth.T.to_numpy().flatten()
    
    return temperature, salinity, depths


def _setup_water_velocity(
    Urelative: Union[float, xr.Dataset],
    do_constant: bool,
    ice_concentration: float,
    initial_keel: float,
    nz: int,
    ni: int,
    nt: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Setup water velocity field (constant or from ADCP).
    
    Parameters
    ----------
    Urelative : float or xr.Dataset
        Either constant velocity or ADCP dataset
    do_constant : bool
        Use constant velocity
    ice_concentration : float
        Sea ice concentration (affects velocity processing)
    initial_keel : float
        Initial keel depth for ADCP processing
    nz, ni, nt : int
        Grid dimensions (depth, icebergs, time)
    
    Returns
    -------
    velocity : ndarray
        Relative water velocity field (nz, ni, nt)
    velocity_unadjusted : ndarray
        Unadjusted velocity (for diagnostics)
    """
    if do_constant:
        velocity = Urelative * np.ones((nz, ni, nt))
        velocity_unadjusted = velocity.copy()
        return velocity, velocity_unadjusted
    
    # Process ADCP data
    velocity = np.nan * np.ones((nz, ni, nt))
    velocity_unadjusted = velocity.copy()
    
    # Find keel depth in ADCP vertical grid
    kdt = cKDTree(Urelative.zadcp.values.reshape(-1, 1))
    keel_index = kdt.query([[np.ceil(initial_keel)]])[1][0]
    
    # Process horizontal velocity
    if ice_concentration == 1.0:
        # 100% sea ice: melange, keep mean flow
        vmadcp = Urelative.vadcp.values
    else:
        # Drifting icebergs: remove mean horizontal flow
        mean_flow = np.nanmean(Urelative.vadcp.values[0:keel_index+1, :], axis=0)
        vmadcp = Urelative.vadcp.values - np.tile(mean_flow, (len(Urelative.zadcp), 1))
    
    # Zero below keel
    vmadcp_unadj = vmadcp.copy()
    vmadcp[keel_index+1:, :] = 0
    vmadcp = np.abs(vmadcp)
    
    # Add vertical velocity
    wvel = float(Urelative.wvel.values.flatten()[0])
    vmadcp += wvel
    vmadcp_unadj += wvel
    
    # Interpolate to simulation grid
    from scipy.interpolate import interp2d
    interp_func = interp2d(Urelative.tadcp.values.flatten(),
                          Urelative.zadcp.values.flatten(), vmadcp)
    interp_func_unadj = interp2d(Urelative.tadcp.values.flatten(),
                                Urelative.zadcp.values.flatten(), vmadcp_unadj)
    
    # Assuming ni=1 for now (single iceberg)
    time_indices = np.arange(1, nt + 1)
    depth_coords = Urelative.zadcp.values  # Need to pass this properly
    
    velocity[:, 0, :] = interp_func(time_indices, depth_coords)
    velocity_unadjusted[:, 0, :] = interp_func_unadj(time_indices, depth_coords)
    
    return velocity, velocity_unadjusted


def _initialize_output_arrays(nz: int, ni: int, nt: int) -> Dict[str, np.ndarray]:
    """
    Initialize all output tracking arrays.
    
    Parameters
    ----------
    nz, ni, nt : int
        Grid dimensions (depth, icebergs, time)
    
    Returns
    -------
    arrays : dict
        Dictionary of initialized arrays
    """
    return {
        # Melt volumes (integrated)
        'Mwave': np.zeros((ni, nt)),
        'Mturbw': np.zeros((nz, ni, nt)),
        'Mturba': np.zeros((ni, nt)),
        'Mfreea': np.zeros((ni, nt)),
        'Mfreew': np.zeros((nz, ni, nt)),
        
        # Melt rates (m/day)
        'melt_rate_wave': np.zeros((ni, nt)),
        'melt_rate_forced_air': np.zeros((ni, nt)),
        'melt_rate_solar': np.zeros((ni, nt)),
        'melt_rate_forced_water': np.zeros((nz, ni, nt)),
        'melt_rate_buoyant': np.zeros((nz, ni, nt)),
        
        # Geometry
        'VOL': np.full((ni, nt), np.nan),
        'LEN': np.full((ni, nt), np.nan),
        'WIDTH': np.full((ni, nt), np.nan),
        'THICK': np.full((ni, nt), np.nan),
        'FREEB': np.full((ni, nt), np.nan),
        'KEEL': np.full((ni, nt), np.nan),
        'SAILVOL': np.full((ni, nt), np.nan),
        'UWVOL': np.full((nz, ni, nt), np.nan),
        'UWL': np.full((nz, ni, nt), np.nan),
        'UWW': np.full((nz, ni, nt), np.nan),
        
        # Diagnostics
        'wave_height': np.zeros((ni, nt)),
    }


def _calculate_wave_melt(
    wind_speed: float,
    sea_ice_conc: float,
    temperature: np.ndarray,
    depths: np.ndarray,
    freeboard: float,
    length: float,
    underwater_length_top: float,
    dz: float,
    timestep: float
) -> Tuple[float, float, float, float, float]:
    """
    Calculate wave erosion melt for one timestep.
    
    Returns
    -------
    melt_rate : float
        Wave melt rate in m/timestep
    melt_volume : float
        Integrated melt volume in m³
    wave_height : float
        Estimated wave height in m
    fraction_above : float
        Melt fraction affecting freeboard
    fraction_below : float
        Melt fraction affecting underwater
    """
    # Get sea surface temperature (0-5m average)
    SST_func = interp1d(depths, temperature)
    SST = np.nanmean(SST_func(np.arange(1, 6)))
    
    # Wave height estimate
    wave_height = const.WAVE_HEIGHT_ESTIMATE_COEFF * np.power(np.abs(wind_speed), 2)
    wave_penetration = np.minimum(freeboard, const.MAX_WAVE_PENETRATION_FACTOR * wave_height)
    
    # Melt rate
    melt_rate = melt.melt_wave(wind_speed, SST, sea_ice_conc)
    melt_rate *= timestep
    
    # Integrated volume
    avg_length = np.nanmean([length, underwater_length_top])
    melt_volume = (const.WAVE_LENGTH_SURFACES * melt_rate * wave_penetration * avg_length +
                  const.WAVE_WIDTH_SURFACES * melt_rate * wave_penetration * avg_length)
    
    # Melt fractions
    fraction_above = wave_penetration / freeboard if freeboard > 0 else 0
    fraction_below = wave_penetration / dz
    
    return melt_rate, melt_volume, wave_height, fraction_above, fraction_below


def _calculate_forced_water_melt(
    keel_layer_index: int,
    depths: np.ndarray,
    temperature: np.ndarray,
    salinity: np.ndarray,
    velocity: np.ndarray,
    underwater_length: np.ndarray,
    underwater_width: np.ndarray,
    dz: float,
    keel: float,
    timestep: float,
    factor: float,
    use_constant_tf: bool,
    constant_tf: Optional[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate forced convection melt in water for all depth layers.
    
    Parameters
    ----------
    keel_layer_index : int
        Index of deepest keel layer
    depths : np.ndarray
        CTD depth coordinates (for interpolation)
    temperature : np.ndarray
        CTD temperature profile
    salinity : np.ndarray
        CTD salinity profile
    velocity : np.ndarray
        Water velocity at each iceberg depth layer
    underwater_length : np.ndarray
        Iceberg length at each depth layer
    underwater_width : np.ndarray
        Iceberg width at each depth layer
    dz : float
        Layer thickness
    keel : float
        Keel depth
    timestep : float
        Timestep duration
    factor : float
        Transfer coefficient factor
    use_constant_tf : bool
        Use constant thermal forcing
    constant_tf : Optional[float]
        Constant thermal forcing value
    
    Returns
    -------
    melt_rates : ndarray
        Melt rates by depth (nz_iceberg,) - sized to iceberg layers
    melt_volumes : ndarray
        Integrated melt volumes by depth (nz_iceberg,) - sized to iceberg layers
    """
    # Size arrays to iceberg depth levels
    nz_iceberg = len(velocity)
    melt_rates = np.zeros(nz_iceberg)
    melt_volumes = np.zeros(nz_iceberg)
    
    # Calculate iceberg depth coordinates for interpolation
    # Use cell centers like original: [dz, 2*dz, 3*dz, ...] not [0, dz, 2*dz, ...]
    iceberg_depths = np.arange(1, nz_iceberg + 1) * dz  # [5, 10, 15, ...] for dz=5
    
    # Full layers
    for k in range(keel_layer_index - 1):
        T_far = interp1d(depths, temperature)(iceberg_depths[k])
        S_far = interp1d(depths, salinity)(iceberg_depths[k])
        
        melt_rates[k], _, _ = melt.melt_forcedwater(
            T_far, S_far, iceberg_depths[k], velocity[k],
            factor=factor, use_constant_tf=use_constant_tf, constant_tf=constant_tf
        )
        melt_rates[k] *= timestep
        
        melt_volumes[k] = (const.FORCED_WATER_LENGTH_SURFACES * melt_rates[k] * dz * underwater_length[k] +
                          const.FORCED_WATER_WIDTH_SURFACES * melt_rates[k] * dz * underwater_width[k])
    
    # Partial keel layer
    if keel_layer_index > 0:
        k = keel_layer_index - 1
        T_far = interp1d(depths, temperature)(iceberg_depths[k])
        S_far = interp1d(depths, salinity)(iceberg_depths[k])
        
        melt_rates[k], _, _ = melt.melt_forcedwater(
            T_far, S_far, iceberg_depths[k], velocity[k],
            factor=factor, use_constant_tf=use_constant_tf, constant_tf=constant_tf
        )
        melt_rates[k] *= timestep
        
        dz_keel = -1 * ((keel_layer_index - 1) * dz - keel)
        melt_volumes[k] = (const.FORCED_WATER_LENGTH_SURFACES * melt_rates[k] * dz_keel * underwater_length[k] +
                          const.FORCED_WATER_WIDTH_SURFACES * melt_rates[k] * dz_keel * underwater_width[k])
    
    return melt_rates, melt_volumes


def _calculate_forced_air_melt(
    air_temp: float,
    wind_speed: float,
    length: float,
    width: float,
    dz: float,
    timestep: float
) -> Tuple[float, float]:
    """
    Calculate forced convection melt in air.
    
    Returns
    -------
    melt_rate : float
        Melt rate in m/timestep
    melt_volume : float
        Integrated melt volume in m³
    """
    melt_rate = melt.melt_forcedair(air_temp, wind_speed, length)
    melt_rate *= timestep
    
    melt_volume = (const.FORCED_AIR_LENGTH_SURFACES * melt_rate * dz * length +
                  const.FORCED_AIR_WIDTH_SURFACES * melt_rate * dz * width +
                  const.FORCED_AIR_TOP_SURFACE_FRACTION * melt_rate * length * width)
    
    return melt_rate, melt_volume


def _calculate_solar_melt(
    solar_flux: float,
    length: float,
    width: float,
    timestep: float
) -> Tuple[float, float]:
    """
    Calculate solar radiation melt.
    
    Returns
    -------
    melt_rate : float
        Melt rate in m/timestep
    melt_volume : float
        Integrated melt volume in m³
    """
    melt_rate = melt.melt_solar(solar_flux)
    melt_rate *= timestep
    
    melt_volume = const.SOLAR_TOP_SURFACE_FRACTION * melt_rate * width * length
    
    return melt_rate, melt_volume


def _calculate_buoyant_melt(
    keel_layer_index: int,
    depths: np.ndarray,
    temperature: np.ndarray,
    salinity: np.ndarray,
    underwater_length: np.ndarray,
    underwater_width: np.ndarray,
    dz: float,
    keel: float,
    timestep: float,
    use_constant_tf: bool,
    constant_tf: Optional[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate buoyant convection melt along vertical walls.
    
    Returns
    -------
    melt_rates : ndarray
        Melt rates by depth (nz_iceberg,) - sized to iceberg layers
    melt_volumes : ndarray
        Integrated melt volumes by depth (nz_iceberg,) - sized to iceberg layers
    """
    # Size arrays to iceberg depth levels
    nz_iceberg = len(underwater_length)
    melt_rates = np.zeros(nz_iceberg)
    melt_volumes = np.zeros(nz_iceberg)
    
    # Calculate iceberg depth coordinates (cell centers like original)
    iceberg_depths = np.arange(1, nz_iceberg + 1) * dz  # [5, 10, 15, ...] for dz=5
    
    # Full layers
    for k in range(keel_layer_index - 1):
        T_far = interp1d(depths, temperature)(iceberg_depths[k])
        S_far = interp1d(depths, salinity)(iceberg_depths[k])
        
        melt_rates[k] = melt.melt_buoyantwater(
            T_far, S_far, 'cis',
            use_constant_tf=use_constant_tf, constant_tf=constant_tf
        )
        melt_rates[k] *= timestep
        
        melt_volumes[k] = (const.BUOYANT_WATER_LENGTH_SURFACES * melt_rates[k] * dz * underwater_length[k] +
                          const.BUOYANT_WATER_WIDTH_SURFACES * melt_rates[k] * dz * underwater_width[k])
    
    # Partial keel layer
    if keel_layer_index > 0:
        k = keel_layer_index - 1
        
        # Calculate melt rate for partial layer (was missing!)
        T_far = interp1d(depths, temperature)(iceberg_depths[k])
        S_far = interp1d(depths, salinity)(iceberg_depths[k])
        
        melt_rates[k] = melt.melt_buoyantwater(
            T_far, S_far, 'cis',
            use_constant_tf=use_constant_tf, constant_tf=constant_tf
        )
        melt_rates[k] *= timestep
        
        dz_keel = -1 * ((keel_layer_index - 1) * dz - keel)
        melt_volumes[k] = (const.BUOYANT_WATER_LENGTH_SURFACES * melt_rates[k] * dz_keel * underwater_length[k] +
                          const.BUOYANT_WATER_WIDTH_SURFACES * melt_rates[k] * dz_keel * underwater_width[k])
    
    return melt_rates, melt_volumes


def _update_geometry_from_melt(
    length: float,
    width: float,
    freeboard: float,
    keel: float,
    underwater_length: np.ndarray,
    underwater_width: np.ndarray,
    underwater_volume: np.ndarray,
    melt_rate_wave: float,
    melt_rate_solar: float,
    melt_rate_forced_air: float,
    melt_rate_forced_water: np.ndarray,
    melt_rate_buoyant: np.ndarray,
    melt_fraction_above: float,
    melt_fraction_below: float,
    keel_layer_index: int,
    dz: float
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Update iceberg geometry after applying melt rates.
    
    Returns
    -------
    length : float
    width : float
    freeboard : float
    keel : float
    underwater_length : ndarray
    underwater_width : ndarray
    underwater_volume : ndarray
    keel_layer_index : int
    """
    # Update vertical dimensions
    freeboard = freeboard - melt_rate_solar - melt_rate_forced_air
    keel = keel - melt_rate_forced_water[keel_layer_index - 1]
    thickness = keel + freeboard
    
    # Update horizontal dimensions (melt from both sides)
    mult = 2
    
    # Note: In MATLAB original, wave melt is applied to ALL underwater layers
    # Line 555: uwL(k) = uwL(k) - (scale .* mtw(k,j)) - (scale .* mb(k,j)) - (scale .* mw(j))
    # So we need to include wave melt in the loop below, not just for layer 0
    
    length = length - mult * melt_rate_forced_air - mult * melt_rate_wave * melt_fraction_above
    
    # Update underwater lengths
    # MATLAB: for k=1:iKeel (indices 1 to iKeel in 1-based indexing)
    # Python: for k in range(iKeel) (indices 0 to iKeel-1 in 0-based indexing)
    # Since our keel_layer_index is already 1-indexed (from ceil), we use range(keel_layer_index)
    for k in range(keel_layer_index):
        underwater_length[k] = (underwater_length[k] - 
                               mult * melt_rate_forced_water[k] - 
                               mult * melt_rate_buoyant[k] -
                               mult * melt_rate_wave)
    
    # Maintain L:W ratio
    underwater_width = underwater_length / const.DEFAULT_LENGTH_TO_WIDTH_RATIO
    width = length / const.DEFAULT_LENGTH_TO_WIDTH_RATIO
    
    # Check if keel layer should be removed
    keel_index_new = int(np.ceil(keel / dz))
    if keel_index_new < keel_layer_index:
        underwater_length[keel_layer_index - 1] = np.nan
        underwater_width[keel_layer_index - 1] = np.nan
        underwater_volume[keel_layer_index - 1] = np.nan
        keel_layer_index = keel_index_new
    
    # Update volumes
    underwater_volume[:keel_layer_index - 1] = (dz * 
                                                underwater_length[:keel_layer_index - 1] * 
                                                underwater_width[:keel_layer_index - 1])
    
    dzk_partial = -1 * ((keel_layer_index - 1) * dz - keel)
    underwater_volume[keel_layer_index - 1] = (dzk_partial * 
                                               underwater_length[keel_layer_index - 1] * 
                                               underwater_width[keel_layer_index - 1])
    
    return (length, width, freeboard, keel, underwater_length, 
            underwater_width, underwater_volume, keel_layer_index)


def _apply_buoyancy_adjustment(
    length: float,
    width: float,
    thickness: float,
    freeboard: float,
    underwater_volume: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Adjust geometry for buoyancy equilibrium.
    
    Matches original melt_functions.py lines 876-881
    
    Parameters
    ----------
    freeboard : float
        Current freeboard (needed for sailV calculation)
    
    Returns
    -------
    freeboard : float
        Adjusted freeboard
    keel : float
        Adjusted keel
    sail_volume : float
        Adjusted sail volume
    total_volume : float
        Total iceberg volume
    """
    density_ratio = const.DENSITY_RATIO_ICE_TO_WATER
    
    # Line 876: sailV = freeB * L * W
    sail_volume = freeboard * length * width
    
    # Line 877: totalV = np.nansum(uwV) + sailV
    total_volume = np.nansum(underwater_volume) + sail_volume
    
    # Line 879: sailV = (1 - ratio_i) * totalV
    sail_volume = (1 - density_ratio) * total_volume
    
    # Line 880: freeB = sailV / (L * W)
    freeboard = sail_volume / (length * width) if (length * width) > 0 else 0
    
    # Line 881: keel = TH - freeB
    keel = thickness - freeboard
    
    return freeboard, keel, sail_volume, total_volume


def _check_stability_and_roll(
    length: float,
    width: float,
    thickness: float,
    total_volume: float,
    underwater_length: np.ndarray,
    underwater_width: np.ndarray,
    underwater_volume: np.ndarray,
    dz: float,
    do_roll: bool,
    quiet: bool,
    timestep_index: int
) -> Tuple[float, float, float, float, float, float, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Check stability and roll iceberg if unstable.
    
    Returns
    -------
    All geometry parameters (potentially updated if rolled)
    """
    if not do_roll:
        keel = thickness - (total_volume * (1 - const.DENSITY_RATIO_ICE_TO_WATER) / (length * width))
        keel_layer_index = int(np.ceil(keel / dz))
        return (length, width, thickness, total_volume, keel, 
                underwater_length, underwater_width, underwater_volume, keel_layer_index)
    
    length_thickness_ratio = length / thickness if thickness > 0 else 0
    
    if length_thickness_ratio < const.STABILITY_WIDTH_FACTOR:
        if not quiet:
            print(f'Iceberg rolling at timestep {timestep_index}')
        
        # Roll iceberg
        thickness = length
        length = np.sqrt(total_volume / (thickness / const.DEFAULT_LENGTH_TO_WIDTH_RATIO))
        width = length / const.DEFAULT_LENGTH_TO_WIDTH_RATIO
        
        density_ratio = const.DENSITY_RATIO_ICE_TO_WATER
        freeboard = (1 - density_ratio) * thickness
        sail_volume = freeboard * length * width
        total_volume = (1 / density_ratio) * sail_volume
        keel = thickness - freeboard
        keel_layer_index = int(np.ceil(keel / dz))
        
        # Reset underwater geometry
        underwater_length[int(keel_layer_index + 1):] = np.nan
        underwater_width[int(keel_layer_index + 1):] = np.nan
        underwater_volume = dz * underwater_length * underwater_width
    else:
        keel = thickness - (total_volume * (1 - const.DENSITY_RATIO_ICE_TO_WATER) / (length * width))
        keel_layer_index = int(np.ceil(keel / dz))
    
    return (length, width, thickness, total_volume, keel,
            underwater_length, underwater_width, underwater_volume, keel_layer_index)


def _package_results(
    arrays: Dict[str, np.ndarray],
    ice_init: list,
    time: np.ndarray,
    timestep: float,
    velocity_unadjusted: np.ndarray
) -> xr.Dataset:
    """
    Package all results into xarray Dataset.
    
    Parameters
    ----------
    arrays : dict
        All output arrays
    ice_init : list
        Initial iceberg geometries
    time : ndarray
        Time coordinate
    timestep : float
        Timestep in seconds
    velocity_unadjusted : ndarray
        Unadjusted water velocity
    
    Returns
    -------
    xr.Dataset
        Complete simulation results
    """
    # Convert melt volumes to freshwater fluxes (m³/s)
    density_ratio = const.RHO_ICE / const.RHO_FRESHWATER
    
    Mwave = (density_ratio * arrays['Mwave']) / timestep
    Mfreea = (density_ratio * arrays['Mfreea']) / timestep
    Mturbw = (density_ratio * arrays['Mturbw']) / timestep
    Mturba = (density_ratio * arrays['Mturba']) / timestep
    Mfreew = (density_ratio * arrays['Mfreew']) / timestep  # FIXED: Now using correct Mfreew data
    
    # Total melt
    ni, nt = arrays['Mwave'].shape
    Mtotal = np.zeros((ni, nt))
    for i in range(ni):
        Mtotal[i, :] = (Mwave[i, :] + Mfreea[i, :] + Mturba[i, :] +
                       np.nansum(Mturbw[:, i, :], axis=0) +
                       np.nansum(Mfreew[:, i, :], axis=0))
    
    # Mean melt rate (matching original calculation exactly)
    # Original: i_mtotalm = np.nanmean(mw) + np.nanmean(mb) + np.nanmean(ms) + np.nanmean(ma) + np.nanmean(mtw)
    # 
    # IMPORTANT: Arrays have index 0 filled with zeros (never updated in loop which starts at j=1)
    # We must exclude index 0 when calculating means, or use only indices 1:nt
    # The loop is: for j in range(1, nt), so indices 1 to nt-1 are filled
    #
    # For 2D arrays (ni, nt): select [:, 1:]
    # For 3D arrays (nz, ni, nt): select [:, :, 1:]
    mean_melt = (np.nanmean(arrays['melt_rate_wave'][:, 1:]) +          # mw: wave (2D)
                np.nanmean(arrays['melt_rate_buoyant'][:, :, 1:]) +     # mb: buoyant (3D) 
                np.nanmean(arrays['melt_rate_solar'][:, 1:]) +          # ms: solar (2D)
                np.nanmean(arrays['melt_rate_forced_air'][:, 1:]) +     # ma: forced air (2D)
                np.nanmean(arrays['melt_rate_forced_water'][:, :, 1:]))  # mtw: forced water (3D)
    
    # Create Dataset
    results = xr.Dataset()
    
    # Melt fluxes (m³/s)
    results['Mwave'] = xr.DataArray(Mwave, coords={"time": time}, dims=["X", "time"],
                                    attrs={'Description': 'Wave melt flux', 'Units': 'm³/s'})
    results['Mfreea'] = xr.DataArray(Mfreea, coords={"time": time}, dims=["X", "time"],
                                     attrs={'Description': 'Solar melt flux', 'Units': 'm³/s'})
    results['Mturbw'] = xr.DataArray(Mturbw, coords={"time": time, "Z": ice_init[0].Z.values},
                                     dims=["Z", "X", "time"],
                                     attrs={'Description': 'Forced water melt flux', 'Units': 'm³/s'})
    results['Mturba'] = xr.DataArray(Mturba, coords={"time": time}, dims=["X", "time"],
                                     attrs={'Description': 'Forced air melt flux', 'Units': 'm³/s'})
    results['Mfreew'] = xr.DataArray(Mfreew, coords={"time": time, "Z": ice_init[0].Z.values},
                                     dims=["Z", "X", "time"],
                                     attrs={'Description': 'Buoyant melt flux', 'Units': 'm³/s'})
    results['Mtotal'] = xr.DataArray(Mtotal, coords={"time": time}, dims=["X", "time"],
                                     attrs={'Description': 'Total freshwater flux', 'Units': 'm³/s'})
    
    # Melt rates (m/day)
    results['i_mwave'] = xr.DataArray(arrays['melt_rate_wave'], coords={"time": time},
                                     dims=["X", "time"], attrs={'Units': 'm/day'})
    results['i_mfreea'] = xr.DataArray(arrays['melt_rate_solar'], coords={"time": time},
                                      dims=["X", "time"], attrs={'Units': 'm/day'})
    results['i_mturbw'] = xr.DataArray(arrays['melt_rate_forced_water'], 
                                      coords={"time": time, "Z": ice_init[0].Z.values},
                                      dims=["Z", "X", "time"], attrs={'Units': 'm/day'})
    results['i_mfreew'] = xr.DataArray(arrays['melt_rate_buoyant'],
                                      coords={"time": time, "Z": ice_init[0].Z.values},
                                      dims=["Z", "X", "time"], attrs={'Units': 'm/day'})
    results['i_mturba'] = xr.DataArray(arrays['melt_rate_forced_air'], coords={"time": time},
                                      dims=["X", "time"], attrs={'Units': 'm/day'})
    results['i_mtotalm'] = xr.DataArray(mean_melt, attrs={'Units': 'm/day'})
    
    # Geometry
    for var in ['VOL', 'FREEB', 'KEEL', 'LEN', 'SAILVOL', 'THICK', 'WIDTH']:
        results[var] = xr.DataArray(arrays[var], coords={"time": time}, dims=["X", "time"])
    
    for var in ['UWL', 'UWVOL', 'UWW']:
        results[var] = xr.DataArray(arrays[var], coords={"time": time, "Z": ice_init[0].Z.values},
                                   dims=["Z", "X", "time"])
    
    # Add initial geometry (2D, no time dimension) for compatibility with original code
    results['uwL'] = ice_init[0].uwL  # Initial underwater length
    results['uwW'] = ice_init[0].uwW  # Initial underwater width
    results['uwV'] = ice_init[0].uwV  # Initial underwater volume
    results['depth'] = ice_init[0].depth  # Depth coordinate
    
    results['Urel'] = xr.DataArray(velocity_unadjusted, 
                                   coords={"time": time, "Z": ice_init[0].Z.values},
                                   dims=["Z", "X", "time"])
    
    results.attrs['Description'] = 'Iceberg melt model from Moon et al. (2018)'
    
    return results


# ==============================================================================
# MAIN SIMULATION FUNCTION (Now much cleaner!)
# ==============================================================================

def iceberg_melt_refactored(
    L: Union[float, np.ndarray],
    dz: float,
    timespan: float,
    ctddata: xr.Dataset,
    IceConc: Union[float, np.ndarray],
    WindSpd: Union[float, np.ndarray],
    Tair: Union[float, np.ndarray],
    SWflx: Union[float, np.ndarray],
    Urelative: Union[float, xr.Dataset],
    do_constantUrel: bool = False,
    factor: float = 4,
    quiet: bool = True,
    do_roll: bool = True,
    do_slab: bool = True,
    use_constant_tf: bool = False,
    constant_tf: Optional[float] = None,
    do_melt: Optional[Dict[str, bool]] = None
) -> xr.Dataset:
    """
    Simulate iceberg melt evolution over time.
    
    This is a refactored version of the original iceberg_melt function,
    broken into smaller helper functions for better maintainability.
    
    Parameters
    ----------
    ... (same as original)
    
    Returns
    -------
    xr.Dataset
        Complete simulation results
    
    Notes
    -----
    This function now delegates to helper functions that each do ONE thing:
    - _prepare_forcing_timeseries: Convert scalars to arrays
    - _prepare_ctd_profiles: Process CTD data
    - _setup_water_velocity: Handle ADCP or constant velocity
    - _calculate_*_melt: Compute each melt mechanism
    - _update_geometry_from_melt: Apply melt to geometry
    - _check_stability_and_roll: Handle rolling
    - _package_results: Create output Dataset
    """
    # Set defaults
    if do_melt is None:
        do_melt = const.DEFAULT_MELT_MECHANISMS.copy()
    
    # ========================================================================
    # SETUP (Lines 1-100 of original)
    # ========================================================================
    
    # Initialize icebergs
    L = np.atleast_1d(L)
    ice_init = [Iceberg(length=length, dz=dz).init_iceberg_size() for length in L]
    
    # Grid dimensions
    nz = len(ice_init[0].Z)
    dz = float(ice_init[0].dz.values)
    timestep_seconds = const.SECONDS_PER_DAY
    t = np.arange(timestep_seconds, timespan + timestep_seconds, timestep_seconds)
    nt = len(t)
    ni = len(L)
    
    # Prepare forcing time series
    sea_ice_conc = _prepare_forcing_timeseries(IceConc, nt)
    wind_velocity = _prepare_forcing_timeseries(WindSpd, nt)
    air_temp = _prepare_forcing_timeseries(Tair, nt)
    solar_radiation = _prepare_forcing_timeseries(SWflx, nt)
    
    # Prepare CTD profiles
    temperature, salinity, depths = _prepare_ctd_profiles(ctddata)
    
    # Setup water velocity
    initial_keel = float(ice_init[0].keel.values)
    velocity, velocity_unadj = _setup_water_velocity(
        Urelative, do_constantUrel, sea_ice_conc[0], 
        initial_keel, nz, ni, nt
    )
    
    # Initialize output arrays
    arrays = _initialize_output_arrays(nz, ni, nt)
    
    # Store initial geometry
    for i, iceberg_data in enumerate(ice_init):
        arrays['VOL'][i, 0] = iceberg_data.totalV.values
        arrays['LEN'][i, 0] = iceberg_data.L.values
        arrays['WIDTH'][i, 0] = iceberg_data.W.values
        arrays['THICK'][i, 0] = iceberg_data.TH.values
        arrays['FREEB'][i, 0] = iceberg_data.freeB.values
        arrays['KEEL'][i, 0] = iceberg_data.keel.values
        arrays['SAILVOL'][i, 0] = iceberg_data.sailV.values
        arrays['UWVOL'][:, i, 0] = iceberg_data.uwV.values.flatten()
        arrays['UWL'][:, i, 0] = iceberg_data.uwL.values.flatten()
        arrays['UWW'][:, i, 0] = iceberg_data.uwW.values.flatten()
    
    # ========================================================================
    # TIME EVOLUTION (Lines 101-350 of original, now much cleaner!)
    # ========================================================================
    
    for i, iceberg_data in enumerate(ice_init):
        # Extract initial state
        depth_coords = iceberg_data.depth.values.copy()
        underwater_length = iceberg_data.uwL.values.copy()
        underwater_width = iceberg_data.uwW.values.copy()
        underwater_volume = iceberg_data.uwV.values.copy()
        total_volume = float(iceberg_data.totalV.values)
        sail_volume = float(iceberg_data.sailV.values)
        width = float(iceberg_data.W.values)
        freeboard = float(iceberg_data.freeB.values)
        length = float(iceberg_data.L.values)
        keel = float(iceberg_data.keel.values)
        thickness = float(iceberg_data.TH.values)
        keel_layer_index = int(iceberg_data.keeli.values)
        
        # TIME LOOP (Now just orchestrates helper functions!)
        for j in range(1, nt):
            keel_layer_index = int(np.ceil(keel / dz))
            
            # Save geometry at START of this timestep (before applying melt)
            # This matches original which calculates Mturbw using uwL before updating it
            uwL_start = underwater_length.flatten().copy()
            uwW_start = underwater_width.flatten().copy()
            
            # Calculate melt for each mechanism
            if do_melt['wave']:
                (arrays['melt_rate_wave'][i, j], arrays['Mwave'][i, j], 
                 arrays['wave_height'][i, j], melt_frac_above, melt_frac_below) = \
                    _calculate_wave_melt(
                        wind_velocity[j], sea_ice_conc[j], temperature, depths,
                        freeboard, length, underwater_length[0, 0], dz, timestep_seconds
                    )
            else:
                melt_frac_above = melt_frac_below = 0
            
            if do_melt['turbw']:
                # Use geometry from START of this timestep (before melt applied)
                # This matches original line 780 which uses uwL[k] before line 851 updates it
                arrays['melt_rate_forced_water'][:, i, j], arrays['Mturbw'][:, i, j] = \
                    _calculate_forced_water_melt(
                        keel_layer_index, depths, temperature, salinity,
                        velocity[:, i, j], uwL_start, uwW_start,
                        dz, keel, timestep_seconds, factor, use_constant_tf, constant_tf
                    )
            
            if do_melt['turba']:
                arrays['melt_rate_forced_air'][i, j], arrays['Mturba'][i, j] = \
                    _calculate_forced_air_melt(
                        air_temp[j], wind_velocity[j], length, width, dz, timestep_seconds
                    )
            
            if do_melt['freea']:
                arrays['melt_rate_solar'][i, j], arrays['Mfreea'][i, j] = \
                    _calculate_solar_melt(solar_radiation[j], length, width, timestep_seconds)
            
            if do_melt['freew']:
                # Use geometry from START of this timestep (before melt applied)
                # This matches original line 827 which uses uwL[k] before line 851 updates it
                arrays['melt_rate_buoyant'][:, i, j], arrays['Mfreew'][:, i, j] = \
                    _calculate_buoyant_melt(
                        keel_layer_index, depths, temperature, salinity,
                        uwL_start, uwW_start,
                        dz, keel, timestep_seconds, use_constant_tf, constant_tf
                    )
            
            # Update geometry
            (length, width, freeboard, keel, underwater_length, underwater_width,
             underwater_volume, keel_layer_index) = _update_geometry_from_melt(
                length, width, freeboard, keel, underwater_length, underwater_width,
                underwater_volume, arrays['melt_rate_wave'][i, j],
                arrays['melt_rate_solar'][i, j], arrays['melt_rate_forced_air'][i, j],
                arrays['melt_rate_forced_water'][:, i, j],
                arrays['melt_rate_buoyant'][:, i, j],
                melt_frac_above, melt_frac_below, keel_layer_index, dz
            )
            
            # Update thickness with post-melt values
            thickness = keel + freeboard
            
            # Apply buoyancy
            freeboard, keel, sail_volume, total_volume = _apply_buoyancy_adjustment(
                length, width, thickness, freeboard, underwater_volume
            )
            
            thickness = keel + freeboard
            
            # Check stability and roll if needed
            (length, width, thickness, total_volume, keel, underwater_length,
             underwater_width, underwater_volume, keel_layer_index) = \
                _check_stability_and_roll(
                    length, width, thickness, total_volume, underwater_length,
                    underwater_width, underwater_volume, dz, do_roll, quiet, j
                )
            
            # Store results
            arrays['VOL'][i, j] = total_volume
            arrays['LEN'][i, j] = length
            arrays['WIDTH'][i, j] = width
            arrays['THICK'][i, j] = thickness
            arrays['FREEB'][i, j] = freeboard
            arrays['KEEL'][i, j] = keel
            arrays['SAILVOL'][i, j] = sail_volume
            arrays['UWVOL'][:, i, j] = underwater_volume.flatten()
            arrays['UWL'][:, i, j] = underwater_length.flatten()
            arrays['UWW'][:, i, j] = underwater_width.flatten()
    
    # ========================================================================
    # PACKAGE RESULTS (Lines 351-400 of original)
    # ========================================================================
    
    return _package_results(arrays, ice_init, t, timestep_seconds, velocity_unadj)


# Alias for backwards compatibility
iceberg_melt = iceberg_melt_refactored

print(__doc__)