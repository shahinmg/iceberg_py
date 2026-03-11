"""
Helheim Glacier Iceberg Melt Simulation - Modernized Version

This script simulates iceberg melting in Sermilik Fjord using the refactored
iceberg_model package. It processes multiple size classes and calculates
heat flux and freshwater contributions.

Original script: helheim_example.py
Modernized to use: IcebergMeltSimulation class with clean interfaces
"""

import numpy as np
import xarray as xr
import scipy.io as sio
import pandas as pd
import geopandas as gpd
import pickle
import os
from pathlib import Path
from typing import Tuple

# Import the new iceberg model
from iceberg_model import Iceberg
from iceberg_model.simulation import IcebergMeltSimulation, EnvironmentalForcing
from iceberg_model import constants as const


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Simulation parameters
SIZE_CLASSES = np.arange(50, 1450, 50)  # Iceberg lengths (m)
LAYER_THICKNESS = 5  # Vertical discretization (m)
TIMESPAN_DAYS = 30  # Simulation duration
FJORD = 'helheim'
RUN_TYPE = 'avg'  # 'min', 'avg', or 'max'

# Atlantic Water depth threshold
AWW_DEPTH = 150  # meters

# Physical constants
LATENT_HEAT_FUSION = 3.34e5  # J/kg
SPECIFIC_HEAT_SEAWATER = 3980  # J/(kg·K)
DENSITY_SEAWATER = 1027  # kg/m³
DENSITY_FRESHWATER = 1000  # kg/m³

# Environmental forcing (constant values)
AIR_TEMPERATURE = 5.5  # °C
SOLAR_FLUX = 306  # W/m²
WIND_SPEED = 2.3  # m/s
SEA_ICE_CONCENTRATION = 1.0  # 0-1 (1.0 = 100% melange)

# Water velocity by scenario
VELOCITY_SCENARIOS = {
    'min': 0.02,  # m/s - slow
    'avg': 0.07,  # m/s - medium
    'max': 0.15   # m/s - fast
}

# Transfer coefficient factor (1 = standard, 4 = Jackson et al. 2020)
TRANSFER_COEFF_FACTOR = 1

# File paths (adjust these to your setup)
CTD_PATH = f'../../data/ctd_data/{RUN_TYPE}_temp_sal_sermilik_fjord.csv'  # Or use full path
ADCP_PATH = f'../../data/ascp_template/ADCP_template.mat'  # Or use full path
ICEBERG_GEOM_DIR = f'../../data/iceberg_geoms/{FJORD}/'
OUTPUT_DIR = f'../../data/test_class_outputs/'


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_ctd_data(filepath: str) -> xr.Dataset:
    """
    Load CTD data from CSV and convert to xarray Dataset.
    
    Parameters
    ----------
    filepath : str
        Path to CTD CSV file
    
    Returns
    -------
    xr.Dataset
        CTD data with temp, salt, depth fields
    """
    print(f"Loading CTD data from {filepath}")
    df = pd.read_csv(filepath, index_col=0)
    
    # Create xarray Dataset in format expected by simulation
    ctd_ds = xr.Dataset({
        'depth': (['Z', 'X'], df['depth'].values.reshape(1, -1)),
        'temp': (['tZ', 'tX'], df['temp'].values.reshape(-1, 1)),
        'salt': (['tZ', 'tX'], df['salt'].values.reshape(-1, 1))
    })
    
    print(f"  Loaded {len(df)} depth levels")
    print(f"  Depth range: {df['depth'].min():.1f} - {df['depth'].max():.1f} m")
    print(f"  Temp range: {df['temp'].min():.2f} - {df['temp'].max():.2f} °C")
    
    return ctd_ds


def load_adcp_data(filepath: str, water_velocity: float) -> xr.Dataset:
    """
    Load ADCP template and set water velocity.
    
    Parameters
    ----------
    filepath : str
        Path to ADCP .mat file
    water_velocity : float
        Water velocity to use (m/s)
    
    Returns
    -------
    xr.Dataset
        ADCP data structure
    """
    print(f"Loading ADCP template from {filepath}")
    adcp = sio.loadmat(filepath)
    
    adcp_ds = xr.Dataset({
        'zadcp': (['adcpX', 'adcpY'], adcp['zadcp']),
        'vadcp': (['adcpX', 'adcpZ'], adcp['vadcp']),
        'tadcp': (['adcpY', 'adcpZ'], adcp['tadcp']),
        'wvel': (['adcpY'], np.array([water_velocity]))
    })
    
    print(f"  Water velocity set to {water_velocity} m/s")
    
    return adcp_ds


def run_size_class_simulation(
    length: float,
    forcing: EnvironmentalForcing,
    timespan: float,
    transfer_coeff_factor: float = 1,
    verbose: bool = True
) -> xr.Dataset:
    """
    Run melt simulation for a single iceberg size class.
    
    Parameters
    ----------
    length : float
        Iceberg length in meters
    forcing : EnvironmentalForcing
        Environmental forcing data
    timespan : float
        Simulation duration in seconds
    transfer_coeff_factor : float
        Adjustment factor for transfer coefficients
    verbose : bool
        Print progress messages
    
    Returns
    -------
    xr.Dataset
        Simulation results
    """
    if verbose:
        print(f"  Simulating {length:.0f}m iceberg...")
    
    # Create iceberg
    iceberg = Iceberg(length=length, dz=LAYER_THICKNESS)
    
    # Run simulation
    sim = IcebergMeltSimulation(iceberg, forcing)
    results = sim.run(
        timespan=timespan,
        do_constant_velocity=True,
        transfer_coeff_factor=transfer_coeff_factor,
        do_roll=True,
        quiet=True
    )
    
    if verbose:
        final = sim.get_final_state()
        print(f"    Initial volume: {results.VOL.values[0, 0]:.2e} m³")
        print(f"    Final volume: {final['volume']:.2e} m³")
    
    return results


def calculate_heat_flux(
    results: xr.Dataset,
    aww_depth: float = 150,
    day: int = 2
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate heat flux and total melt from iceberg.
    
    Parameters
    ----------
    results : xr.Dataset
        Simulation results
    aww_depth : float
        Atlantic Water depth threshold (m)
    day : int
        Which day to analyze
    
    Returns
    -------
    heat_flux : xr.DataArray
        Heat flux per depth layer (W)
    total_melt : xr.DataArray
        Total melt rate per depth layer (m³/s)
    """
    # Get keel depth
    keel = results.KEEL.sel(time=86400 * day).values[0]
    
    # Get melt rates below keel
    Mfreew = results.Mfreew.sel(Z=slice(None, keel), time=86400 * day)
    Mturbw = results.Mturbw.sel(Z=slice(None, keel), time=86400 * day)
    
    # Total underwater melt (integrated over faces)
    total_melt = np.mean(Mfreew + Mturbw, axis=1)  # m³/s per layer
    
    # Calculate heat flux: Q = m * L * ρ
    heat_flux = total_melt * LATENT_HEAT_FUSION * DENSITY_FRESHWATER  # W
    
    return heat_flux, total_melt


def calculate_ensemble_statistics(
    melt_dict: dict,
    size_counts: pd.Series,
    aww_depth: float = 150
) -> dict:
    """
    Calculate ensemble statistics for iceberg distribution.
    
    Parameters
    ----------
    melt_dict : dict
        Dictionary of simulation results by size class
    size_counts : pd.Series
        Number of icebergs in each size class
    aww_depth : float
        Atlantic Water depth threshold
    
    Returns
    -------
    stats : dict
        Ensemble statistics
    """
    print("\nCalculating ensemble statistics...")
    
    heat_flux_totals = {}
    melt_totals = {}
    volume_totals = {}
    
    for length, results in melt_dict.items():
        if length not in size_counts.index:
            continue
        
        count = size_counts[length]
        
        # Heat flux below AWW depth
        heat_flux, total_melt = calculate_heat_flux(results, aww_depth)
        heat_flux_sum = float(np.nansum(heat_flux.sel(Z=slice(aww_depth, None))))
        melt_sum = float(np.nansum(total_melt.sel(Z=slice(aww_depth, None))))
        
        heat_flux_totals[length] = heat_flux_sum * count
        melt_totals[length] = melt_sum * count
        
        # Volume below AWW depth
        vol = results.UWVOL.sel(Z=slice(aww_depth, None))
        vol_sum = float(np.nansum(vol))
        volume_totals[length] = vol_sum * count
    
    # Total ensemble values
    total_heat_flux = np.nansum(list(heat_flux_totals.values()))
    total_melt_rate = np.nansum(list(melt_totals.values()))
    total_volume = np.nansum(list(volume_totals.values()))
    
    # Mean melt rates across all classes
    mean_melt_rates = {}
    integrated_melt_rates = {}
    
    for length, results in melt_dict.items():
        if length not in size_counts.index:
            continue
        
        count = size_counts[length]
        mean_melt_rates[length] = float(results.i_mtotalm.values) * count
        integrated_melt_rates[length] = float(results.Mtotal.mean().values) * count
    
    total_mean_melt = np.nansum(list(mean_melt_rates.values()))
    total_integrated_melt = np.nansum(list(integrated_melt_rates.values()))
    
    print(f"  Total heat flux: {total_heat_flux:.2e} W")
    print(f"  Total melt rate: {total_melt_rate:.2e} m³/s")
    print(f"  Total ice volume (below {aww_depth}m): {total_volume:.2e} m³")
    
    return {
        'total_heat_flux': total_heat_flux,
        'total_melt_rate': total_melt_rate,
        'total_volume': total_volume,
        'mean_melt_rate': total_mean_melt,
        'integrated_melt_rate': total_integrated_melt,
        'heat_flux_by_class': heat_flux_totals,
        'melt_by_class': melt_totals,
        'volume_by_class': volume_totals
    }


def save_results(
    melt_dict: dict,
    stats: dict,
    ctd_data: xr.Dataset,
    date_str: str,
    water_velocity: float,
    transfer_factor: float,
    output_dir: str,
    fjord: str,
    run_type: str,
    aww_depth: float = 150
):
    """
    Save simulation results to files.
    
    Parameters
    ----------
    melt_dict : dict
        Simulation results by size class
    stats : dict
        Ensemble statistics
    ctd_data : xr.Dataset
        CTD data used
    date_str : str
        Date string for filename
    water_velocity : float
        Water velocity used (m/s)
    transfer_factor : float
        Transfer coefficient factor
    output_dir : str
        Base output directory
    fjord : str
        Fjord name
    run_type : str
        Run type ('min', 'avg', 'max')
    aww_depth : float
        Atlantic Water depth threshold
    """
    print("\nSaving results...")
    
    # Create output directories
    berg_model_dir = Path(output_dir) / 'iceberg_classes_output' / fjord / run_type
    berg_model_dir.mkdir(parents=True, exist_ok=True)
    
    model_output_dir = Path(output_dir) / 'iceberg_model_output' / fjord / run_type
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full melt dictionary
    urel_str = str(water_velocity).replace('.', '')
    pkl_file = berg_model_dir / f'{date_str}_urel{urel_str}_ctd_data_bergs_coeff{transfer_factor}.pkl'
    
    with open(pkl_file, 'wb') as f:
        pickle.dump(melt_dict, f)
    
    print(f"  Saved melt dictionary: {pkl_file}")
    
    # Create summary dataset
    date = pd.to_datetime(date_str)
    aww_temp = float(np.mean(ctd_data.temp.sel(tZ=slice(aww_depth, None))).values)
    
    summary_ds = xr.Dataset({
        'Qib': stats['total_heat_flux'],
        'iceberg_date': date,
        'transfer_coeff_factor': transfer_factor,
        'ice_vol': stats['total_volume'],
        'average_aww_temp': aww_temp,
        'melt_rate_avg': stats['mean_melt_rate'],
        'melt_rate_integrated': stats['integrated_melt_rate'],
    })
    
    # Add attributes
    summary_ds.Qib.attrs = {
        'units': 'W',
        'description': f'Total heat flux from icebergs below {aww_depth}m'
    }
    summary_ds.ice_vol.attrs = {
        'units': 'm³',
        'description': f'Total ice volume below {aww_depth}m'
    }
    summary_ds.average_aww_temp.attrs = {
        'units': '°C',
        'description': f'Average water temperature below {aww_depth}m'
    }
    summary_ds.melt_rate_avg.attrs = {
        'units': 'm/day',
        'description': 'Mean melt rate over all time, depths, processes, classes'
    }
    summary_ds.melt_rate_integrated.attrs = {
        'units': 'm³/s',
        'description': 'Average total freshwater flux for all iceberg classes'
    }
    
    # Save summary
    urel_filename = str(water_velocity).split('.')[1] if '.' in str(water_velocity) else str(water_velocity)
    nc_file = model_output_dir / f'{date_str}_{fjord}_coeff_{transfer_factor}_CTD_constant_UREL_{urel_filename}.nc'
    
    summary_ds.to_netcdf(nc_file)
    print(f"  Saved summary dataset: {nc_file}")


# ==============================================================================
# MAIN SIMULATION
# ==============================================================================

def main():
    """Run the Helheim iceberg melt simulation."""
    
    print("="*80)
    print("HELHEIM GLACIER ICEBERG MELT SIMULATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Fjord: {FJORD}")
    print(f"  Run type: {RUN_TYPE}")
    print(f"  Size classes: {len(SIZE_CLASSES)} ({SIZE_CLASSES[0]}-{SIZE_CLASSES[-1]}m)")
    print(f"  Layer thickness: {LAYER_THICKNESS}m")
    print(f"  Simulation duration: {TIMESPAN_DAYS} days")
    print(f"  Transfer coefficient factor: {TRANSFER_COEFF_FACTOR}")
    print(f"  Water velocity: {VELOCITY_SCENARIOS[RUN_TYPE]} m/s")
    
    # Load CTD data
    print("\n" + "-"*80)
    ctd_data = load_ctd_data(CTD_PATH)
    
    # Set water velocity for this scenario
    water_velocity = VELOCITY_SCENARIOS[RUN_TYPE]
    
    # Load ADCP data (or use constant velocity)
    # adcp_data = load_adcp_data(ADCP_PATH, water_velocity)
    # For now, just use constant velocity
    
    # Create environmental forcing
    print("\n" + "-"*80)
    print("Creating environmental forcing...")
    forcing = EnvironmentalForcing(
        ctd_data=ctd_data,
        ice_concentration=SEA_ICE_CONCENTRATION,
        wind_speed=WIND_SPEED,
        air_temperature=AIR_TEMPERATURE,
        solar_flux=SOLAR_FLUX,
        water_velocity=water_velocity
    )
    print(f"  Air temperature: {AIR_TEMPERATURE}°C")
    print(f"  Wind speed: {WIND_SPEED} m/s")
    print(f"  Solar flux: {SOLAR_FLUX} W/m²")
    print(f"  Sea ice concentration: {SEA_ICE_CONCENTRATION*100}%")
    
    # Get iceberg size distribution
    print("\n" + "-"*80)
    print("Loading iceberg size distribution...")
    
    # Look for iceberg geometry files
    geom_dir = Path(ICEBERG_GEOM_DIR)
    if geom_dir.exists():
        gdf_files = sorted([f for f in geom_dir.glob('*.gpkg')])
        print(f"  Found {len(gdf_files)} geometry files")
    else:
        print(f"  Warning: Geometry directory not found: {geom_dir}")
        print(f"  Running simulation without size distribution weighting")
        gdf_files = []
    
    # Process each date (or just run once if no files)
    if gdf_files:
        date_files = gdf_files
    else:
        # Create a dummy entry for single run
        date_files = ['2020-01-01']
    
    for berg_file in date_files:
        if isinstance(berg_file, Path):
            date_str = berg_file.stem[:10]
            print(f"\n{'='*80}")
            print(f"Processing: {date_str}")
            print(f"{'='*80}")
            
            # Load iceberg distribution
            icebergs_gdf = gpd.read_file(berg_file)
            size_counts = icebergs_gdf['binned'].value_counts()
            print(f"  Total icebergs: {len(icebergs_gdf)}")
            print(f"  Size classes represented: {len(size_counts)}")
        else:
            date_str = berg_file
            print(f"\n{'='*80}")
            print(f"Running single simulation (no size distribution)")
            print(f"{'='*80}")
            size_counts = pd.Series({length: 1 for length in SIZE_CLASSES})
        
        # Run simulation for each size class
        print(f"\nRunning {len(SIZE_CLASSES)} size class simulations...")
        print("-"*80)
        
        melt_dict = {}
        for length in SIZE_CLASSES:
            results = run_size_class_simulation(
                length=length,
                forcing=forcing,
                timespan=TIMESPAN_DAYS * 86400,
                transfer_coeff_factor=TRANSFER_COEFF_FACTOR,
                verbose=True
            )
            melt_dict[length] = results
        
        # Calculate ensemble statistics
        print("-"*80)
        stats = calculate_ensemble_statistics(melt_dict, size_counts, AWW_DEPTH)
        
        # Save results
        print("-"*80)
        save_results(
            melt_dict=melt_dict,
            stats=stats,
            ctd_data=ctd_data,
            date_str=date_str,
            water_velocity=water_velocity,
            transfer_factor=TRANSFER_COEFF_FACTOR,
            output_dir=OUTPUT_DIR,
            fjord=FJORD,
            run_type=RUN_TYPE,
            aww_depth=AWW_DEPTH
        )
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()