#!/usr/bin/env python3
"""
Detailed Diagnostic: Find Source of Heat Flux Difference

This script loads individual iceberg results to pinpoint where differences arise.
"""

import xarray as xr
import numpy as np
import pickle

print("="*80)
print("DETAILED DIAGNOSTIC: Finding Source of Differences")
print("="*80)

# Load the pickle files with individual iceberg results
OLD_PKL = '/home/laserglaciers/icebergPy/data/iceberg_classes_output_bug_fix_ctd/helheim/avg/2016-04-24_urel0.07_ctd_data_bergs_coeff1.pkl'
NEW_PKL = '/home/laserglaciers/icebergPy/data/test_class_outputs/iceberg_classes_output/helheim/avg/2016-04-24_urel007_ctd_data_bergs_coeff1.pkl'

print(f"\nLoading old (fixed bug) results...")
with open(OLD_PKL, 'rb') as f:
    old_results = pickle.load(f)

print(f"Loading new (refactored) results...")
with open(NEW_PKL, 'rb') as f:
    new_results = pickle.load(f)

# Compare a single iceberg (200m)
LENGTH = 200
print(f"\n{'='*80}")
print(f"COMPARING {LENGTH}m ICEBERG")
print(f"{'='*80}")

old = old_results[LENGTH]
new = new_results[LENGTH]

# Check what variables exist
print(f"\nOLD variables: {list(old.data_vars)}")
print(f"NEW variables: {list(new.data_vars)}")

# Check dimensions
print(f"\nOLD Mfreew shape: {old.Mfreew.shape}")
print(f"NEW Mfreew shape: {new.Mfreew.shape}")

print(f"\nOLD Mturbw shape: {old.Mturbw.shape}")
print(f"NEW Mturbw shape: {new.Mturbw.shape}")

# Check if we have uwV
print(f"\nOLD has uwV: {'uwV' in old.data_vars}")
print(f"NEW has uwV: {'uwV' in new.data_vars}")

# Compare melt rates at day 2
print(f"\n{'='*80}")
print(f"MELT RATES AT DAY 2 (time=172800)")
print(f"{'='*80}")

time_day2 = 86400 * 2

# Get keel depth at day 2
keel_old = old.KEEL.sel(time=time_day2).values[0]
keel_new = new.KEEL.sel(time=time_day2).values[0]

print(f"\nKeel depth at day 2:")
print(f"  OLD: {keel_old:.2f} m")
print(f"  NEW: {keel_new:.2f} m")
print(f"  Diff: {keel_new - keel_old:.2f} m")

# Get Mfreew at day 2, below keel
Mfreew_old = old.Mfreew.sel(Z=slice(None, keel_old), time=time_day2)
Mfreew_new = new.Mfreew.sel(Z=slice(None, keel_new), time=time_day2)

print(f"\nMfreew at day 2 (below keel):")
print(f"  OLD shape: {Mfreew_old.shape}")
print(f"  NEW shape: {Mfreew_new.shape}")
print(f"  OLD mean: {np.mean(Mfreew_old.values):.6e} m³/s")
print(f"  NEW mean: {np.mean(Mfreew_new.values):.6e} m³/s")

# Get Mturbw at day 2, below keel
Mturbw_old = old.Mturbw.sel(Z=slice(None, keel_old), time=time_day2)
Mturbw_new = new.Mturbw.sel(Z=slice(None, keel_new), time=time_day2)

print(f"\nMturbw at day 2 (below keel):")
print(f"  OLD shape: {Mturbw_old.shape}")
print(f"  NEW shape: {Mturbw_new.shape}")
print(f"  OLD mean: {np.mean(Mturbw_old.values):.6e} m³/s")
print(f"  NEW mean: {np.mean(Mturbw_new.values):.6e} m³/s")

# Calculate total_iceberg_melt as in original
total_melt_old = np.mean(Mfreew_old + Mturbw_old, axis=1)
total_melt_new = np.mean(Mfreew_new + Mturbw_new, axis=1)

print(f"\nTotal melt (mean over axis=1):")
print(f"  OLD shape: {total_melt_old.shape}")
print(f"  NEW shape: {total_melt_new.shape}")
print(f"  OLD sum: {np.nansum(total_melt_old):.6e} m³/s")
print(f"  NEW sum: {np.nansum(total_melt_new):.6e} m³/s")
print(f"  Diff: {np.nansum(total_melt_new - total_melt_old):.6e} m³/s")

# Calculate heat flux
l_heat = 3.34e5  # J/kg
p_fw = 1000  # kg/m³

Qib_old = total_melt_old * l_heat * p_fw
Qib_new = total_melt_new * l_heat * p_fw

print(f"\nHeat flux (per layer):")
print(f"  OLD sum: {np.nansum(Qib_old):.6e} W")
print(f"  NEW sum: {np.nansum(Qib_new):.6e} W")
print(f"  Diff: {np.nansum(Qib_new - Qib_old):.6e} W")

# Now check below 150m only
Qib_old_150 = Qib_old.sel(Z=slice(150, None))
Qib_new_150 = Qib_new.sel(Z=slice(150, None))

print(f"\nHeat flux (below 150m only):")
print(f"  OLD sum: {np.nansum(Qib_old_150):.6e} W")
print(f"  NEW sum: {np.nansum(Qib_new_150):.6e} W")
print(f"  Diff: {np.nansum(Qib_new_150 - Qib_old_150):.6e} W")

# Check total melt (Mtotal)
print(f"\n{'='*80}")
print(f"TOTAL MELT (Mtotal)")
print(f"{'='*80}")

print(f"\nMtotal mean over time:")
print(f"  OLD: {old.Mtotal.mean().values:.6e} m³/s")
print(f"  NEW: {new.Mtotal.mean().values:.6e} m³/s")
print(f"  Diff: {(new.Mtotal.mean().values - old.Mtotal.mean().values):.6e} m³/s")

# Check i_mtotalm
print(f"\ni_mtotalm (mean melt rate in m/day):")
print(f"  OLD: {old.i_mtotalm.values:.6f} m/day")
print(f"  NEW: {new.i_mtotalm.values:.6f} m/day")
print(f"  Diff: {(new.i_mtotalm.values - old.i_mtotalm.values):.6f} m/day")

# Check individual mechanisms
print(f"\n{'='*80}")
print(f"INDIVIDUAL MELT MECHANISMS (mean over time)")
print(f"{'='*80}")

mechanisms = {
    'Wave (Mwave)': 'Mwave',
    'Forced water (Mturbw)': 'Mturbw',
    'Forced air (Mturba)': 'Mturba',
    'Solar (Mfreea)': 'Mfreea',
    'Buoyant water (Mfreew)': 'Mfreew'
}

for name, var in mechanisms.items():
    old_mean = old[var].mean().values
    new_mean = new[var].mean().values
    
    # Handle different dimensions
    if np.isscalar(old_mean):
        diff = new_mean - old_mean
        pct_diff = 100 * diff / old_mean if old_mean != 0 else 0
    else:
        old_mean_scalar = float(old_mean)
        new_mean_scalar = float(new_mean)
        diff = new_mean_scalar - old_mean_scalar
        pct_diff = 100 * diff / old_mean_scalar if old_mean_scalar != 0 else 0
    
    print(f"\n{name}:")
    print(f"  OLD: {old_mean:.6e}")
    print(f"  NEW: {new_mean:.6e}")
    print(f"  Diff: {diff:.6e} ({pct_diff:.2f}%)")

print(f"\n{'='*80}")
print(f"DIAGNOSIS")
print(f"{'='*80}")

# Calculate percentage differences
qib_pct = 100 * np.nansum(Qib_new_150 - Qib_old_150) / np.nansum(Qib_old_150)
mtotal_pct = 100 * (new.Mtotal.mean().values - old.Mtotal.mean().values) / old.Mtotal.mean().values

print(f"\nFor {LENGTH}m iceberg:")
print(f"  Heat flux difference: {qib_pct:.2f}%")
print(f"  Total melt difference: {mtotal_pct:.2f}%")

if abs(qib_pct) > 5:
    print(f"\n⚠️  WARNING: Heat flux differs by {abs(qib_pct):.1f}% - significant!")
    print(f"    This suggests Mfreew and/or Mturbw are different")
    
if abs(mtotal_pct) > 5:
    print(f"\n⚠️  WARNING: Total melt differs by {abs(mtotal_pct):.1f}% - significant!")
    print(f"    This suggests a fundamental difference in melt calculations")

print(f"\n{'='*80}")
print(f"NEXT STEPS")
print(f"{'='*80}")
print("""
If differences are >5%:
1. Check if both files are using the FIXED melt_functions.py
2. Check transfer_coeff_factor in both (should be 1)
3. Check if time selection is correct (day 2 = 172800 seconds)
4. Compare individual depth layers to find where divergence occurs
""")