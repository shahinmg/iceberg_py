#!/usr/bin/env python3
"""
Check if geometry is being used correctly across all timesteps
"""

import numpy as np
import pickle

print("="*80)
print("GEOMETRY USAGE DIAGNOSTIC ACROSS ALL TIMESTEPS")
print("="*80)

# Load results
OLD_PKL = '/home/laserglaciers/icebergPy/data/iceberg_classes_output_bug_fix/helheim/avg/2016-04-24_urel007_ctd_data_bergs_coeff1.pkl'
NEW_PKL = '/home/laserglaciers/icebergPy/data/test_class_outputs/iceberg_classes_output/helheim/avg/2016-04-24_urel007_ctd_data_bergs_coeff1.pkl'

with open(OLD_PKL, 'rb') as f:
    old_results = pickle.load(f)

with open(NEW_PKL, 'rb') as f:
    new_results = pickle.load(f)

# Compare 200m iceberg
LENGTH = 200
old = old_results[LENGTH]
new = new_results[LENGTH]

print(f"\nComparing {LENGTH}m iceberg across all timesteps")
print("-"*80)

# Check forced water melt at multiple timesteps
timesteps_to_check = [1, 5, 10, 15, 20, 25, 29]

print(f"\n{'Day':<8} {'Mturbw_OLD':<15} {'Mturbw_NEW':<15} {'Diff':<15} {'%':<10}")
print("-"*70)

for t_idx in timesteps_to_check:
    time_sec = (t_idx + 1) * 86400  # +1 because timestep 1 is day 1
    day = time_sec / 86400
    
    try:
        Mturbw_old = float(old.Mturbw.isel(time=t_idx).mean().values)
        Mturbw_new = float(new.Mturbw.isel(time=t_idx).mean().values)
        
        diff = Mturbw_new - Mturbw_old
        pct = 100 * diff / Mturbw_old if Mturbw_old != 0 else 0
        
        print(f"{day:<8.0f} {Mturbw_old:<15.6e} {Mturbw_new:<15.6e} {diff:<15.6e} {pct:<10.3f}")
    except:
        pass

print("\n" + "="*80)
print("BUOYANT WATER MELT ACROSS TIMESTEPS")
print("="*80)

print(f"\n{'Day':<8} {'Mfreew_OLD':<15} {'Mfreew_NEW':<15} {'Diff':<15} {'%':<10}")
print("-"*70)

for t_idx in timesteps_to_check:
    time_sec = (t_idx + 1) * 86400
    day = time_sec / 86400
    
    try:
        Mfreew_old = float(old.Mfreew.isel(time=t_idx).mean().values)
        Mfreew_new = float(new.Mfreew.isel(time=t_idx).mean().values)
        
        diff = Mfreew_new - Mfreew_old
        pct = 100 * diff / Mfreew_old if Mfreew_old != 0 else 0
        
        print(f"{day:<8.0f} {Mfreew_old:<15.6e} {Mfreew_new:<15.6e} {diff:<15.6e} {pct:<10.3f}")
    except:
        pass

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

# Calculate mean differences
Mturbw_old_mean = float(old.Mturbw.mean().values)
Mturbw_new_mean = float(new.Mturbw.mean().values)
Mturbw_diff_pct = 100 * (Mturbw_new_mean - Mturbw_old_mean) / Mturbw_old_mean

Mfreew_old_mean = float(old.Mfreew.mean().values)
Mfreew_new_mean = float(new.Mfreew.mean().values)
Mfreew_diff_pct = 100 * (Mfreew_new_mean - Mfreew_old_mean) / Mfreew_old_mean

print(f"\nOverall mean differences:")
print(f"  Forced water (Mturbw): {Mturbw_diff_pct:.2f}%")
print(f"  Buoyant water (Mfreew): {Mfreew_diff_pct:.2f}%")

print("\n" + "="*80)
