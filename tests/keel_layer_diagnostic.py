#!/usr/bin/env python3
"""
Check if keel layer index or keel depth differs
"""

import numpy as np
import pickle

print("="*80)
print("KEEL LAYER DIAGNOSTIC")
print("="*80)

# Load results
OLD_PKL = '/home/laserglaciers/icebergPy/data/iceberg_classes_output_bug_fix/helheim/avg/2016-04-24_urel0.07_ctd_data_bergs_coeff1.pkl'
NEW_PKL = '/home/laserglaciers/icebergPy/data/test_class_outputs/iceberg_classes_output/helheim/avg/2016-04-24_urel007_ctd_data_bergs_coeff1.pkl'

with open(OLD_PKL, 'rb') as f:
    old_results = pickle.load(f)

with open(NEW_PKL, 'rb') as f:
    new_results = pickle.load(f)

LENGTH = 200
old = old_results[LENGTH]
new = new_results[LENGTH]

print(f"\nComparing {LENGTH}m iceberg")
print("\nKeel depth at each timestep:")
print(f"{'Timestep':<10} {'OLD':<20} {'NEW':<20} {'Diff':<20}")
print("-"*70)

for t_idx in range(min(10, len(old.time))):
    keel_old = float(old.KEEL.isel(time=t_idx).values[0])
    keel_new = float(new.KEEL.isel(time=t_idx).values[0])
    diff = keel_new - keel_old
    
    print(f"{t_idx:<10} {keel_old:<20.10f} {keel_new:<20.10f} {diff:<20.10f}")

# Check how many non-NaN values in Mturbw at each timestep
print("\n" + "="*80)
print("NUMBER OF NON-NAN LAYERS IN MTURBW")
print("="*80)

print(f"\n{'Timestep':<10} {'OLD':<10} {'NEW':<10} {'Diff':<10}")
print("-"*40)

for t_idx in range(min(10, len(old.time))):
    # Count non-NaN values
    old_count = np.sum(~np.isnan(old.Mturbw.isel(time=t_idx).values))
    new_count = np.sum(~np.isnan(new.Mturbw.isel(time=t_idx).values))
    
    print(f"{t_idx:<10} {old_count:<10} {new_count:<10} {new_count - old_count:<10}")

# Check the actual values at the keel layer
print("\n" + "="*80)
print("MTURBW AT KEEL LAYER")
print("="*80)

dz = 5
for t_idx in range(1, min(6, len(old.time))):
    keel_old = float(old.KEEL.isel(time=t_idx).values[0])
    keel_new = float(new.KEEL.isel(time=t_idx).values[0])
    
    keel_layer_old = int(np.ceil(keel_old / dz))
    keel_layer_new = int(np.ceil(keel_new / dz))
    
    print(f"\nTimestep {t_idx}:")
    print(f"  Keel OLD: {keel_old:.6f} m → layer {keel_layer_old}")
    print(f"  Keel NEW: {keel_new:.6f} m → layer {keel_layer_new}")
    
    if keel_layer_old > 0:
        k_old = keel_layer_old - 1
        Mturbw_keel_old = float(old.Mturbw.isel(time=t_idx, Z=k_old).values[0])
        print(f"  Mturbw at keel layer OLD: {Mturbw_keel_old:.10e}")
    
    if keel_layer_new > 0:
        k_new = keel_layer_new - 1
        Mturbw_keel_new = float(new.Mturbw.isel(time=t_idx, Z=k_new).values[0])
        print(f"  Mturbw at keel layer NEW: {Mturbw_keel_new:.10e}")
    
    if keel_layer_old == keel_layer_new and keel_layer_old > 0:
        diff = Mturbw_keel_new - Mturbw_keel_old
        pct = 100 * diff / Mturbw_keel_old if Mturbw_keel_old != 0 else 0
        print(f"  Difference: {diff:.10e} ({pct:.6f}%)")

# Check a specific problematic timestep (t=3)
print("\n" + "="*80)
print("DETAILED CHECK AT TIMESTEP 3")
print("="*80)

t_idx = 3
print(f"\nAll Mturbw values at timestep {t_idx}:")
print(f"{'Layer':<8} {'Depth':<10} {'OLD':<20} {'NEW':<20} {'Diff':<20} {'%':<10}")
print("-"*90)

for z_idx in range(30):  # First 30 layers
    depth = (z_idx + 1) * dz
    Mturbw_old = float(old.Mturbw.isel(time=t_idx, Z=z_idx).values[0])
    Mturbw_new = float(new.Mturbw.isel(time=t_idx, Z=z_idx).values[0])
    
    if not (np.isnan(Mturbw_old) and np.isnan(Mturbw_new)):
        diff = Mturbw_new - Mturbw_old
        pct = 100 * diff / Mturbw_old if (Mturbw_old != 0 and not np.isnan(Mturbw_old)) else 0
        
        print(f"{z_idx:<8} {depth:<10} {Mturbw_old:<20.10e} {Mturbw_new:<20.10e} {diff:<20.10e} {pct:<10.6f}")

print("\n" + "="*80)
