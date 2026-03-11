#!/usr/bin/env python3
"""
Keel Depth Divergence Diagnostic

Track exactly when and why the keel depths start to differ.
"""

import xarray as xr
import numpy as np
import pickle

print("="*80)
print("KEEL DEPTH DIVERGENCE DIAGNOSTIC")
print("="*80)

# Load results
OLD_PKL = '/home/laserglaciers/icebergPy/data/iceberg_classes_output_bug_fix/helheim/avg/2016-04-24_urel0.07_ctd_data_bergs_coeff1.pkl'
NEW_PKL = '/home/laserglaciers/icebergPy/data/test_class_outputs/iceberg_classes_output/helheim/avg/2016-04-24_urel007_ctd_data_bergs_coeff1.pkl'

with open(OLD_PKL, 'rb') as f:
    old_results = pickle.load(f)

with open(NEW_PKL, 'rb') as f:
    new_results = pickle.load(f)

# Compare 200m iceberg
LENGTH = 200
old = old_results[LENGTH]
new = new_results[LENGTH]

print(f"\nComparing {LENGTH}m iceberg:")
print("-"*80)

# Check INITIAL geometry
print("\nINITIAL GEOMETRY (t=0):")
print("-"*80)

initial_vars = ['VOL', 'KEEL', 'FREEB', 'LEN', 'WIDTH', 'THICK']
for var in initial_vars:
    old_val = float(old[var].isel(time=0).values[0])
    new_val = float(new[var].isel(time=0).values[0])
    diff = new_val - old_val
    pct = 100 * diff / old_val if old_val != 0 else 0
    
    print(f"{var:<10} OLD: {old_val:12.6f}  NEW: {new_val:12.6f}  DIFF: {diff:12.6f} ({pct:6.3f}%)")

# Check initial underwater geometry
print("\nINITIAL UNDERWATER GEOMETRY:")
print("-"*80)

uwvol_old_0 = old.UWVOL.isel(time=0).values
uwvol_new_0 = new.UWVOL.isel(time=0).values

print(f"UWVOL sum (t=0):")
print(f"  OLD: {np.nansum(uwvol_old_0):.6e} m³")
print(f"  NEW: {np.nansum(uwvol_new_0):.6e} m³")
print(f"  DIFF: {np.nansum(uwvol_new_0 - uwvol_old_0):.6e} m³")

# Track keel evolution over time
print("\nKEEL EVOLUTION OVER TIME:")
print("-"*80)

times = old.time.values
print(f"{'Time (days)':<15} {'OLD Keel (m)':<15} {'NEW Keel (m)':<15} {'Diff (m)':<15} {'Diff (%)':<10}")
print("-"*80)

for i, t in enumerate(times[::3]):  # Every 3rd timestep
    day = t / 86400
    old_keel = float(old.KEEL.isel(time=i*3).values[0])
    new_keel = float(new.KEEL.isel(time=i*3).values[0])
    diff = new_keel - old_keel
    pct = 100 * diff / old_keel if old_keel != 0 else 0
    
    print(f"{day:<15.1f} {old_keel:<15.6f} {new_keel:<15.6f} {diff:<15.6f} {pct:<10.6f}")

# Check first timestep in detail
print("\nFIRST TIMESTEP (t=1 day) DETAILED:")
print("-"*80)

t1_vars = ['VOL', 'KEEL', 'FREEB', 'LEN', 'WIDTH', 'THICK', 'SAILVOL']
for var in t1_vars:
    old_t0 = float(old[var].isel(time=0).values[0])
    old_t1 = float(old[var].isel(time=1).values[0])
    new_t0 = float(new[var].isel(time=0).values[0])
    new_t1 = float(new[var].isel(time=1).values[0])
    
    old_change = old_t1 - old_t0
    new_change = new_t1 - new_t0
    diff_change = new_change - old_change
    
    print(f"\n{var}:")
    print(f"  OLD: {old_t0:.6f} → {old_t1:.6f} (change: {old_change:.6f})")
    print(f"  NEW: {new_t0:.6f} → {new_t1:.6f} (change: {new_change:.6f})")
    print(f"  DIFF in change: {diff_change:.6f}")

# Check melt rates at first timestep
print("\nMELT RATES AT FIRST TIMESTEP (t=1 day):")
print("-"*80)

melt_vars = ['Mwave', 'Mfreea', 'Mturba', 'Mturbw', 'Mfreew', 'Mtotal']
for var in melt_vars:
    old_val = old[var].isel(time=1).values
    new_val = new[var].isel(time=1).values
    
    # Handle different dimensions
    old_mean = float(np.nanmean(old_val))
    new_mean = float(np.nanmean(new_val))
    diff = new_mean - old_mean
    pct = 100 * diff / old_mean if old_mean != 0 else 0
    
    print(f"{var:<10} OLD: {old_mean:.6e}  NEW: {new_mean:.6e}  DIFF: {diff:.6e} ({pct:6.3f}%)")

# Check individual melt rates (m/day) at first timestep
print("\nINDIVIDUAL MELT RATES (m/day) AT FIRST TIMESTEP:")
print("-"*80)

i_melt_vars = ['i_mwave', 'i_mfreea', 'i_mturbw', 'i_mfreew', 'i_mturba']
for var in i_melt_vars:
    # These are 2D (X, time) arrays
    old_val = float(old[var].isel(time=1).values.flatten()[0])
    new_val = float(new[var].isel(time=1).values.flatten()[0])
    diff = new_val - old_val
    pct = 100 * diff / old_val if old_val != 0 else 0
    
    print(f"{var:<12} OLD: {old_val:.6f}  NEW: {new_val:.6f}  DIFF: {diff:.6f} ({pct:6.3f}%)")

# Check underwater volume evolution
print("\nUNDERWATER VOLUME AT FIRST TIMESTEP:")
print("-"*80)

uwvol_old_1 = old.UWVOL.isel(time=1).values
uwvol_new_1 = new.UWVOL.isel(time=1).values

print(f"UWVOL sum (t=1):")
print(f"  OLD: {np.nansum(uwvol_old_1):.6e} m³")
print(f"  NEW: {np.nansum(uwvol_new_1):.6e} m³")
print(f"  DIFF: {np.nansum(uwvol_new_1 - uwvol_old_1):.6e} m³")

# Check if there are NaN differences
print("\nNaN PATTERN CHECK:")
print("-"*80)

nan_old = np.isnan(uwvol_old_1).sum()
nan_new = np.isnan(uwvol_new_1).sum()
print(f"NaN count in UWVOL (t=1):")
print(f"  OLD: {nan_old}")
print(f"  NEW: {nan_new}")
print(f"  DIFF: {nan_new - nan_old}")

# Check depth layers
print("\nDEPTH LAYER INFORMATION:")
print("-"*80)

if 'depth' in old.data_vars:
    old_depths = old.depth.values.flatten()
    new_depths = new.depth.values.flatten()
    
    print(f"OLD depths: {old_depths[:10]} ... (first 10)")
    print(f"NEW depths: {new_depths[:10]} ... (first 10)")
    print(f"Match: {np.allclose(old_depths, new_depths)}")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)

# Determine if initial geometry matches
initial_match = True
for var in initial_vars:
    old_val = float(old[var].isel(time=0).values[0])
    new_val = float(new[var].isel(time=0).values[0])
    if abs(old_val - new_val) > 1e-6:
        initial_match = False
        break

if not initial_match:
    print("\n❌ INITIAL GEOMETRY DOES NOT MATCH!")
    print("   The iceberg initialization is different between old and new code.")
    print("   Check iceberg.py initialization.")
else:
    print("\n✓ Initial geometry matches.")
    print("\n❌ But keel diverges over time.")
    print("   Likely causes:")
    print("   1. Different melt rate calculations")
    print("   2. Different geometry update logic")
    print("   3. Different buoyancy adjustment")
    print("   4. Different stability/rolling logic")
    
    # Check which is most likely
    first_step_melt_diff = abs(new_mean - old_mean) / old_mean if 'old_mean' in locals() else 0
    if first_step_melt_diff > 0.01:
        print(f"\n   → First timestep melt differs by {first_step_melt_diff*100:.2f}%")
        print("     This suggests melt calculation differences.")

print("\n" + "="*80)