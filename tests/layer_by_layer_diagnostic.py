#!/usr/bin/env python3
"""
Layer-by-Layer Forced Water Melt Diagnostic

Compare forced water melt calculations at each depth layer to find
exactly where the 0.435% difference originates.
"""

import numpy as np
import pickle
import xarray as xr

print("="*80)
print("LAYER-BY-LAYER FORCED WATER MELT DIAGNOSTIC")
print("="*80)

# Load results
OLD_PKL = '/home/laserglaciers/icebergPy/data/iceberg_classes_output_bug_fix_ctd/helheim/avg/2016-04-24_urel0.07_ctd_data_bergs_coeff1.pkl'
NEW_PKL = '/home/laserglaciers/icebergPy/data/test_class_outputs/iceberg_classes_output/helheim/avg/2016-04-24_urel007_ctd_data_bergs_coeff1.pkl'

with open(OLD_PKL, 'rb') as f:
    old_results = pickle.load(f)

with open(NEW_PKL, 'rb') as f:
    new_results = pickle.load(f)

# Compare 200m iceberg at first timestep
LENGTH = 200
old = old_results[LENGTH]
new = new_results[LENGTH]

print(f"\nComparing {LENGTH}m iceberg at t=1 day (timestep index 1)")
print("-"*80)

# Get the forced water melt at t=1
t_idx = 1

# Extract Mturbw (integrated volume m³/s) and i_mturbw (rate m/day)
Mturbw_old = old.Mturbw.isel(time=t_idx).values  # (Z, X)
Mturbw_new = new.Mturbw.isel(time=t_idx).values  # (Z, X)

i_mturbw_old = old.i_mturbw.isel(time=t_idx).values  # (Z, X)
i_mturbw_new = new.i_mturbw.isel(time=t_idx).values  # (Z, X)

print(f"\nMturbw shape: {Mturbw_old.shape}")
print(f"i_mturbw shape: {i_mturbw_old.shape}")

# Get depth coordinates
if 'depth' in old.data_vars:
    depths_old = old.depth.values.flatten()
    depths_new = new.depth.values.flatten()
else:
    depths_old = old.Z.values
    depths_new = new.Z.values

print(f"\nDepth coordinates (first 10):")
print(f"  OLD: {depths_old[:10]}")
print(f"  NEW: {depths_new[:10]}")
print(f"  Match: {np.allclose(depths_old, depths_new)}")

# Get keel depth and layer index
keel_old = float(old.KEEL.isel(time=t_idx).values[0])
keel_new = float(new.KEEL.isel(time=t_idx).values[0])

print(f"\nKeel depth at t=1:")
print(f"  OLD: {keel_old:.6f} m")
print(f"  NEW: {keel_new:.6f} m")
print(f"  Match: {np.isclose(keel_old, keel_new)}")

# Calculate keel layer index (assuming dz=5)
dz = 5
keel_layer_old = int(np.ceil(keel_old / dz))
keel_layer_new = int(np.ceil(keel_new / dz))

print(f"\nKeel layer index:")
print(f"  OLD: {keel_layer_old}")
print(f"  NEW: {keel_layer_new}")

# Get underwater geometry at t=1
UWL_old = old.UWL.isel(time=t_idx).values.flatten()
UWW_old = old.UWW.isel(time=t_idx).values.flatten()
UWL_new = new.UWL.isel(time=t_idx).values.flatten()
UWW_new = new.UWW.isel(time=t_idx).values.flatten()

print(f"\nUnderwater geometry at t=1 (first 5 layers):")
print(f"  UWL OLD: {UWL_old[:5]}")
print(f"  UWL NEW: {UWL_new[:5]}")
print(f"  UWW OLD: {UWW_old[:5]}")
print(f"  UWW NEW: {UWW_new[:5]}")

# Compare layer by layer
print("\n" + "="*80)
print("LAYER-BY-LAYER COMPARISON")
print("="*80)
print(f"{'Depth':<8} {'i_mturbw_OLD':<15} {'i_mturbw_NEW':<15} {'Diff':<12} {'%':<8} {'Mturbw_OLD':<15} {'Mturbw_NEW':<15} {'Diff':<12} {'%':<8}")
print("-"*140)

total_Mturbw_old = 0
total_Mturbw_new = 0
total_i_mturbw_old = 0
total_i_mturbw_new = 0

for k in range(min(keel_layer_old, keel_layer_new)):
    depth = depths_old[k]
    
    # Melt rate (m/day)
    i_old = i_mturbw_old[k, 0]
    i_new = i_mturbw_new[k, 0]
    i_diff = i_new - i_old
    i_pct = 100 * i_diff / i_old if i_old != 0 else 0
    
    # Integrated volume (m³/s)
    M_old = Mturbw_old[k, 0]
    M_new = Mturbw_new[k, 0]
    M_diff = M_new - M_old
    M_pct = 100 * M_diff / M_old if M_old != 0 else 0
    
    if not (np.isnan(i_old) or np.isnan(M_old)):
        total_i_mturbw_old += i_old
        total_i_mturbw_new += i_new
        total_Mturbw_old += M_old
        total_Mturbw_new += M_new
        
        # Only print layers with non-zero melt
        if abs(i_old) > 1e-10 or abs(i_new) > 1e-10:
            print(f"{depth:<8.1f} {i_old:<15.6e} {i_new:<15.6e} {i_diff:<12.6e} {i_pct:<8.3f} {M_old:<15.6e} {M_new:<15.6e} {M_diff:<12.6e} {M_pct:<8.3f}")

print("-"*140)
print(f"{'TOTAL':<8} {total_i_mturbw_old:<15.6e} {total_i_mturbw_new:<15.6e} {total_i_mturbw_new-total_i_mturbw_old:<12.6e} {100*(total_i_mturbw_new-total_i_mturbw_old)/total_i_mturbw_old:<8.3f} {total_Mturbw_old:<15.6e} {total_Mturbw_new:<15.6e} {total_Mturbw_new-total_Mturbw_old:<12.6e} {100*(total_Mturbw_new-total_Mturbw_old)/total_Mturbw_old:<8.3f}")

# Check the conversion from i_mturbw to Mturbw
print("\n" + "="*80)
print("CONVERSION CHECK: i_mturbw → Mturbw")
print("="*80)
print("\nOriginal formula: Mturbw[k] = 2 * (mtw[k] * dz * uwL[k]) + 1 * (mtw[k] * dz * uwW[k])")
print("Where mtw[k] = melt_rate * dt (m/day)")
print("Then Mturbw converted: Mturbw = (rho_i/rho_fw) * Mturbw / dt")

rho_i_fw = 917 / 1000
dt = 86400

print(f"\nChecking a few layers manually:")
for k in [0, 5, 10]:
    if k < len(i_mturbw_old):
        i_old = i_mturbw_old[k, 0]
        M_old = Mturbw_old[k, 0]
        
        if not np.isnan(i_old) and abs(i_old) > 1e-10:
            # Calculate what Mturbw should be
            uwL = UWL_old[k]
            uwW = UWW_old[k]
            
            # mtw is in m/day, which is i_mturbw
            mtw = i_old
            
            # Integrated volume before conversion
            Mturbw_before = 2 * (mtw * dz * uwL) + 1 * (mtw * dz * uwW)
            
            # After conversion to m³/s
            Mturbw_after = (rho_i_fw * Mturbw_before) / dt
            
            print(f"\nLayer {k} (depth {depths_old[k]}m):")
            print(f"  i_mturbw: {i_old:.6e} m/day")
            print(f"  UWL: {uwL:.6f}, UWW: {uwW:.6f}")
            print(f"  Expected Mturbw: {Mturbw_after:.6e} m³/s")
            print(f"  Actual Mturbw:   {M_old:.6e} m³/s")
            print(f"  Match: {np.isclose(Mturbw_after, M_old)}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

# Check if the difference is uniform across layers
i_diffs = []
M_diffs = []
for k in range(min(keel_layer_old, keel_layer_new)):
    i_old = i_mturbw_old[k, 0]
    i_new = i_mturbw_new[k, 0]
    M_old = Mturbw_old[k, 0]
    M_new = Mturbw_new[k, 0]
    
    if not (np.isnan(i_old) or np.isnan(M_old)) and abs(i_old) > 1e-10:
        i_pct = 100 * (i_new - i_old) / i_old
        M_pct = 100 * (M_new - M_old) / M_old
        i_diffs.append(i_pct)
        M_diffs.append(M_pct)

if i_diffs:
    print(f"\ni_mturbw differences across layers:")
    print(f"  Mean: {np.mean(i_diffs):.4f}%")
    print(f"  Std:  {np.std(i_diffs):.4f}%")
    print(f"  Min:  {np.min(i_diffs):.4f}%")
    print(f"  Max:  {np.max(i_diffs):.4f}%")
    
    print(f"\nMturbw differences across layers:")
    print(f"  Mean: {np.mean(M_diffs):.4f}%")
    print(f"  Std:  {np.std(M_diffs):.4f}%")
    print(f"  Min:  {np.min(M_diffs):.4f}%")
    print(f"  Max:  {np.max(M_diffs):.4f}%")
    
    # Check if differences are uniform
    uniform = np.std(i_diffs) < 0.01
    if uniform:
        print("\n✓ Differences are UNIFORM across layers")
        print("  → Suggests a systematic offset in the melt rate calculation")
        print("  → Check: interpolation, velocity, or melt_forcedwater function")
    else:
        print("\n⚠ Differences VARY across layers")
        print("  → Suggests depth-dependent issue")
        print("  → Check: depth interpolation, pressure calculation")

print("\n" + "="*80)
