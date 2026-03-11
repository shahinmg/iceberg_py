#!/usr/bin/env python3
"""
Detailed Geometry Comparison - Find exactly where geometry diverges
"""

import numpy as np
import pickle

print("="*80)
print("DETAILED GEOMETRY COMPARISON AT EACH TIMESTEP")
print("="*80)

# Load results
OLD_PKL = '/home/laserglaciers/icebergPy/data/iceberg_classes_output_bug_fix_ctd/helheim/avg/2016-04-24_urel0.07_ctd_data_bergs_coeff1.pkl'
NEW_PKL = '/home/laserglaciers/icebergPy/data/test_class_outputs/iceberg_classes_output/helheim/avg/2016-04-24_urel007_ctd_data_bergs_coeff1.pkl'

with open(OLD_PKL, 'rb') as f:
    old_results = pickle.load(f)

with open(NEW_PKL, 'rb') as f:
    new_results = pickle.load(f)

# Compare 200m iceberg
LENGTH = 200
old = old_results[LENGTH]
new = new_results[LENGTH]

print(f"\nComparing {LENGTH}m iceberg")
print("-"*80)

# Check UWL (underwater length) at each timestep
print("\nUWL (underwater length) at layer 0 (depth 5m):")
print(f"{'Timestep':<10} {'OLD':<20} {'NEW':<20} {'Diff':<20} {'%':<10}")
print("-"*80)

for t_idx in range(min(10, len(old.time))):
    uwL_old = float(old.UWL.isel(time=t_idx, Z=0).values[0])
    uwL_new = float(new.UWL.isel(time=t_idx, Z=0).values[0])
    diff = uwL_new - uwL_old
    pct = 100 * diff / uwL_old if uwL_old != 0 else 0
    
    print(f"{t_idx:<10} {uwL_old:<20.10f} {uwL_new:<20.10f} {diff:<20.10f} {pct:<10.6f}")

print("\n" + "="*80)
print("MELT RATES AT EACH TIMESTEP (layer 0, depth 5m)")
print("="*80)

print(f"\n{'Timestep':<10} {'i_mturbw_OLD':<20} {'i_mturbw_NEW':<20} {'Diff':<20} {'%':<10}")
print("-"*80)

for t_idx in range(min(10, len(old.time))):
    mturbw_old = float(old.i_mturbw.isel(time=t_idx, Z=0).values[0])
    mturbw_new = float(new.i_mturbw.isel(time=t_idx, Z=0).values[0])
    diff = mturbw_new - mturbw_old
    pct = 100 * diff / mturbw_old if mturbw_old != 0 else 0
    
    print(f"{t_idx:<10} {mturbw_old:<20.10e} {mturbw_new:<20.10e} {diff:<20.10e} {pct:<10.6f}")

print("\n" + "="*80)
print("GEOMETRY EVOLUTION DETAILS")
print("="*80)

# At timestep 1 (day 2), check if UWL matches at t=0 and t=1
print("\nTimestep 0 (initial):")
print(f"  UWL[0] OLD: {float(old.UWL.isel(time=0, Z=0).values[0]):.10f}")
print(f"  UWL[0] NEW: {float(new.UWL.isel(time=0, Z=0).values[0]):.10f}")

print("\nTimestep 1 (after first melt):")
uwL_0_old = float(old.UWL.isel(time=0, Z=0).values[0])
uwL_1_old = float(old.UWL.isel(time=1, Z=0).values[0])
uwL_0_new = float(new.UWL.isel(time=0, Z=0).values[0])
uwL_1_new = float(new.UWL.isel(time=1, Z=0).values[0])

print(f"  UWL[0] OLD: {uwL_1_old:.10f} (change: {uwL_1_old - uwL_0_old:.10f})")
print(f"  UWL[0] NEW: {uwL_1_new:.10f} (change: {uwL_1_new - uwL_0_new:.10f})")
print(f"  Diff in change: {(uwL_1_new - uwL_0_new) - (uwL_1_old - uwL_0_old):.10f}")

print("\nTimestep 2:")
uwL_2_old = float(old.UWL.isel(time=2, Z=0).values[0])
uwL_2_new = float(new.UWL.isel(time=2, Z=0).values[0])

print(f"  UWL[0] OLD: {uwL_2_old:.10f} (change from t=1: {uwL_2_old - uwL_1_old:.10f})")
print(f"  UWL[0] NEW: {uwL_2_new:.10f} (change from t=1: {uwL_2_new - uwL_1_new:.10f})")
print(f"  Diff in change: {(uwL_2_new - uwL_1_new) - (uwL_2_old - uwL_1_old):.10f}")

# Check if Mturbw at t=1 uses UWL from t=0 or t=1
print("\n" + "="*80)
print("WHICH GEOMETRY IS USED FOR MTURBW?")
print("="*80)

print("\nAt timestep 1:")
i_mturbw_1_old = float(old.i_mturbw.isel(time=1, Z=0).values[0])
Mturbw_1_old = float(old.Mturbw.isel(time=1, Z=0).values[0])

# Calculate what Mturbw should be using UWL from t=0
uwL_t0 = float(old.UWL.isel(time=0, Z=0).values[0])
uwW_t0 = float(old.UWW.isel(time=0, Z=0).values[0])
dz = 5
rho_i_fw = 917 / 1000
dt = 86400

# Mturbw = (rho_i/rho_fw) * (2 * mtw * dz * uwL + 1 * mtw * dz * uwW) / dt
# where mtw = i_mturbw (already in m/day)
Mturbw_calc_t0 = (rho_i_fw * (2 * i_mturbw_1_old * dz * uwL_t0 + 1 * i_mturbw_1_old * dz * uwW_t0)) / dt

# Calculate what Mturbw should be using UWL from t=1  
uwL_t1 = float(old.UWL.isel(time=1, Z=0).values[0])
uwW_t1 = float(old.UWW.isel(time=1, Z=0).values[0])
Mturbw_calc_t1 = (rho_i_fw * (2 * i_mturbw_1_old * dz * uwL_t1 + 1 * i_mturbw_1_old * dz * uwW_t1)) / dt

print(f"  i_mturbw at t=1: {i_mturbw_1_old:.10e} m/day")
print(f"  Actual Mturbw at t=1: {Mturbw_1_old:.10e} m³/s")
print(f"\n  If using UWL from t=0: {Mturbw_calc_t0:.10e} m³/s")
print(f"  If using UWL from t=1: {Mturbw_calc_t1:.10e} m³/s")
print(f"\n  Match with t=0? {np.isclose(Mturbw_1_old, Mturbw_calc_t0)}")
print(f"  Match with t=1? {np.isclose(Mturbw_1_old, Mturbw_calc_t1)}")

print("\n" + "="*80)
