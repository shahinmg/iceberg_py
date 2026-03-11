#!/usr/bin/env python3
"""
Check Mturbw (integrated volumes) at each timestep
"""

import numpy as np
import pickle

print("="*80)
print("MTURBW (INTEGRATED VOLUMES) AT EACH TIMESTEP")
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
print("\nMturbw at layer 0 (depth 5m):")
print(f"{'Timestep':<10} {'OLD':<20} {'NEW':<20} {'Diff':<20} {'%':<10}")
print("-"*80)

for t_idx in range(min(10, len(old.time))):
    Mturbw_old = float(old.Mturbw.isel(time=t_idx, Z=0).values[0])
    Mturbw_new = float(new.Mturbw.isel(time=t_idx, Z=0).values[0])
    diff = Mturbw_new - Mturbw_old
    pct = 100 * diff / Mturbw_old if Mturbw_old != 0 else 0
    
    print(f"{t_idx:<10} {Mturbw_old:<20.10e} {Mturbw_new:<20.10e} {diff:<20.10e} {pct:<10.6f}")

# Now check the SUM over all depths
print("\n" + "="*80)
print("MTURBW SUMMED OVER ALL DEPTHS")
print("="*80)

print(f"\n{'Timestep':<10} {'OLD':<20} {'NEW':<20} {'Diff':<20} {'%':<10}")
print("-"*80)

for t_idx in range(min(10, len(old.time))):
    Mturbw_old_sum = float(old.Mturbw.isel(time=t_idx).sum().values)
    Mturbw_new_sum = float(new.Mturbw.isel(time=t_idx).sum().values)
    diff = Mturbw_new_sum - Mturbw_old_sum
    pct = 100 * diff / Mturbw_old_sum if Mturbw_old_sum != 0 else 0
    
    print(f"{t_idx:<10} {Mturbw_old_sum:<20.10e} {Mturbw_new_sum:<20.10e} {diff:<20.10e} {pct:<10.6f}")

# Check what the stored UWL values are at t=1
print("\n" + "="*80)
print("CHECK: Which UWL is stored in results?")
print("="*80)

print("\nAt t=1, layer 0:")
uwL_1_old = float(old.UWL.isel(time=1, Z=0).values[0])
uwL_1_new = float(new.UWL.isel(time=1, Z=0).values[0])

print(f"  UWL OLD: {uwL_1_old:.10f}")
print(f"  UWL NEW: {uwL_1_new:.10f}")
print(f"  Match: {uwL_1_old == uwL_1_new}")

# The stored UWL should be AFTER melt is applied
# Let's verify by checking if Mturbw at t=2 uses UWL from t=1
print("\nAt timestep 2, checking which UWL was used:")
i_mturbw_2_old = float(old.i_mturbw.isel(time=2, Z=0).values[0])
Mturbw_2_old = float(old.Mturbw.isel(time=2, Z=0).values[0])

uwL_t1 = float(old.UWL.isel(time=1, Z=0).values[0])
uwW_t1 = float(old.UWW.isel(time=1, Z=0).values[0])
uwL_t2 = float(old.UWL.isel(time=2, Z=0).values[0])
uwW_t2 = float(old.UWW.isel(time=2, Z=0).values[0])

dz = 5
rho_i_fw = 917 / 1000
dt = 86400

Mturbw_calc_t1 = (rho_i_fw * (2 * i_mturbw_2_old * dz * uwL_t1 + 1 * i_mturbw_2_old * dz * uwW_t1)) / dt
Mturbw_calc_t2 = (rho_i_fw * (2 * i_mturbw_2_old * dz * uwL_t2 + 1 * i_mturbw_2_old * dz * uwW_t2)) / dt

print(f"  Actual Mturbw at t=2: {Mturbw_2_old:.10e}")
print(f"  If using UWL from t=1: {Mturbw_calc_t1:.10e}")
print(f"  If using UWL from t=2: {Mturbw_calc_t2:.10e}")
print(f"  Match with t=1? {np.isclose(Mturbw_2_old, Mturbw_calc_t1)}")
print(f"  Match with t=2? {np.isclose(Mturbw_2_old, Mturbw_calc_t2)}")

print("\n" + "="*80)
