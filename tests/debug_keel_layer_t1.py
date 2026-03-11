#!/usr/bin/env python3
"""
Debug keel layer calculation at timestep 1
"""

import numpy as np
import pickle

print("="*80)
print("KEEL LAYER CALCULATION DEBUG - TIMESTEP 1")
print("="*80)

OLD_PKL = '/home/laserglaciers/icebergPy/data/iceberg_classes_output_bug_fix/helheim/avg/2016-04-24_urel0.07_ctd_data_bergs_coeff1.pkl'
NEW_PKL = '/home/laserglaciers/icebergPy/data/test_class_outputs/iceberg_classes_output/helheim/avg/2016-04-24_urel007_ctd_data_bergs_coeff1.pkl'

with open(OLD_PKL, 'rb') as f:
    old = pickle.load(f)[200]
with open(NEW_PKL, 'rb') as f:
    new = pickle.load(f)[200]

dz = 5
rho_i_fw = 917 / 1000
dt = 86400

print("\nAt timestep 0 (initial):")
keel_0_old = float(old.KEEL.isel(time=0).values[0])
keel_0_new = float(new.KEEL.isel(time=0).values[0])
print(f"  Keel OLD: {keel_0_old:.10f} m")
print(f"  Keel NEW: {keel_0_new:.10f} m")
print(f"  Match: {keel_0_old == keel_0_new}")

keel_layer_old = int(np.ceil(keel_0_old / dz))
keel_layer_new = int(np.ceil(keel_0_new / dz))
print(f"  Keel layer OLD: {keel_layer_old} (depth {keel_layer_old * dz}m)")
print(f"  Keel layer NEW: {keel_layer_new} (depth {keel_layer_new * dz}m)")

print("\nAt timestep 1:")
keel_1_old = float(old.KEEL.isel(time=1).values[0])
keel_1_new = float(new.KEEL.isel(time=1).values[0])
print(f"  Keel OLD: {keel_1_old:.10f} m")
print(f"  Keel NEW: {keel_1_new:.10f} m")
print(f"  Diff: {keel_1_new - keel_1_old:.10f} m")

# Calculate dz_keel for timestep 1
# Original formula: dz_keel = -1*((keeli-1) * dz - keel)
# At timestep 1, keeli is calculated from keel at t=0

k_old = keel_layer_old - 1  # Layer index (0-based)
k_new = keel_layer_new - 1

dz_keel_old = -1 * ((keel_layer_old - 1) * dz - keel_0_old)
dz_keel_new = -1 * ((keel_layer_new - 1) * dz - keel_0_new)

print(f"\nCalculating dz_keel for timestep 1:")
print(f"  OLD: dz_keel = -1 * (({keel_layer_old}-1) * {dz} - {keel_0_old:.6f})")
print(f"       dz_keel = -1 * ({(keel_layer_old-1) * dz} - {keel_0_old:.6f})")
print(f"       dz_keel = -1 * ({(keel_layer_old-1) * dz - keel_0_old:.6f})")
print(f"       dz_keel = {dz_keel_old:.6f} m")

print(f"\n  NEW: dz_keel = -1 * (({keel_layer_new}-1) * {dz} - {keel_0_new:.6f})")
print(f"       dz_keel = -1 * ({(keel_layer_new-1) * dz} - {keel_0_new:.6f})")
print(f"       dz_keel = -1 * ({(keel_layer_new-1) * dz - keel_0_new:.6f})")
print(f"       dz_keel = {dz_keel_new:.6f} m")

print(f"\n  Diff: {dz_keel_new - dz_keel_old:.10f} m")
print(f"  Match: {np.isclose(dz_keel_old, dz_keel_new)}")

# Get the melt rate at keel layer
i_mturbw_old = float(old.i_mturbw.isel(time=1, Z=k_old).values[0])
i_mturbw_new = float(new.i_mturbw.isel(time=1, Z=k_new).values[0])

print(f"\nMelt rate at keel layer (layer {k_old}):")
print(f"  i_mturbw OLD: {i_mturbw_old:.10e} m/day")
print(f"  i_mturbw NEW: {i_mturbw_new:.10e} m/day")
print(f"  Match: {i_mturbw_old == i_mturbw_new}")

# Get UWL and UWW at t=0 for keel layer
uwL_0_old = float(old.UWL.isel(time=0, Z=k_old).values[0])
uwL_0_new = float(new.UWL.isel(time=0, Z=k_new).values[0])
uwW_0_old = float(old.UWW.isel(time=0, Z=k_old).values[0])
uwW_0_new = float(new.UWW.isel(time=0, Z=k_new).values[0])

print(f"\nGeometry at keel layer at t=0:")
print(f"  UWL OLD: {uwL_0_old:.10f} m")
print(f"  UWL NEW: {uwL_0_new:.10f} m")
print(f"  UWW OLD: {uwW_0_old:.10f} m")
print(f"  UWW NEW: {uwW_0_new:.10f} m")

# Calculate what Mturbw should be
# Mturbw = (rho_i/rho_fw) * (2 * mtw * dz_keel * uwL + 1 * mtw * dz_keel * uwW) / dt

Mturbw_calc_old = (rho_i_fw * (2 * i_mturbw_old * dz_keel_old * uwL_0_old + 
                                1 * i_mturbw_old * dz_keel_old * uwW_0_old)) / dt

Mturbw_calc_new = (rho_i_fw * (2 * i_mturbw_new * dz_keel_new * uwL_0_new + 
                                1 * i_mturbw_new * dz_keel_new * uwW_0_new)) / dt

Mturbw_actual_old = float(old.Mturbw.isel(time=1, Z=k_old).values[0])
Mturbw_actual_new = float(new.Mturbw.isel(time=1, Z=k_new).values[0])

print(f"\nMturbw calculation:")
print(f"  OLD calculated: {Mturbw_calc_old:.10e} m³/s")
print(f"  OLD actual:     {Mturbw_actual_old:.10e} m³/s")
print(f"  Match: {np.isclose(Mturbw_calc_old, Mturbw_actual_old)}")

print(f"\n  NEW calculated: {Mturbw_calc_new:.10e} m³/s")
print(f"  NEW actual:     {Mturbw_actual_new:.10e} m³/s")
print(f"  Match: {np.isclose(Mturbw_calc_new, Mturbw_actual_new)}")

print(f"\n  Diff (calc): {Mturbw_calc_new - Mturbw_calc_old:.10e} m³/s")
print(f"  Diff (actual): {Mturbw_actual_new - Mturbw_actual_old:.10e} m³/s")
print(f"  % diff: {100*(Mturbw_actual_new - Mturbw_actual_old)/Mturbw_actual_old:.6f}%")

# Break down the difference
print("\n" + "="*80)
print("ISOLATING THE DIFFERENCE")
print("="*80)

if dz_keel_old != dz_keel_new:
    print(f"\n⚠ dz_keel differs by {dz_keel_new - dz_keel_old:.10f} m")
    print(f"  This should NOT happen since keel at t=0 is identical!")
    
if uwL_0_old != uwL_0_new:
    print(f"\n⚠ UWL at t=0 differs!")
    
if i_mturbw_old != i_mturbw_new:
    print(f"\n⚠ Melt rate differs!")

print("\n" + "="*80)
