#!/usr/bin/env python3
"""
Compare melt_forcedwater Function Inputs

Directly compare what gets passed to melt_forcedwater in both codes
at the first timestep.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/laserglaciers/icebergPy/src')

# Import both melt modules
from iceberg_model import melt as new_melt

# Also need to check the original
import importlib.util
spec = importlib.util.spec_from_file_location("old_melt", "/home/laserglaciers/icebergPy/src/iceberg_model/melt_functions.py")
old_melt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_melt_module)

print("="*80)
print("MELT_FORCEDWATER FUNCTION COMPARISON")
print("="*80)

# Test with same inputs (as arrays)
T_far = np.array([2.5])  # Temperature
S_far = np.array([34.5])  # Salinity
depth = np.array([50.0])  # Depth/pressure
U_rel = np.array([0.1])  # Velocity
factor = 1.0  # Transfer coefficient factor

print(f"\nTest inputs:")
print(f"  T_far: {T_far[0]} °C")
print(f"  S_far: {S_far[0]} psu")
print(f"  depth: {depth[0]} m")
print(f"  U_rel: {U_rel[0]} m/s")
print(f"  factor: {factor}")

# Call both functions
old_result, old_Tsh, old_Tfp = old_melt_module.melt_forcedwater(
    T_far, S_far, depth, U_rel, factor, use_constant_tf=False, constant_tf=None
)

new_result, new_Tsh, new_Tfp = new_melt.melt_forcedwater(
    T_far[0], S_far[0], depth[0], U_rel[0], factor, use_constant_tf=False, constant_tf=None
)

print(f"\nOLD melt_forcedwater:")
print(f"  Melt rate: {old_result[0]:.10e} m/s")
print(f"  T_sh: {old_Tsh[0]:.10e}")
print(f"  T_fp: {old_Tfp[0]:.10e}")

print(f"\nNEW melt_forcedwater:")
print(f"  Melt rate: {new_result:.10e} m/s")
print(f"  T_sh: {new_Tsh:.10e}")
print(f"  T_fp: {new_Tfp:.10e}")

print(f"\nDifferences:")
print(f"  Melt rate: {(new_result - old_result[0]):.10e} m/s ({100*(new_result-old_result[0])/old_result[0]:.6f}%)")
print(f"  T_sh: {(new_Tsh - old_Tsh[0]):.10e}")
print(f"  T_fp: {(new_Tfp - old_Tfp[0]):.10e}")

if abs(new_result - old_result[0]) < 1e-12:
    print("\n✓ Functions are IDENTICAL!")
else:
    print(f"\n❌ Functions differ by {100*(new_result-old_result[0])/old_result[0]:.6f}%")
    print("   → The melt.py file may have been modified from the original")
    print("   → Check if constants or equations differ")

print("\n" + "="*80)