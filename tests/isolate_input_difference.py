#!/usr/bin/env python3
"""
Call melt_forcedwater directly to isolate the differing input
"""

import numpy as np
import pickle
import sys
sys.path.insert(0, '/home/laserglaciers/icebergPy/src')
from iceberg_model import melt
from scipy.interpolate import interp1d

print("="*80)
print("DIRECT MELT_FORCEDWATER CALL - ISOLATE DIFFERING INPUT")
print("="*80)

# This simulates what SHOULD happen at timestep 1, keel layer
# We'll reconstruct the exact inputs from the stored results

OLD_PKL = '/home/laserglaciers/icebergPy/data/iceberg_classes_output_bug_fix/helheim/avg/2016-04-24_urel0.07_ctd_data_bergs_coeff1.pkl'
NEW_PKL = '/home/laserglaciers/icebergPy/data/test_class_outputs/iceberg_classes_output/helheim/avg/2016-04-24_urel007_ctd_data_bergs_coeff1.pkl'

with open(OLD_PKL, 'rb') as f:
    old_results = pickle.load(f)
with open(NEW_PKL, 'rb') as f:
    new_results = pickle.load(f)

old = old_results[200]
new = new_results[200]

# Parameters
k = 25  # keel layer index (0-based)
depth_target = 130.0  # meters
dz = 5
factor = 1.0
use_constant_tf = False
constant_tf = None

# Get velocity
velocity_old = float(old.Urel.isel(Z=k, time=1).values.item())
velocity_new = float(new.Urel.isel(Z=k, time=1).values.item())

print(f"\nInputs:")
print(f"  Depth: {depth_target} m")
print(f"  Velocity OLD: {velocity_old} m/s")
print(f"  Velocity NEW: {velocity_new} m/s")
print(f"  Factor: {factor}")

# Now we need to get the CTD data and interpolate T and S
# The CTD data should be the same for both, but let's verify

# Try to get CTD-like data from the results
# The original would have depth, temp, salt arrays
# Let's see if we can reconstruct them

# Actually, let's just test with some example values first
# to see if melt_forcedwater gives the same result

T_test = 2.5
S_test = 34.5

print(f"\nTest with fixed T/S:")
print(f"  T_far: {T_test} °C")
print(f"  S_far: {S_test} psu")

melt_old, _, _ = melt.melt_forcedwater(
    T_test, S_test, depth_target, velocity_old,
    factor=factor, use_constant_tf=use_constant_tf, constant_tf=constant_tf
)

melt_new, _, _ = melt.melt_forcedwater(
    T_test, S_test, depth_target, velocity_new,
    factor=factor, use_constant_tf=use_constant_tf, constant_tf=constant_tf
)

print(f"\nMelt rate with test inputs:")
print(f"  OLD: {melt_old:.10e} m/s")
print(f"  NEW: {melt_new:.10e} m/s")
print(f"  Match: {melt_old == melt_new}")

# Convert to m/day
melt_old_day = melt_old * 86400
melt_new_day = melt_new * 86400

print(f"\nIn m/day:")
print(f"  OLD: {melt_old_day:.10e} m/day")
print(f"  NEW: {melt_new_day:.10e} m/day")

# Compare with actual stored values
i_mturbw_old = float(old.i_mturbw.isel(time=1, Z=k).values[0])
i_mturbw_new = float(new.i_mturbw.isel(time=1, Z=k).values[0])

print(f"\nActual stored i_mturbw:")
print(f"  OLD: {i_mturbw_old:.10e} m/day")
print(f"  NEW: {i_mturbw_new:.10e} m/day")

print(f"\nDoes our test match the stored values?")
print(f"  OLD match: {np.isclose(melt_old_day, i_mturbw_old)}")
print(f"  NEW match: {np.isclose(melt_new_day, i_mturbw_new)}")

# If they don't match, then T/S must be different
# Let's try to work backwards from the stored melt rate

print("\n" + "="*80)
print("WORKING BACKWARDS FROM STORED MELT RATES")
print("="*80)

# If the stored melt rates are 0.1918 and 0.1948 m/day,
# and we're using the same velocity, depth, and factor,
# then T or S must differ

# The melt rate is roughly proportional to thermal forcing
# So if NEW is 1.6% higher, thermal forcing might be 1.6% higher

# Let's try different T values to see which gives us the stored melt rates

from scipy.optimize import fsolve

def find_temp(T, target_melt, S, depth, vel):
    """Find T that gives target melt rate"""
    melt_rate, _, _ = melt.melt_forcedwater(T, S, depth, vel, factor=1.0)
    return (melt_rate * 86400 - target_melt)

# Find T for OLD
T_for_old = fsolve(find_temp, 2.5, args=(i_mturbw_old, S_test, depth_target, velocity_old))[0]
T_for_new = fsolve(find_temp, 2.5, args=(i_mturbw_new, S_test, depth_target, velocity_new))[0]

print(f"\nRequired T to match stored melt rates (assuming S={S_test}):")
print(f"  OLD needs T = {T_for_old:.6f} °C")
print(f"  NEW needs T = {T_for_new:.6f} °C")
print(f"  Difference: {T_for_new - T_for_old:.6f} °C")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if abs(T_for_new - T_for_old) > 0.01:
    print(f"\n⚠ Temperature or salinity interpolation differs!")
    print(f"  The NEW code is using T ~{T_for_new - T_for_old:.3f}°C warmer")
    print(f"  This causes {100*(i_mturbw_new/i_mturbw_old - 1):.2f}% higher melt rate")
else:
    print("\n✓ T and S should be the same")
    print("  Check for other parameter differences")

print("\n" + "="*80)