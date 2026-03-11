#!/usr/bin/env python3
"""
Check if the difference is from NaNs vs zeros at index 0
"""

import pickle
import numpy as np

OLD_PKL = '/home/laserglaciers/icebergPy/data/iceberg_classes_output_bug_fix_ctd/helheim/avg/2016-04-24_urel0.07_ctd_data_bergs_coeff1.pkl'
NEW_PKL = '/home/laserglaciers/icebergPy/data/test_class_outputs/iceberg_classes_output/helheim/avg/2016-04-24_urel007_ctd_data_bergs_coeff1.pkl'

with open(OLD_PKL, 'rb') as f:
    old_results = pickle.load(f)
with open(NEW_PKL, 'rb') as f:
    new_results = pickle.load(f)

old = old_results[200]
new = new_results[200]

print("="*80)
print("CHECKING INDEX 0 FOR NaNs vs ZEROS")
print("="*80)

# Check Mturbw at index 0
old_idx0 = old.Mturbw.values[:, 0, 0]
new_idx0 = new.Mturbw.values[:, 0, 0]

print("\nMturbw at index 0:")
print(f"  OLD shape: {old_idx0.shape}")
print(f"  NEW shape: {new_idx0.shape}")

# Count zeros and NaNs
old_zeros = np.sum(old_idx0 == 0)
old_nans = np.sum(np.isnan(old_idx0))
old_nonzero = np.sum(old_idx0 != 0)

new_zeros = np.sum(new_idx0 == 0)
new_nans = np.sum(np.isnan(new_idx0))
new_nonzero = np.sum(new_idx0 != 0)

print(f"\n  OLD: {old_zeros} zeros, {old_nans} NaNs, {old_nonzero} non-zeros")
print(f"  NEW: {new_zeros} zeros, {new_nans} NaNs, {new_nonzero} non-zeros")

# Show actual values if there are non-zeros
if old_nonzero > 0:
    print(f"\n  OLD non-zero values:")
    non_zero_idx = np.where(old_idx0 != 0)[0]
    for idx in non_zero_idx[:10]:  # Show first 10
        print(f"    Layer {idx}: {old_idx0[idx]:.6e}")

if new_nonzero > 0:
    print(f"\n  NEW non-zero values:")
    non_zero_idx = np.where(new_idx0 != 0)[0]
    for idx in non_zero_idx[:10]:  # Show first 10
        print(f"    Layer {idx}: {new_idx0[idx]:.6e}")

# Check mean calculation with NaNs
print("\n" + "="*80)
print("MEAN CALCULATION COMPARISON")
print("="*80)

# np.mean vs np.nanmean
print("\nFor Mturbw:")
print(f"  OLD np.mean():    {np.mean(old.Mturbw.values):.6e}")
print(f"  OLD np.nanmean(): {np.nanmean(old.Mturbw.values):.6e}")
print(f"  NEW np.mean():    {np.mean(new.Mturbw.values):.6e}")
print(f"  NEW np.nanmean(): {np.nanmean(new.Mturbw.values):.6e}")

# xarray mean (ignores NaNs by default)
print(f"\n  OLD xarray mean: {float(old.Mturbw.mean().values):.6e}")
print(f"  NEW xarray mean: {float(new.Mturbw.mean().values):.6e}")

# Calculate what the mean SHOULD be if index 0 has all zeros
print("\n" + "="*80)
print("EXPECTED MEAN IF INDEX 0 IS ALL ZEROS")
print("="*80)

# Mean of indices 1-29 only
old_mean_excl0 = np.nanmean(old.Mturbw.values[:, :, 1:])
new_mean_excl0 = np.nanmean(new.Mturbw.values[:, :, 1:])

print(f"\nMean of indices 1-29:")
print(f"  OLD: {old_mean_excl0:.6e}")
print(f"  NEW: {new_mean_excl0:.6e}")

# If index 0 is zeros, mean should be 29/30 of the mean without index 0
expected_old_with0 = old_mean_excl0 * 29 / 30
expected_new_with0 = new_mean_excl0 * 29 / 30

print(f"\nExpected mean if index 0 is zeros (29/30 factor):")
print(f"  OLD: {expected_old_with0:.6e}")
print(f"  NEW: {expected_new_with0:.6e}")

actual_old = float(old.Mturbw.mean().values)
actual_new = float(new.Mturbw.mean().values)

print(f"\nActual mean:")
print(f"  OLD: {actual_old:.6e}")
print(f"  NEW: {actual_new:.6e}")

print(f"\nDoes expected match actual?")
print(f"  OLD: {np.isclose(expected_old_with0, actual_old, rtol=0.001)}")
print(f"  NEW: {np.isclose(expected_new_with0, actual_new, rtol=0.001)}")

# Calculate the implied percentage of zeros
print("\n" + "="*80)
print("REVERSE CALCULATION")
print("="*80)

# If mean = x * (n_nonzero / n_total)
# Then n_nonzero / n_total = mean / mean_excl0
ratio_old = actual_old / old_mean_excl0
ratio_new = actual_new / new_mean_excl0

print(f"\nImplied fraction of non-zero values:")
print(f"  OLD: {ratio_old:.4f} ({ratio_old * 30:.1f} out of 30 timesteps)")
print(f"  NEW: {ratio_new:.4f} ({ratio_new * 30:.1f} out of 30 timesteps)")

print(f"\nImplied number of zero timesteps:")
print(f"  OLD: {30 - ratio_old * 30:.1f}")
print(f"  NEW: {30 - ratio_new * 30:.1f}")
