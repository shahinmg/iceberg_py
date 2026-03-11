#!/usr/bin/env python3
"""
Check which specific timesteps differ between OLD and NEW
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
print("TIMESTEP-BY-TIMESTEP COMPARISON")
print("="*80)

print("\nMturbw mean at each timestep (averaged over depth and iceberg dimensions):")
print(f"{'Index':<8} {'Time(days)':<12} {'OLD':<15} {'NEW':<15} {'Diff':<15} {'%Diff':<10}")
print("-"*80)

for j in range(30):
    time_days = (j + 1) if j > 0 else 0  # Index 0 is initial, others are day 1, 2, etc.
    
    # Average over depth and iceberg dimensions
    old_mean = np.nanmean(old.Mturbw.values[:, :, j])
    new_mean = np.nanmean(new.Mturbw.values[:, :, j])
    
    diff = new_mean - old_mean
    pct = 100 * diff / old_mean if old_mean != 0 else 0
    
    marker = ""
    if abs(pct) < 0.01:
        marker = " ✓ MATCH"
    elif abs(pct) > 5:
        marker = " ⚠️ BIG DIFF"
    
    print(f"{j:<8} {time_days:<12} {old_mean:<15.6e} {new_mean:<15.6e} {diff:<15.6e} {pct:<10.2f}{marker}")

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

# Calculate statistics for indices 1-29 only
diffs = []
pcts = []

for j in range(1, 30):
    old_mean = np.nanmean(old.Mturbw.values[:, :, j])
    new_mean = np.nanmean(new.Mturbw.values[:, :, j])
    
    if old_mean != 0:
        diff = new_mean - old_mean
        pct = 100 * diff / old_mean
        diffs.append(diff)
        pcts.append(pct)

print(f"\nFor indices 1-29:")
print(f"  Mean % difference: {np.mean(pcts):.2f}%")
print(f"  Std % difference: {np.std(pcts):.2f}%")
print(f"  Min % difference: {np.min(pcts):.2f}%")
print(f"  Max % difference: {np.max(pcts):.2f}%")

# Check if the difference is constant or growing
print(f"\n  First 5 timesteps avg: {np.mean(pcts[:5]):.2f}%")
print(f"  Last 5 timesteps avg: {np.mean(pcts[-5:]):.2f}%")

if abs(np.mean(pcts[-5:]) - np.mean(pcts[:5])) > 0.5:
    print("  → Difference is GROWING over time (accumulation error)")
else:
    print("  → Difference is CONSTANT (systematic offset)")

# Check geometry
print("\n" + "="*80)
print("GEOMETRY COMPARISON")
print("="*80)

print("\nKeel depth at each timestep:")
print(f"{'Index':<8} {'OLD':<12} {'NEW':<12} {'Diff(m)':<12}")
print("-"*50)

for j in [0, 1, 2, 5, 10, 20, 29]:
    old_keel = old.KEEL.values[0, j]
    new_keel = new.KEEL.values[0, j]
    diff = new_keel - old_keel
    
    print(f"{j:<8} {old_keel:<12.2f} {new_keel:<12.2f} {diff:<12.4f}")

print("\n" + "="*80)
print("CHECKING FOR CUMULATIVE DRIFT")
print("="*80)

# Calculate cumulative melt
old_cumsum = np.nancumsum(np.nanmean(old.Mturbw.values, axis=(0, 1)))
new_cumsum = np.nancumsum(np.nanmean(new.Mturbw.values, axis=(0, 1)))

print("\nCumulative melt at selected timesteps:")
print(f"{'Index':<8} {'OLD':<15} {'NEW':<15} {'Diff':<15} {'%Diff':<10}")
print("-"*70)

for j in [0, 1, 5, 10, 20, 29]:
    diff = new_cumsum[j] - old_cumsum[j]
    pct = 100 * diff / old_cumsum[j] if old_cumsum[j] != 0 else 0
    
    print(f"{j:<8} {old_cumsum[j]:<15.6e} {new_cumsum[j]:<15.6e} {diff:<15.6e} {pct:<10.2f}")
