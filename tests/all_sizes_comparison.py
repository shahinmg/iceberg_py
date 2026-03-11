#!/usr/bin/env python3
"""
All Size Classes Comparison

Check if the 1-2% difference is consistent across all iceberg sizes.
"""

import numpy as np
import pickle

print("="*80)
print("ALL SIZE CLASSES COMPARISON")
print("="*80)

# Load results
OLD_PKL = '/home/laserglaciers/icebergPy/data/iceberg_classes_output_bug_fix_ctd/helheim/avg/2016-04-24_urel0.07_ctd_data_bergs_coeff1.pkl'
NEW_PKL = '/home/laserglaciers/icebergPy/data/test_class_outputs/iceberg_classes_output/helheim/avg/2016-04-24_urel007_ctd_data_bergs_coeff1.pkl'

with open(OLD_PKL, 'rb') as f:
    old_results = pickle.load(f)

with open(NEW_PKL, 'rb') as f:
    new_results = pickle.load(f)

# Get all size classes
size_classes = sorted(old_results.keys())

print(f"\nFound {len(size_classes)} size classes: {size_classes[:5]} ... {size_classes[-5:]}")
print("\n" + "-"*80)

# Compare each size class
results = []

print(f"{'Size':<8} {'Keel_t0':<12} {'Keel_t2':<12} {'Keel_Diff':<12} {'Mtotal_OLD':<15} {'Mtotal_NEW':<15} {'Mtotal_%':<10} {'Mturbw_%':<10} {'Mfreew_%':<10}")
print("-"*120)

for length in size_classes:
    old = old_results[length]
    new = new_results[length]
    
    # Initial keel
    keel_0_old = float(old.KEEL.isel(time=0).values[0])
    keel_0_new = float(new.KEEL.isel(time=0).values[0])
    keel_0_diff = keel_0_new - keel_0_old
    
    # Keel at day 2
    try:
        keel_2_old = float(old.KEEL.sel(time=86400*2).values[0])
        keel_2_new = float(new.KEEL.sel(time=86400*2).values[0])
        keel_2_diff = keel_2_new - keel_2_old
    except:
        keel_2_old = keel_2_new = keel_2_diff = np.nan
    
    # Total melt (mean over time)
    Mtotal_old = float(old.Mtotal.mean().values)
    Mtotal_new = float(new.Mtotal.mean().values)
    Mtotal_pct = 100 * (Mtotal_new - Mtotal_old) / Mtotal_old if Mtotal_old != 0 else 0
    
    # Forced water melt
    Mturbw_old = float(old.Mturbw.mean().values)
    Mturbw_new = float(new.Mturbw.mean().values)
    Mturbw_pct = 100 * (Mturbw_new - Mturbw_old) / Mturbw_old if Mturbw_old != 0 else 0
    
    # Buoyant water melt
    Mfreew_old = float(old.Mfreew.mean().values)
    Mfreew_new = float(new.Mfreew.mean().values)
    Mfreew_pct = 100 * (Mfreew_new - Mfreew_old) / Mfreew_old if Mfreew_old != 0 else 0
    
    results.append({
        'length': length,
        'keel_0_diff': keel_0_diff,
        'keel_2_diff': keel_2_diff,
        'Mtotal_pct': Mtotal_pct,
        'Mturbw_pct': Mturbw_pct,
        'Mfreew_pct': Mfreew_pct
    })
    
    print(f"{length:<8} {keel_0_diff:<12.6f} {keel_2_diff:<12.6f} {keel_2_diff:<12.6f} {Mtotal_old:<15.6e} {Mtotal_new:<15.6e} {Mtotal_pct:<10.3f} {Mturbw_pct:<10.3f} {Mfreew_pct:<10.3f}")

# Statistical summary
print("\n" + "="*80)
print("STATISTICAL SUMMARY ACROSS ALL SIZE CLASSES")
print("="*80)

keel_0_diffs = [r['keel_0_diff'] for r in results]
keel_2_diffs = [r['keel_2_diff'] for r in results if not np.isnan(r['keel_2_diff'])]
Mtotal_pcts = [r['Mtotal_pct'] for r in results]
Mturbw_pcts = [r['Mturbw_pct'] for r in results]
Mfreew_pcts = [r['Mfreew_pct'] for r in results]

print(f"\nInitial keel (t=0) differences:")
print(f"  Mean: {np.mean(keel_0_diffs):.6f} m")
print(f"  Std:  {np.std(keel_0_diffs):.6f} m")
print(f"  Max:  {np.max(np.abs(keel_0_diffs)):.6f} m")

print(f"\nKeel at day 2 differences:")
print(f"  Mean: {np.mean(keel_2_diffs):.6f} m")
print(f"  Std:  {np.std(keel_2_diffs):.6f} m")
print(f"  Max:  {np.max(np.abs(keel_2_diffs)):.6f} m")

print(f"\nTotal melt (Mtotal) % differences:")
print(f"  Mean: {np.mean(Mtotal_pcts):.3f}%")
print(f"  Std:  {np.std(Mtotal_pcts):.3f}%")
print(f"  Range: [{np.min(Mtotal_pcts):.3f}%, {np.max(Mtotal_pcts):.3f}%]")

print(f"\nForced water (Mturbw) % differences:")
print(f"  Mean: {np.mean(Mturbw_pcts):.3f}%")
print(f"  Std:  {np.std(Mturbw_pcts):.3f}%")
print(f"  Range: [{np.min(Mturbw_pcts):.3f}%, {np.max(Mturbw_pcts):.3f}%]")

print(f"\nBuoyant water (Mfreew) % differences:")
print(f"  Mean: {np.mean(Mfreew_pcts):.3f}%")
print(f"  Std:  {np.std(Mfreew_pcts):.3f}%")
print(f"  Range: [{np.min(Mfreew_pcts):.3f}%, {np.max(Mfreew_pcts):.3f}%]")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

# Check if patterns are consistent
keel_0_matches = np.max(np.abs(keel_0_diffs)) < 1e-6
keel_2_close = np.max(np.abs(keel_2_diffs)) < 0.01

Mtotal_consistent = np.std(Mtotal_pcts) < 0.5
Mturbw_consistent = np.std(Mturbw_pcts) < 0.5

if keel_0_matches:
    print("\n✓ Initial keel matches perfectly across ALL size classes")
else:
    print("\n❌ Initial keel differs - initialization bug!")

if keel_2_close:
    print("✓ Keel at day 2 is very close (<1cm) across ALL size classes")
else:
    print(f"⚠ Keel at day 2 varies significantly (max: {np.max(np.abs(keel_2_diffs)):.3f}m)")

if Mtotal_consistent:
    print(f"✓ Total melt differences are CONSISTENT across sizes (std: {np.std(Mtotal_pcts):.3f}%)")
    print("  → Suggests a systematic difference, not size-dependent bug")
else:
    print(f"⚠ Total melt differences VARY across sizes (std: {np.std(Mtotal_pcts):.3f}%)")
    print("  → Suggests size-dependent bug in calculations")

if Mturbw_consistent:
    print(f"✓ Forced water differences are CONSISTENT (std: {np.std(Mturbw_pcts):.3f}%)")
else:
    print(f"⚠ Forced water differences VARY (std: {np.std(Mturbw_pcts):.3f}%)")

# Check if larger icebergs have larger differences
from scipy.stats import pearsonr
sizes = [r['length'] for r in results]
corr_Mtotal, p_Mtotal = pearsonr(sizes, Mtotal_pcts)
corr_Mturbw, p_Mturbw = pearsonr(sizes, Mturbw_pcts)

print(f"\nCorrelation with iceberg size:")
print(f"  Mtotal vs size: r={corr_Mtotal:.3f} (p={p_Mtotal:.3e})")
print(f"  Mturbw vs size: r={corr_Mturbw:.3f} (p={p_Mturbw:.3e})")

if abs(corr_Mtotal) > 0.5:
    print("  → Strong correlation! Differences are SIZE-DEPENDENT")
else:
    print("  → Weak correlation. Differences are size-independent")

print("\n" + "="*80)
