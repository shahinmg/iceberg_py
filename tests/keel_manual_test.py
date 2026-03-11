import numpy as np
import pickle

OLD_PKL = '/home/laserglaciers/icebergPy/data/iceberg_classes_output_bug_fix/helheim/avg/2016-04-24_urel0.07_ctd_data_bergs_coeff1.pkl'
NEW_PKL = '/home/laserglaciers/icebergPy/data/test_class_outputs/iceberg_classes_output/helheim/avg/2016-04-24_urel007_ctd_data_bergs_coeff1.pkl'

with open(OLD_PKL, 'rb') as f:
    old = pickle.load(f)[200]
with open(NEW_PKL, 'rb') as f:
    new = pickle.load(f)[200]

# Check layers 24-26 at timesteps 1-5
for t in [1, 2, 3]:
    print(f"\nTimestep {t}:")
    for z in [24, 25, 26]:
        old_val = old.Mturbw.isel(time=t, Z=z).values[0]
        new_val = new.Mturbw.isel(time=t, Z=z).values[0]
        print(f"  Layer {z}: OLD={old_val:.6e}, NEW={new_val:.6e}")
        
#%%

import pickle
import numpy as np
from scipy.interpolate import interp1d

OLD_PKL = '/home/laserglaciers/icebergPy/data/iceberg_classes_output_bug_fix/helheim/avg/2016-04-24_urel0.07_ctd_data_bergs_coeff1.pkl'
NEW_PKL = '/home/laserglaciers/icebergPy/data/test_class_outputs/iceberg_classes_output/helheim/avg/2016-04-24_urel007_ctd_data_bergs_coeff1.pkl'

with open(OLD_PKL, 'rb') as f:
    old = pickle.load(f)[200]
with open(NEW_PKL, 'rb') as f:
    new = pickle.load(f)[200]

# Get the velocity at keel layer at t=1
k = 25  # keel layer index
Urel_old = old.Urel[k, 0, 1]
Urel_new = new.Urel[k, 0, 1]

print(f"Velocity at keel layer (k={k}, t=1):")
print(f"  OLD: {Urel_old}")
print(f"  NEW: {Urel_new}")
print(f"  Match: {Urel_old == Urel_new}")