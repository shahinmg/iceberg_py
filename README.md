## icebergPy

icebergPy is a Python-based iceberg melt rate model based on the [Moon et al., 2018](https://www.nature.com/articles/s41561-017-0018-z) model. Currently this repo is set up to run the model and re-create the main figures of my manuscript.

## How to use the model

Within `src/iceberg_model/` there is `melt_functions.py`. This file contains all the model physics. The other files are example scripts to run the model with the naming convention `{glacier_name}_example.py`.

## Recreate Figures 

In the `figure_pys` dir there are three scripts to recreate figures 4-6. These figures are "data figures" and not schematics or site maps. Please download the accopanying Zenodo archive (link coming soon) to download the data for this repo to recreate the figures.

## Tables

Notebooks to recreate tables are in the `notebooks` directory. For the iceberg volume proportions, see the `aw_vs_total_vol_table_gen.ipynb` in `notebooks/volume_table`. For the Qib tables (avg, min, and max) see the notebooks in `notebooks\qib_tables`.

## Image segmentation example

In the `notebooks` directory, there is a `.ipynb` file that contains an example of how to use [SAM](https://github.com/facebookresearch/segment-anything) (version 1) to segment icebergs. 


## ⚠️ Warning ⚠️

I plan to make icebergPy a package, so this structure will change. The more maintainable package might live elsewhere. Importantly, the code structure in `melt_functions.py` will also change.
