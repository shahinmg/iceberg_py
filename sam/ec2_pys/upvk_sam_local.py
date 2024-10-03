#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 13:49:06 2024

@author: laserglaciers
"""

import dask.distributed
import dask.utils
import numpy as np
import planetary_computer as pc
import xarray as xr
from IPython.display import display
from pystac_client import Client
import geopandas as gpd
import requests
from pystac.extensions.eo import EOExtension as eo
from odc.stac import configure_rio, stac_load
from pathlib import Path
import pickle
from contextlib import contextmanager  
import rasterio
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.features import geometry_mask, geometry_window
from rasterio.plot import show, reshape_as_raster, reshape_as_image

import cv2
from segment_anything import SamPredictor, SamAutomaticMaskGenerator,sam_model_registry

import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
import pandas as pd
#%%

if __name__ == '__main__':

    client = dask.distributed.Client()
    configure_rio(cloud_defaults=True, client=client)
    
    date2use = pd.to_datetime('1999', utc=True)
    to_use = []
    with open('/media/laserglaciers/upernavik/iceberg_py/sam/item_pkls/upvk_image_item_list.pkl', 'rb') as f:
        helheim_stack_list = pickle.load(f)
    
        for item in helheim_stack_list:
            datetime = pd.to_datetime(item.datetime, utc=True)
            if datetime > date2use:
                to_use.append(item)
    
    
    grid_path = '/media/laserglaciers/upernavik/iceberg_py/geoms/upernavik/upvk_3x8_grid_utm22N_v2.gpkg'
    grid = gpd.read_file(grid_path)
    
    
    resolution = 10
    SHRINK = 1
    if client.cluster.workers[0].memory_manager.memory_limit < dask.utils.parse_bytes("4G"):
        SHRINK = 8  # running on Binder with 2Gb RAM
    
    if SHRINK > 1:
        resolution = resolution * SHRINK
    
    
    image_stac = stac_load(
        to_use,
        bands=["red", "green", "blue"],
        resolution=resolution,
        chunks={"x": 2048, "y": 2048},
        patch_url=pc.sign,
        # force dtype and nodata
        dtype="uint16",
        nodata=0,
        geopolygon=grid
    )
    
    affine = image_stac.red.isel(time=0).odc.affine
    height, width = image_stac.red.isel(time=0).shape
    #%%
    #https://rasterio.groups.io/g/main/topic/memoryfile_workflow_should/32634761
    @contextmanager
    def mem_raster(data, **profile):
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dataset_writer:
                dataset_writer.write(data)
     
            with memfile.open() as dataset_reader:
                yield dataset_reader
    
    def cv2Norm(band):
        img_u8 = cv2.normalize(band, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        return img_u8
    
    
    profile_rgb = {'driver':'GTiff', 'count':3, 
                          'transform':affine, 'crs':image_stac.odc.crs, 
                          'width':width, 'height': height, 
                          'dtype':np.uint8
    }
    
    #%%
    
    def sam_segment(image_chunk):
        FEATURE_OF_INTEREST = 'icebergs' #icebergs,crevasse,terminus,supraglacial_lakes,planet, sentinel-2, sentinel-1, timelapse
        MODEL_TYPE = 'vit_h'
        MODEL_WEIGHTS = 'sam_vit_h_4b8939.pth' # sam_vit_b_01ec64.pth,sam_vit_h_4b8939.pth,sam_vit_l_0b3195.pth
        OUTPUT_FOLDER = 'predict_no_prompt' # predict_with_prompt, predict_no_prompt
        
        # OUTPUT_PATH = os.path.join(BASE_PATH,'%s'%(OUTPUT_FOLDER))
    
        sam = sam_model_registry["%s"%(MODEL_TYPE)](checkpoint="/opt/atlas/segment-anything/models/%s"%(MODEL_WEIGHTS))
        predictor = SamPredictor(sam)
        predictor.set_image(image_chunk)
    
    
        mask_generator = SamAutomaticMaskGenerator(sam)
        
        masks = mask_generator.generate(image_chunk)
        # out_path = '/media/laserglaciers/upernavik/iceberg_py/outfiles/helheim/sam_output/2023_07_27.png'
        if FEATURE_OF_INTEREST == 'terminus':
            for num in range(len(masks)):
                im = Image.fromarray(masks[num]['segmentation'])
                # im.save(f'{OUTPUT_PATH}/{fileName.split(".")[0]}{num}_predict.png')
        
        else:
            binary_pred_zeros = np.zeros_like(masks[1]['segmentation'])
            for num in range(len(masks)):
                # 25% or higher number of pixels are True, that means it is a potential representation of background
                # and not of icebergs
                if np.count_nonzero(masks[num]['segmentation'])>(0.25*(masks[num]['segmentation']).size):
                    continue
                else:               
                    binary_pred_zeros[masks[num]['segmentation']==1]=1
            im = Image.fromarray(binary_pred_zeros)
            # # im.save(f'{OUTPUT_PATH}/{fileName.split(".")[0]}_predict_{MODEL_TYPE}.png')
            # im.save(f'{out_path}')
    
        
        return im
    
    #%%
    
    for time in image_stac.time:
        
        rgb = np.dstack((image_stac.red.sel(time=time).values,
                         image_stac.green.sel(time=time).values,
                         image_stac.blue.sel(time=time).values))
        rgb_norm = cv2Norm(rgb)
        rgb_norm_raster = reshape_as_raster(rgb_norm)
        # image_chunk_dict = {}
        date = str(image_stac.red.sel(time=time).time.dt.date.data)
        print(date)
        print(f'{rgb_norm_raster.dtype}')
        with mem_raster(rgb_norm_raster, **profile_rgb) as ds:
            for i, geom in enumerate(grid.geometry): 
                print(f'geom number: {i}')
                window = geometry_window(ds, [geom])
                window_read = ds.read([1,2,3], window=window)
                
                win_transform = ds.window_transform(window)
                window_image = reshape_as_image(window_read)
                # show(w,transform=win_transform)
                # image_chunk_dict[win_transform] = w
                im = sam_segment(window_image)
                
                nim = np.array(im) #numpy image
                
                profile_out = {'driver':'GTiff', 'count':1, 
                              'transform':win_transform, 'crs':32624, 
                              'width':nim.shape[1], 'height': nim.shape[0], 
                              'dtype':np.float64
                                }
                
                out_name = f'{date}_upvk_{i}.tif'
                out_path = f"/opt/atlas/iceberg_segment/upvk/{date}"
                Path(out_path).mkdir(parents=True, exist_ok=True)
                
                with rasterio.open(f'{out_path}/{out_name}', mode='w', **profile_out) as dst:
                    dst.write(nim,1)



