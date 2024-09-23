'''
This code generates masks for images of icebergs, terminus, crevasses, and supraglacial lakes
in optical and SAR imagery using Segment Anything Model (SAM).

It is a NON-prompt based generation of masks.

The code adds all the instances detected to a new 2D array of same shape as the original.
The 2D array will only consist of 0's and will add values at specific indices from
every instance detection.

The code will also remove instances larger than 25% of the original image as that
suggests a background detection which we are not interested in for icebergs and lakes
segmentation.

@Siddharth Shankar
'''
# from /media/laserglaciers/upernavik/segment-anything/pys/segment_SAR_no_prompt_v3_ortho.py

from segment_anything import SamPredictor, SamAutomaticMaskGenerator,sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import torch
import rasterio
from rasterio.windows import Window
# Define the main working directories, models, paths, and fileNames
# ********************************************************************************
FEATURE_OF_INTEREST = 'icebergs' #icebergs,crevasse,terminus,supraglacial_lakes,planet, sentinel-2, sentinel-1, timelapse
MODEL_TYPE = 'vit_h'
MODEL_WEIGHTS = 'sam_vit_h_4b8939.pth' # sam_vit_b_01ec64.pth,sam_vit_h_4b8939.pth,sam_vit_l_0b3195.pth
OUTPUT_FOLDER = 'predict_no_prompt' # predict_with_prompt, predict_no_prompt

BASE_PATH = f'/media/laserglaciers/upernavik/Helheim_ortho_photos/ortho_images/jpgs/'
# OUTPUT_PATH = os.path.join(BASE_PATH,'%s'%(OUTPUT_FOLDER))
OUTPUT_PATH = f'{BASE_PATH}/OUTPUT_FOLDER'
fileName = '7P14_M42_A4545V_239_ortho.jpg'

# *********************************************************************************


def cv2Norm(band):
    img_u8 = cv2.normalize(band, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    
    return img_u8


 
# fileName = 'S1B_IW_GRDH_1SDH_20190502T091100_20190502T091125_016063_01E364_ADEC_subset.png'    
# sam = sam_model_registry["vit_h"](checkpoint="C:/segment-anything/sam_vit_h_4b8939.pth")
sam = sam_model_registry["%s"%(MODEL_TYPE)](checkpoint="/media/laserglaciers/upernavik/segment-anything/models/%s"%(MODEL_WEIGHTS))

device = "cuda"
sam.to(device=device)
image = cv2.imread(f'{BASE_PATH}/{fileName}')
# image = cv2.imread(os.path.join(BASE_PATH,r'%s'%fileName))
# image_src = rasterio.open(f'{BASE_PATH}/{fileName}')
# b1 = cv2Norm(image_src.read(1))
# b2 = cv2Norm(image_src.read(2))
# b3 = cv2Norm(image_src.read(3))

# image = np.stack([b1,b2,b3],axis=2)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor = SamPredictor(sam)
predictor.set_image(image)


# # Enable if need to auto-generate the masks from default settings
# #-----------------------------------------------------------------

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)

# masks_list = rasterio_sliding_window(os.path.join(BASE_PATH,r'%s'%fileName))


# Create a zeroes 2D array of same shape as image and transfer the mask 
# instances to the the new 2D array. Filter out any instance larger than 25%
# of the image size, for background instance removal as we do not need it for
# the final segmentation results.

if FEATURE_OF_INTEREST == 'terminus':
    for num in range(len(masks)):
        im = Image.fromarray(masks[num]['segmentation'])
        im.save(f'{OUTPUT_PATH}/{fileName.split(".")[0]}{num}_predict.png')

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
    im.save(f'{OUTPUT_PATH}/{fileName.split(".")[0]}_predict_{MODEL_TYPE}.png')



# image_src.close()

























'''
# Plot the image

for num in range(len(masks)):
    im = Image.fromarray(masks[num]['segmentation'])
    # im.save(os.path.join(OUTPUT_PATH,'terminus_test2_%s.png'%(num)))
'''


'''
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 
# torch.cuda.empty_cache()


binary_masks = []
for num in range(len(masks)):
    binary_masks.append(masks[num]['segmentation'])

final_binary_mask = sum(binary_masks)

# Enable to get binary classification
# final_binary_mask[final_binary_mask>0]=255
final_binary_image = Image.fromarray(final_binary_mask)

# output_path = r'C:\segment-anything\images\icebergs\predict'
# final_binary_image.save(os.path.join(output_path,'%s_predict_vit_h_defaultsetting.png'%(fileName.split('.')[0])))
final_binary_image.save(os.path.join(OUTPUT_PATH,'%s_predict_%s_%s.png'%(fileName.split('.')[0],MODEL_TYPE,OUTPUT_FOLDER)))

# cv2.imwrite('test.jpg',final_binary_mask)
'''