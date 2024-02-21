# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:42:08 2024

@author: 14055
"""
import os 
os.chdir(r'C:\Users\14055\Desktop\sam_experiments')

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    
image = cv2.imread(r'C:\Users\14055\Downloads\IMG_20240221_113827.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('off')
plt.show()


sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)

masks = mask_generator.generate(highlighted_image)

masks[0]['segmentation']

print(len(masks))
print(masks[0].keys())

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 


area_sum = 0
for i in range(len(masks)):
    area_sum =area_sum+ masks[i]['area']  # Corrected from =+ to +=




masks[4]['area'] 

'''
mask generation params
'''
'''
mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks2)
plt.axis('off')
plt.show() 

'''
#------------------------------------------------
#------------------------------------------------
#------------------------------------------------
#-----------find the rectangularity of all the masks and print them, maybe a histogram 
#----------for the ones with highest rectangularity - return their indices---------
#------------------merge them with the largest mask around 
#--------also suggest a


def calculate_rectangularity(mask_object):
    """
    Calculate the rectangularity of an object represented by a mask object.
    
    Args:
    - mask_object (dict): A dictionary containing the segmentation mask ('segmentation' key)
                          and other properties of the object.
    
    Returns:
    - rectangularity (float): A measure of how close the object is to a perfect rectangle.
    """
    # Use the pre-calculated area of the object
    object_area = mask_object['area']
    # Extract the bounding box
    bbox = mask_object['bbox']  # bbox format is assumed to be [x, y, width, height]
    # Calculate the area of the bounding box
    bbox_area = bbox[2] * bbox[3]
    # Calculate rectangularity as the ratio of object area to bounding box area
    rectangularity = object_area / bbox_area    
    return rectangularity

rect_area_list=[]
for i in range(len(masks)):
    mask_object = masks[i]  # Assuming 'masks' is your list of mask objects
    rectangularity = calculate_rectangularity(mask_object)
    rect_area_list.append([rectangularity, mask_object['area']])

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks[2:3])
plt.axis('off')
plt.show() 



def merge_overlapping_masks(masks):
    """
    Merge overlapping masks based on their bounding boxes.

    Args:
    - masks (list of dicts): A list of mask objects, each containing a 'segmentation' key with a numpy array,
                             and a 'bbox' key with the bounding box [x, y, width, height].

    Returns:
    - merged_masks (list of dicts): A list of merged mask objects.
    """
    merged_masks = []
    while masks:
        current_mask = masks.pop(0)
        current_bbox = current_mask['bbox']
        current_segmentation = current_mask['segmentation']
        overlaps = []

        for i, other_mask in enumerate(masks):
            other_bbox = other_mask['bbox']
            # Check if current_bbox overlaps with other_bbox
            if (current_bbox[0] < other_bbox[0] + other_bbox[2] and
                current_bbox[0] + current_bbox[2] > other_bbox[0] and
                current_bbox[1] < other_bbox[1] + other_bbox[3] and
                current_bbox[1] + current_bbox[3] > other_bbox[1]):
                overlaps.append(i)

        # If there are overlaps, merge them
        for i in sorted(overlaps, reverse=True):
            # Merge segmentation arrays by logical OR operation
            current_segmentation = np.logical_or(current_segmentation, masks[i]['segmentation'])
            # Remove the merged mask from the list
            del masks[i]

        # Recalculate area and bounding box for the merged mask
        new_area = np.sum(current_segmentation)
        rows = np.any(current_segmentation, axis=1)
        cols = np.any(current_segmentation, axis=0)
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]
        new_bbox = [col_min, row_min, col_max - col_min + 1, row_max - row_min + 1]

        # Update the current mask object with the merged segmentation, area, and bbox
        current_mask['segmentation'] = current_segmentation
        current_mask['area'] = new_area
        current_mask['bbox'] = new_bbox

        # Add the updated mask to the merged_masks list
        merged_masks.append(current_mask)

    return merged_masks

merged_masks = merge_overlapping_masks(masks)


plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(merged_masks)
plt.axis('off')
plt.show() 

area_sum = 0
for i in range(len(merged_masks)):
    area_sum =area_sum+ merged_masks[i]['area']  # Corrected from =+ to +=


#--------------create a boolean mask  

def create_binary_mask_image(masks, image_shape):
    """
    Create a binary image from a list of masks and the shape of the original image.
    
    Args:
    - masks (list of dicts): A list of mask objects, each containing a 'segmentation' key
                             with a numpy array indicating masked areas.
    - image_shape (tuple): The shape of the original image (height, width).
    
    Returns:
    - binary_image (np.array): A binary image where pixels in any mask are 1 and all other pixels are 0.
    """
    # Initialize a binary image with zeros (shape of the original image)
    binary_image = np.zeros(image_shape, dtype=np.uint8)
    
    # Iterate through each mask and mark the masked areas in the binary image
    for mask in masks:
        segmentation = mask['segmentation']
        # Use logical OR to combine the current mask with the binary image
        binary_image = np.logical_or(binary_image, segmentation)
    
    return binary_image.astype(np.uint8)

# Example usage:
# Assuming `masks` is your list of mask objects and `input_image_shape` is the shape of your input image


input_image_shape = (image.shape[0], image.shape[1])  # Replace with your actual image shape
binary_mask_image = create_binary_mask_image(masks, input_image_shape)

# Note: If you need the shape of an actual image, you can get it using image.shape if you have the image loaded with a library like OpenCV or PIL.

plt.imshow(binary_mask_image)




#--- display the image only with the green component highlighted - and then
#---- retain only the masks with high green componenets



import cv2


def highlight_green_parts(image, threshold=150, highlight_color=(0, 255, 0)):
    """
    Highlight parts of an image with high green color values.

    Args:
    - image_path (str): Path to the input RGB image.
    - threshold (int): Threshold value to consider a pixel as 'high green'. Default is 150.
    - highlight_color (tuple): Color used for highlighting. Default is bright green.

    Returns:
    - highlighted_image (numpy.ndarray): The image with high green parts highlighted.
    """


    # Split the image into its color channels
    blue_channel, green_channel, red_channel = cv2.split(image)

    # Find pixels where the green channel is significantly higher than both
    # the red and blue channels
    mask = (green_channel > threshold) & (green_channel > red_channel) & (green_channel > blue_channel)

    # Create an all-zero image for highlighting
    highlight = np.zeros_like(image)
    highlight[mask] = highlight_color  # Apply the highlight color to the mask

    # Combine the highlight with the original image
    highlighted_image = cv2.addWeighted(image, 1, highlight, 0.5, 0)

    return highlighted_image


def highlight_green_parts(image, threshold=150, highlight_color=(0, 255, 0)):
    """
    Highlight parts of an image with high green color values.

    Args:
    - image_path (str): Path to the input RGB image.
    - threshold (int): Threshold value to consider a pixel as 'high green'. Default is 150.
    - highlight_color (tuple): Color used for highlighting. Default is bright green.

    Returns:
    - highlighted_image (numpy.ndarray): The image with high green parts highlighted.
    """


    # Split the image into its color channels
    blue_channel, green_channel, red_channel = cv2.split(image)

    # Find pixels where the green channel is significantly higher than both
    # the red and blue channels
    mask = (green_channel > threshold) & (green_channel > red_channel) & (green_channel > blue_channel)

    # Create an all-zero image for highlighting
    highlight = np.zeros_like(image)
    highlight[mask] = highlight_color  # Apply the highlight color to the mask

    # Combine the highlight with the original image
    highlighted_image = cv2.addWeighted(image, 1, highlight, 0.5, 0)

    return highlighted_image

highlighted_image = highlight_green_parts(image)

# Display the result
plt.imshow(highlighted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()













































