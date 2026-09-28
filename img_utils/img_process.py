
import pandas as pd
import argparse
import glob
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt
from monai.transforms import LoadImage, Orientation, Spacing
import monai.transforms as transforms
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes
import scipy.ndimage
# Hello! crop_to_nonzero is the function you are looking for. Ignore the rest.
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, crop_to_bbox, bounding_box_to_slice


from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops
import random
from sklearn.cluster import KMeans

def find_well_distributed_points(mask, num_points=10, method='distance'):
    """
    Find well-distributed points within the mask using different methods.
    """
    # Get all points in the mask
    mask_points = np.array(np.where(mask == 1)).T
    
    if len(mask_points)==0:
        return 'No points'
        
    if len(mask_points) < num_points:
        # Handle the small-mask case
        # E.g., just return all the points or pick the existing points
        return [tuple(p) for p in mask_points]

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_points, random_state=42)
    kmeans.fit(mask_points)

    # Get cluster centers and convert to integers
    points = [tuple(map(int, center)) for center in kmeans.cluster_centers_]
    #print(len(points))
    # Verify points are within the mask
    points = [p for p in points if mask[p] == 1]
    #print(len(points))

    # If some points fell outside the mask, find the nearest valid points
    if len(points) < num_points:
        remaining_points = num_points - len(points)
        valid_points = np.array(np.where(mask == 1)).T

        for center in kmeans.cluster_centers_:
            if len(points) >= num_points:
                break

            if mask[tuple(map(int, center))] == 0:
                # Find the nearest valid point
                distances = np.sum((valid_points - center) ** 2, axis=1)
                nearest_idx = np.argmin(distances)
                points.append(tuple(valid_points[nearest_idx]))

    return points

##https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/preprocessing/cropping/cropping.py
def create_nonzero_mask(data):
    """

    :param data:
    :return: the mask is True where the data is nonzero
    """
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = data[0] != 0
    for c in range(1, data.shape[0]):
        nonzero_mask |= data[c] != 0
    return binary_fill_holes(nonzero_mask)


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)
    nonzero_mask = nonzero_mask[slicer][None]
    
    slicer = (slice(None), ) + slicer
    data = data[slicer]
    if seg is not None:
        seg = seg[slicer]
        #seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    else:
        seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label))
    return data, seg, bbox



def get_cropped_data(nd_array, ms_array):
    if ms_array is not None:
        new_array = np.expand_dims(nd_array, axis=0)
        new_array2 = np.expand_dims(ms_array, axis=0)
        cropped_data, updated_seg, bbox = crop_to_nonzero(new_array, new_array2)
        cropped_data = cropped_data[0,:,:,:]
        updated_seg = updated_seg[0,:,:,:]
        return cropped_data, updated_seg
    else:
        new_array = np.expand_dims(nd_array, axis=0)
        cropped_data, updated_seg, bbox = crop_to_nonzero(new_array, None)
        cropped_data = cropped_data[0,:,:,:]
        return cropped_data


fix_z = 256
def preprocess_zaxis(array, array2):  
    current_z = array.shape[-1]
    add_z = fix_z-current_z
    
    if array2 is not None:
        if add_z>0:
            pad_width = ((0, 0), (0, 0), (0, add_z))  # No padding for x and y axes, padding 156 zeros at the end of z axis
            padded_volume = np.pad(array, pad_width, mode='constant', constant_values=0)
            padded_volume2 = np.pad(array2, pad_width, mode='constant', constant_values=0)
            return padded_volume, padded_volume2
        else:
            return array[:,:,:fix_z], array2[:,:,:fix_z]
    else:
        if add_z>0:
            pad_width = ((0, 0), (0, 0), (0, add_z))  # No padding for x and y axes, padding 156 zeros at the end of z axis
            padded_volume = np.pad(array, pad_width, mode='constant', constant_values=0)
            return padded_volume
        else:
            return array[:,:,:fix_z]
        

def pad_to_square(array, array2):
    """
    Pad the first two dimensions of a 3D array to make them equal, ensuring the array has a square base (x, y).

    :param array: A 3D numpy array of shape (x, y, z)
    :return: A 3D numpy array with x and y dimensions padded to be equal
    """
    x, y, z = array.shape
    max_dim = max(x, y)
    
    # Calculate padding for x and y dimensions
    pad_x = (max_dim - x) // 2, (max_dim - x) - ((max_dim - x) // 2)
    pad_y = (max_dim - y) // 2, (max_dim - y) - ((max_dim - y) // 2)
    
    if array2 is not None:
        # Apply padding to the first two dimensions and no padding to the third dimension
        padded_array = np.pad(array, (pad_x, pad_y, (0, 0)), mode='constant', constant_values=0)
        
        padded_array2 = np.pad(array2, (pad_x, pad_y, (0, 0)), mode='constant', constant_values=0)
        
        return padded_array, padded_array2
    else:
        padded_array = np.pad(array, (pad_x, pad_y, (0, 0)), mode='constant', constant_values=0)
        
        return padded_array


# In[11]:

###low resolution
def resize_array(array,array2,  new_shape=(256, 256, fix_z)):
    """
    Resize a 3D numpy array to a new shape using interpolation.

    :param array: A 3D numpy array of any shape.
    :param new_shape: A tuple of three integers, the desired output shape (x, y, z).
    :return: A 3D numpy array resized to new_shape.
    """
    # Calculate the zoom factors for each dimension
    if array2 is not None:
        zoom_factors = [n / o for n, o in zip(new_shape, array.shape)]
        
        # Use scipy.ndimage.zoom to resize the array
        resized_array = scipy.ndimage.zoom(array, zoom_factors, order=3)  # order=3 uses cubic interpolation
        resized_array2 = scipy.ndimage.zoom(array2, zoom_factors, order=0)  # order=3 uses cubic interpolation
        
        return resized_array, resized_array2
    else:
        zoom_factors = [n / o for n, o in zip(new_shape, array.shape)]
        
        # Use scipy.ndimage.zoom to resize the array
        resized_array = scipy.ndimage.zoom(array, zoom_factors, order=3)  # order=3 uses cubic interpolation
        
        return resized_array


def calculate_label_info(msk_data):
    """
    Calculate pixel count and bounding box for each unique label in a NIfTI file.

    Parameters:
        nifti_file: str
            Path to the NIfTI file.

    Returns:
        dict: Dictionary with label info, including pixel count, bounding box, and center of mass.
    """

    image_data = msk_data.copy()
    
    # Find unique labels in the image
    unique_labels = np.unique(image_data)

    label_info = {}
    for label in unique_labels:
        if label == 0:  # Skip background
            continue

        # Get indices of the current label

        mask = (image_data == label).astype(np.uint8)
        points = find_well_distributed_points(mask, num_points=10, method='distance')
        points_ = points[0]

        label_indices = np.argwhere(image_data == label)
        x_min, y_min, z_min = label_indices.min(axis=0)
        x_max, y_max, z_max = label_indices.max(axis=0)
        bounding_box = [int(x_min), int(x_max), int(y_min), int(y_max), int(z_min), int(z_max)]
        #print(center_of_mass)
        label_info[int(label)] = {
            #"pixel_count": int(pixel_count),
            "bounding_box": bounding_box,
            "point_list": points_
        }

    return label_info
    
    
def process_image(nii_data, ms_data=None , require_ms=True):
    """input:
        nii_data: resampled image volume 
        ms_data: paired mask volume
      
      output:
        out_data: preprocessed image volume
        coordinates: coordinates of center of mass of organs
    
    """
    if require_ms==True:
        cropped_data, cropped_ms = get_cropped_data(nii_data, ms_data)
        padded_data, padded_ms = pad_to_square(cropped_data, cropped_ms)
        fixed_z_data, fixed_z_ms = preprocess_zaxis(padded_data, padded_ms)
        resized_data, resized_ms = resize_array(fixed_z_data, fixed_z_ms)
        
        coordinate_dict = calculate_label_info(resized_ms)
        return resized_data, resized_ms, coordinate_dict 
    else:
        cropped_data = get_cropped_data(nii_data, ms_data)
        padded_data = pad_to_square(cropped_data, None)
        fixed_z_data = preprocess_zaxis(padded_data, None)
        resized_data = resize_array(fixed_z_data, None)
        
        return resized_data

    