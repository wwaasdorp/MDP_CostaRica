# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 12:18:33 2024

@author: Prins
"""



import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import cv2 as cv
from scipy.signal import convolve2d
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters import threshold_otsu
from skimage.filters import threshold_multiotsu
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import pandas as pd
os.chdir('C:/Users/Prins/OneDrive - Delft University of Technology/Desktop/Master_Thesis/Scripts')
from CreationTime import StartTime


def compute_shoreline_over_windows(reduced_timestack, window_sizes):
    num_rows, total_time = reduced_timestack.shape[:2]
    
    # Initialize the result array with zeros
    # It will accumulate results from different window sizes
    final_results = np.zeros((len(window_sizes), total_time))
    
    # Process each window size
    for i,window_size in enumerate(window_sizes):
        # Temporary results for the current window size
        
        shoreline = np.zeros((total_time))
        # Perform sliding window shoreline computation
        for start in range(0, total_time, window_size):
            end = start + window_size
            if end > total_time:
                end = total_time
            
            # Extract the window
            current_window = reduced_timestack[:, start:end]
            
            # Compute shoreline for the current window
            shoreline_window, WD_windo = Shoreline_EntropySaturation(current_window)
            # Place the computed shoreline in the corresponding positions
      
            final_results[i,start:end]  = shoreline_window
    

    
    return final_results


def normalize_columns(array):
    # Compute the minimum and maximum values for each column
    col_min = np.min(array, axis=0)
    col_max = np.max(array, axis=0)
    
    # Normalize each column
    normalized_array = (array - col_min) / (col_max - col_min)
    
    return normalized_array

def Shoreline_EntropySaturation(stack):
    '''
    Function computes the wet/dry boudnary and runup in pixel coordinates for 
    timestack images of dissipative beaches that have a clear effluent line

    Parameters
    ----------
    stack : array
            Timestack with shape (nt, nx, 3) with 
            nt the time dimension,  
            nx the spacial pixel dimension, and
            3 the RGB channels
        DESCRIPTION.

    Returns
    -------
    runup : array
        Instatanious waterline with dimension nt
    WD : array
        Instantanious wet/dry boundary with dimension nt
        
    '''
    
    # ENTROPY PART: -----------------------------------------------------------
    # Grascale image 
    grayscale_image = cv.cvtColor(stack, cv.COLOR_RGB2GRAY)
    #grayscale_image = normalize_columns(grayscale_image)
    # Gradient over time
    gradient_columns = np.gradient(grayscale_image, axis=1).astype(np.uint8)
    #gradient_columns = normalize_columns(gradient_columns)
    # Entropy
    entropy_img = entropy(gradient_columns, disk(5))
    entropy_img = normalize_columns(entropy_img)
    # Entropy threshold binarization --> need for clear bimodality
    thresh_entropy = threshold_otsu(entropy_img)
    binary_entropy = entropy_img >= thresh_entropy

    
    # RED - BLUE PART: --------------------------------------------------------
    # Red - Blue 
    I_RmB = stack[:,:,0] - stack[:,:,1]
    # Deleten foam (highest peak)
    thresholds_RmB = threshold_multiotsu(I_RmB, 3)
    I_RmB_mFoam = np.copy(I_RmB)
    I_RmB_mFoam[I_RmB_mFoam>thresholds_RmB[1]] = 0
    
    # With saturayion instead --------------------------
    I_RmB_mFoam = cv.cvtColor(stack, cv.COLOR_BGR2HSV)[:,:,1]
    I_RmB_mFoam = normalize_columns(I_RmB_mFoam)
    # Blur image over time axis
    I_RmB_blur = cv.GaussianBlur(I_RmB_mFoam,(201,1),0)
    # Binarize wet from dry 
    thresh = threshold_otsu(I_RmB_blur)
    binary_wd = np.where((I_RmB_blur <= thresh), 0, 1)
    # COmpute boundary
    WD = np.sum(binary_wd, axis=0)
    
    
    # Put results together = delete noise above effluent line -----------------
    binary_entropy_mWD = np.copy(binary_entropy)
    binary_entropy_mWD[binary_wd == 1] = 0
    
    # Compute runup line
    runup = stack.shape[0] - np.sum(binary_entropy_mWD, axis=0)
    # Smooth runup
    runup = gaussian_filter1d(runup, 2)

    
# =============================================================================
#     if __name__ == "__main__":
# 
#         t0 = datetime.strptime("00:00", "%M:%S")
#         timestamps = [t0 + timedelta(seconds=0.5*i) for i in range(stack.shape[1])]
#         x_ticks_indices = np.arange(0, stack.shape[1], 100)
#         x_ticks = [timestamps[i].strftime('%M:%S') for i in x_ticks_indices]
#         
#         y_ticks_indices = np.arange(0, stack.shape[0], 100)
# 
#         
#         plt.figure(figsize=(18,4))
#         plt.imshow(stack)
#         plt.plot(WD, color='r', linestyle='--', label='Wet/Dry boundary')
#         plt.xticks(x_ticks_indices, x_ticks, rotation=45, ha='right')
#         plt.yticks(y_ticks_indices, y_ticks_indices)
#         plt.plot(runup, color='g', label='runup')
#         plt.legend()
#         plt.tight_layout()
# 
# =============================================================================
    return runup, WD
    
    
#%%
if __name__ == "__main__":
    
    beach = 'SW'
    stack_name = 'SW_20231114_0850_GX050084.npy'
    
    
    stack_dir = 'C:/Users/Prins/OneDrive - Delft University of Technology/Desktop/Master_Thesis/Processed/Timestacks'
    timestack = np.load(os.path.join(stack_dir, beach, stack_name))

    # Plot entire timestack to choose appropriate window
    plt.figure(figsize=(10,6))
    plt.imshow(timestack)


    #%%

    # Choose window for processing
    timestack_window = timestack[2500:7000]
    n = 10  # Reduction factor for the first axis
    height = timestack_window.shape[0] // n
    width = timestack_window.shape[1]
    reshaped = timestack_window.reshape(height, n, width, timestack_window.shape[2])
    reduced_timestack = np.mean(reshaped, axis=1).astype(np.uint8)
    #%%

    
    case = 'Normalized_entropysaturation'
    save_dir = 'C:/Users/Prins/OneDrive - Delft University of Technology/Desktop/Master_Thesis/Processed/Runup_FromEntropy/Normalize_col'

    
    #%%
    
    shoreline = compute_shoreline_over_windows(reduced_timestack, [500])
    
    #%%
    np.save(save_dir + '/shoreline_entropysaturation_w500_' + case + '.npy', shoreline)
    
    #%%
    stacks_dict = {}
    stacks_dict['SW_20231113_1346_GX010081.npy'] = [2300,7000]
    stacks_dict['SW_20231113_1346_GX020081.npy'] = [1800,6500]
    stacks_dict['SW_20231113_1346_GX030081.npy'] = [1500,6000]
    stacks_dict['SW_20231114_0850_GX010084.npy'] = [4500,10000]
    stacks_dict['SW_20231114_0850_GX020084.npy'] = [5000,10000]
    stacks_dict['SW_20231114_0850_GX030084.npy'] = [4500,9500]
    stacks_dict['SW_20231114_0850_GX040084.npy'] = [3500,8500]
    stacks_dict['SW_20231114_0850_GX050084.npy'] = [2000,7500]
    stacks_dict['SW_20231114_0850_GX060084.npy'] = [2500,7000]
    
    for stack in stacks_dict:
        
        bounds = stacks_dict.get(stack)
        stack_dir = 'C:/Users/Prins/OneDrive - Delft University of Technology/Desktop/Master_Thesis/Processed/Timestacks'
        beach = stack[:2]
        day = stack[3:11]
        timestack = np.load(os.path.join(stack_dir, beach, stack))



        # Choose window for processing
        timestack_window = timestack[bounds[0]:bounds[1]]
        n = 10  # Reduction factor for the first axis
        height = timestack_window.shape[0] // n
        width = timestack_window.shape[1]
        reshaped = timestack_window.reshape(height, n, width, timestack_window.shape[2])
        reduced_timestack = np.mean(reshaped, axis=1).astype(np.uint8)
        
        
        shoreline = compute_shoreline_over_windows(reduced_timestack, [500])
        
        pix2world = np.load('C:/Users/Prins/OneDrive - Delft University of Technology/Desktop/Master_Thesis/Processed/Pixel2World/' + beach + '_' + day + '.npy')
        pix2world = pd.DataFrame(pix2world, columns=['pixel_id', 'U', 'V', 'x', 'y', 'z'])
        
        runup_original_stack = shoreline * 10 + bounds[0]
        runup_original_stack = runup_original_stack.squeeze()
        
        shoreline_real = pix2world[['x','y','z']].iloc[runup_original_stack]                 # real xyz value of shoreline position in time
        shoreline_real['pixel'] = runup_original_stack
        
        start_time = StartTime(stack[:-4])
        timestamps = [start_time + timedelta(seconds=i * 0.5) for i in range(reduced_timestack.shape[1])]
        
        shoreline_real['Time'] = timestamps
        
        save_dir = 'C:/Users/Prins/OneDrive - Delft University of Technology/Desktop/Master_Thesis/Processed/Runup_FromEntropy/_Final'
        save_path = save_dir + '/Entropy_' + stack
        np.save(save_path, shoreline_real)
        