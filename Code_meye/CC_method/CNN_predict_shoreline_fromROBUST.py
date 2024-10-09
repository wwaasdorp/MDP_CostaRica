# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:30:48 2024

@author: Prins
"""

import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import os
import glob
os.chdir('C:/Users/Prins/OneDrive - Delft University of Technology/Desktop/Master_Thesis/Scripts')
from CNN_Single_Channel_Input import create_model, segment_image,\
    reconstruct_from_patches, Prepare_data,\
         Validate, Kfold_Test_Train, reconstruct_from_patches
from CNN_all_Combined import preprocess_channels
from sklearn.metrics import f1_score
import pandas as pd
from sklearn import metrics
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import load_model
from scipy.ndimage import gaussian_filter1d
from CreationTime import StartTime
from datetime import datetime, timedelta



#%%

def prepare(stack_name, bounds):
    
    stack_dir = 'C:/Users/Prins/OneDrive - Delft University of Technology/Desktop/Master_Thesis/Processed/Timestacks'
    beach = stack_name[:2]

    # LOAD DATA:
    timestack = np.load(os.path.join(stack_dir, beach, stack_name))
    # Take upper and lower bounds as defined in handpick
    timestack_window = timestack[bounds[0]:bounds[1], :]

    # Reshape to by averaging over n values in space for computational costs
    n = 10  # Reduction factor for the first axis
    height = timestack_window.shape[0] // n
    width = timestack_window.shape[1]
    reshaped = timestack_window.reshape(height, n, width, timestack_window.shape[2])
    reduced_timestack = np.mean(reshaped, axis=1).astype(np.uint8)

    return reduced_timestack




# =============================================================================
# stack_name = 'TT_20231114_1045_GX050002.npy'
# 
# # LOAD DATA:
# beach = stack_name[:2]
# stack_dir = 'C:/Users/Prins/OneDrive - Delft University of Technology/Desktop/Master_Thesis/Processed/Timestacks'
# timestack = np.load(os.path.join(stack_dir, beach, stack_name))
# 
# upper = 1400
# lower = 5000
# 
# plt.figure(figsize=(18,10))
# plt.imshow(np.repeat(timestack[upper:lower,:500], 5, axis=1))
# #plt.imshow(timestack[upper:lower,:])
# =============================================================================


#%%
stacks_dict = {}
stacks_dict['SW_20231116_0920_GX010085.npy'] = [6000,12000]
stacks_dict['SW_20231116_0920_GX020085.npy'] = [6000,12000]
stacks_dict['SW_20231116_0920_GX030085.npy'] = [5800,12000]
stacks_dict['SW_20231116_0920_GX040085.npy'] = [5500,10000]
stacks_dict['SW_20231116_0920_GX050085.npy'] = [4500,9000]
stacks_dict['SW_20231116_0920_GX060085.npy'] = [3500,7500]

# =============================================================================
# stacks_dict = {}
# stacks_dict['TT_20231026_1335_GX011364.npy'] = [2000,6000]
# stacks_dict['TT_20231114_1045_GX010002.npy'] = [4400,8500]
# stacks_dict['TT_20231114_1045_GX020002.npy'] = [3000,8500]
# stacks_dict['TT_20231114_1045_GX030002.npy'] = [2500,7500]
# stacks_dict['TT_20231114_1045_GX040002.npy'] = [1800,7000]
# =============================================================================


Channel_inputs = ['Grayscale', 'Grayscale_overSpace', 'S', 'Entropy', 'Entropy_overTime']


model = load_model('C:/Users/Prins/OneDrive - Delft University of Technology/Desktop/Master_Thesis/CNN/RESULTS/Robust/Final_trained9/SW_20231114_Grayscale_Grayscale_overSpace_S_Entropy_Entropy_overTime/Model_trainedon9.h5')

for stack_name in stacks_dict:
    
    
    
    bounds = stacks_dict.get(stack_name)
    
    reduced_timestack = prepare(stack_name, bounds)
    
    data = preprocess_channels(Channel_inputs, reduced_timestack)
    X_patches = segment_image(data)
    
    
    
    predictions = model.predict(X_patches)
    predicted_masks = predictions > 0.5
    reconstructed_prediction = reconstruct_from_patches(predicted_masks, reduced_timestack.shape[:-1])
    
    
    parts = stack_name.split('_')
    beach = parts[0]
    day = parts[1]
    stack_dir = 'C:/Users/Prins/OneDrive - Delft University of Technology/Desktop/Master_Thesis/Processed/Timestacks'
    original_stack = np.load(os.path.join(stack_dir, beach, stack_name))
    
    
    mask = (reconstructed_prediction == 0)
    lowest_indices = np.argmax(mask, axis=0)
    runup_smooth = gaussian_filter1d(lowest_indices, 6)
    # runup times series on original timestack
    
    runup_original_stack = runup_smooth * 10 + bounds[0]
    
    
    # Transpose to real-world coordinates
    pix2world = np.load('C:/Users/Prins/OneDrive - Delft University of Technology/Desktop/Master_Thesis/Processed/Pixel2World/' + beach + '_' + day + '.npy')
    pix2world = pd.DataFrame(pix2world, columns=['pixel_id', 'U', 'V', 'x', 'y', 'z'])
    
    
    save_path = 'C:/Users/Prins/OneDrive - Delft University of Technology/Desktop/Master_Thesis/Processed/Runup_FromML/' + beach + '/' + stack_name
    
    
    
    shoreline_real = pix2world[['x','y','z']].iloc[runup_original_stack]                 # real xyz value of shoreline position in time
    shoreline_real['pixel'] = runup_original_stack
    
    start_time = StartTime(stack_name[:-4])
    timestamps = [start_time + timedelta(seconds=i * 0.5) for i in range(reduced_timestack.shape[1])]
    
    shoreline_real['Time'] = timestamps
    
    np.save(save_path, shoreline_real)
    
    
    x_ticks_indices = np.arange(0, len(timestamps), 1000)
    x_ticks = [timestamps[i].strftime('%H:%M:%S') for i in x_ticks_indices]
    #%%

    plt.figure(figsize=(18,6))
    plt.xticks(x_ticks_indices, x_ticks, rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    plt.imshow(np.repeat(reduced_timestack,5, axis=1))
    plt.plot(runup_smooth, color='firebrick')
    plt.title(stack_name, fontsize=18)
    plt.ylabel('x (pix)', fontsize=18)
    plt.xlabel('Time [hh:mm:ss]', fontsize=18)
    plt.tight_layout()
    plt.xlim(0,1000)
    plt.plot()
    plt.savefig(save_path[:-4] + 'stack_runupml.png')

    
    plt.figure(figsize=(10,5))
    plt.plot(shoreline_real.Time, shoreline_real.z)
    plt.title(stack_name, fontsize=18)
    plt.ylabel('Elevation (m)', fontsize=18)
    plt.xlabel('Time [hh:mm:ss]', fontsize=18)
    plt.tight_layout()
    plt.grid()
    plt.savefig(save_path[:-4] + 'runup_timeseries.png')
