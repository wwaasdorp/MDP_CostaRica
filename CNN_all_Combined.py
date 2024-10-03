# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:05:54 2024

@author: Prins
"""


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import cv2 as cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.filters.rank import entropy
from skimage.morphology import disk
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from scipy import stats
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
os.chdir('C:/Users/Prins/OneDrive - Delft University of Technology/Desktop/Master_Thesis/Scripts')
from CNN_Single_Channel_Input import create_model, segment_image,\
    reconstruct_from_patches, Prepare_data,\
         Validate, Kfold_Test_Train
import itertools        
from math import comb
from tensorflow.keras.backend import clear_session
from itertools import combinations

def preprocess_channels(selected_channels, reduced_timestack):
    processed_channels = []

    for channel in selected_channels:
        if channel == 'Grayscale':
            data = reduced_timestack
            data = cv.cvtColor(data, cv.COLOR_RGB2GRAY)
            data = data[:, :, np.newaxis]  # Add a new axis to make it 3D
            
        elif channel == 'RGB':
            data = reduced_timestack  # Already 3D, do not add new axis
            
        elif channel == 'Grayscale_overTime':
            data = cv.GaussianBlur(reduced_timestack, (3, 3), 0)
            data = np.gradient(cv.cvtColor(data, cv.COLOR_RGB2GRAY), axis=1)
            data = data[:, :, np.newaxis]  # Add a new axis to make it 3D
            
        elif channel == 'Grayscale_overSpace':
            data = cv.GaussianBlur(reduced_timestack, (3, 1), 0)
            data = np.gradient(cv.cvtColor(data, cv.COLOR_RGB2GRAY), axis=0)
            data = data[:, :, np.newaxis]  # Add a new axis to make it 3D
            
        elif channel == 'Entropy':
            data = cv.GaussianBlur(reduced_timestack, (3, 3), 0)
            data = entropy(cv.cvtColor(data, cv.COLOR_RGB2GRAY), disk(3))
            data = data[:, :, np.newaxis]  # Add a new axis to make it 3D
            
        elif channel == 'Entropy_overTime':
            data = cv.GaussianBlur(reduced_timestack, (3, 3), 0)
            entropy_data = entropy(cv.cvtColor(data, cv.COLOR_RGB2GRAY), disk(3))
            data = np.gradient(entropy_data, axis=1)
            data = data[:, :, np.newaxis]  # Add a new axis to make it 3D
            
        elif channel == 'S':
            data = cv.GaussianBlur(reduced_timestack, (3, 3), 0)
            data = cv.cvtColor(data, cv.COLOR_BGR2HSV)[:, :, 1]
            data = data[:, :, np.newaxis]  # Add a new axis to make it 3D
            

        
        elif channel == 'H':
            data = cv.GaussianBlur(reduced_timestack, (3, 3), 0)
            data = cv.cvtColor(data, cv.COLOR_BGR2HSV)[:, :, 0]
            data = data[:, :, np.newaxis]  # Add a new axis to make it 3D


        data_min = np.min(data)
        data_max = np.max(data)
        data = (data - data_min) / (data_max - data_min + 1e-6)
        

        processed_channels.append(data)
    
    if not processed_channels:
        return np.empty((reduced_timestack.shape[0], reduced_timestack.shape[1], 0))  # Return an empty array with the right shape but zero channels

    # Stack all processed channels along the last axis, handling RGB case appropriately
    return np.concatenate(processed_channels, axis=-1)


# =============================================================================
# def preprocess_channels(selected_channels, reduced_timestack):
# 
#         
#     processed_channels = []
#     
#     for channel in selected_channels:
#         
#         if channel == 'Grayscale':
#             data = reduced_timestack
#             data = cv.cvtColor(data, cv.COLOR_RGB2GRAY)
#             data = data[:, :, np.newaxis]  # Add a new axis to make it 3D
#         elif channel == 'RGB':
#             data = reduced_timestack
#             data = data  # Already 3D, do not add new axis
#         elif channel == 'Grayscale_overTime':
#             data = cv.GaussianBlur(reduced_timestack,(3,3),0)
#             data = np.gradient(cv.cvtColor(data, cv.COLOR_RGB2GRAY), axis=1)
#             data = data[:, :, np.newaxis]  # Add a new axis to make it 3D
#         elif channel == 'Grayscale_overSpace':
#             data = cv.GaussianBlur(reduced_timestack,(3,1),0)
#             data = np.gradient(cv.cvtColor(data, cv.COLOR_RGB2GRAY), axis=0)
#             data = data[:, :, np.newaxis]  # Add a new axis to make it 3D
#         elif channel == 'Entropy':
#             data = cv.GaussianBlur(reduced_timestack,(3,3),0)
#             data = entropy(cv.cvtColor(data, cv.COLOR_RGB2GRAY), disk(3))
#             data = data[:, :, np.newaxis]  # Add a new axis to make it 3D
#         elif channel == 'Entropy_overTime':
#             data = cv.GaussianBlur(reduced_timestack,(3,3),0)
#             entropy_data =  entropy(cv.cvtColor(data, cv.COLOR_RGB2GRAY), disk(3))
#             data = np.gradient(entropy_data, axis=1)
#             data = data[:, :, np.newaxis]  # Add a new axis to make it 3D
#         elif channel == 'S':
#             data = cv.GaussianBlur(reduced_timestack,(3,3),0)
#             data = cv.cvtColor(data, cv.COLOR_BGR2HSV)[:,:,1]
#             data = data[:, :, np.newaxis]  # Add a new axis to make it 3D
#         elif channel == 'H':
#             data = cv.GaussianBlur(reduced_timestack,(3,3),0)
#             data = cv.cvtColor(data, cv.COLOR_BGR2HSV)[:,:,0]
#             data = data[:, :, np.newaxis]  # Add a new axis to make it 3D
#         
#         
#         #data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-6)
#         
#         col_min = np.min(data, axis=0)
#         col_max = np.max(data, axis=0)
#         
#         # Normalize the data with broadcasting
#         data = (data - col_min) / (col_max - col_min + 1e-6)
#         processed_channels.append(data)
#         
#     if not processed_channels:
#         return np.empty((reduced_timestack.shape[0], reduced_timestack.shape[1], 0))  # Return an empty array with the right shape but zero channels
# 
#     
#     # Stack all processed channels along the last axis, handling RGB case appropriately
#     return np.concatenate(processed_channels, axis=-1)
# =============================================================================


#%%

if __name__ == "__main__":

    # Load data  --------------------------------------------------------------

    beach = 'SW'
    stack_name_train = 'SW_20231114_0850_GX060084.npy'
    bound_train = [2000, 7500]  # offet for handpick method
    stack_name_validate = 'SW_20231114_0850_GX050084.npy'
    bound_validate = [2000, 7500]  # offet for handpick method

    reduced_timestack_train, reduced_timestack_validate, labels_train, labels_validate =\
        Prepare_data(beach, stack_name_train, stack_name_validate,
                     bound_train, bound_validate)

# %%
    Case_name = 'Combinations_All_channels_All_possible_comb'
    
    
    # Make new saving dir for case
    base_path = 'C:\\Users\\Prins\\OneDrive - Delft University of Technology\\Desktop\\Master_Thesis\\CNN'
    full_path = os.path.join(base_path, Case_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"Directory '{Case_name}' created at: {full_path}")
    else:
        print(f"Directory already exists: {full_path}")
    
    
    # Prepare -----------------------------------------------------------------
    # Preprocess input channel
    Channel_inputs = ['Grayscale', 'Grayscale_overTime', 'Grayscale_overSpace',\
                       'S', 'RGB', 'Entropy', 'Entropy_overTime']
    
    all_combinations = []

    # Generate all possible combinations for all possible lengths
    for r in range(1, len(Channel_inputs) + 1):
        combinations_of_r = list(combinations(Channel_inputs, r))
        all_combinations.extend(combinations_of_r)
    
    # Convert the combinations from tuples to lists
    all_combinations = [list(comb) for comb in all_combinations]
    
# =============================================================================
#     all_combinations = ['Grayscale', 'Grayscale_overTime', 'Grayscale_overSpace',\
#                        'S', 'RGB', 'Entropy', 'Entropy_overTime']
# =============================================================================
    #%%
    Y_patches_train = segment_image(labels_train)

    for i,combination in enumerate(all_combinations):
        

        data_train = preprocess_channels(combination, reduced_timestack_train)
            
        X_patches_train = segment_image(data_train)
            
        # K-fold training
        filename = str(i) + '_' + '_'.join(combination) + '.txt'
        save_path = os.path.join(full_path, filename)

        Kfold_Test_Train(X_patches_train, Y_patches_train, save_path)
        
    
#%%

    file_save = 'C:/Users/Prins/OneDrive - Delft University of Technology/Desktop/Master_Thesis/CNN/Combinations_All_channels_All_possible_comb/Combinations/Combinations.txt'
    
    with open(file_save, 'w') as file:
        for combination in all_combinations:
            # Join each combination into a single string separated by commas
            # and write each one to a new line
            file.write(', '.join(combination) + '\n')
