# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:22:17 2024

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
import itertools
from tensorflow.keras.backend import clear_session


#%%
def create_model(n_input=1):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=(60, 60, n_input)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2DTranspose(64, (3, 3), strides=(2, 2),
                        padding='same', activation='relu'),
        Conv2DTranspose(32, (3, 3), strides=(2, 2),
                        padding='same', activation='relu'),
        Conv2D(1, (1, 1), activation='sigmoid', padding='same'),
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def segment_image(image, patch_size=(60, 60)):
    # Determine if the image is grayscale or multi-channel by checking its dimension
    if image.ndim == 2:
        img_height, img_width = image.shape
        num_channels = 1  # It's a grayscale image
    elif image.ndim == 3:
        img_height, img_width, num_channels = image.shape
    else:
        raise ValueError("Image must be either 2 or 3 dimensions")

    patches = []

    # Loop through the image to extract patches
    for i in range(0, img_height, patch_size[0]):
        for j in range(0, img_width, patch_size[1]):
            # Check to ensure only full patches are taken
            if (i + patch_size[0] <= img_height) and (j + patch_size[1] <= img_width):
                if num_channels == 1:
                    patch = image[i:i + patch_size[0], j:j + patch_size[1]]
                    patches.append(patch.reshape(*patch_size, 1))  # Add channel dimension for consistency
                else:
                    patch = image[i:i + patch_size[0], j:j + patch_size[1], :]
                    patches.append(patch)
    
    return np.array(patches)


def reconstruct_from_patches(patches, original_dims, patch_size=(60, 60)):
    """
    Reconstructs an image from its patches.

    Args:
    - patches: The array of patches.
    - original_dims: The dimensions of the original image (height, width).
    - patch_size: The size of each patch (height, width).

    Returns:
    - A numpy array representing the reconstructed image.
    """
    reconstructed_image = np.zeros(original_dims)
    patch_idx = 0
    for i in range(0, original_dims[0], patch_size[0]):
        for j in range(0, original_dims[1], patch_size[1]):
            if i + patch_size[0] <= original_dims[0] and j + patch_size[1] <= original_dims[1]:
                reconstructed_image[i:i + patch_size[0], j:j +
                                    patch_size[1]] = patches[patch_idx].reshape(patch_size)
                patch_idx += 1
    return reconstructed_image





def Prepare_data(beach, stack_name_train, stack_name_validate, bound_train, bound_validate):

    stack_dir = 'C:/Users/Prins/OneDrive - Delft University of Technology/Desktop/Master_Thesis/Processed/Timestacks'

    # LOAD DATA:
    timestack_train = np.load(os.path.join(stack_dir, beach, stack_name_train))
    timestack_validate = np.load(os.path.join(
        stack_dir, beach, stack_name_validate))
    # Take upper and lower bounds as defined in handpick
    timestack_window_train = timestack_train[bound_train[0]:bound_train[1], :]
    timestack_window_validate = timestack_validate[bound_validate[0]                                                   :bound_validate[1], :]

    # Reshape to by averaging over n values in space for computational costs
    n = 10  # Reduction factor for the first axis
    height = timestack_window_train.shape[0] // n
    width = timestack_window_train.shape[1]
    reshaped = timestack_window_train.reshape(
        height, n, width, timestack_window_train.shape[2])
    reduced_timestack_train = np.mean(reshaped, axis=1).astype(np.uint8)

    height = timestack_window_validate.shape[0] // n
    width = timestack_window_validate.shape[1]
    reshaped = timestack_window_validate.reshape(
        height, n, width, timestack_window_validate.shape[2])
    reduced_timestack_validate = np.mean(reshaped, axis=1).astype(np.uint8)

    plt.figure(figsize=(18, 4))
    plt.imshow(reduced_timestack_train)

    plt.figure(figsize=(18, 4))
    plt.imshow(reduced_timestack_validate)

    # LOAD LABELS:'
    # train
    dir_labels = 'C:/Users/Prins/OneDrive - Delft University of Technology/Desktop/Master_Thesis/Processed/Runup_FromManual/SW'
    shoreline = np.load(dir_labels + '/' + stack_name_train, allow_pickle=True)
    runup = shoreline[:, 3] - bound_train[0]
    labels = np.ones(
        (timestack_window_train.shape[0], timestack_window_train.shape[1]))
    # Create a mask where True indicates row indices in labels above the row indices in runup
    mask = np.arange(labels.shape[0])[:, np.newaxis] > runup
    # Set values in labels to 0 where the mask is True
    labels[mask] = 0
    labels_train = labels[::10]

    # validate
    shoreline = np.load(dir_labels + '/' +
                        stack_name_validate, allow_pickle=True)
    runup = shoreline[:, 3] - bound_validate[0]
    labels = np.ones(
        (timestack_window_validate.shape[0], timestack_window_validate.shape[1]))
    # Create a mask where True indicates row indices in labels above the row indices in runup
    mask = np.arange(labels.shape[0])[:, np.newaxis] > runup
    # Set values in labels to 0 where the mask is True
    labels[mask] = 0
    labels_validate = labels[::10]

    plt.subplots(figsize=(18, 4))
    plt.imshow(labels_train)

    plt.figure(figsize=(18, 4))
    plt.imshow(labels_validate)

    return reduced_timestack_train, reduced_timestack_validate, labels_train, labels_validate


def Kfold_Test_Train(X, y, save_path):
    

    start_time = time.time()
    num_folds = 5
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # Placeholder for model scores
    fold_no = 1
    acc_per_fold = []
    loss_per_fold = []
    
    n_input = X.shape[-1]

    t0 = time.time()

    for train, test in kfold.split(X, y):
        # Generate a print
        print(f'Training model for fold {fold_no}...')

        # Create model for the current fold
        model = create_model(n_input)


        # Fit data to model
        model.fit(X[train], y[train],
                  batch_size=32,
                  epochs=10,
                  verbose=0,
                  validation_data=(X[test], y[test]))

        # Generate generalization metrics
        scores = model.evaluate(X[test], y[test], verbose=0)
        
        if __name__ == "__main__":
            print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no += 1
        
        # clear session to avoid memory slowdown
        clear_session()
    
    t1 = time.time()
    
    print(f'IT TOOK US {(t1-t0)/60:.2f} minutes')
    
    if __name__ == "__main__":
        # == Provide average scores ==
        print('------------------------------------------------------------------------')
        print('Score per fold')
        for i in range(0, len(acc_per_fold)):
            print('------------------------------------------------------------------------')
            print(
                f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(acc_per_fold)}% (+- {np.std(acc_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)}')
        print('------------------------------------------------------------------------')

    # write to file:
    with open(save_path, 'w') as file:
        file.write(
            '------------------------------------------------------------------------\n')
        file.write('Score per fold\n')
        for i in range(len(acc_per_fold)):
            file.write(
                '------------------------------------------------------------------------\n')
            file.write(
                f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%\n')
        file.write(
            '------------------------------------------------------------------------\n')
        file.write('Average scores for all folds:\n')
        mean_acc = np.mean(acc_per_fold)
        std_acc = np.std(acc_per_fold)
        mean_loss = np.mean(loss_per_fold)
        file.write(f'> Accuracy: {mean_acc:.2f}% (+- {std_acc:.2f})\n')
        file.write(f'> Loss: {mean_loss:.2f}\n')
        file.write(
            '------------------------------------------------------------------------\n')
        end_time = time.time()
        elapsed_time = end_time - start_time
        file.write(f'> Computation Time: {elapsed_time}')


def Validate(X, y, X_new, y_new, file_name, shape):

    save_path = 'C:/Users/Prins/OneDrive - Delft University of Technology/Desktop/Master_Thesis/CNN/k_fold_SingleInputChannels/' + file_name + '.txt'

    model = create_model()

    model.fit(X, y, batch_size=32, epochs=10)

    loss, accuracy = model.evaluate(X_new, y_new)
    print(f"Validation Loss: {loss}")
    print(f"Validation Accuracy: {accuracy}")

    predictions = model.predict(X_new)

    predicted_masks = predictions > 0.5

    reconstructed_prediction = reconstruct_from_patches(predicted_masks, shape)

    plt.figure(figsize=(18, 4))
    plt.imshow(reconstructed_prediction)

    return reconstructed_prediction


# %%
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

    # Prepare -----------------------------------------------------------------
    # Preprocess input channel
    Channel_inputs = ['Grayscale', 'Grayscale_overTime', 'Grayscale_overSpace',\
                      'Entropy', 'Entropy_overTime', 'Gaussian_blur', 'Red-Blue', 'Red-Blue_overSpace', 'RGB']
    
    
    Input = Channel_inputs[1]
    
    #%%
    
    grayscale_train = cv.cvtColor(reduced_timestack_train, cv.COLOR_RGB2GRAY)
    grayscale_validate = cv.cvtColor(reduced_timestack_validate, cv.COLOR_RGB2GRAY)
    
    
    
    if Input == Channel_inputs[0]:
        # Grayscale image
        data_train = grayscale_train
        data_validate = grayscale_validate
    if Input == Channel_inputs[1]:
        # Gradient grayscale in time
        data_train = np.gradient(grayscale_train, axis=1)
        data_validate = np.gradient(grayscale_validate, axis=1)
    if Input == Channel_inputs[2]:
        # Gradient grayscale in space
        data_train = np.gradient(grayscale_train, axis=0)
        data_validate = np.gradient(grayscale_validate, axis=0)
        
    if Input == Channel_inputs[3]:
        # Local enytropy
        data_train = entropy(grayscale_train, disk(3))
        data_validate = entropy(grayscale_validate, disk(3))
    if Input == Channel_inputs[4]:
        # Local entropy gradient in time
        data_train = np.gradient(entropy(grayscale_train, disk(3)), axis=1)
        data_validate = np.gradient(entropy(grayscale_validate, disk(3)), axis=1)
        
    if Input == Channel_inputs[5]:
        # Gaussian blur in time
        data_train = cv.GaussianBlur(grayscale_train,(3,1),0)
        data_validate = cv.GaussianBlur(grayscale_train,(3,1),0)
    
        
        
    if Input == Channel_inputs[6]:
        # R-B
        data_train = reduced_timestack_train[:,:,0] - reduced_timestack_train[:,:,1]
        data_validate = reduced_timestack_validate[:,:,0] - reduced_timestack_validate[:,:,1]
    if Input == Channel_inputs[7]:
        # R-B in space
        data_train = np.gradient(reduced_timestack_train[:,:,0] - reduced_timestack_train[:,:,1], axis=0)
        data_validate = np.gradient(reduced_timestack_validate[:,:,0] - reduced_timestack_validate[:,:,1], axis=0)
        
        
    
    X_patches_train = segment_image(data_train)
    Y_patches_train = segment_image(labels_train)
    

    plt.figure(figsize=(18, 4))
    plt.imshow(data_train, cmap='gray')
    plt.title(Input)

    # %%
    # K-fold training
    save_path = 'C:/Users/Prins/OneDrive - Delft University of Technology/Desktop/Master_Thesis/CNN/k_fold_SingleInputChannels/' + Input + '.txt'
    Kfold_Test_Train(X_patches_train, Y_patches_train, save_path)
    
    
    # %%

    # Prepare for validation --------------------------------------------------
    X_patches_validate = segment_image(data_validate)
    Y_patches_validate = segment_image(labels_validate)

    Y_predicted = Validate(X_patches_train, Y_patches_train,
                           X_patches_validate, Y_patches_validate, Input, data_validate.shape)

    # %%

    runup_prediction = np.sum(Y_predicted, axis=0)
    runup_real = np.sum(labels_validate, axis=0)
    fig, axs = plt.subplots(3, 1, figsize=(18, 6))  # 2 Rows, 1 Column

    # Plot the first image
    # 'cmap' might be unnecessary depending on your image
    axs[0].imshow(labels_validate, cmap='gray')
    axs[0].set_title('Validation labels')
    axs[0].plot(runup_prediction)
    axs[0].plot(runup_real)
    axs[0].axis('off')  # Disable axis

    # Plot the second image
    axs[1].imshow(Y_predicted, cmap='gray')
    axs[1].set_title('Prediction')
    axs[1].axis('off')  # Disable axis

    axs[2].imshow(Y_predicted - labels_validate, cmap='bwr')
    axs[2].set_title('Prediction')
    axs[2].axis('off')  # Disable axis

    # Display the plot
    plt.tight_layout()  # Adjust the layout to make sure there's no overlap
    plt.show()
    

    # %%

    cm = confusion_matrix(labels_validate.flat, Y_predicted.flat)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_percentage = cm_normalized * 100

    plt.figure(figsize=(10, 8))  # Increase figure size for better visibility
    ax = sns.heatmap(cm_percentage, annot=True, fmt=".1f", linewidths=.5, square=True, cmap='Blues', cbar=False,
                     xticklabels=['water', 'beach'],
                     yticklabels=['water', 'beach'],
                     annot_kws={"size": 30})  # Increase font size for annotations
    
    # Increase font size for labels
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 30)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 30)
    
    # Optionally, adjust the title and label sizes
    plt.title('Confusion Matrix (%)', fontsize=40)
    plt.xlabel('Predicted', fontsize=35)
    plt.ylabel('True', fontsize=35)
    
    plt.show()

    # %%

    accuracies_A = np.array(
        [97.01979160308838, 96.2160587, 96.734464, 96.6966152, 96.2102413])
    accuracies_B = np.array(
        [94.188106, 92.30807423, 94.6089386, 94.5519089698, 94.433593])

    accuracies_A = np.array(
        [96.144789457, 96.244794130, 96.22760415, 94.91909742, 96.62204980])
    accuracies_B = np.array(
        [93.30147504, 94.14157867, 95.7835078, 93.7671005, 89.22473788])

    # Calculate mean and standard deviation
    mean_A = np.mean(accuracies_A)
    mean_B = np.mean(accuracies_B)
    std_A = np.std(accuracies_A, ddof=1)
    std_B = np.std(accuracies_B, ddof=1)

    # Perform a paired t-test
    t_stat, p_value = stats.ttest_rel(accuracies_A, accuracies_B)

    # Calculate Cohen's d for effect size
    effect_size = (mean_A - mean_B) / np.sqrt((std_A**2 + std_B**2) / 2)

    # Print results
    print(f"Mean accuracy for Model A: {mean_A}, Model B: {mean_B}")
    print(f"Paired t-test p-value: {p_value}")
    print(f"Effect size (Cohen's d): {effect_size}")
