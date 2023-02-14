##############################################################################################
# Filename: svm_training.py                                                                  #
# Authors: Bryce Whitney, Shen Juin Lee, Zenan Chen                                          #
#                                                                                            #
# Description: This file contains the code for training a SVM classifier to detect           #
# car damage. It uses the HOG features of the image and the SVM classifier from sklearn.     #
#                                                                                            #
# Resources: The following resources were helpful in creating this code:                    #
#    - https://www.kaggle.com/code/mehmetlaudatekman/support-vector-machine-object-detection #
##############################################################################################

# Imports 
import os
import cv2
import json
import numpy as np
import pickle
from sklearn.svm import SVC
from skimage.feature import hog

# Constants
WINDOW_IMG_WIDTH = 96
WINDOW_IMG_HEIGHT= 96
CLASSES = ['Dent', 'Scratch', 'Crack', 'Glass shatter', 'Lamp broken', 'Tire flat']

#############
# FUNCTIONS #
#############

def load_train_data(image_dir, labels_dir):
    """Loads the training data from the given directories
    and return the training images and labels along with the testing labels. 
    Training images are cropped around their bounding box and resized to be used for training
    later with a sliding window approach. 

    Args:
        image_dir (str): Path to the images directory
        labels_dir (str): Path to the labels directory

    Returns:
        train_images (np.ndarray): numpy array of the training data images
        train_labels (np.ndarray): numpy array of the training data labels
        test_labels (dict): Dictionary containing the testing labels classification, bounding box, and segmentation info
    """
    # Generate the training and testing image paths
    train_image_path = os.path.join(image_dir, 'train')
    train_labels_path = os.path.join(labels_dir, 'train.json')
    test_labels_path = os.path.join(labels_dir, 'test.json')
    
    # Process the labels from the JSON file
    train_labels_dict = extract_labels(train_labels_path)
    test_labels_dict = extract_labels(test_labels_path)
    
    # Get the training images and labels
    train_images = []
    train_labels = []
    for image_file in os.listdir(train_image_path):
        # Get the image ID
        image_id = int(image_file.split('.')[0])
        
        # Load the image
        image = cv2.imread(os.path.join(train_image_path, image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Ensure it is RGB
        
        # Crop the image at the bounding box for each label
        for i, unique_label in enumerate(train_labels_dict['classification'][image_id]):
            # Get the bounding box for the label
            (x, y, w, h) = train_labels_dict['bbox'][image_id][i]

            # Crop the image at the bounding box
            cropped_image = image[int(np.floor(x)) : int(np.ceil(x+w)), int(np.floor(y)) : int(np.ceil(y+h))]
            
            # Ensure the width and height are big enough, otherwise it will cause problems
            if(cropped_image.shape[0] <= 0 or cropped_image.shape[1] <= 0):
                continue
            # Resize image to a standard size
            cropped_image = cv2.resize(cropped_image, (WINDOW_IMG_WIDTH, WINDOW_IMG_HEIGHT))
            
            # Extract the HOG features from the cropped image
            img_features = hog(cropped_image, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2), channel_axis=2)
            
            # Append the image and labels to the datasets
            train_images.append(img_features)
            train_labels.append(unique_label)  
       
    # Return the data 
    return np.array(train_images), np.array(train_labels), test_labels_dict


def extract_labels(labels_file):
    """Helper method to extract the labels from the JSON file

    Args:
        labels_file (str): Path to the JSON file containing the labels

    Returns:
        labels (dict): dictionary of the classification, bounding box, and segmentation labels
    """
    with(open(labels_file)) as label_data:
        annot_df = json.load(label_data)
        
    labels = {
        "classification": {}, # Store image id and list of labels for that image
        'bbox' : {},          # Store image id and list of bounding boxes for that image
        'segementation' : {}  # Store image id and list of segementations for that image
    }
    
    # Add labels to the dictionary
    for annotations in annot_df['annotations']:
        image_id = annotations['image_id']
        
        # Classification
        if(image_id in labels['classification'].keys()):
            labels['classification'][image_id].append(annotations['category_id'])
        else:
            labels['classification'][image_id] = [annotations['category_id']]
            
        # Bounding Box
        if(image_id in labels['bbox'].keys()):
            labels['bbox'][image_id].append(annotations['bbox'])
        else:
            labels['bbox'][image_id] = [annotations['bbox']]
        
        # Segmentation
        if(image_id in labels['segementation'].keys()):
            labels['segementation'][image_id].append(annotations['segmentation'])
        else:
            labels['segementation'][image_id] = [annotations['segmentation']]
    
    # Return the labels
    return labels

def train_model(images, labels, random_state=0):
    """Method to train the SVC classifier on the training data

    Args:
        images (np.ndarray): List of the training images
        labels (np.ndarray): List of the training labels
        random_state (int, optional): Random seed. Defaults to 0.

    Returns:
        model (sklearn.svm.SVC): Trained SVC model
    """
    # Instatite the model
    model = SVC(probability=True, random_state=random_state)
    
    # Train the model
    model.fit(images, labels)
    
    # Return the model
    return model


###############
# MAIN METHOD #
###############

if __name__ == '__main__':
    # Load the training data
    print("Loading the data...")
    train_images, train_labels, test_labels = load_train_data(image_dir='../data/images', labels_dir='../data/images/annotations')
    
    # Train the SVC model
    print("Training the model... This may take a while...")
    model = train_model(train_images, train_labels)
    
    # Save the model
    print("Saving the model...")
    pickle.dump(model, open('../models/SVM_MODEL.pkl', 'wb'))
    print("Done!")