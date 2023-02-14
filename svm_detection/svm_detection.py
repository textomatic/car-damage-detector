#############################################################################################
# Filename: svm_detection.py                                                                #
# Authors: Bryce Whitney, Shen Juin Lee, Zenan Chen                                         #
#                                                                                           #
# Description: This file contains the code for detecting car damage using a SVM classifier. #
#                                                                                           #
# Resources: The following resources were helpful in creating this code:                    #
#   - https://www.kaggle.com/code/mehmetlaudatekman/support-vector-machine-object-detection #
#   - https://blog.paperspace.com/mean-average-precision/                                   #
#   - https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection #
#############################################################################################

# Imports
import os
import cv2
import pickle
import numpy as np
from PIL import Image
from skimage.feature import hog
from heatmap import Heatmap
from svm_training import extract_labels

# Constants
WINDOW_IMG_WIDTH = 96
WINDOW_IMG_HEIGHT= 96
CLASSES = ['Dent', 'Scratch', 'Crack', 'Glass shatter', 'Lamp broken', 'Tire flat']

#############
# FUNCTIONS #
#############
def slideExtract(image, windowSize=(WINDOW_IMG_WIDTH, WINDOW_IMG_HEIGHT), step=12):
    """ Implements a sliding window appraoch where the coordinates and feature of each window are extracted.
    Adapted from https://www.kaggle.com/code/mehmetlaudatekman/support-vector-machine-object-detection

    Args:
        image (np.ndarray): _description_
        windowSize (tuple, optional): The width and height of the sliding window. Defaults to (WINDOW_IMG_WIDTH, WINDOW_IMG_HEIGHT).
        step (int, optional): How many steps the window will take each time. Defaults to 12.

    Returns:
        coords (list): A list of the coordinates for each window across the image
        features (np.ndarray): A list of the features for each of the windows
    """
    # We'll store coords and features in these lists
    coords = []
    features = []
    
    hIm,wIm = image.shape[:2] 

    
    # W1 will start from 0 to end of image - window size
    # W2 will start from window size to end of image
    # We'll use step (stride) like convolution kernels.
    for w1,w2 in zip(range(0,wIm-windowSize[0],step),range(windowSize[0],wIm,step)):
       
        for h1,h2 in zip(range(0,hIm-windowSize[1],step),range(windowSize[1],hIm,step)):
            window = image[h1:h2,w1:w2]
            features_of_window = hog(window,orientations=9,pixels_per_cell=(16,16),
                                     cells_per_block=(2,2), channel_axis=2
                                    )
            
            coords.append((w1,w2,h1,h2))
            features.append(features_of_window)
    
    return coords, np.asarray(features)

def detection(model, val_images_dir, step=24, threshold=0.5):
    """This method performs the object detection using the SVM classifier. 
    It makes a prediction for each image and returns the predicted bounding boxes of damage and their
    associated label. 

    Args:
        model (sklearn.svm.SVC): Trained SVC model
        val_images_dir (str): path to the validation/test images
        step (int, optional): How many steps the sliding window should take each time. Defaults to 24.
        threshold (float, optional): Threshold to determine if a prediction is valid or not. Defaults to 0.5.

    Returns:
        pred_classes (dict): Dictionary of predicted classes for each image
        pred_bboxes (dict): Dictionary of predicted bounding boxes for each image
    """
    pred_classes = {}
    pred_bboxes = {}
    
    for file in os.listdir(val_images_dir):
        # Open the image
        image_id = int(file.split('.')[0])
        image = np.asarray(Image.open(os.path.join(val_images_dir, file)))
        
        # Extracting features and initalizing heatmap
        coords,features = slideExtract(image, step=step)
        htmp = Heatmap(image)
        
        # Go through all the windows
        for i in range(len(features)):
            # If region is positive then add some heat
            pred = model.predict_proba([features[i]])
            if(max(pred[0]) > threshold):
                htmp.incValOfReg(coords[i], np.argmax(pred[0]) + 1)
            else:
                htmp.decValOfReg(coords[i])
        
        # Compiling heatmap
        mask = htmp.compileHeatmap()
        
        cont,_ = cv2.findContours(mask,1,2)[:2]
        for c in cont:
            # If a contour is small don't consider it
            if cv2.contourArea(c) < 70*70:
                continue
            
            (x,y,w,h) = cv2.boundingRect(c)
            
            if (htmp.labels[y,x] != 0):
                if image_id in pred_bboxes.keys():
                    pred_bboxes[image_id].append([x,y,w,h])
                else:
                    pred_bboxes[image_id] = [[x,y,w,h]]
                    
                if image_id in pred_classes.keys():
                    pred_classes[image_id].append(htmp.labels[y,x])
                else:
                    pred_classes[image_id] = [htmp.labels[y,x]]
                        
    return pred_classes, pred_bboxes

def IoU(boxA, boxB):
    """Calculates the IoU of two bounding boxes. 
    Taken from https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

    Args:
        boxA (np.ndarray): The first bounding box
        boxB (np.ndarray): The second bounding box
        
    Returns:
        iou (float): The IoU of the two bounding boxes
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
	# rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
    return iou

def calculate_ap(true_classes, pred_classes, true_bboxes, pred_bboxes, threshold=0.5):
    """Function to calculate the average precision for each class. This takes both
    the bounding box and classifcation into account, and cant be considered a true 
    positive unless both are accurate. 

    Args:
        true_classes (dict): true classifications for each image
        pred_classes (dict): predicted classifications for each image
        true_bboxes (dict): true bounding boxes for each image
        pred_bboxes (dict): predicted bounding boxes for each image

    Returns:
        class_aps : average precision (AP) for each class
    """
    # Store the aps
    class_aps = np.zeros(len(CLASSES))
        
    # Track TP, FP, and FN for each class
    class_TPS = np.zeros(len(CLASSES))
    class_FPS = np.zeros(len(CLASSES))
    class_FNS = np.zeros(len(CLASSES))
        
    # Loop through the images
    for image_id in pred_classes.keys():
        
        # Get the true and predicted classes for this image
        true_classes_image = true_classes[image_id]
        pred_classes_image = pred_classes[image_id]
        
        # Get the true and predicted bboxes for this image
        true_bboxes_image = true_bboxes[image_id]
        pred_bboxes_image = pred_bboxes[image_id]
    
        # For each true category in the image
        for cat in true_classes_image:
            # If there are no predictions
            if(len(pred_classes_image) == 0):
                class_FNS[cat - 1] += len(true_classes_image)
                continue
            
            # Calculate number of FNS
            if(len(true_classes_image) > len(pred_classes_image)):
                class_FNS[cat - 1] += len(true_classes_image) - len(pred_classes_image)
            
            # Calculate TPS and FPS
            for j in range(len((true_bboxes_image))):
                for k in range(j, len(pred_bboxes_image)):
                    true_bbox = true_bboxes_image[j]
                    pred_bbox = pred_bboxes_image[k]
                    
                    # Calculate IoU
                    iou = IoU(true_bbox, pred_bbox)
                    
                    # If the IoU is greater than the threshold
                    if(iou > threshold):
                        # If the predicted class is the same as the true class
                        if(int(pred_classes_image[k]) == cat):
                            class_TPS[cat - 1] += 1
                        else:
                            class_FPS[cat - 1] += 1
                    else:
                        class_FPS[cat - 1] += 1

    # Calculate Precision and Recall for each class  
    for cat in range(len(CLASSES)):
        precision = np.nan_to_num(class_TPS[cat] / (class_TPS[cat] + class_FPS[cat]), nan=0.1, posinf=1, neginf=0)
        recall = np.nan_to_num(class_TPS[cat] / (class_TPS[cat] + class_FNS[cat]), nan=0.1, posinf=1, neginf=0)
        class_aps[cat] += np.nan_to_num(precision*recall, nan=0.1, posinf=1, neginf=0)
    
    # Return the mean AP for each class
    return class_aps

###############
# MAIN METHOD #
###############

if __name__ == '__main__':
    # Get the test data labels
    print("Loading the test data...")
    test_labels = extract_labels('../data/images/annotations/test.json')
    
    # Load the model
    print("Loading the model...")
    model = pickle.load(open('../models/svm_model.pkl', 'rb'))
    
    # Evaluate the model at different thresholds and calculate the AP for each class
    print("Performing object detection... This may take a long time...")
    class_aps = np.zeros(len(CLASSES))
    for threshold in np.arange(0.1, 1.0, 0.2):
        pred_classes, pred_bboxes = detection(model, val_images_dir='../data/images/test', step=36, threshold=threshold)
        class_aps += (calculate_ap(true_classes=test_labels['classification'], pred_classes=pred_classes, true_bboxes=test_labels['bbox'], pred_bboxes=pred_bboxes, threshold=threshold)) * 100
    
    # Print the results
    print("\n")
    print("Average Precision (AP) for each class:")
    for cat in CLASSES:
        print(f"{cat} AP: {class_aps[CLASSES.index(cat)] / 5:.2f}")