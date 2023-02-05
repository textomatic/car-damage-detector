# Imports
import os
import json
import cv2
import pandas as pd
from torch.utils.data import Dataset

# Define the custom dataset
class CarDD_Dataset(Dataset):
    def __init__(self, data_dir: str, labels_path: str, transform=None):
        """
        Args:
            data_dir (str): Path to the folder containing the images
            labels_path (str): path to the annotation json file with the label information
            transform (callable, optional): Optional transform to be applied to images
        """
        # Load the JSON file
        with open(labels_path) as label_data:
            annot_df = json.load(label_data)
            
        # Extract the mapping of intergers to classes
        self.class_map = {cat["id"] : cat["name"] for cat in annot_df["categories"]}
        
        # Extract labels, bounding boxes, and pixel segmentation for each image
        self.image_labels = [image["category_id"]for image in annot_df["annotations"]]
        self.image_bboxes = [image["bbox"]for image in annot_df["annotations"]]
        self.image_segmentations = [image["segmentation"]for image in annot_df["annotations"]]
        
        # Store image filenames for the __get-tem__ method
        self.image_files = [image["file_name"] for image in annot_df["images"]]
        
        # Store data for later
        self.data_dir = data_dir
        self.transform = transform
                    
    def __len__(self):
        """Returns the number of images in the dataset"""
        return len(self.labels)
        
    def __getitem__(self, idx):
        """
        Returns the image and corresponding label, bbox, and segmentation for the image located
        at index idx within the dataset

        Args:
            idx (int): index value for which to retrieve the image and label
        """
        # Load the image
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        image = cv2.imread(image_path)
        
        # Apply transforms to the image if necessary
        if self.transform is not None:
            image = self.transform(image)
        
        # Load the target data
        label = self.image_labels[idx]
        bbox = self.image_bboxes[idx]
        segmentation = self.image_segmentations[idx]
        
        # Return the image and label
        return image, label, bbox, segmentation
        
    
if __name__ == "__main__":
    x = CarDD_Dataset(data_dir="data/CarDD/train2017", labels_path="data/CarDD/annotations/instances_train2017.json")