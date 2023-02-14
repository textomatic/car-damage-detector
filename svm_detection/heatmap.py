###############################################################################################
# Filename: heatmap.py                                                                        #
#                                                                                             #
# Descripion: This file contains methods for tracking what bounding boxes                     #
# have strong predictions and ensuring we do not have multiple bounding                       #
# boxes for the same object.                                                                  #
#                                                                                             #
# Resources: This code is adapted from the following resource:                                #
#     - https://www.kaggle.com/code/mehmetlaudatekman/support-vector-machine-object-detection #
###############################################################################################

# Imports
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Heatmap():
    
    def __init__(self,original_image):
        """Constructor for the Heatmap class. Initializes a mask for the heatmap
        and the associated labels that will be updated later.

        Args:
            original_image (np.ndarray): Image that the mask will represent
        """
        # Mask attribute is the heatmap initialized with zeros
        self.mask = np.zeros(original_image.shape[:2])
        self.labels = np.zeros(original_image.shape[:2])
    
    # Increase value of region function will add some heat to heatmap
    def incValOfReg(self,coords, label):
        """Increases the value of the mask in the given coordinates, 
        making it more likely a bounding box will be drawn here. It also stores the 
        label associated with the bounding box in the labels attribute.
        
        Args:
            coords (tuple): coordiantes of the bounding box
            label (int): label associated with the bounding box
        """
        w1,w2,h1,h2 = coords
        self.mask[h1:h2,w1:w2] = self.mask[h1:h2,w1:w2] + 30
        self.labels[h1:h2, w1:w2] = label
    
    def decValOfReg(self,coords):
        """Decrease the vale of the region in the mask, making
        it less likely a bounding box will be drawn here
        Args:
            coords (_type_): _description_
        """
        w1,w2,h1,h2 = coords
        self.mask[h1:h2,w1:w2] = self.mask[h1:h2,w1:w2] - 30
    
    def compileHeatmap(self):
        """Apply a threshold to the mask to determine which bounding boxes are 
        the most likely to be accurate and return the mask representing those boxes

        Returns:
            self.mask (np.ndarray): a mask represneting where the predicted bounding boxes are on the image
        """
        # As you know,pixel values must be between 0 and 255 (uint8)
        # Now we'll scale our values between 0 and 255 and convert it to uint8
        
        # Scaling between 0 and 1 
        scaler = MinMaxScaler()
        
        self.mask = scaler.fit_transform(self.mask)
        
        
        # Scaling between 0 and 255
        self.mask = np.asarray(self.mask * 255).astype(np.uint8)
        
        # Now we'll threshold our mask, if a value is higher than 170, it will be white else
        # it will be black
        self.mask = cv2.inRange(self.mask,170,255)
        
        return self.mask