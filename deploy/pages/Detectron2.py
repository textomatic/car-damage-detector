import os
import requests
import streamlit as st
from PIL import Image
import cv2
import torch
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.logger import setup_logger


# Define constants
_TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
_CUDA_VERSION = torch.__version__.split("+")[-1]
_DETECTRON_VERSION = detectron2.__version__
_MODEL_CACHE = './model_cache_detectron'
_MODEL_1 = (
    'maskrcnn_r101_fpn_3x',
    'https://duke.box.com/shared/static/wajctmp0twllhskpgekm539rsy7je1za.yaml',
    'https://duke.box.com/shared/static/5isq1mrfoojvrr0n0ty8xkkht7ftkntg.pth'
)
_MODEL_2 = (
    'maskrcnn_r101_dc5_3x',
    'https://duke.box.com/shared/static/y760ykf8g8cl7yjlaiq2nsj1y2dtg0j1.yaml',
    'https://duke.box.com/shared/static/g76bawaig4zwtfty3yk3715tzaovvkin.pth'
)


class CarDamageMetadata:
    '''
    Metadata for the car damage dataset. Primary method is to return the class labels.
    '''

    def get(self, _):
        '''
        Returns the class labels of the dataset.

        Args:
            None
        
        Returns:
            List(str): list of class labels.
        '''
        return ['dent','scratch','crack','glass shatter', 'lamp broken', 'tire flat'] # Class labels


class DetectronModel:
    '''
    Helper class to load a Detectron2 model.
    '''

    def __init__(self, model_choice, threshold=0.5):
        self.cfg = get_cfg() # Get default configuration
        cfg_yaml_loadpath = os.path.join(_MODEL_CACHE, model_choice + '.yaml')
        self.cfg.merge_from_file(cfg_yaml_loadpath) # Load configuration from trained model
        logger.info(f"Model configuration YAML file has been loaded from {cfg_yaml_loadpath}.")
        self.cfg.MODEL.WEIGHTS = os.path.join(_MODEL_CACHE, model_choice + '.pth') # Load weights of trained model
        logger.info(f"Model weights have been loaded from {os.path.join(_MODEL_CACHE, model_choice + '.pth')}.")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold # Set custom test threshold
        self.cfg.MODEL.DEVICE = 'cpu' # Inference is done on CPU
        self.predictor = DefaultPredictor(self.cfg)

    def predict(self, image_path):
        '''
        Generates prediction of bounding box and instance segmentation given an image.

        Args:
            image_path(str): String containing path to the target image.

        Returns:
            None.
        '''

        image = cv2.imread(image_path) # Read in image
        predictions = self.predictor(image) # Obtain output from predictor
        viz = Visualizer(image[:, :, ::-1], metadata=CarDamageMetadata, scale=1.0, instance_mode=ColorMode.IMAGE_BW) # Instantiate visualizer for viewing annotated images
        output = viz.draw_instance_predictions(predictions['instances'].to('cpu')) # Annotate prediction on output image
        result = 'detectron_result.jpg'
        cv2.imwrite(result, output.get_image()[:,:,::-1]) # Save output image


if __name__ == '__main__':
    st.markdown('# Car Damage Detector - Detectron2 🤖')

    # Display PyTorch and Detectron2 versions as sanity checks
    print('torch: ', _TORCH_VERSION, '; cuda: ', _CUDA_VERSION)
    print("detectron2: ", _DETECTRON_VERSION)
    logger = setup_logger()

    # Download the models if they don't exist locally yet
    if not os.path.exists(_MODEL_CACHE):
        with st.spinner('Model(s) not found locally. Downloading from remote...'):
            os.makedirs(_MODEL_CACHE)
            for model in [_MODEL_1, _MODEL_2]:
                response_cfg = requests.get(model[1], allow_redirects=True)
                open(os.path.join(_MODEL_CACHE, model[0] + '.yaml'), 'wb').write(response_cfg.content)
                response_weights = requests.get(model[2], allow_redirects=True)
                open(os.path.join(_MODEL_CACHE, model[0] + '.pth'), 'wb').write(response_weights.content)

    # Dropdown box to select a model
    model_choice = st.selectbox('Pick a model', [_MODEL_1[0], _MODEL_2[0]])
    # Slider to select an inference threshold
    confidence_threshold = st.slider('Pick a confidence threshold', 0.0, 1.0, 0.1)
    # Load a custom trained Detectron2 model
    detector = DetectronModel(model_choice, threshold=confidence_threshold)
    
    # Image uploader
    image_file = st.file_uploader("Upload an Image", type=['jpeg', 'jpg', 'png'])

    if image_file is not None:
        # Show basic meta data of image
        file_details = {"FileName": image_file.name, "FileType": image_file.type}
        st.write(file_details)
        # Display uploaded image
        img = Image.open(image_file)
        st.image(img, caption='Uploaded Image')
        # Save uploaded image to enable inference by model
        with open(image_file.name, mode = 'wb') as f: 
            f.write(image_file.getbuffer())         
        st.success('File uploaded successfully!')
        
        # Generate prediction on image
        with st.spinner('Performing inference on image...'):
            detector.predict(image_file.name)

        # Display prediction on image
        img_result = Image.open('detectron_result.jpg')
        st.image(img_result, caption='Processed Image')