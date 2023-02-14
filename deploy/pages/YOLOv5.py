import requests
import os
from PIL import Image
import streamlit as st
import torch
import subprocess
import sys


# Define constants
_TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
_CUDA_VERSION = torch.__version__.split("+")[-1]
_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
_PARENT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
_REPO_CACHE = os.path.join(_CURRENT_DIR, 'repo_cache_yolo/yolov5')
_MODEL_CACHE = os.path.join(_CURRENT_DIR, 'model_cache_yolo')
_MODEL_NAME = 'yolov5s-seg-best.pt'
_MODEL_WEIGHTS = 'https://duke.box.com/shared/static/dr5nir98p7pbt0bijp6mlfd2rul3g8g7.pt'


if __name__ == '__main__':
    st.markdown('# Car Damage Detector - YOLOv5 👀')

    # Display PyTorch version as sanity check
    print('torch: ', _TORCH_VERSION, '; cuda: ', _CUDA_VERSION)

    # Download the models if they don't exist locally yet
    if not os.path.exists(_MODEL_CACHE):
        with st.spinner('Model(s) not found locally. Downloading from remote...'):
            os.makedirs(_MODEL_CACHE)
            response_weights = requests.get(_MODEL_WEIGHTS, allow_redirects=True)
            open(os.path.join(_MODEL_CACHE, _MODEL_NAME), 'wb').write(response_weights.content)
            # Clone git repo of YOLOv5 for the detection scripts
            subprocess.call(f'git clone https://github.com/ultralytics/yolov5.git {_REPO_CACHE}', shell=True)

    # Dropdown box to select a model
    model_choice = st.selectbox('Pick a model', [_MODEL_NAME])
    # Slider to select an inference threshold
    confidence_threshold = st.slider('Pick a confidence threshold', 0.0, 1.0, 0.7, 0.1)
    # Load a custom trained YOLOv5 model
    detector = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join(_MODEL_CACHE, _MODEL_NAME))
    detector.conf = confidence_threshold
    detector.cpu()
        
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
            # Execute detection script to get inference of bounding box and segmentation mask
            subprocess.run([f"{sys.executable}", f"{_REPO_CACHE}/segment/predict.py", "--weights", _MODEL_CACHE + "/" + _MODEL_NAME, "--conf", str(confidence_threshold), "--source", image_file.name, "--exist-ok"])

        # Display prediction on image
        pred_loc = os.path.join(_REPO_CACHE + '/runs/predict-seg/exp', image_file.name)
        img_result = Image.open(pred_loc)
        st.image(img_result, caption='Processed Image')