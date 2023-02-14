import streamlit as st
import os
import warnings
warnings.filterwarnings('ignore')

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    st.set_page_config(page_title='Car Damage Detector')
    st.title('Car Damage Detector')
    st.subheader('This web app demonstrates the detection of damage(s) on the exterior of a car using object detection and instance segmentation deep learning models.')
    st.markdown("### Detectron2")
    st.write("Training was done on the Mask R-CNN ResNet-101 FPN 3x and Mask R-CNN ResNet-101 DC5 3x instance segmentation models available from the model zoo.")
    st.markdown("### YOLOv5")
    st.write("Training was done on the YOLOv5s-seg instance segmentation model available from the repository.")
    st.markdown("### Types of Damage")
    st.write("Dent, Scratch, Crack, Shattered Glass, Broken Lamp, Flat Tire")
    st.markdown("### Examples")
    st.image(_CURRENT_DIR + '/images/example1.png', caption='Example of scratch detection')
    st.image(_CURRENT_DIR + '/images/example2.png', caption='Example of shattered glass detection')
    st.image(_CURRENT_DIR + '/images/example3.jpg', caption='Example of scratch detection')