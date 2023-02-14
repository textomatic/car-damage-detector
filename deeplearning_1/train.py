import argparse
import os
import json
import random
from datetime import datetime
from zipfile import ZipFile
import matplotlib.pyplot as plt
import torch
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from customtrainer import CustomTrainer


# Define constants
_TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
_CUDA_VERSION = torch.__version__.split("+")[-1]
_DETECTRON_VERSION = detectron2.__version__
_IMAGE_DATA_DIR = '../data'
_IMAGE_ARCHIVE_NAME = 'images2'
_TRAIN_DATASET_NAME = 'cardamage_train'
_TOTAL_CLASSES = 6
_DEFAULT_MODEL_NAME = 'model_final'
_NUM_WORKERS = 2
_MINI_BATCH_SIZE = 4
_ROIHEAD_BATCH_SIZE = 512


def register_image():
    '''
    Decompresses dataset archive if necessary, defines paths to the training image dataset, and registers it with Detectron2.

    Args:
        None
    
    Returns:
        None
    '''
    # Define path to image files
    data_zip = os.path.join(_IMAGE_DATA_DIR, _IMAGE_ARCHIVE_NAME + '.zip')
    image_dir = os.path.join(_IMAGE_DATA_DIR, _IMAGE_ARCHIVE_NAME)
    train_dir = os.path.join(image_dir, 'train')
    annot_dir = os.path.join(image_dir, 'annotations')

    # Extract images if still in archive
    if not os.path.exists(image_dir):
        with ZipFile(data_zip, 'r') as z_object:
            z_object.extractall(_IMAGE_DATA_DIR)
    
    # Register train and test sets with Detectron2
    register_coco_instances(_TRAIN_DATASET_NAME, {}, os.path.join(annot_dir, 'train.json'), train_dir)


def visualize_image(output_dir, train_data, train_metadata, amount):
    '''
    Visualize a random sample of the images in training data. Annotations (bounding boxes and segmentation masks) are included in the images.

    Args:
        output_dir(str): Name of the directory where training artifacts and metrics will be stored in.
        train_data(torch.utils.data.Dataset): A PyTorch dataset of the training images, can be obtained by using :func:`DatasetCatalog.get`.
        train_metadata(torch.utils.data.Dataset): A PyTorch dataset of the metadata of the training images, can be obtained by using :func:`MetadataCatalog.get`.
        amount(int): the amount of random images to view.
    
    Returns:
        None
    '''
    # Draw random samples of training images and visualize with matplotlib
    for d in random.sample(train_data, amount):
        img = plt.imread(d['file_name'])
        visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=0.5) # Instantiate Detectron2 visualizer
        out = visualizer.draw_dataset_dict(d)
        img_save_path = output_dir + '/' + d['file_name'].split('/')[-1] # Indicate location to save image to
        plt.imsave(img_save_path, out.get_image()[:, :, ::-1])
        logger.info(f"Random training image has been saved to {img_save_path}.")


def view_losses(output_dir):
    '''
    Plots the training losses over all iterations.

    Args:
        output_dir(str): name of the training output directory containing `metrics.json`.

    Returns:
        None
    '''
    metrics = []
    with open(output_dir + '/metrics.json', 'r') as f:
        for line in f:
            metrics.append(json.loads(line))

    img_save_path = output_dir + '/losses.png'
    plt.plot([m['iteration'] for m in metrics], [m['total_loss'] for m in metrics])
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.savefig(img_save_path)
    logger.info(f"Training losses chart has been saved to {img_save_path}.")


def train(pretrained_model_name, train_data, output_dir, epochs=10, learning_rate=0.001):
    '''
    Loads a pretrained COCO instance segmentation model from Detectron2 and fine tunes it with your training data

    Args:
        pretrained_model_name(str): Name of the pretrained model. Example: "mask_rcnn_R_101_FPN_3x".
        train_data(torch.utils.data.Dataset): A PyTorch dataset of the training images, can be obtained by using :func:`DatasetCatalog.get`.
        output_dir(str): Name of the directory to store training artifacts and metrics in.
        epochs(int): Number of epochs to train the model for. Defaults to 10.
        learning_rate(float): Learning rate during training. Defaults to 0.001.

    Returns:
        None
    '''
    pretrained_model_config = 'COCO-InstanceSegmentation/' + pretrained_model_name + '.yaml'
    # Customizing configurations
    cfg = get_cfg() # Get default configuration
    cfg.OUTPUT_DIR = output_dir # Set output directory for this training session
    cfg.merge_from_file(model_zoo.get_config_file(pretrained_model_config)) # Merge existing configs from pretrained model
    cfg.DATASETS.TRAIN = (_TRAIN_DATASET_NAME,) # Define training dataset
    cfg.DATASETS.TEST = () # Leave test dataset empty because evaluation is carried out separately
    cfg.DATALOADER.NUM_WORKERS = _NUM_WORKERS # Number of workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(pretrained_model_config)  # Obtain weights from pretrained model
    cfg.SOLVER.IMS_PER_BATCH = _MINI_BATCH_SIZE  # Size of mini-batch
    train_size = len(train_data) # Size of training data
    iter_per_epoch = int(train_size / cfg.SOLVER.IMS_PER_BATCH) # Calculates how many mini-batches in one epoch
    cfg.SOLVER.MAX_ITER = epochs * iter_per_epoch # Defines the "total iterations" needed for training
    cfg.SOLVER.BASE_LR = learning_rate  # Learning rate
    cfg.SOLVER.STEPS = [] # Reduces LR by a gamma factor of 0.1 from specifc steps onwards
    cfg.MODEL.MASK_ON = True # Enable instance segmentation
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = _ROIHEAD_BATCH_SIZE   # Batch size of the RoIHead
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = _TOTAL_CLASSES  # Number of classes of car damage
    # Save configurations to a YAML file for ease of evaluation later on
    cfg_yaml = cfg.dump()
    cfg_yaml_savepath = os.path.join(output_dir, _DEFAULT_MODEL_NAME + '.yaml')
    with open(cfg_yaml_savepath, 'w') as file:
        file.write(cfg_yaml)
    logger.info(f"Model configuration YAML file has been saved to {cfg_yaml_savepath}.")
    # Instantiate custom trainer and initiate training
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def main(args):
    '''
    Main method which orchestrates registering the data, triggering training, and plotting sample training images and loss chart.

    Args:
        args(List(str)): command line arguments passed in.

    Returns:
        None.
    '''
    # Register image dataset
    register_image()

    # Obtain train dataset and metadataset by name
    train_data = DatasetCatalog.get(_TRAIN_DATASET_NAME)
    train_metadata = MetadataCatalog.get(_TRAIN_DATASET_NAME)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=False)

    # Visualize sample images from train set
    if args.visualize_train:
        visualize_image(args.output_dir, train_data, train_metadata, args.visualize_train_amount)

    # Initialize training
    train(args.pretrained_model_name, train_data, args.output_dir, args.epochs, args.learning_rate)

    # View training losses
    view_losses(args.output_dir)

    # Indicate end of training
    logger.info('Training has completed. Please run evaluation to verify.')


if __name__ == '__main__':

    # Display PyTorch and Detectron2 versions as sanity checks
    print("torch: ", _TORCH_VERSION, "; cuda: ", _CUDA_VERSION)
    print("detectron2: ", _DETECTRON_VERSION)
    logger = setup_logger()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog="Training script for car damage detection dataset using Detectron2",
        epilog="Example usage: train.py --pretrained-model-name mask_rcnn_R_101_FPN_3x --epochs 5 --learning-rate 0.0025 --visualize-train --visualize-train-amount 5"
    )
    parser.add_argument("--pretrained-model-name", default="mask_rcnn_R_101_FPN_3x", help="Name of the pretrained model in Detectron2. Defaults to Mask R-CNN ResNet101-FPN 3x.")
    parser.add_argument("--output-dir", default='experiment_' + datetime.now().strftime('%Y%m%dT%H%M%S'), help="Name of the directory to store training outputs in. Directory will be created if does not exist yet.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for. Defaults to 10.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate during training. Defaults to 0.001.")
    parser.add_argument("--visualize-train", default=False, action="store_true", help="Visualize a sample of training data with annotations.")
    parser.add_argument("--visualize-train-amount", type=int, default=3, help="Number of sample training images to view. Defaults to 3.")
    args = parser.parse_args()
    print("Command Line Args: ", args)

    # Pass command line arguments to main function
    main(args)