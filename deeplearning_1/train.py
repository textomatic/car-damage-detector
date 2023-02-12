import os
import json
import random
from zipfile import ZipFile
import matplotlib.pyplot as plt
import torch
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
setup_logger()

from customtrainer import CustomTrainer

# GLOBAL CONSTANTS
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
IMAGE_ARCHIVE_NAME = 'images2'
TRAIN_DATASET_NAME = 'cardamage_train'
TEST_DATASET_NAME = 'cardamage_test'


def register_image():
    '''
    Decompresses dataset archive if necessary, defines paths to the training and test image datasets, and registers them with Detectron2.

    Args:
        None
    
    Returns:
        None
    '''
    # Define path to image files
    data_dir = '../data'
    data_zip = os.path.join(data_dir, IMAGE_ARCHIVE_NAME + '.zip')
    image_dir = os.path.join(data_dir, IMAGE_ARCHIVE_NAME)
    train_dir = os.path.join(image_dir, 'train')
    test_dir = os.path.join(image_dir, 'test')
    annot_dir = os.path.join(image_dir, 'annotations')

    # Extract images if still in archive
    if not os.path.exists(image_dir):
        with ZipFile(data_zip, 'r') as z_object:
            z_object.extractall(data_dir)
    
    # Register train and test sets with Detectron2
    register_coco_instances(TRAIN_DATASET_NAME, {}, os.path.join(annot_dir, 'train.json'), train_dir)
    register_coco_instances(TEST_DATASET_NAME, {}, os.path.join(annot_dir, 'test.json'), test_dir)


def visualize_image(train_data, train_metadata, amount=3):
    '''
    Visualize a random sample of the images in training data. Annotations (bounding boxes and segmentation masks) are included in the images.

    Args:
        train_data(torch.utils.data.Dataset): A PyTorch dataset of the training images, can be obtained by using :func:`DatasetCatalog.get`.
        train_metadata(torch.utils.data.Dataset): A PyTorch dataset of the metadata of the training images, can be obtained by using :func:`MetadataCatalog.get`.
        amount(int): the amount of random images to view. Defaults to 3.
    
    Returns:
        None
    '''
    # Draw random samples of training images and visualize with matplotlib
    for d in random.sample(train_data, amount):
        img = plt.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.show()


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

    plt.plot([m['iteration'] for m in metrics], [m['total_loss'] for m in metrics])
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.show()


def train(pretrained_model_name, train_data, output_dir, epochs=10):
    '''
    
    '''
    pretrained_model_config = 'COCO-InstanceSegmentation' + pretrained_model_name + '.yaml'
    cfg = get_cfg() # Get default configuration
    cfg.OUTPUT_DIR = output_dir # Set output directory for this training session
    cfg.merge_from_file(model_zoo.get_config_file(pretrained_model_config)) # Merge existing configs from pretrained model
    cfg.DATASETS.TRAIN = (TRAIN_DATASET_NAME,) # Define training dataset
    cfg.DATASETS.TEST = () # Leave test dataset empty because evaluation is carried out separately
    cfg.DATALOADER.NUM_WORKERS = 2 # Number of workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(pretrained_model_config)  # Obtain weights from pretrained model
    cfg.SOLVER.IMS_PER_BATCH = 4  # Size of mini-batch
    train_size = len(train_data) # Size of training data
    iter_per_epoch = int(train_size / cfg.SOLVER.IMS_PER_BATCH) # Calculates how many mini-batches in one epoch
    cfg.SOLVER.MAX_ITER = epochs * iter_per_epoch # Defines the "total iterations" needed for training
    cfg.SOLVER.BASE_LR = 0.0025  # Learning rate
    cfg.SOLVER.STEPS = [3000, 6000] # Reduces LR by a gamma factor of 0.1 from Steps 3000 and 6000
    cfg.MODEL.MASK_ON = True # Enable instance segmentation
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # Batch size of the RoIHead
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # Number of classes of car damage

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=False)
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def main():
    # Register image dataset
    register_image()

    # Obtain dataset and metadataset by name
    train_data = DatasetCatalog.get(TRAIN_DATASET_NAME)
    train_metadata = MetadataCatalog.get(TRAIN_DATASET_NAME)

    # Visualize sample images from train set
    visualize_image(train_data, train_metadata)

    # Initialize training
    output_dir = 'experiment_1'
    train(train_data, output_dir, 10)

    # View training losses
    view_losses(output_dir)


if __name__ == '__main__':
    # Display PyTorch and Detectron2 versions as sanity checks
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
    print("detectron2:", detectron2.__version__)

    main()