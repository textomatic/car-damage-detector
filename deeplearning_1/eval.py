import argparse
import os
from zipfile import ZipFile
import random
import matplotlib.pyplot as plt
import torch
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.logger import setup_logger


# Define constants
_TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
_CUDA_VERSION = torch.__version__.split("+")[-1]
_DETECTRON_VERSION = detectron2.__version__
_IMAGE_DATA_DIR = '../data'
_IMAGE_ARCHIVE_NAME = 'images2'
_TEST_DATASET_NAME = 'cardamage_test'
_DEFAULT_MODEL_NAME = 'model_final'


def register_image():
    '''
    Decompresses dataset archive if necessary, defines paths to the test image dataset, and registers it with Detectron2.

    Args:
        None
    
    Returns:
        None
    '''
    # Define path to image files
    data_zip = os.path.join(_IMAGE_DATA_DIR, _IMAGE_ARCHIVE_NAME + '.zip')
    image_dir = os.path.join(_IMAGE_DATA_DIR, _IMAGE_ARCHIVE_NAME)
    test_dir = os.path.join(image_dir, 'test')
    annot_dir = os.path.join(image_dir, 'annotations')

    # Extract images if still in archive
    if not os.path.exists(image_dir):
        with ZipFile(data_zip, 'r') as z_object:
            z_object.extractall(_IMAGE_DATA_DIR)
    
    # Register test set with Detectron2
    register_coco_instances(_TEST_DATASET_NAME, {}, os.path.join(annot_dir, 'test.json'), test_dir)


def get_predictor(model_dir, threshold):
    '''
    Decompresses dataset archive if necessary, defines paths to the test image dataset, and registers it with Detectron2.

    Args:
        model_dir(str): Name of the directory where the trained model is stored in.
        threshold(int): Threshold used to filter out low-scored, predicted bounding boxes.
    
    Returns:
        None
    '''
    cfg = get_cfg() # Obtain default configurations
    cfg_yaml_loadpath = os.path.join(model_dir, _DEFAULT_MODEL_NAME + '.yaml')
    cfg.merge_from_file(cfg_yaml_loadpath)
    logger.info(f"Model configuration YAML file has been loaded from {cfg_yaml_loadpath}.")
    cfg.MODEL.WEIGHTS = os.path.join(model_dir, _DEFAULT_MODEL_NAME + '.pth') # Path to the trained model weights
    logger.info(f"Model weights have been loaded from {os.path.join(model_dir, _DEFAULT_MODEL_NAME + '.pth')}.")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold # Set a custom testing threshold
    predictor = DefaultPredictor(cfg) # Instantiate predictor object
    return predictor, cfg


def eval_model(model_dir, predictor, cfg):
    '''
    Evaluates a trained Detectron2 model with a COCO-style evaluator.

    Args:
        model_dir(str): Name of the directory where the trained model is stored in.
        predictor(detectron2.engine.DefaultPredictor): A Detectron2 predictor object for generating predictions.
        cfg(detectron2.config.CfgNode): A Detectron2 configuration object containing the configurations for the trained model.
    
    Returns:
        None
    '''
    evaluator = COCOEvaluator(_TEST_DATASET_NAME, output_dir=model_dir) # Instantiate evaluator object
    test_loader = build_detection_test_loader(cfg, _TEST_DATASET_NAME) # Create test loader with custom test dataset
    logger.info(inference_on_dataset(predictor.model, test_loader, evaluator)) # Print evaluation results


def visualize_preds(model_dir, test_data, test_metadata, predictor, amount):
    '''
    Visualize a random sample of the images in test data. Predicted annotations (bounding boxes and segmentation masks) are included in the images.

    Args:
        model_dir(str): Name of the directory where the trained model is stored in.
        test_data(torch.utils.data.Dataset): A PyTorch dataset of the test images, can be obtained by using :func:`DatasetCatalog.get`.
        test_metadata(torch.utils.data.Dataset): A PyTorch dataset of the metadata of the test images, can be obtained by using :func:`MetadataCatalog.get`.
        predictor(detectron2.engine.DefaultPredictor): A Detectron2 predictor object for generating predictions.
        amount(int): The amount of random images to generate predictions on.
    
    Returns:
        None
    '''
    for d in random.sample(test_data, amount):    
        im = plt.imread(d["file_name"]) # Retrieve image from test set
        outputs = predictor(im)  # Generate predictions from model

        # Instantiate Detectron2 visualizers 
        v1 = Visualizer(im[:, :, ::-1], metadata=test_metadata, scale=0.5, instance_mode=ColorMode.IMAGE_BW)  # ColorMode.IMAGE_BW removes the colors of unsegmented pixels
        v2 = Visualizer(im[:, :, ::-1], metadata=test_metadata, scale=0.5) # Second visualizer to visualize ground truth annotations
        out_pred = v1.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_gt = v2.draw_dataset_dict(d)

        img_save_path = model_dir + '/pred_' + d['file_name'].split('/')[-1] # Indicate location to save image to
        fig, ax = plt.subplots(1, 2, figsize=(10, 10)) # Generate subplots with prediction and ground-truth side-by-side
        ax[0].imshow(out_pred.get_image()[:, :, ::-1])
        ax[0].set_title('Prediction')
        ax[1].imshow(out_gt.get_image()[:, :, ::-1])
        ax[1].set_title('Ground Truth')
        fig.savefig(img_save_path)

        logger.info(f"Random test image prediction and ground truth have been saved to {img_save_path}.")


def main(args):
    '''
    Main method which orchestrates registering the data, triggering evaluation, and plotting sample test images.

    Args:
        args(List(str)): command line arguments passed in.

    Returns:
        None.
    '''
    # Register image dataset
    register_image()

    # Obtain test dataset and metadataset by name
    test_data = DatasetCatalog.get(_TEST_DATASET_NAME)
    test_metadata = MetadataCatalog.get(_TEST_DATASET_NAME)

    # Obtain predictor
    predictor, cfg = get_predictor(args.model_dir, args.threshold)

    # Initialize evaluation
    eval_model(args.model_dir, predictor, cfg)

    # Visualize sample images from train set
    if args.visualize_preds:
        visualize_preds(args.model_dir, test_data, test_metadata, predictor, args.visualize_preds_amount)

    # Indicate end of evaluation
    logger.info('Evaluation has completed. Please check prediction images if necessary.')


if __name__ == '__main__':

    # Display PyTorch and Detectron2 versions as sanity checks
    print("torch: ", _TORCH_VERSION, "; cuda: ", _CUDA_VERSION)
    print("detectron2: ", _DETECTRON_VERSION)
    logger = setup_logger()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog="Evaluation script for car damage detection dataset using Detectron2",
        epilog="Example usage: eval.py --model-dir ./experiment_20230213T040136 --threshold 0.5 --visualize-preds --visualize-preds-amount 5"
    )
    parser.add_argument("--model-dir", default='./output', help="Name of the directory where the trained model's weights are stored in.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold used to filter out low-scored, predicted bounding boxes. Defaults to 0.5.")
    parser.add_argument("--visualize-preds", default=False, action="store_true", help="Visualize a sample of predictions with annotations.")
    parser.add_argument("--visualize-preds-amount", type=int, default=3, help="Number of predicted images to view. Defaults to 3.")
    args = parser.parse_args()
    print("Command Line Args: ", args)

    # Pass command line arguments to main function
    main(args)