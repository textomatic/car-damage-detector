# Description
This directory contains all the scripts required to train and evaluate an instance segmentation model using the Detectron2 library. A prediction refers to a bounding box around the damage, a segmentation mask indicating the region of damage, and the type of damage to the car. The effectiveness of these predictions is measured through the average precision (AP) for each class. Separate APs are reported for the bounding boxes and segmentation masks. You can find these in the results section.

# Setup
1. Change directory to this folder:
```bash
cd car-damage-detector/deeplearning_1
```

2. Create a new Python 3.8 environment in Conda:
```bash
conda create -n deeplearning1 python=3.8
```

3. Activate the Conda environment:
```bash
conda activate deeplearning1
```

4. Install from requirements.txt:
```bash
pip install -r requirements.txt
```

# Data
Once you have downloaded the dataset, decompress it and place it in the `deeplearning_1/data` directory.

# Training
Execute the Python script `/deeplearning_1/train.py` with your preferred options. For example:
```bash
python train.py --pretrained-model-name mask_rcnn_R_101_FPN_3x --epochs 5 --learning-rate 0.0025 --visualize-train --visualize-train-amount 5"
```

You may specify a custom output directory to store the training results and model artifacts in. Simply pass in a directory string or path to the script using the `--output-dir` option. The default output directory is a newly created directory named as `experiment_<current_timestamp>`.

The list of available pretrained models can be found in the [Detectron2 model zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md). Simply supply the name of your preferred instance segmentation model as an argument to the script.

You may also adjust the learning rate and number of epochs to train the model for. By default, a weight decay of `0.0001` and a momentum of `0.9` are applied during training. If you would like to change them to different values, you may edit the `train.py` script to specify them as additional configurations. And by default, stepped learning rates are not enabled. If you are interested in enabling that, you may edit the script to specify the milestone steps that the learning rate should be adjusted at.

The `visualize-train` option allows you to view some sample images randomly selected from the training dataset with bounding boxes and segmentation masks annotated on them. The  `visualize-train-amount` option lets you specify how many random samples you would like to view. The images will be saved in the same directory as `--output-dir`.

At the end of the training, a chart plotting the loss over total iterations would be saved to the same working directory. You may view it to see how training improves over time.

You may enter `python train.py --help` in a shell terminal to view descriptions of each argument option and what the default values are.

# Evaluation
Upon completion of training, you may run the Python script `/deeplearning_1/eval.py` with your preferred options to evaluate the trained model. For example:
```bash
python eval.py --model-dir ./experiment_20230213T040136 --threshold 0.5 --visualize-preds --visualize-preds-amount 5
```

You should tell the script where the output directory of your training is so that the model weights and configuration file could be properly located. Specify the location by passing in a string value to the `--model-dir` option.

The `--threshold` option allows you to specify a custom threshold for filtering out low-confidence bounding boxes predicted by the Region of Interest (ROI) head of the model during inference time. Predictions with a confidence score above the threshold value is kept, and the remaining are discarded.

The `visualize-preds` option allows you to view some sample images randomly selected from the test dataset with predictions of bounding boxes and segmentation masks annotated on them. The  `visualize-preds-amount` option lets you specify how many random test samples you would like to view. The images will be saved in the same directory as `--model-dir`.

# Results
We trained two different models from the model zoo, namely `Mask R-CNN ResNet-101 FPN 3x` and `Mask R-CNN ResNet-101 DC5 3x`. The training was done for 24 epochs with an initial learning rate of 0.01. Stepped learning rate was used to scale down the LR by a factor of 10 from epoch 17 and again from epoch 23. No changes were made to the default weight decay factor of 0.0001 and momentum of 0.9. A number of image augmentation techniques were applied to the training data loader, including resizing shortest edges, random brightness, random contrast, random saturation, random horizontal flip, and random vertical flip. Random cropping was not performed as we were wary of accidental omission of damage features which could possibly be important to model training. 

The evaluation results of the trained models were as follows:

### Mask R-CNN ResNet-101 FPN 3x

Category AP: Bounding Box / Segmentation Mask
|Dent AP|Scratch AP|Crack AP|Glass Shattered AP|Lamp broken AP|Tire Flat AP|
|---|---|---|---|---|---|
28.4/28.3|34.4/28.4|26.4/13.0|88.5/89.0|56.4/58.0|84.5/85.7|

mAP Bounding Box: `53.1`

maP Segmentation Mask: `50.4`

### Mask R-CNN ResNet-101 DC5 3x

Category AP: Bounding Box / Segmentation Mask
|Dent AP|Scratch AP|Crack AP|Glass Shattered AP|Lamp broken AP|Tire Flat AP|
|---|---|---|---|---|---|
30.9/30.5|35.5/30.3|26.2/10.6|88.9/88.7|62.3/65.2|85.7/89.0|

mAP Bounding Box: `54.9`

maP Segmentation Mask: `52.4`