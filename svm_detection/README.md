# Description

This directory contains all the scripts required to train and evaluate an SVM classifier on the car damage detection dataset. A prediction consists of a bounding box around the damage and the type of damage to the car. The effectiveness of these predictions is measured through the average precision (AP) for each class. You can find these in the results section.

# Setup

Run the following commands to create and instantiate an environment for running the scripts for SVM object detection

1. Change directory to this folder:
```bash
cd car-damage-detection/svm_detection
```

2. Create a new Python 3.8 environment in Conda:
```bash
conda create -n svm_detection python=3.8
```

3. Activate the Conda environment:
```bash
conda activate svm_detection
```

4. Insatll the requirements
```bash 
pip install -r requirements.txt
```

# Training the model
An SVC model from the sklearn library is trained for this object detection path. To train the model, run the following command:

```bash
python svm_training.py
```

The `svm_training.py` script has two functions: load the data and train the model. As a warning, the training process can take a little while to run depending on your system. Afterwards, the model is saved in `models/SVM_MODEL.pkl`. 

# Evaluating the model
Once the model is trained, it can be evaluated on the test data. This is accomplished with the `svm_detection.py` script. To run the script, run the following command in the terminal:

```bash
python svm_detection.py
```

The result will be the bounding box AP for each of the six classes: dent, scratch, crack, glass shatter, lamp broken, and tire flat. 

# Results

The bounding box AP for each class is reported below:

|Dent BB AP|Scratch BB AP|Crack BB AP|Glass Shattered BB AP|Lamp broken BB AP|Tire Flat BB AP|
|---|---|---|---|---|---|
|23.05|26.36|N/A|20.46|19.87|20|

These results are not nearly as good as the deep learning techniques used, and for that reason we do not have the segmentation AP. Since the bounding box AP was so much worse, we did not feel it was worth it to explore any further.

