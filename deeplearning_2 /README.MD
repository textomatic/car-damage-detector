# Setup

1. Change directory to this folder:
```bash
cd car/damage-detection/deeplearning_2
```

2. Create a new Python 3.8 environment in Conda:
```bash
conda create -n deeplearning2 python=3.8
```

3. Activate the Conda environment:
```bash
conda activate deeplearning2
```

4. Insatll YOLOv5 model  
```bash 
git clone https://github.com/ultralytics/yolov5.git -b v7.0
```

5. Install requirements :
```bash
conda install -r requirements.txt
```

# DataProcessing 
```bash
├── CarD(Ori_data)
│         ├── annotations  (json files )   
│         └── train2017    (TrainImage)
│         └── test2017     (TestImage)
│         └── val2017      (ValImage)
```
```bash
run python dataprocessing.py #Generate YOLOv5dataset 
```
```bash
├── CarSeg_Data (YOLOv5 )
│         ├── images   
│         │     ├── train (TrainImage)
│         │     └── val   (ValImage)
│         └── labels    
│               ├── train  (TrainTxt)
│               └── val    (ValTxt)  
```

# Training
```bash
cd YOLOv5
python segment/train.py --img 640 --batch 16 --epochs 500 --data ./deeplearning/car.yaml --weights yolov5s-seg.pt 
```

# Predict 
```bash 
cd YOLOv5
python segment/predict.py --weights ./deeplearning/weight/best.pt --img 640 --conf 0.25 --source data/images
```

# YOLOv5-TrainResult
```
├── TrainResult.csv
```
