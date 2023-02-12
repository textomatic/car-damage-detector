import os
import random
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer, ColorMode

cfg = get_cfg()
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
predictor = DefaultPredictor(cfg)


evaluator = COCOEvaluator("cardamage_test", output_dir=cfg.OUTPUT_DIR)
test_loader = build_detection_test_loader(cfg, "cardamage_test")
print(inference_on_dataset(predictor.model, test_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`


test_data = DatasetCatalog.get("cardamage_test")
test_metadata = MetadataCatalog.get("cardamage_test")
for d in random.sample(test_data, 3):    
    im = plt.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v1 = Visualizer(im[:, :, ::-1],
                   metadata=test_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    v2 = Visualizer(im[:, :, ::-1], metadata=test_metadata, scale=0.5)
    out_pred = v1.draw_instance_predictions(outputs["instances"].to("cpu"))
    out_gt = v2.draw_dataset_dict(d)
    # print(outputs["instances"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(out_pred.get_image()[:, :, ::-1])
    ax[0].set_title('Prediction')
    ax[1].imshow(out_gt.get_image()[:, :, ::-1])
    ax[1].set_title('Ground Truth')
    plt.show()

