from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetMapper, build_detection_train_loader, transforms as T

class CustomTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        '''
        Overrides the default build_train_loader() to instantiate a custom data loader for training. 
        A list of image augmentation transformations are used, including:
            - Resize (to 800x800)
            - Random Brightness (from half to double)
            - Random Contrast (from half to double)
            - Random Horizontal Flip
            - Random Vertical Flip
        
        Args:
            cfg(detectron2.config): Detectron2 configuration object

        Returns:
            (torch.utils.data.DataLoader): custom dataloader
        '''
        mapper = DatasetMapper(
            cfg,
            is_train=True,
            augmentations=[
                T.ResizeShortestEdge(short_edge_length=(800, 800), max_size=1333, sample_style='choice'), 
                T.RandomBrightness(0.5, 2),
                T.RandomContrast(0.5, 2),
                T.RandomSaturation(0.5, 2),
                T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                T.RandomFlip(prob=0.5, horizontal=False, vertical=True)
            ],
            use_instance_mask=True,
            recompute_boxes=True
        )
        
        return build_detection_train_loader(cfg, mapper=mapper)