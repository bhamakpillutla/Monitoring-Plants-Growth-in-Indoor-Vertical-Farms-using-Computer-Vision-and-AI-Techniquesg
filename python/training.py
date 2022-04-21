#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 19:25:36 2022

@author: Bhama Pillutla

Model training 

"""

from detectron2.engine import DefaultTrainer,DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt
import cv2
import os
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

import random
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer 
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
cfg = get_cfg()
cfg.merge_from_file("/home/psych256lab/Downloads/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.DATASETS.TRAIN = ("august_train",)
cfg.DATASETS.TEST = ("august_val",)
cfg.DATALOADER.NUM_WORKERS = 2

## already trained model
#cfg.MODEL.WEIGHTS = '/home/psych256lab/Downloads/Scripts/model_final.pth' # Set path model .pth

## fresh model
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # # Let training initialize from model zoo
#cfg.OUTPUT_DIR = "home/psych256lab/Downloads/Scripts/models/"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02  # pick a good LR
cfg.SOLVER.MAX_ITER = 2000 # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.SOLVER.CHECKPOINT_PERIOD = 500

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False) ## resume = True for resuming training from last checkpoint
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

my_dataset_test_metadata = MetadataCatalog.get("august_train")
from detectron2.utils.visualizer import ColorMode
dataset_dicts = DatasetCatalog.get("august_val")
for d in random.sample(dataset_dicts, 5):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=my_dataset_test_metadata, 
                   scale=0.5, 
#                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow(out.get_image()[:, :, ::-1])

#import the COCO Evaluator to use the COCO Metrics
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

#Call the COCO Evaluator function and pass the Validation Dataset
evaluator = COCOEvaluator("august_val", cfg, False, output_dir="./output/")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") 
val_loader = build_detection_test_loader(cfg, "august_val")

#Use the created predicted model in the previous step
inference_on_dataset(trainer.model, val_loader, evaluator)

"""
resume_dir = "/content/output" # get the last resume checkpoint file path
trainer_2 = DefaultTrainer(cfg) 
trainer_2.resume_or_load(resume= True)# pass the resume_dir path here 
trainer_2.train()
## torch load and save checkpoint

#DetectionCheckpointer(model).load(file_path_or_url)  
#checkpointer = DetectionCheckpointer(model, save_dir="output")
#checkpointer.save("model_999")  # save to output/model_999.pth
"""

