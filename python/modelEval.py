#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:06:43 2022


@author: Bhama Pillutla

Evaluate models performance using COCO Evaluator

"""

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.utils.visualizer import ColorMode, Visualizer
import glob

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import os
import numpy as np
import json
from detectron2.structures import BoxMode
import time
from PIL import Image
import time
cfg= get_cfg()
cfg.merge_from_file("/home/psych256lab/Downloads/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#from detectron2.modeling import build_model
#model = build_model(cfg)
#cfg.MODEL.DEVICE='cuda:0'
cfg.MODEL.WEIGHTS = "/home/psych256lab/Downloads/Scripts/output/model_final.pth"

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.10

predictor = DefaultPredictor(cfg)

    
    
#import the COCO Evaluator to use the COCO Metrics
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

#Call the COCO Evaluator function and pass the Validation Dataset
evaluator = COCOEvaluator("august_val", cfg, False, output_dir="/home/psych256lab/Downloads/Scripts/output/")
val_loader = build_detection_test_loader(cfg, "august_val")

#Use the created predicted model in the previous step
inference_on_dataset(predictor.model, val_loader, evaluator)