
#did'nt run it entirely since annotated image is not available


import cv2
#import numpy as np
#import tensorflow as tf
#from keras.models import load_model
import os
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

path = "C:\\Users\sudo\Downloads\\yolo fire n smoke dataset combined\data.yaml"


#Todo train yolo with images [] export model
#Todo since label is not found it is recommended to use transferlearning or label custom data
#train = model.
#result = model.predict(source=r"Image Dataset\Fire\0.jpg", classes='fire' ,show=True, save=True)
model.train(task='detect', mode='train', epochs=20,model='yolov8n.pt',batch =5,imgsz=640,patience=5, data=path)
#data_custom.yaml
model.export(format='onnx')
#yolov8n-seg.pt is a segmentation model downloaded from yolo repo