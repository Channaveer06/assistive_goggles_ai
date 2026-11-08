import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
print(f"COCO Class 63: {model.names[63]}")
print(f"COCO Class 67: {model.names[67]}")
