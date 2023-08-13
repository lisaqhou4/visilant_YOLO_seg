from ultralytics import YOLO
import os

path = "/home/lisahou/Desktop/segmentation/visilant_YOLO_seg/runs/segment/train7/weights/best.pt"
test_path = "/home/lisahou/Desktop/segmentation/visilant_YOLO_seg/dataset/images/test"
img_path = "/home/lisahou/Desktop/segmentation/visilant_YOLO_seg/dataset/images/test/0ee74d61-9a8e-4f9d-b70f-41c8565706cd.jpg"
model = YOLO(path)
img_files = os.listdir(img_path)
#results = model(img_files)

