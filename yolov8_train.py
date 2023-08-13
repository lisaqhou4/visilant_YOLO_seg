from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')
yaml_path = "/home/lisahou/Desktop/segmentation/visilant_YOLO_seg/eye_yolov8.yaml"

results = model.train(data=yaml_path, epochs=20, imgsz=640)
