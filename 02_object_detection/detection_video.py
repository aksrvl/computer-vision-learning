# Task 2.4 — YOLO на відео:
# - Обробляти відео покадрово
# - Отримувати detection для кожного frame
# - Виводити class, confidence та bounding box
# - Візуалізувати bounding boxes
# - НЕ використовувати tracking / model.track()
from ultralytics import YOLO
import numpy as np

model = YOLO("yolo26n.pt")
results = model('02_object_detection/test_video.mp4', show = True)
frame_count = 0
for result in results:
    frame_count += 1
    print(f"Frame {frame_count}")
    name = [result.names[cls.item()] for cls in result.boxes.cls]
    conf = result.boxes.conf
    box = result.boxes.xyxy.numpy()
    for detection in zip(name, conf, box):
        print(f"{detection[0]} | confidence: {detection[1]} | box: {detection[2]}")