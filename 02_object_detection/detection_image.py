# Task 2.1 — YOLO на зображенні:
# - Завантажити pretrained YOLO detection model
# - Запустити detection на image.jpg
# - Візуалізувати результати
#
# Task 2.2 — Розібрати результати:
# - Отримати для кожного об'єкта class, confidence, x1, y1, x2, y2
# - Знайти центр bounding box
# - Розібратися з result, boxes, xyxy, conf, cls
#
# Task 2.3 — Confidence threshold:
# - Запустити detection з threshold 0.25, 0.50, 0.75
# - Порівняти кількість знайдених об'єктів
# - Пояснити, чому вона змінюється

from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model('02_object_detection/image.jpg')
results[0].show()

for result in results:
    name = [result.names[cls.item()] for cls in result.boxes.cls]
    conf = result.boxes.conf
    box = result.boxes.xyxy
    center = result.boxes.xywh[:, :2]
    for detection in zip(name, conf, box, center):
        print(f"{detection[0]} | confidence: {detection[1]} | box: {detection[2]} | center: {detection[3]}")