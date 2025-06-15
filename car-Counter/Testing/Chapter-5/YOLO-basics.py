import cv2
from ultralytics import YOLO

model = YOLO('../../../yolo-weights/yolov8l.pt')

results = model('./bus.jpg',show = True)
results = model('./cars.jpg',show = True)
results = model('./bikes.jpg',show = True)
cv2.waitKey(0)