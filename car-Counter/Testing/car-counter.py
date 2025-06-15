import math
import cv2
import cvzone
import numpy as np
import torch
from ultralytics import YOLO
from sort import *
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

capture = cv2.VideoCapture("../cars.mp4")

capture.set(3,1280)
capture.set(4,720)

model = YOLO('../../yolo-weights/yolov8l.pt').to(device)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

car_mask = cv2.imread("../carmask.png")

tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)
while True:
    success,img  = capture.read()
    #image_region = cv2.bitwise_and(img,car_mask)
    img = cv2.bitwise_and(img, car_mask)
    if not success:
        continue
    # results = model(img,stream=True)
    results = model(img, stream=True)
    detections = np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Using CVZONE
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            w, h = x2 - x1, y2 - y1
            bbox = (x1, y1, w, h)
            conf = math.ceil(box.conf[0]*100)
            cls = int(box.cls[0])
            currentCls = classNames[cls]

            if currentCls in ["car", "bicycle", "bus", "truck"] and conf > 0.3:
                print(bbox,"\n confidence:",conf)
                cvzone.putTextRect(img,f'{currentCls } : {conf} ' ,(max(0,x1),max(35,y1)),scale=2.6,offset=3)
                cvzone.cornerRect(img, bbox)
                currArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currArray))

    results_tracker = tracker.update(detections)
    for result in results_tracker:
        x1, y1, x2, y2, Id = result
        print(result)

    cv2.imshow("Video : ", img)
    #cv2.imshow("VideoRegion : ", image_region)
    cv2.waitKey(1)
