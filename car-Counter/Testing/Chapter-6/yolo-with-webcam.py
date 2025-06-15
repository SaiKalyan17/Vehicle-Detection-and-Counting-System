import math
import cv2
import cvzone
import torch
from ultralytics import YOLO

capture = cv2.VideoCapture(0) #For WebCam
# capture = cv2.VideoCapture("./Videos/bikes.mp4") #THESE ALL FOR DIFFERENT VIDEOS
# capture = cv2.VideoCapture("./Videos/cars.mp4")
# capture = cv2.VideoCapture("./Videos/motorbikes.mp4")
# capture = cv2.VideoCapture("./Videos/ppe-1-1.mp4")
# capture = cv2.VideoCapture("./Videos/ppe-2-1.mp4")
# capture = cv2.VideoCapture("./Videos/ppe-3-1.mp4")
capture.set(3,1280)
capture.set(4,720)

model = YOLO('../../../yolo-weights/yolov8l.pt')
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
while True:
    success,img  = capture.read()
    if not success:
        continue
    results = model(img,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Using CVZONE
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            w, h = x2 - x1, y2 - y1
            bbox = (x1, y1, w, h)
            cvzone.cornerRect(img, bbox)
            conf = math.ceil(box.conf[0]*100)
            cls = int(box.cls[0])
            print(bbox,"\n confidence:",conf)
            cvzone.putTextRect(img,f'{classNames[cls] } : {conf}',(max(0,x1),max(35,y1)),scale=2.6)

    cv2.imshow("Image : ", img)
    cv2.waitKey(1)
