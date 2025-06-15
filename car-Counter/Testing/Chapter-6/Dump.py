#Yolo with webcame Using cv2 Module of generating output
import math
from ultralytics import YOLO
import cvzone
import cv2
import math


capture = cv2.VideoCapture(0)
capture.set(3,1280)
capture.set(4,720)

model = YOLO('../../../yolo-weights/yolov8n.pt')
while True:
    success,img  = capture.read()
    if not success:
        continue
    results = model(img,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #Using CV2
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            bbox = (x1, y1, x2, y2)
            conf = math.ceil(box.conf[0]*100)
            print(bbox,"\n",conf)
    cv2.imshow("Image : ", img)
    cv2.waitKey(1)


while True:
    success,img  = capture.read()
    if not success:
        continue
    results = model(img,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            #Using CV2
            w, h = x2 - x1, y2 - y1
            bbox = (x1, y1, w, h)
            cvzone.cornerRect(img, bbox)
            conf = math.ceil(box.conf[0]*100)
            print(bbox,"\n",conf)

    cv2.imshow("Image : ", img)
    cv2.waitKey(1)
