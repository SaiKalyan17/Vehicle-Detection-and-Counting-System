import math
import cv2
import cvzone
import numpy as np
import torch
from ultralytics import YOLO
from sort import *

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load video and set resolution
capture = cv2.VideoCapture("cars.mp4")
capture.set(3, 1280)
capture.set(4, 720)

# Load YOLO model
model = YOLO('../yolo-weights/yolov8l.pt').to(device)

# Define class names
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
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

# Load mask
car_mask = cv2.imread("carmask.png")

# Create SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Counting line and counted vehicle IDs
line_position = 450
offset = 20
total_count = set()

# Main loop
while True:
    success, img = capture.read()
    if not success:
        break

    # Apply mask
    # img = cv2.bitwise_and(img, car_mask)

    # Get YOLO detections
    results = model(img, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            bbox = (x1, y1, w, h)
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            currentCls = classNames[cls]

            # Filter for vehicles
            if currentCls in ["car", "bicycle", "bus", "truck"] and conf > 0.3:
                cvzone.putTextRect(img, f'{currentCls} : {int(conf * 100)}%', (max(0, x1), max(35, y1)),
                                   scale=1.5, offset=3)
                cvzone.cornerRect(img, bbox)
                currArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currArray))

    # Update tracker
    results_tracker = tracker.update(detections)

    for result in results_tracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        Id = int(Id)

        # Center point
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Draw ID & center
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        cvzone.putTextRect(img, f'ID: {Id}', (x1, y1 - 10), scale=1, thickness=1, offset=2)

        # Check crossing
        if (line_position - offset) < cy < (line_position + offset):
            if Id not in total_count:
                total_count.add(Id)
                print(f"Vehicle ID {Id} counted. Total = {len(total_count)}")

    # Draw counting line
    cv2.line(img, (0, line_position), (1280, line_position), (0, 255, 255), 2)
    cvzone.putTextRect(img, f'Count: {len(total_count)}', (50, 50), scale=2, thickness=2, offset=10)

    # Show result
    cv2.imshow("Vehicle Counter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
