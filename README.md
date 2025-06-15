How to run this project ?
Load this project into python Intrepeter or Visual Studio Code 
Navigate to ../../../ObjectDetection/car-Counter/counter_code.py 
Run this counter_code.py

Project Overview:

This project uses a combination of YOLOv8 object detection and the SORT tracking algorithm to detect, track, and count vehicles in a video stream. It processes frames from a video, identifies vehicles such as cars, buses, trucks, and bicycles, tracks them using unique IDs, and counts them when they cross a predefined virtual line.
Key Features
•	Accurate vehicle detection using YOLOv8
•	Real-time vehicle tracking with SORT
•	Vehicle counting via virtual line crossing
•	Overlay of object ID, confidence score, and bounding boxes.
•	Runs on CPU, GPU, or Apple’s MPS
Requirements

•	ultralytics
•	opencv-python
•	cvzone
•	numpy
•	torch
Implementation
1. Model Setup
•	Loads the YOLOv8 model from ultralytics.
•	Loads SORT tracker for multi-object tracking.
2. Detection and Tracking
•	Each video frame is passed through YOLOv8 to detect vehicles.
•	SORT tracks each detected vehicle and assigns a unique ID.
3. Vehicle Counting
•	A horizontal line is defined on the frame.
•	When a vehicle’s center crosses this line, it's counted once.
4. Visualization
•	Bounding boxes, IDs, and vehicle types are drawn on the video.
•	Total vehicle count is shown in real time.
 
Classes Detected
Only the following classes from COCO dataset are considered for counting:
•	car
•	bus
•	truck
•	bicycle
 
Logic for Counting
python
CopyEdit
if (line_position - offset) < cy < (line_position + offset):
    if Id not in total_count:
        total_count.add(Id)
•	cy: Vertical center of the vehicle's bounding box.
•	offset: Margin to handle frame variations.
•	total_count: Set to avoid double-counting same object.
 
 Sample Output
•	Real-time video stream with:
o	Bounding boxes on detected vehicles
o	IDs and confidence scores
o	Line for vehicle counting
o	Total vehicle count displayed
# ![image](https://github.com/user-attachments/assets/7849ab16-605b-4cea-8855-43bc04997951)
