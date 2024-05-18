from ultralytics import YOLO
import cv2
from util import *
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import numpy as np


results = {}
mot_tracker = Sort()

# LOAD MODELS
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./license_plate_detector.pt')

# LOAD VIDEO
cap = cv2.VideoCapture('./traffic_vid1.mp4')

vehicles = [2, 3, 5, 7]

"""
In the line vehicles = [2, 3, 5, 7], a list named vehicles is defined, which contains class IDs corresponding to different types of vehicles. In object detection tasks, each detected object is assigned a class ID based on a predefined list of classes. In this case, the list [2, 3, 5, 7] likely corresponds to specific class IDs for different types of vehicles in the YOLO model being used.

For example:

Class ID 2 might represent cars.
Class ID 3 might represent trucks.
Class ID 5 might represent buses.
Class ID 7 might represent motorcycles.

By specifying these class IDs, the code ensures that only objects detected as vehicles are considered for further processing, such as tracking and license plate detection. Other detected objects that do not have class IDs listed in vehicles are ignored.
"""

# READ FRAMES
                                                    # https://chat.openai.com/share/f4fe23a0-acd8-499b-bc98-10aae629ad56
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}

####### DETECT VEHICLES
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

####### TRACK VEHICLES
        track_ids = mot_tracker.update(np.asarray(detections_))

####### DETECT LICENSE PLATES
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

########### ASSIGN LICENSE PLATE TO CAR
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

############### CROP LICENSE PLATE
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

############### PROCESS LICENSE PLATE
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY),
                license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

############### READ LICENSE PLATE NUMBER
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

# WRITE RESULTS
write_csv(results, './test.csv')