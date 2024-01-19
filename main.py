from ultralytics import YOLO
import cv2
import math
import numpy as np
from time import time

# start webcam
cap = cv2.VideoCapture(4)
cap.set(3, 720)
cap.set(4, 480)

# model
model = YOLO("onboard_cam.pt")

# object classes
classNames =  ("plant")

# Initialize variables for FPS calculation
prev_time = 0
fps = 0

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = (x1, y1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
            
    # Calculate FPS
    current_time = time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display confidence and FPS text
    #cv2.putText(img, f'Confidence: {confidence}', (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f'FPS: {fps:.2f}', (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('realsence', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
