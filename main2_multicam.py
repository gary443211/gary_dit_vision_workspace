from ultralytics import YOLO
import cv2
import math
import numpy as np
import pyrealsense2 as rs
from time import time

# Create a RealSense pipeline
# ...from Camera 1
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device('949122070603')
config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline_1.start(config_1)

# Model
model = YOLO("onboard_cam.pt")

# Object classes
classNames = ("plant")

# Initialize variables for FPS calculation
prev_time = 0
fps = 0

# Set the desired window resolution
window_width = 640
window_height = 360

# Create an OpenCV window
cv2.namedWindow('RealSense YOLO', cv2.WINDOW_NORMAL)
cv2.resizeWindow('RealSense YOLO', window_width, window_height)

while True:
    # Wait for a RealSense frame
    frames = pipeline_1.wait_for_frames()
    color_frame = frames.get_color_frame()

    # Convert RealSense frame to OpenCV format
    img = np.asanyarray(color_frame.get_data())

    # Resize the image to match the window resolution
    img = cv2.resize(img, (window_width, window_height))

    results = model(img, stream=True)

    # Initialize confidence variable
    confidence = 0

    # Coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

            # Draw bounding box on the frame
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # Confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            # Class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # Object details
            org = (x1, y1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 1

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    # Calculate FPS
    current_time = time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display confidence and FPS text
    cv2.putText(img, f'Confidence: {confidence}', (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f'FPS: {fps:.2f}', (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('RealSense YOLO', img)
    if cv2.waitKey(1) == ord('q'):
        break

pipeline_1.stop()
cv2.destroyAllWindows()
