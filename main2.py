from ultralytics import YOLO
import cv2
import math
import numpy as np
import pyrealsense2 as rs
from time import time

# Create a RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

# Model
model = YOLO("best.pt")

# Object classes
classNames = ("can", "dark", "light")

# Initialize variables for FPS calculation
prev_time = 0
fps = 0

# Set the desired window resolution
window_width = 1280
window_height = 720

# Create an OpenCV window
cv2.namedWindow('RealSense YOLO', cv2.WINDOW_NORMAL)
cv2.resizeWindow('RealSense YOLO', window_width, window_height)

while True:
    # Wait for a RealSense frame
    frames = pipeline.wait_for_frames()
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
    cv2.putText(img, f'Confidence: {confidence}', (10, 660), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f'FPS: {fps:.2f}', (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('RealSense YOLO', img)
    if cv2.waitKey(1) == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
