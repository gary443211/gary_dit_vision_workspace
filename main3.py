from ultralytics import YOLO
import cv2
import math
import numpy as np
import pyrealsense2 as rs
from time import time
from decimal import Decimal, ROUND_HALF_UP

# Create a RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile=pipeline.start(config)

theta=0 #camera depression angle

align_to = rs.stream.color
align = rs.align(align_to)

# get camera intrinsics, focal distance, optical center
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# Model
model = YOLO("onboard_cam.pt")

# Object classes
classNames = ("plant")

# Initialize variables for FPS calculation
prev_time = 0
fps = 0

# Set the desired window resolution
window_width = 640
window_height = 480

# Create an OpenCV window
cv2.namedWindow('RealSense YOLO', cv2.WINDOW_NORMAL)
cv2.resizeWindow('RealSense YOLO', window_width, window_height)

# # test
# frames = pipeline.wait_for_frames()
# print("Number of frames:", len(frames))
# depth_frame = frames.get_depth_frame()
# if depth_frame:
#     Depth = depth_frame.get_distance(320, 180)
#     print("central depth --->", Depth)
# else:
#     print("Depth frame is None")


while True:
    # Wait for a RealSense frame
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

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

            # Get distance(depth) of the center of the box
            d1, d2 = int((x1+x2)/2), int(y1/4 + y2*3/4)
            dist = depth_frame.get_distance(int(d1),int(d2))  # by default realsense returns distance in meters 

            #calculate real world coordinates
            Xtemp = dist*(int(d1) -intr.ppx)/intr.fx
            Ytemp = dist*(int(d2) -intr.ppy)/intr.fy
            Ztemp = dist

            #coordinate transformation, theta is the camera deprssion angle
            #Xtarget = Xtemp - 35 #35 is RGB camera module offset from the center of the realsense
            Xtarget = Xtemp #seems like we don't have the offset here
            Ytarget = -(Ztemp*math.sin(theta) + Ytemp*math.cos(theta))
            Ztarget = Ztemp*math.cos(theta) + Ytemp*math.sin(theta)
            
            # #no rounding
            # coordinates_text = "(" + str(Xtarget) + \
            #                     ", " + str(Ytarget) + \
            #                     ", " + str(Ztarget) + ")"
            
            #rounding
            coordinates_text = "(" + str(Decimal(str(Xtarget)).quantize(Decimal('.000'), rounding=ROUND_HALF_UP)) + \
                                ", " + str(Decimal(str(Ytarget)).quantize(Decimal('.000'), rounding=ROUND_HALF_UP)) + \
                                ", " + str(Decimal(str(Ztarget)).quantize(Decimal('.000'), rounding=ROUND_HALF_UP)) + ")"

            # Object details
            org = (x1, y1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 1

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
            #cv2.putText(img, Depth, org, font, fontScale, color, thickness)
            cv2.putText(img, coordinates_text, (int(d1)-160, int(d2)), font, fontScale, color, 2)

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

pipeline.stop()
cv2.destroyAllWindows()
