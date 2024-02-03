#!/usr/bin/python3
import cv2
import math
import numpy as np
import pyrealsense2 as rs
import rospy
from yolo.msg import yolomsg
from ultralytics import YOLO

xoffset = 0 
zoffset = 0
THETA = 0
WIN_WIDTH, WIN_HEIGHT = 640, 480

class YoloDetector:
    def __init__(self):
        self.model = YOLO("src/yolo/weight/onboard_cam_low.pt")
        self.model.fuse()
        self.class_names = ("plant")

    def detect_objects(self, img):
        results = self.model(img, stream=True)
        return results

    def draw_bounding_boxes(self, img, results, depth_frame, intr, theta):
        msg = yolomsg()
        msg.x = []
        msg.y = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x, y = int((x1 + x2) / 2), int(y1 / 4 + y2 * 3 / 4)
                depth = depth_frame.get_distance(x, y)

                Xtarget, Ztarget = self.transform_coordinates(depth, x, y, intr, theta)
                msg.x.append(Xtarget)
                msg.y.append(Ztarget)
                self.cv2_draw(img, x, y, x1, y1, x2, y2, Xtarget, Ztarget)
                
        pub.publish(msg)
        return img

    def transform_coordinates(self, depth, x, y, intr, theta):
        Xtemp = depth * (x - intr.ppx) / intr.fx
        Ytemp = depth * (y - intr.ppy) / intr.fy
        Ztemp = depth

        Xtarget = Xtemp + xoffset
        Ztarget = Ztemp*math.cos(math.radians(theta)) + zoffset

        return Xtarget, Ztarget

    def cv2_draw(self, img, x, y, x1, y1, x2, y2, Xtarget, Ztarget):
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.circle(img, (x, y), 3, (0, 0, 255), 2)
        cv2.putText(img, "({:.3f}, {:.3f})".format(Xtarget, Ztarget), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

class RealsenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        #self.config.enable_device('949122070603')
        self.config.enable_stream(rs.stream.color, WIN_WIDTH, WIN_HEIGHT, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, WIN_WIDTH, WIN_HEIGHT, rs.format.z16, 30)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    def wait_for_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        return color_frame, depth_frame

if __name__ == "__main__":
    rospy.init_node("new_ultra")
    pub = rospy.Publisher("yolo", yolomsg, queue_size=10)

    yolo_detector = YoloDetector()
    realsense_camera = RealsenseCamera()

    cv2.namedWindow('RealSense YOLO', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('RealSense YOLO', WIN_WIDTH, WIN_HEIGHT)

    try:
        while rospy.is_shutdown() == False:
            color_frame, depth_frame = realsense_camera.wait_for_frames()
            img = cv2.resize(np.asanyarray(color_frame.get_data()), (WIN_WIDTH, WIN_HEIGHT))

            results = yolo_detector.detect_objects(img)
            img_with_boxes = yolo_detector.draw_bounding_boxes(img, results, depth_frame, realsense_camera.intr, THETA)

            cv2.imshow('RealSense YOLO', img_with_boxes)
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        realsense_camera.pipeline.stop()
        cv2.destroyAllWindows()