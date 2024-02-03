#!/usr/bin/python3
#Vision
from ultralytics import YOLO
import cv2

#ROS
import rospy
from yolo.msg import yolomsg
from sensor_msgs.msg import Image

#Img msg parser to cv
from cv_bridge import CvBridge

#Other tools 
import math
import numpy as np
import pyrealsense2 as rs

xoffset = 0 # x distance with machine center
zoffset = 0 # Z distance with machine center
THETA = 0 # camera depression angle in degree

class Node:
    def __init__(self):
        ###Adding Param

        #CvBridge
        self.bridge = CvBridge()
        #YOLO
        self.model = YOLO("src/yolo/weight/onboard_cam.pt")
        self.model.fuse() # Fuse for speed
        self.result_img = None

        #Publisher 
        rospy.init_node("vison_node")
        self.pub = rospy.Publisher("/yolo/plant", yolomsg, queue_size=10)
        self.bgrm_pub = rospy.Publisher("/yolo/bgrm", Image, queue_size=10)
        self.result_pub = rospy.Publisher("/yolo/result", Image, queue_size=10)

        #Subscriber
        self.col1_msg = None
        self.dep1_msg = None
        self.sub_col1 = rospy.Subscriber("/cam1/color/image_raw", Image, self.col_callback1)
        self.sub_dep1 = rospy.Subscriber("/cam1/aligned_depth_to_color/image_raw", Image, self.dep_callback1) 
        #self.sub_coln = rospy.Subscriber("/camn/color/image_raw", Image, self.col_callbackn)
        #self.sub_depn = rospy.Subscriber("/camn/depth/image_rect_raw", Image, self.dep_callbackn) 

    def col_callback1(self, msg):
        self.col1_msg = msg
    def dep_callback1(self, msg):
        self.dep1_msg = msg
    '''
    def col_callbackn(self, msg):
        self.col1_msg = msg
    def dep_callbackn(self, msg):
        self.dep1_msg = msg
    '''
    def bg_removal(self, col1_msg:Image, dep1_msg:Image):
        """
        Input:
        col_msg : 'Image'-- RGB
        dep_msg : 'Image'-- Z16
        
        Return:
        bgrm_img : 'Image' -- RGB
        """
        #Convert col_msg
        cv_col1_img = self.bridge.imgmsg_to_cv2(col1_msg, desired_encoding = "bgr8")
        np_col1_img = np.array(cv_col1_img, dtype = np.uint8)
        #coln_image = CvBridge.imgmsg_to_cv2(coln_msg, desired_encoding = 'bgr8')

        #Convert dep_msg
        cv_dep1_img = self.bridge.imgmsg_to_cv2(dep1_msg, desired_encoding = "passthrough")
        np_dep1_img = np.array(cv_dep1_img, dtype = np.uint16)

        #3d depth image to match bgr color image #stack 1 dim into 3
        dep_3d_img = np.dstack((np_dep1_img, np_dep1_img, np_dep1_img)) 

        #bg removal 
        grey = 153
        bgrm_img = np.where((dep_3d_img > 500) | (np.isnan(dep_3d_img)), grey, np_col1_img)
        return bgrm_img

    def publish(self):
        pass
    #    pub_msg = yolomsg()
    #    pub_msg.x = Xtarget
    #    pub_msg.y = Ytarget
    #    pub_msg.z = Ztarget
    #    self.pub.publish(pub_msg)

    def yolo(self):
        if self.col1_msg is not None and self.dep1_msg is not None:
            #bg removal
            bgrm_img = self.bg_removal(self.col1_msg, self.dep1_msg)
            self.bgrm_pub.publish(self.bridge.cv2_to_imgmsg(bgrm_img, encoding="bgr8"))

            #YOLO detection
            results = self.model(source = bgrm_img)

            msg = yolomsg()
            msg.x = []
            msg.y = []
            
            for object in results:
                self.result_img = object.plot()
                self.result_pub.publish(self.bridge.cv2_to_imgmsg(self.result_img, encoding="bgr8")) 
                boxes = object.boxes

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x, y = int((x1 + x2) / 2), int(y1 / 4 + y2 * 3 / 4)
                    depth = self.np_dep1_img[y, x]

                    Xtarget, Ztarget = self.transform_coordinates(depth, x, y, self.intr, THETA)
                    msg.x.append(Xtarget)
                    msg.y.append(Ztarget)

                self.pub.publish(msg)

    def transform_coordinates(self, depth, x, y, intr, theta):
        Xtemp = depth * (x - intr.ppx) / intr.fx
        Ytemp = depth * (y - intr.ppy) / intr.fy
        Ztemp = depth

        Xtarget = Xtemp + xoffset
        Ztarget = Ztemp*math.cos(math.radians(theta)) + zoffset

        return Xtarget, Ztarget
    
class RealsenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        #self.config.enable_device('949122070603')
        self.profile = self.pipeline.start(self.config)
        self.intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()


if __name__ == "__main__":
    
    #Class
    realsense_camera = RealsenseCamera()
    vision_node = Node()
    #cv2.namedWindow('RealSense YOLO', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('RealSense YOLO', WIN_WIDTH, WIN_HEIGHT)

    try:
        while not rospy.is_shutdown():

            vision_node.yolo()
            #cv2.imshow('RealSense YOLO', results)
            #if cv2.waitKey(1) == ord('q'):
            #    break

    except KeyboardInterrupt:
        pass
    finally:
        #cv2.destroyAllWindows()
        pass