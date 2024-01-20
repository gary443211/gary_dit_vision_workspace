import pyrealsense2 as rs
import cv2
import math
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from PIL import Image
from time import time 

#main 函數的開始。theta 是相機角度，config 是Intel RealSense配置。
def main():
    theta = 0
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

#創建RealSense管道和配置文件，以及對齊配置，以獲得對齊的彩色和深度圖像。
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

#獲取相機的內部參數，即內部校準。
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    #=========== YOLOv8 weights and config file paths =======================
#配置YOLOv8模型，並使用CUDA進行推論。
    weightsPath = "yolov8.weights"
    configPath = "yolov8.cfg"

    net = cv2.dnn.readNet(weightsPath, configPath)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    model = cv2.dnn_DetectionModel(net)

#定義一個函數 YOLOv8_video，用於對圖像進行YOLOv8目標檢測。
    def YOLOv8_video(pred_image):
        model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
        image_test = cv2.cvtColor(pred_image, cv2.COLOR_RGBA2RGB)
        image = image_test.copy()
        print('image', image.shape)
        confThreshold = 0.5
        nmsThreshold = 0.4
        classes, confidences, boxes = model.detect(image, confThreshold, nmsThreshold)

        return classes, confidences, boxes

#初始化一些變數，包括按鍵、標籤、顏色和時間。
    key = ' '
    LABELS = ['plant']
    COLORS = [[0, 0, 255]]
    prev_frame_time = 0
    new_frame_time = 0

#循環處理RealSense的幀，如果深度幀或彩色幀不存在，則繼續下一次迭代。
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not depth_frame or not color_frame:
            continue

#將RealSense彩色圖像轉換為NumPy數組，然後使用YOLOv8進行目標檢測。
        color_image = np.asanyarray(color_frame.get_data())
        image = Image.fromarray(color_image)
        img = np.asarray(image)
        classes, confidences, boxes = YOLOv8_video(img)

        print("predict:", classes, boxes)

#如果檢測到目標，則迭代並繪製檢測框。
        if len(boxes) > 0:
            for cl, score, (x_min, y_min, x_max, y_max) in zip(classes, confidences, boxes):
                start_pooint = (int(x_min), int(y_min))
                end_point = (int(x_max), int(y_max))

#計算目標的中心坐標，並在圖像上繪製相應的形狀和標籤。
                x = int(x_min + (x_max - x_min) / 2)
                y = int(y_min + (y_max - y_min) / 2)
                color = COLORS[0]
                img = cv2.rectangle(img, start_pooint, end_point, color, 3)
                img = cv2.circle(img, (x, y), 5, [0, 0, 255], 5)
                text = f'{LABELS[int(cl)]}: {score:0.2f}'
                cv2.putText(img, text, (int(x_min), int(y_min - 7)), cv2.FONT_ITALIC, 1, COLORS[0], 2)

#計算目標到相機的距離，將深度值轉換為毫米。
                x = round(x)
                y = round(y)
                dist = depth_frame.get_distance(int(x), int(y)) * 1000  # convert to mm

# 計算目標的實際世界座標。
                Xtemp = dist * (x - intr.ppx) / intr.fx
                Ytemp = dist * (y - intr.ppy) / intr.fy
                Ztemp = dist

                Xtarget = Xtemp - 35  # 35 is RGB camera module offset from the center of the realsense
                Ytarget = -(Ztemp * math.sin(theta) + Ytemp * math.cos(theta))
                Ztarget = Ztemp * math.cos(theta) + Ytemp * math.sin(theta)

#顯示目標的實際世界座標和相機到目標的距離。
                coordinates_text = "(" + str(Decimal(str(Xtarget)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) + \
                                   ", " + str(Decimal(str(Ytarget)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) + \
                                   ", " + str(Decimal(str(Ztarget)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) + ")"

                coordinat = (Decimal(str(Ztarget)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
                print("Distance to Camera at (class : {0}, score : {1:0.2f}): distance : {2:0.2f} mm".format(LABELS[int(cl)], score, coordinat), end="\r")
                cv2.putText(img, "Distance: " + str(round(coordinat, 2)) + 'm', (int(x_max - 180), int(y_max + 30)),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

#顯示每秒的幀數和檢測結果的圖像。
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time

                print('FPS : %.2f  ' % fps)
                cv2.imshow("Image", img)

        else:
            cv2.imshow("Image", img)

        cv2.waitKey(1)

    cv2.destroyAllWindows()

    print("\nFINISH")

if __name__ == "__main__":
    main()
