#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/07/17
# @author:aiden
# Face detection(人脸检测)
import cv2
import time
import rospy
import signal
import queue
import numpy as np
import mediapipe as mp
from rosman_sdk import common
from threading import RLock
from sensor_msgs.msg import Image
from std_srvs.srv import Empty, EmptyResponse
from rosman_interfaces.msg import ObjectInfo, ObjectsInfo

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

class FaceDetectNode:
    def __init__(self, name):
        # Initalize node(初始化节点)
        rospy.init_node(name, log_level=rospy.INFO)
        self.name = name
        
        self.enter = False
        self.running = True
        self.image_sub = None
        self.start_detect = False
        
        self.lock = RLock()
        self.image_queue = queue.Queue(maxsize=2)
        signal.signal(signal.SIGINT, self.shutdown)

        # Obtain parameter(获取参数)
        self.debug = rospy.get_param('~debug', False)
        self.camera = rospy.get_param('/camera')
        
        # Instantiate face detection(实例化人脸检测)
        self.face_detection = mp_face_detection.FaceDetection(min_detection_confidence=rospy.get_param('~confidence', 0.5))

        # Detect image publisher(检测图像发布)
        self.image_pub = rospy.Publisher('~image_result', Image, queue_size=1)

        # Publish object's position and posture(物体位姿发布)
        self.face_info_pub = rospy.Publisher('/object/pixel_coords', ObjectsInfo, queue_size=1)
        
        rospy.Service('~enter', Empty, self.enter_func)
        rospy.Service('~exit', Empty, self.exit_func)
        rospy.Service('~start', Empty, self.start_func)
        rospy.Service('~stop', Empty, self.stop_func)

        self.enable_display = rospy.get_param('~enable_display', False)
        time.sleep(0.2)
        common.loginfo("%s init finish"%self.name)
        
        if self.debug:
            self.enter_func(None)
            self.start_func(None)

    def shutdown(self, signum, frame):
        with self.lock:
            self.running = False
            
            common.loginfo('%s shutdown'%self.name)
    
    # Enable subscription(开启订阅)
    def enter_func(self, msg):
        # Obtain parameter(获取参数)
        with self.lock:
            if self.image_sub is None:
                self.image_sub = rospy.Subscriber('/{}/{}'.format(self.camera['camera_name'], self.camera['image_topic']), Image, self.image_callback)  # 订阅图像
            self.enter = True 
        common.loginfo("%s enter" % self.name)
        
        return EmptyResponse()

    # Disable subscription(注销订阅)
    def exit_func(self, msg):
        with self.lock:
            self.stop_func(Empty())
            if self.image_sub is not None:
                self.image_sub.unregister()
                self.image_sub = None
            self.enter = False
        common.loginfo('%s exit' % self.name)

        return EmptyResponse()

    # Enable detection(开启检测)
    def start_func(self, msg):
        with self.lock:
            self.start_detect = True
        
        common.loginfo("%s start" % self.name)

        return EmptyResponse()

    # Stop detection(停止检测)
    def stop_func(self, msg):
        with self.lock:
            self.start_detect = False
        
        common.loginfo("%s stop" % self.name)

        return EmptyResponse()

    def run(self):
        while self.running:
            with self.lock:
                if self.enter:
                    image = self.image_queue.get(block=True, timeout=1)
                    if self.start_detect:  # If enable detection（如果开启检测）
                        time_start = rospy.get_time()
                        results = self.face_detection.process(image)
                        frame_result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        faces_info = []
                        if results.detections:
                            for detection in results.detections:
                                xmin = detection.location_data.relative_bounding_box.xmin
                                ymin = detection.location_data.relative_bounding_box.ymin
                                width = detection.location_data.relative_bounding_box.width
                                height = detection.location_data.relative_bounding_box.height
                                
                                landmarks = np.array([(xmin + width/2)*640, (ymin + height/2)*480]) 
                                center = landmarks.astype(int)
                                face_info = ObjectInfo()
                                face_info.label = 'face'
                                face_info.type = 'rect'
                                face_info.x = center[0]
                                face_info.y = center[1]
                                face_info.width = 640
                                face_info.height = 480
                                faces_info.append(face_info)
                                mp_drawing.draw_detection(frame_result, detection)

                        self.face_info_pub.publish(faces_info)  # Publish position and posture(发布位姿)
                        time_d = 0.03 - (rospy.get_time() - time_start)
                        if time_d > 0:
                            time.sleep(time_d)
                    else:
                        frame_result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        time.sleep(0.03)
                    ros_image = common.cv2_image2ros(frame_result, self.name)  # Convert OpenCV format to ROS(opencv格式转为ros)
                    self.image_pub.publish(ros_image)  # Publish image(发布图像)
                    if self.enable_display:
                        cv2.imshow('face_detection', frame_result)
                        key = cv2.waitKey(1)
                        if key != -1:
                            self.running = False
                else:
                    time.sleep(0.01)

    def image_callback(self, ros_image):
        rgb_image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8,
                           buffer=ros_image.data)  # Convert the image information from ROS format to OpenCV format(将ros格式图像消息转化为opencv格式)
        if self.image_queue.full():
            # If the queue is full, discard the oldest image(如果队列已满，丢弃最旧的图像)
            self.image_queue.get()
        # Put the image into the queue(将图像放入队列)
        self.image_queue.put(rgb_image)

if __name__ == '__main__':
    FaceDetectNode('face_detect').run()
