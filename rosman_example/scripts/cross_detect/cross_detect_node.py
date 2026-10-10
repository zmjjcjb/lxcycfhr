#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/07/03
# @author:aiden
# Crossroad detection(十字路口检测)
import time
import rospy
import signal
from rosman_sdk import common
from rosman_example.color_common import Common
# 导入巡线十字检测需要的模块
from rosman_example.visual_patrol import VisualPatrol
from rosman_interfaces.msg import ObjectsInfo, ColorDetect, ColorsDetect

class CrossDetectNode(Common):
    # Check three ROIs in order, and immediately exit if a black line is detected(按顺序检测三个roi，如果检测到黑线立刻跳出)
    # y_min, y_max, x_min, and x_max represent respectively the proportion of the image occupied. The actual size is y_min*height(y_min, y_max, x_min, x_max分别表示占图像的比例, 即实际大小为y_min*height)
    # Three line detection region from top to bottom(3个线条检测区，从上到下)
    line_roi = [(5 / 12, 6 / 12, 1 / 4, 3 / 4),
                (6 / 12, 7 / 12, 1 / 4, 3 / 4),
                (7 / 12, 8 / 12, 1 / 4, 3 / 4)
                ]
    # One cross detection region(1个十字检测区域)
    cross_roi = [1 / 4, 3 / 4, 0, 1]
    # Enter the image threshold for cross detection, and adjust based on the actual situation(进入十字检测的图像阈值，视情况调整)
    enter_cross_detect_y = 240 / 480
    # Size of the image after down-sampling, and the detection accuracy can be adjusted(下采样后图像大小，可调整检测精度)
    image_process_size = [160, 120]
    # Name of the action for raising up the hand(抬手动作名称)
    raise_right_hand_action_name = 'raise_right_hand'

    def __init__(self, name):
        # Initialize ROS node(初始化ROS节点)
        rospy.init_node(name)
        self.name = name
        # Flag used to indicate whether the program should continue running(是否继续运行的标志)
        self.running = True
        self.count = 0
        # Store information about detected objects(存储检测到的对象信息)
        self.objects_info = []
        # Define two states(定义两个状态)
        self.current_state = 'visual_patrol'  #Line following state(巡线状态)
        self.next_state = 'cross_detect'      #Crossing state(十字路口状态)
        # The parameters for each state: head posture, color detection parameter, initialization flag(状态对应的参数:头部姿态,颜色检测参数,是否初始化)
        # Achieving configurations under different states through a dictionary: "visual_patrol" line following state; "cross_detect" crossroad state(通过字典实现不同状态下的配置：visual_patrol巡线状态；cross_detect十字路口状态)
        self.state = {'visual_patrol':  [[500, 260], ['black', self.line_roi, self.image_process_size, self.set_visual_patrol_color], False],
                      'cross_detect':   [[500, 260], ['black', self.cross_roi, self.image_process_size, self.set_cross_color], False]}
        
        # Obtain state corresponding to head initial posture(获取状态对应的头部初始姿态)
        self.head_pan_init = self.state[self.current_state][0][0]   # Initial value of the left-right servo(左右舵机的初始值)
        self.head_tilt_init = self.state[self.current_state][0][1]  # Initial value of the up-down servo(上下舵机的初始值)
        # Initialize the parent class and set the initial angle of the head(初始化父类,设置头部角度)
        super().__init__(name, self.head_pan_init, self.head_tilt_init)
        # Create line following control instance(创建巡线控制实例)
        self.visual_patrol = VisualPatrol(self.gait_manager)

        # Decrease the stride of line following, improving the stability of other flag detection(减小巡线步幅，提高其他标志检测稳定性)
        self.visual_patrol.update_go_gait(dsp=[300, 0.2, 0.02], x_max=0.01)
        self.visual_patrol.update_turn_gait(dsp=[400, 0.2, 0.02], x_max=0.01) 

        # Ctrl+C exit(Ctrl+C 退出)
        signal.signal(signal.SIGINT, self.shutdown)
        
        # Subscribe to color recognition result(订阅颜色识别结果)
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)
        # Create a publisher to publish color recognition parameter(创建发布颜色识别参数的publisher）
        self.detect_pub = rospy.Publisher("/color_detection/update_detect", ColorsDetect, queue_size=1)

        self.motion_manager.run_action('walk_ready')  # Ready to stand up(准备站立)
        # Whether to automatically start(是否自动开始)
        if rospy.get_param('~start', True):
            # Notify the color recognition to prepare, at this time only display the camera original image(通知颜色识别准备，此时只显示摄像头原画)
            self.enter_func(None)
            #self.set_color_srv_callback('black', self.cross_roi)
            self.start_srv_callback(None)  # Start color recognition(开启颜色识别)
            common.loginfo('start cross detect')
    
    # Ctrl+C exit callback(Ctrl+C 退出回调)
    def shutdown(self, signum, frame):
        with self.lock:
            self.running = False
            common.loginfo('%s shutdown' % self.name)
     # Generate color recognition configuration based on line following parameter(根据巡线参数生成颜色识别配置)
    def set_visual_patrol_color(self, color, roi, image_process_size):
        # Set line following color(设置巡线颜色)
        line_param = ColorDetect()
        line_param.color_name = color
        line_param.use_name = True
        line_param.detect_type = 'line'
        line_param.image_process_size = image_process_size
        line_param.line_roi.up.y_min = int(roi[0][0] * image_process_size[1])
        line_param.line_roi.up.y_max = int(roi[0][1] * image_process_size[1])
        line_param.line_roi.up.x_min = int(roi[0][2] * image_process_size[0])
        line_param.line_roi.up.x_max = int(roi[0][3] * image_process_size[0])

        line_param.line_roi.center.y_min = int(roi[1][0] * image_process_size[1])
        line_param.line_roi.center.y_max = int(roi[1][1] * image_process_size[1])
        line_param.line_roi.center.x_min = int(roi[1][2] * image_process_size[0])
        line_param.line_roi.center.x_max = int(roi[1][3] * image_process_size[0])

        line_param.line_roi.down.y_min = int(roi[2][0] * image_process_size[1])
        line_param.line_roi.down.y_max = int(roi[2][1] * image_process_size[1])
        line_param.line_roi.down.x_min = int(roi[2][2] * image_process_size[0])
        line_param.line_roi.down.x_max = int(roi[2][3] * image_process_size[0])

        line_param.min_area = 1
        line_param.max_area = image_process_size[0] * image_process_size[1]
        
        return line_param
    
    # Generate cross and color detection parameters(生成十字检测颜色检测参数)
    def set_cross_color(self, color, roi, image_process_size):
        cross_param = ColorDetect()
        cross_param.color_name = color
        cross_param.use_name = True
        cross_param.detect_type = 'cross'
        cross_param.image_process_size = image_process_size
        cross_param.roi.y_min = int(roi[0] * image_process_size[1])
        cross_param.roi.y_max = int(roi[1] * image_process_size[1])
        cross_param.roi.x_min = int(roi[2] * image_process_size[0])
        cross_param.roi.x_max = int(roi[3] * image_process_size[0])
        cross_param.min_area = 10
        cross_param.max_area = image_process_size[0]*image_process_size[1]
        
        return cross_param


    # Initialize state(状态初始化)
    def state_init(self, current_state, next_state):
        # Initialization in different phrase(不同阶段的初始化)
        if self.state[current_state][2] == False:
            # Set head posture(设置头部姿态)
            self.state[current_state][2] = True
            self.init_action(self.state[current_state][0][0], self.state[current_state][0][1])  # Head posture(头部姿态)
            # Generate detection parameter and publish(生成检测参数并发布)
            param1 = self.state[current_state][1][3](self.state[current_state][1][0], self.state[current_state][1][1], self.state[current_state][1][2])
            param2 = self.state[next_state][1][3](self.state[next_state][1][0], self.state[next_state][1][1], self.state[next_state][1][2])
            self.detect_pub.publish([param1, param2])  

    # The condition for entering cross detection(进入十字检测的条件)
    def enter_cross_detect(self, cross_data):
        # Determine based on the cross region detection result(根据十字区域检测结果判断)
        if cross_data is not None:
            
            if max(cross_data.y, cross_data.left_point[1], cross_data.right_point[1]) > self.enter_cross_detect_y * cross_data.height:
                self.count += 1
                print(self.count)
                if self.count > 2:
                    self.count = 0
                    self.gait_manager.stop()
                    common.loginfo('cross detect')
                    return True
            else:
                self.count = 0
        else:
            self.count = 0
        return False

    # The action after exiting the cross detection(退出十字检测后的动作)
    def exit_cross_detect(self):
        # Play action(播放动作)
        # Update gait(更新步态)
        # Enter line following state(进入巡线状态)
        self.gait_manager.disable()
        self.motion_manager.run_action(self.raise_right_hand_action_name)
        walking_param = self.gait_manager.get_gait_param()
        walking_param['body_height'] = 0.015
        walking_param['pelvis_offset'] = 7
        walking_param['step_height'] = 0.02
        walking_param['hip_pitch_offset'] = 10
        walking_param['z_swap_amplitude'] = 0.006
        self.gait_manager.set_step([600, 0.2, 0.04], 0.02, 0, 0, walking_param, 0, 4)
        self.motion_manager.run_action('stand')
        return True

    # Color recognition callback(颜色识别回调)
    def get_color_callback(self, msg):
        # Obtain color recognition result(获取颜色识别结果)
        self.objects_info = msg.data

    def run(self):
        while self.running:
            if self.start:
                line_data = None
                cross_data = None
                for object_info in self.objects_info:
                    print(object_info.type)
                    if object_info.type == 'line':
                        line_data = object_info
                    if object_info.type == 'cross':
                        cross_data = object_info
                
                if self.current_state == 'visual_patrol':
                    if line_data is not None:
                        self.visual_patrol.process(line_data.x, line_data.width)
                elif self.current_state == 'cross_detect':
                    if self.exit_cross_detect():
                        self.current_state = 'visual_patrol'
                        self.next_state = 'cross_detect'
                        self.state[self.current_state][2] = False
                
                if self.next_state == 'cross_detect':
                    if self.enter_cross_detect(cross_data):
                        self.current_state = 'cross_detect'
                        self.next_state = 'visual_patrol'
                
                self.state_init(self.current_state, self.next_state)

                time.sleep(0.01)
            else:
                time.sleep(0.01)

        self.init_action(self.head_pan_init, self.head_tilt_init)
        self.stop_srv_callback(None)
        rospy.signal_shutdown('shutdown')

if __name__ == "__main__":
    CrossDetectNode('cross_detect').run()
