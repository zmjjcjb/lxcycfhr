#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/09/27
# @author:aiden

# 步态姿态单一来源，Constants 里要引用它，故 import 必须在 class 之前
from rosman_example import gait_posture

# 【新增】常量定义，替代魔法数字
class Constants:
    # 坐标常量
    CAMERA_CENTER_X = 320
    CAMERA_CENTER_Y = 240
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480

    # 步态参数
    DEFAULT_DSP = gait_posture.DSP
    DEFAULT_X_MAX = gait_posture.X_MAX

    # 动作延迟（秒）
    SHORT_DELAY = 0.1
    MEDIUM_DELAY = 0.3
    LONG_DELAY = 0.5
    EXTRA_LONG_DELAY = 2.0

    # 计数阈值
    BLOCK_DETECT_COUNT = 10
    GO_AROUND_THRESHOLD = 10
    STAIRS_ENTER_COUNT = 5
    STAIRS_DESCEND_COUNT = 2
    HURDLES_ENTER_COUNT = 3

    # Y坐标阈值（像素）
    CLIMB_STAIRS_ENTER_Y = 275
    CLIMB_STAIRS_Y_STOP = 400
    DESCEND_STAIRS_ENTER_Y = 350
    DESCEND_STAIRS_Y_STOP = 250
    HURDLES_ENTER_Y = 250
    HURDLES_Y_STOP = 345
    GO_AROUND_Y = 300

    # X坐标阈值（像素）
    GO_AROUND_X = 450
    INTERSECTION_X_THRESHOLD = 550
    BLOCK_X_THRESHOLD = 480

    # 转向步数
    HURDLES_TURN_RIGHT_COUNT = 7
    CRAWL_LEFT_TURN_COUNT = 22
    ALIGN_TURN_COUNT = 3

    # 前进步数
    TRANSITION_STEP_COUNT = 3
    FORWARD_AFTER_ALIGN_COUNT = 15

    # YOLO相关
    YOLO_TIMEOUT = 15
    FORWARD_TO_TARGET_COUNT = 26
    SIDE_STEP_COUNT = 16

import rospy
import signal
import time
from rosman_sdk import misc, common
from rosman_example.color_common import Common
from rosman_example.visual_patrol import VisualPatrol
from rosman_example.approach_object import ApproachObject
from rosman_interfaces.srv import SetString
from rosman_interfaces.msg import ObjectsInfo, ColorDetect, ROI
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import subprocess

ai_x,ai_y=320,240
is_save_img=False
saved_img_num=0
go_left_signal=False
ii=0

def image_callback(msg):
    global ai_x,ai_y,is_save_img,saved_img_num,ii

    ai_x,ai_y=320,140

    max_area=1000
    max_area_index=None

    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    
    if is_save_img:
        cv2.imwrite(f'/home/ubuntu/ros_ws/src/rosman_example/scripts/img__/{saved_img_num}.jpg',cv_image)
        ii=ii+1
        if ii>12:
            ii=0
        saved_img_num=saved_img_num+1
        if saved_img_num>5:
            is_save_img=False
    lab_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab_img)
    
    # 计算各通道阈值掩码
    mask_L = cv2.inRange(L, 0, 255)
    mask_A = cv2.inRange(A, 148, 255)
    mask_B = cv2.inRange(B, 118, 255)
    
    # 三通道掩码交集（只有全部满足才保留）
    mask = cv2.bitwise_and(mask_L, cv2.bitwise_and(mask_A, mask_B))

    contours, _ = cv2.findContours(
        mask, 
        cv2.RETR_EXTERNAL,  # 只提取外轮廓
        cv2.CHAIN_APPROX_SIMPLE  # 简化轮廓（减少顶点数）
    )
    if not contours:
        return
    
    for i,cnt in enumerate(contours):
        _, _, w, h = cv2.boundingRect(cnt)
        rect_area = w * h  # 计算矩形面积
        if rect_area > max_area: 
            max_area=rect_area
            max_area_index=i
    if max_area_index is not None:
        x, y, w, h = cv2.boundingRect(contours[max_area_index])
        ai_x=x+int(w/2)
        ai_y=y+int(h/2)
        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 5)
    
    cv2.imshow("cs", cv_image)
    cv2.waitKey(1)

class CompetitionNode(Common):
    # 按顺序检测三个roi，如果检测到黑线立刻跳出
    line_roi = [(5 / 12, 6 / 12, 1 / 4, 3 / 4),
                (6 / 12, 7 / 12, 0, 1),
                (7 / 12, 8 / 12, 0, 1)
                ]

    stairs_roi = [1 / 5, 1, 0.15, 0.85]
    hurdles_roi = [1 / 5, 1, 0.3, 0.7]
    block_roi = [0, 270 / 480, 0, 1]
    intersection_roi = [100 / 480, 300 / 480, 0, 1]
    
    # 所需动作的名称
    hurdles_action_name = 'hurdles_competition_2'
    climb_stairs_action_name = 'climb_stairs'
    descend_stairs_action_name = 'descend_stairs'
    crawl_left_action_name = 'crawl_two_hands_and_stand'   
    crawl_right_action_name = 'crawl_right_2025' 
    place_block_action_name = 'place_block' 
    forward_step = 'forward_one_step'   
    back_step = 'back_step_1'             
    move_left = 'move_left_1'             
    move_right = 'move_right_1'           
    
    move_left_1 = 'move_left_hand_up'             
    move_right_1 = 'move_right_hand_up' 

    go_turn_left = 'go_turn_left_20'      
    turn_right = 'turn_right_30'
    turn_left = 'turn_left_30'
    
    turn_right_open = 'turn_right_30_open_arm'
    turn_left_open = 'turn_left_30_open_arm'

    go_turn_right_open = 'go_turn_right_20_open_arm'
    go_turn_left_open = 'go_turn_left_20_open_arm'

    throw_action_name = 'throw_sandbag'
    crawl_military_name ='throw_two_hands'
    go_turn_right = 'go_turn_right_20'
    forward_step_1 = 'forward_step'
    forward_step_open_arm = 'forward_step_open_arm_1'
    place_big_pink_block_action_name = 'place_big_pink_block'
    stand_open_arm_slow='stand_open_arm_slow'
    
    # 图像处理时缩放到这个分辨率， 不建议修改
    image_process_size = [160, 120]
    
    enter_climb_stairs_y = 275 / 480  
    climb_stairs_x_stop = 0.5  
    climb_stairs_y_stop = 400 / 480  
    climb_stairs_yaw_stop = 5  
    
    enter_descend_stairs_y = 350/480
    descend_stairs_x_stop = 0.45
    descend_stairs_y_stop = 250 / 480
    descend_stairs_yaw_stop = 0
    
    enter_hurdles_y = 250 / 480
    hurdles_x_stop = 0.5
    hurdles_y_stop = 380 / 480
    hurdles_yaw_stop = 0
    
    enter_crawl_left_y = 80 / 480 
    crawl_left_x_stop = 195 / 640  
    crawl_left_y_stop = 225 / 480  
    crawl_left_yaw_stop = 0        
    
    enter_crawl_right_y = 110 / 480
    crawl_right_x_stop = 390 / 640 
    crawl_right_y_stop = 255 / 480
    crawl_right_yaw_stop = 0
    
    enter_place_block_y = 125 / 480
    place_block_x_stop = 320 / 640
    place_block_y_stop = 250 / 480
    place_block_yaw_stop = 0
    
    enter_throw_y=450/480
    enter_throw_x=320/640
    exit_throw_y=450/480
    exit_throw_x=320/640
    throw_x_offset=0
    throw_y_offset=0
    
    # 【修复】：提早绕行触发的视觉 Y 轴门槛
    go_around_y=300/480
    go_around_x=450/480
    
    hurdles_approach_signal = False
    climb_stairs_approach_signal = False
    descend_stairs_approach_signal = False
    place_block_approach_signal = True
    go_around_forward_signal = True
    enter_crawl_left_signal = False
    
    exist_stairs_count = 0
    exist_right_block_count = 0
    go_around_forward_signal = False
    start_forward_signal = True
    go_around_signal = 0
    turn_left_count = 0
    forward_step_count = 0
    hurdles_forward_count = 0
    the_first_y = 0
    temp_arg=0
    line_data_count = 0

    # 【新增】通用动作序列执行函数
    def execute_action_sequence(self, action_name, forward_steps=3, reset_gait=True):
        """执行标准动作序列：禁用步态 → 执行动作 → 立正复位 → 过渡步数 → 重置步态"""
        self.gait_manager.disable()
        rospy.sleep(Constants.MEDIUM_DELAY)
        common.loginfo(action_name)
        self.motion_manager.run_action(action_name)
        rospy.sleep(Constants.LONG_DELAY)
        self.motion_manager.run_action('walk_ready')
        rospy.sleep(Constants.LONG_DELAY)
        for _ in range(forward_steps):
            self.motion_manager.run_action(self.forward_step_1)
        if reset_gait:
            self.visual_patrol.update_go_gait(dsp=Constants.DEFAULT_DSP, x_max=Constants.DEFAULT_X_MAX)
            self.visual_patrol.update_turn_gait(dsp=Constants.DEFAULT_DSP, x_max=Constants.DEFAULT_X_MAX)

    # 【新增】重复执行动作的辅助函数
    def run_action_repeated(self, action_name, count):
        for _ in range(count):
            self.motion_manager.run_action(action_name)

    def __init__(self, name):
        rospy.init_node(name)
        self.calib_config = common.get_yaml_data('/home/ubuntu/ros_ws/src/rosman_example/config/calib.yaml')
        self.name = name
        self.count = 0
        self.running = True
        self.slow = True
        self.forward_signal = True      
        self.objects_info = []
        self.delay_time = 0
        self.current_state = "visual_patrol"            
        self.next_state = "go_around"

        self.state = {'visual_patrol':  [[500, 260], ['white', self.line_roi, self.image_process_size, self.set_visual_patrol_color], False],        
                      'climb_stairs':   [[500, 260], ['green', self.stairs_roi, self.image_process_size, self.set_stairs_color], False],               
                      'descend_stairs': [[500, 260], ['green', self.stairs_roi, self.image_process_size, self.set_stairs_color], False],               
                      'hurdles': [[500, 260], ['blue', self.hurdles_roi, self.image_process_size, self.set_hurdles_color], False],                   
                      'crawl_left':   [[500, 260], ['white', self.intersection_roi, self.image_process_size, self.set_intersection_color], False],                    
                      'crawl_right': [[500, 260], ['green', self.block_roi, self.image_process_size, self.set_block_color], False],                     
                      'place_block':   [[500, 450], ['white', self.intersection_roi, self.image_process_size, self.set_intersection_color], False],     
                      'go_around':  [[500, 260],['red', self.block_roi, self.image_process_size, self.set_block_color], False],      
                      'throw':   [[500, 260], ['purple', self.block_roi, self.image_process_size, self.set_block_color], False],        
                      'ai-task':   [[500, 260], ['pink', self.block_roi, self.image_process_size, self.set_block_color], False] 
                      }     
        
        self.head_pan_init = self.state[self.current_state][0][0]   
        self.head_tilt_init = self.state[self.current_state][0][1]  
        super().__init__(name, self.head_pan_init, self.head_tilt_init)

        self.approach_object = ApproachObject(self.gait_manager)
        self.visual_patrol = VisualPatrol(self.gait_manager)
        
        # 步幅统一由 gait_posture.X_MAX 决定，VisualPatrol 构造时已生效。
        # 这里原来写的是 update_go_gait/update_turn_gait(x_max=0.02)，把步幅顶回厂家上限，
        # 是巡线前倾摔倒的直接推手之一，故删除。

        signal.signal(signal.SIGINT, self.shutdown)
        
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)
        rospy.Service('~set_color', SetString, self.set_color_srv_callback)  
        self.motion_manager.run_action('walk_ready')

        if rospy.get_param('~start', True):
            self.enter_func(None)
            self.start_srv_callback(None)
            common.loginfo('start combination')

    def shutdown(self, signum, frame):
        self.running = False
        common.loginfo('%s shutdown' % self.name)

    def set_visual_patrol_color(self, color, roi, image_process_size):
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

    def set_block_color(self, color, roi, image_process_size):
        block_param = ColorDetect()
        block_param.color_name = color
        block_param.detect_type = 'circle'
        block_param.use_name = True
        block_param.image_process_size = self.image_process_size
        block_param.roi.y_min = int(roi[0] * image_process_size[1])
        block_param.roi.y_max = int(roi[1] * image_process_size[1])
        block_param.roi.x_min = int(roi[2] * image_process_size[0])
        block_param.roi.x_max = int(roi[3] * image_process_size[0])
        block_param.min_area = 10
        block_param.max_area = image_process_size[0]*image_process_size[1]
        return block_param

    def set_intersection_color(self, color, roi, image_process_size):
        intersection_param = ColorDetect()
        intersection_param.color_name = color
        intersection_param.detect_type = 'intersection'
        intersection_param.use_name = True
        intersection_param.image_process_size = image_process_size
        intersection_param.roi.y_min = int(roi[0] * image_process_size[1])
        intersection_param.roi.y_max = int(roi[1] * image_process_size[1])
        intersection_param.roi.x_min = int(roi[2] * image_process_size[0])
        intersection_param.roi.x_max = int(roi[3] * image_process_size[0])
        intersection_param.min_area = 10
        intersection_param.max_area = image_process_size[0] * image_process_size[1]
        return intersection_param

    def set_stairs_color(self, color, roi, image_process_size):
        stairs_param = ColorDetect()
        stairs_param.color_name = color
        stairs_param.detect_type = 'side'
        stairs_param.use_name = True
        stairs_param.image_process_size = image_process_size
        stairs_param.roi.y_min = int(roi[0] * image_process_size[1])
        stairs_param.roi.y_max = int(roi[1] * image_process_size[1])
        stairs_param.roi.x_min = int(roi[2] * image_process_size[0])
        stairs_param.roi.x_max = int(roi[3] * image_process_size[0])
        stairs_param.min_area = 10*20
        stairs_param.max_area = image_process_size[0]*image_process_size[1]
        return stairs_param

    def set_hurdles_color(self, color, roi, image_process_size):
        hurdles_param = ColorDetect()
        hurdles_param.color_name = color
        hurdles_param.detect_type = 'side'
        hurdles_param.use_name = True
        hurdles_param.image_process_size = image_process_size
        hurdles_param.roi.y_min = int(roi[0] * image_process_size[1])
        hurdles_param.roi.y_max = int(roi[1] * image_process_size[1])
        hurdles_param.roi.x_min = int(roi[2] * image_process_size[0])
        hurdles_param.roi.x_max = int(roi[3] * image_process_size[0])
        hurdles_param.min_area = 10 * 20
        hurdles_param.max_area = image_process_size[0] * image_process_size[1]
        return hurdles_param

    def set_color_srv_callback(self, msg):
        block_param = self.set_block_color(self.state['crawl_left'][1][0])
        stairs_param = self.set_stairs_color(self.state['climb_stairs'][1][0])
        line_param = self.set_visual_patrol_color(self.state['visual_patrol'][1][0])
        hurdles_param = self.set_hurdles_color(self.state['hurdles'][1][0])
        intersection_param = self.set_intersection_color(self.state['place_block'][1][0])

        self.detect_pub.publish([line_param, stairs_param, hurdles_param, intersection_param, block_param])
        common.loginfo('%s set_color' % self.name)
        return [True, 'set_color']

    def get_color_callback(self, msg):
        self.objects_info = msg.data

    def state_init(self, current_state, next_state):
        if self.state[current_state][2] == False:
            self.state[current_state][2] = True
            self.init_action(self.state[current_state][0][0], self.state[current_state][0][1])  
            param1 = self.state[current_state][1][3](self.state[current_state][1][0], self.state[current_state][1][1], self.state[current_state][1][2])
            param2 = self.state[next_state][1][3](self.state[next_state][1][0], self.state[next_state][1][1], self.state[next_state][1][2])
            self.detect_pub.publish([param1, param2])  
            common.loginfo(current_state + ' init')
            
    def exit_climb_stairs(self, stairs_data):
        if stairs_data is not None:
            if self.approach_object.process(max(stairs_data.y, stairs_data.left_point[1], stairs_data.right_point[1]), stairs_data.x, stairs_data.angle,
                                        self.climb_stairs_y_stop*stairs_data.height, self.climb_stairs_x_stop*stairs_data.width, self.climb_stairs_yaw_stop, stairs_data.width, stairs_data.height):
                # 根据距离决定是否先走一步
                if stairs_data.y < Constants.CLIMB_STAIRS_Y_STOP:
                    self.motion_manager.run_action(self.forward_step)
                self.execute_action_sequence(self.climb_stairs_action_name)
                return True

        if self.climb_stairs_approach_signal:
            self.execute_action_sequence(self.climb_stairs_action_name)
            return True

        self.climb_stairs_approach_signal = True
        return False
        
    def exit_descend_stairs(self, stairs_data):
        if stairs_data is not None:
            if self.approach_object.process(max(stairs_data.y, stairs_data.left_point[1], stairs_data.right_point[1]), stairs_data.x + self.calib_config['center_x_offset'], stairs_data.angle,
                                        self.descend_stairs_y_stop*stairs_data.height, self.descend_stairs_x_stop*stairs_data.width, self.descend_stairs_yaw_stop, stairs_data.width, stairs_data.height):
                # 根据距离决定额外前进步数
                self.run_action_repeated(self.forward_step_1, 1)
                if stairs_data.y < Constants.DESCEND_STAIRS_Y_STOP:
                    self.run_action_repeated(self.forward_step_1, 1)

                self.execute_action_sequence(self.descend_stairs_action_name)
                self.slow = True
                return True

        if self.descend_stairs_approach_signal:
            self.execute_action_sequence(self.descend_stairs_action_name)
            self.slow = True
            return True

        self.descend_stairs_approach_signal = True
        return False
        
    def exit_hurdles(self, hurdles_data):
        if hurdles_data is not None:
            if not self.hurdles_approach_signal:
                if self.approach_object.process(max(hurdles_data.y, hurdles_data.left_point[1], hurdles_data.right_point[1]),
                                                hurdles_data.x + self.calib_config['center_x_offset'], hurdles_data.angle,
                                                self.hurdles_y_stop*hurdles_data.height, self.hurdles_x_stop*hurdles_data.width,
                                                self.hurdles_yaw_stop, hurdles_data.width, hurdles_data.height):
                    self.gait_manager.disable()
                    common.loginfo('hurdles')
                    self.hurdles_approach_signal = True
                return False

            # 接近时前进一步调整位置
            if hurdles_data.y < Constants.HURDLES_Y_STOP and self.hurdles_forward_count < 2 and self.the_first_y < hurdles_data.y:
                self.the_first_y = hurdles_data.y
                self.run_action_repeated(self.forward_step_1, 1)
                self.hurdles_forward_count += 1
                return False

            self.execute_action_sequence(self.hurdles_action_name)
            self.slow = False
            return True

        if self.hurdles_approach_signal:
            self.execute_action_sequence(self.hurdles_action_name)
            self.slow = False
            return True

        self.hurdles_approach_signal = True
        return False
        
    def enter_climb_stairs(self, stairs_data):
        if stairs_data is not None:
            self.exist_stairs_count += 1
            if stairs_data.y > self.stairs_roi[0] * stairs_data.height and self.exist_stairs_count > 3:
                self.visual_patrol.update_go_gait(dsp=Constants.DEFAULT_DSP, x_max=Constants.DEFAULT_X_MAX)
                self.visual_patrol.update_turn_gait(dsp=Constants.DEFAULT_DSP, x_max=Constants.DEFAULT_X_MAX)
                self.slow = False
            if max(stairs_data.y, stairs_data.left_point[1], stairs_data.right_point[1]) > self.enter_climb_stairs_y * stairs_data.height:
                self.count += 1
                if self.count > Constants.STAIRS_ENTER_COUNT:
                    self.count = 0
                    self.gait_manager.disable()
                    self.approach_object.update_approach_stop_value(20, 30, 6)
                    self.motion_manager.run_action('hand_back')
                    rospy.sleep(Constants.LONG_DELAY)
                    return True
            else:
                self.count = 0
        return False

    def enter_descend_stairs(self, stairs_data):
        if stairs_data is not None:
            if max(stairs_data.y, stairs_data.left_point[1], stairs_data.right_point[1]) > self.enter_descend_stairs_y * stairs_data.height:
                self.count += 1
                if self.count > Constants.STAIRS_DESCEND_COUNT:
                    self.count = 0
                    self.gait_manager.disable()
                    self.approach_object.update_approach_stop_value(20, 30, 6)
                    self.motion_manager.run_action('hand_back')
                    rospy.sleep(Constants.LONG_DELAY)
                    return True
            else:
                self.count = 0
        return False
        
    def enter_hurdles(self, hurdles_data):
        if hurdles_data is not None:
            if hurdles_data.y > self.hurdles_roi[0] * hurdles_data.height and self.slow:
                self.visual_patrol.update_go_gait(dsp=Constants.DEFAULT_DSP, x_max=Constants.DEFAULT_X_MAX)
                self.visual_patrol.update_turn_gait(dsp=Constants.DEFAULT_DSP, x_max=Constants.DEFAULT_X_MAX)
                self.slow = False
            if max(hurdles_data.y, hurdles_data.left_point[1], hurdles_data.right_point[1]) > self.enter_hurdles_y * hurdles_data.height:
                self.count += 1
                if self.count > Constants.HURDLES_ENTER_COUNT:
                    self.count = 0
                    self.gait_manager.disable()
                    self.approach_object.update_approach_stop_value(x_approach_value=30, yaw_approach_value=5)
                    self.motion_manager.run_action('hand_back')
                    self.run_action_repeated(self.turn_right, Constants.HURDLES_TURN_RIGHT_COUNT)
                    return True
            else:
                self.count = 0
        return False

    def exit_crawl_left(self, line_data):
        if line_data is not None:
            if self.place_block_approach_signal and self.approach_object.process(max(line_data.y, line_data.left_point[1], line_data.right_point[1]), line_data.x, line_data.angle, 
                                            self.place_block_y_stop*line_data.height, self.place_block_x_stop*line_data.width, self.place_block_yaw_stop, line_data.width, line_data.height):
                self.place_block_approach_signal = False
            
            if line_data.y < 300:
                self.motion_manager.run_action(self.forward_step_1)
                return False
            return True
        return True

    def enter_crawl_left(self, line_data):
        if line_data is not None and abs(line_data.angle) < 10:
            self.line_data_count += 1
            self.visual_patrol.update_go_gait(dsp=Constants.DEFAULT_DSP, x_max=Constants.DEFAULT_X_MAX)
            if line_data.y > 250 and self.line_data_count > 5:
                return True
        return False

    def enter_throw(self, block_data):
        return True
    
    def exit_throw(self, block_data):
        if block_data is not None:
            # 对准目标
            if block_data.y - 180 < -5:
                self.motion_manager.run_action(self.forward_step_1)
                return False
            if block_data.x - Constants.CAMERA_CENTER_X > 5:
                self.motion_manager.run_action(self.move_right)
                return False
            if block_data.x - Constants.CAMERA_CENTER_X < -5:
                self.motion_manager.run_action(self.move_left)
                return False

            # 执行投掷动作序列
            self.gait_manager.disable()
            rospy.sleep(Constants.MEDIUM_DELAY)
            self.motion_manager.run_action('hand_back')
            self.motion_manager.run_action(self.crawl_left_action_name)
            rospy.sleep(Constants.EXTRA_LONG_DELAY)
            self.run_action_repeated(self.forward_step_open_arm, 7)
            self.run_action_repeated(self.turn_right_open, Constants.ALIGN_TURN_COUNT)
            rospy.sleep(Constants.SHORT_DELAY)
            self.motion_manager.run_action(self.throw_action_name)

            # 复位
            self.motion_manager.run_action('walk_ready')
            rospy.sleep(Constants.LONG_DELAY)

            # 重新初始化步态
            self.visual_patrol.update_go_gait(arm_swap=0)
            self.visual_patrol.update_turn_gait(arm_swap=0)
            self.visual_patrol.update_go_gait(dsp=Constants.DEFAULT_DSP, x_max=Constants.DEFAULT_X_MAX)
            self.visual_patrol.update_turn_gait(dsp=Constants.DEFAULT_DSP, x_max=Constants.DEFAULT_X_MAX)
            self.slow = False
            return True

        return False
    
    def enter_go_around(self, block_data):
        if block_data is not None:
            if block_data.y > self.block_roi[0] * block_data.height and self.exist_right_block_count > Constants.BLOCK_DETECT_COUNT:
                self.visual_patrol.update_go_gait(dsp=Constants.DEFAULT_DSP, x_max=Constants.DEFAULT_X_MAX)
                self.visual_patrol.update_turn_gait(dsp=Constants.DEFAULT_DSP, x_max=Constants.DEFAULT_X_MAX)
                self.slow = False
            self.go_around_signal += 1
            if self.go_around_signal > Constants.GO_AROUND_THRESHOLD:
                common.loginfo('[绕行] 连续%d帧检测到障碍物，进入绕行状态' % self.go_around_signal)
                self.gait_manager.disable()
                return True
            return False

        time.sleep(0.01)
        return False

    # v9 绕行逻辑：转向结束后如果看不到红色障碍物，立即进入下一流程
    def exit_go_around(self, block_data):
        global go_left_signal

        if block_data is not None:
            # 还未靠近障碍物，继续前进靠近
            if block_data.y < 20:
                common.loginfo('[绕行] 靠近障碍物中... y=%d' % block_data.y)
                self.run_action_repeated(self.forward_step_1, 1)
                return False

            # 阶段1：左转离开危险区并前进绕过
            if block_data.x <= Constants.BLOCK_X_THRESHOLD and not go_left_signal:
                common.loginfo('[绕行] 检测到障碍物 x=%d, 开始左转绕行' % block_data.x)
                self.run_action_repeated(self.go_turn_left, 4)
                go_left_signal = True
                common.loginfo('[绕行] 左转4步完成，前进8步绕过障碍物')
                self.run_action_repeated(self.forward_step_1, 9)
                # 左转+前进完成，检查障碍物是否还在视野
                return False

            # 阶段2：右转回正对准终点横线
            if go_left_signal and self.go_around_forward_signal is False:
                common.loginfo('[绕行] 右转4步，回正对准终点横线')
                self.run_action_repeated(self.go_turn_right, 4)
                self.go_around_forward_signal = True
                # 右转完成，检查障碍物是否还在视野
                return False

            # 绕行中，继续前进
            common.loginfo('[绕行] 前进中... 障碍物位置 x=%d y=%d' % (block_data.x, block_data.y))
            self.run_action_repeated(self.forward_step_1, 1)
            return False

        # 转向结束后看不到红色障碍物，立即进入下一流程
        if go_left_signal:
            common.loginfo('[绕行] 障碍物已消失（已完成%d次转向），退出绕行' %
                           (2 if self.go_around_forward_signal else 1))
            go_left_signal = False
            self.go_around_forward_signal = True
            return True

        return False
            
    def start_forward(self, step_count):
        # 原来这里组好 walking_param 却给 set_step 传了 None，上面 7 行全部作废；
        # 而且第 6 个位置参数是 arm_swap(单位度)，填 0.5 等于手臂基本不摆(radians(30)≈0.52)。
        # 改为套用 gait_posture 的统一姿态，不再在这里另写一份 init_x_offset=-0.02 的前倾值。
        self.gait_manager.set_step(gait_posture.DSP_FAST, gait_posture.X_MAX, 0, 0,
                                   gait_posture.build(self.gait_manager),
                                   arm_swap=gait_posture.ARM_SWAP, step_num=step_count)
        
    # 【新增】统一处理 objects_info 过滤，避免重复代码
    def parse_objects_info(self):
        line_data = None
        side_data = None
        block_data = None
        intersection_data = None
        for object_info in self.objects_info:
            if object_info.type == 'line':
                line_data = object_info
            elif object_info.type == 'side':
                side_data = object_info
            elif object_info.type == 'circle':
                block_data = object_info
            elif object_info.type == 'intersection':
                intersection_data = object_info
        return line_data, side_data, block_data, intersection_data

    def run(self):
        global is_save_img

        while self.running:
            if self.start:
                line_data, side_data, block_data, intersection_data = self.parse_objects_info()
                
                if self.start_forward_signal:
                    for i in range(5):
                        self.motion_manager.run_action(self.forward_step_1)
                    self.start_forward_signal = False

                if self.current_state == 'visual_patrol':
                    if block_data is not None:
                        # 第二个参数是图像宽度(ObjectInfo.width, color_detection 填的是 img_w)，
                        # 原来传 block_data.y 等于把像素行当宽度，center_x 会算错
                        self.visual_patrol.process(block_data.x, block_data.width)
                    else:
                        # 丢线必须主动接管，否则连续步态会沿用上一条已饱和的转向继续前冲
                        self.visual_patrol.on_no_line()
                elif self.current_state == 'go_around':
                    if self.exit_go_around(block_data):
                        common.loginfo('[绕行] 绕行完成，开始寻找十字交叉线')

                        # 原来这里组了 7 行 walking_param 但从未传给任何 set_step，是死代码
                        # （里面还藏着一个 init_x_offset=-0.015 的前倾值，容易误导后续调参），已删。
                        self.init_action(500, 350)
                        param1 = self.state['visual_patrol'][1][3](self.state['visual_patrol'][1][0], self.state['visual_patrol'][1][1], self.state['visual_patrol'][1][2])
                        param2 = self.state['place_block'][1][3](self.state['place_block'][1][0], self.state['place_block'][1][1], self.state['place_block'][1][2])
                        self.detect_pub.publish([param1, param2])
                        
                        while intersection_data is None:
                            self.motion_manager.run_action(self.go_turn_right)
                            line_data, side_data, block_data, intersection_data = self.parse_objects_info()
                        while 1:
                            line_data, side_data, block_data, intersection_data = self.parse_objects_info()
                            if intersection_data is not None:
                                if intersection_data.x > Constants.INTERSECTION_X_THRESHOLD:
                                    self.motion_manager.run_action(self.go_turn_right)
                                if intersection_data.y < Constants.CAMERA_CENTER_Y:
                                    self.motion_manager.run_action(self.forward_step_1)
                                else:
                                    break
                            else:
                                self.motion_manager.run_action(self.go_turn_right)
                                break
                        rospy.sleep(0.1)
                        while line_data is None:
                            self.motion_manager.run_action(self.forward_step_1)
                            line_data, side_data, block_data, intersection_data = self.parse_objects_info()
                        self.init_action(500, 300)
                        line_data=None
                        while line_data is None:
                            self.motion_manager.run_action(self.forward_step_1)
                            line_data, side_data, block_data, intersection_data = self.parse_objects_info()
                        
                        # 【修复】：回正多转 2 步，越过障碍物多走 15 步
                        self.run_action_repeated(self.go_turn_left, 2)
                        self.run_action_repeated(self.forward_step_1, Constants.FORWARD_AFTER_ALIGN_COUNT)
                            
                        line_data=None
                        while line_data is None:
                            time.sleep(0.1)
                            self.motion_manager.run_action(self.go_turn_right)
                            line_data, side_data, block_data, intersection_data = self.parse_objects_info()
                                    
                        self.current_state = 'visual_patrol'
                        self.next_state = 'hurdles'
                        self.state[self.current_state][2] = False  
                        common.loginfo('exit go_around ---> enter visual_patrol')
                    else:
                        rospy.sleep(0.8)

                if self.current_state == 'visual_patrol':
                    if line_data is not None:
                        self.visual_patrol.process(line_data.x, line_data.width)
                    else:
                        self.visual_patrol.on_no_line()
                elif self.current_state == 'hurdles':
                    if self.exit_hurdles(side_data):
                        self.run_action_repeated(self.turn_right, Constants.ALIGN_TURN_COUNT)
                        self.current_state = 'visual_patrol'
                        self.next_state = 'climb_stairs'
                        self.state[self.current_state][2] = False  
                        common.loginfo('exit hurdles ---> enter visual_patrol')
                    else:
                        rospy.sleep(0.8)

                if self.current_state == 'visual_patrol':
                    if line_data is not None:
                        self.visual_patrol.process(line_data.x, line_data.width)
                    else:
                        self.visual_patrol.on_no_line()
                elif self.current_state == 'climb_stairs':
                    if self.exit_climb_stairs(side_data):
                        self.current_state = 'visual_patrol'
                        self.next_state = 'descend_stairs'
                        self.state[self.current_state][2] = False  
                        common.loginfo('exit climb_stairs ---> enter visual_patrol')
                    else:
                        rospy.sleep(0.8)

                elif self.current_state == 'descend_stairs':
                    if self.exit_descend_stairs(side_data):
                        self.current_state = 'visual_patrol'
                        self.next_state = 'crawl_left'
                        self.state[self.current_state][2] = False
                        common.loginfo('exit descend_stairs ---> enter visual_patrol')
                    else:
                        rospy.sleep(0.8)

                elif self.current_state == 'crawl_left':
                    if self.exit_crawl_left(intersection_data):
                        for i in range(2):
                            self.motion_manager.run_action(self.turn_right)
                        self.current_state = 'visual_patrol'
                        self.next_state = 'crawl_right'
                        self.state[self.current_state][2] = False
                        rospy.sleep(1)
                        self.running = False
                        common.loginfo('end')
                    else:
                        rospy.sleep(0.8)  

                elif self.current_state == 'throw':
                    if self.exit_throw(block_data):
                        with open('/home/ubuntu/ros_ws/src/rosman_example/scripts/txt/YOLO_bridge_ros.txt','w') as f:
                            f.write('')
                        with open('/home/ubuntu/ros_ws/src/rosman_example/scripts/txt/YOLO_class.txt','w') as f:
                            f.write('')
                        print('中转文件清空')
                        self.current_state = 'visual_patrol'
                        self.next_state = 'crawl_right'
                        self.state[self.current_state][2] = False
                        rospy.sleep(1)
                        param1 = self.state['visual_patrol'][1][3](self.state['visual_patrol'][1][0], self.state['visual_patrol'][1][1], self.state['visual_patrol'][1][2])
                        param2 = self.state['ai-task'][1][3](self.state['ai-task'][1][0], self.state['ai-task'][1][1], self.state['ai-task'][1][2])
                        self.detect_pub.publish([param1, param2])
                        
                        rospy.Subscriber('/camera/image_raw', Image, image_callback)
                        print('相机话题订阅完成')
                        self.run_action_repeated(self.turn_left, Constants.CRAWL_LEFT_TURN_COUNT)
                        self.init_action(500, 400)
                        
                        start=time.time()
                        while 1:      
                            now=time.time()
                            if (ai_x - Constants.CAMERA_CENTER_X) > 60:
                                self.motion_manager.run_action(self.turn_right)
                            elif (ai_x - Constants.CAMERA_CENTER_X) < -60:
                                self.motion_manager.run_action(self.turn_left)
                            elif ai_y < 430:
                                self.motion_manager.run_action(self.forward_step)
                            elif ai_y >= 430:
                                if (now - start) > Constants.YOLO_TIMEOUT:
                                    self.init_action(500, 260)
                                    while 1:
                                        if (ai_x - Constants.CAMERA_CENTER_X) > 60:
                                            self.motion_manager.run_action(self.turn_right)
                                        elif (ai_x - Constants.CAMERA_CENTER_X) < -60:
                                            self.motion_manager.run_action(self.turn_left)
                                        elif ai_y < 140:
                                            self.motion_manager.run_action(self.forward_step)
                                        else:
                                            self.run_action_repeated(self.forward_step, Constants.ALIGN_TURN_COUNT)
                                            break
                                break
                        
                        self.init_action(500, 260)
                        is_save_img=True
                        time.sleep(3)   
                        print('完成保存图片')
                        with open('/home/ubuntu/ros_ws/src/rosman_example/scripts/txt/YOLO_bridge_ros.txt','w') as f:
                            f.write('true')
                        
                        subprocess.run(['sudo','/root/miniconda3/envs/yolo/bin/python', '/home/ubuntu/ros_ws/src/rosman_example/scripts/YOLO_cls.py'])
                        time.sleep(2)   
                        print('推理完成')

                        class_=''
                        while 1:
                            with open('/home/ubuntu/ros_ws/src/rosman_example/scripts/txt/YOLO_class.txt','r') as f:
                                class_=f.readline()
                                if class_=='':
                                    time.sleep(1)
                                    continue
                            break

                        self.motion_manager.run_action(self.crawl_military_name)
                        time.sleep(2)
                        self.motion_manager.run_action(self.stand_open_arm_slow)
                        # 根据YOLO分类结果选择路径
                        if class_ == 'equipment':
                            self.run_action_repeated(self.forward_step_open_arm, Constants.FORWARD_TO_TARGET_COUNT)
                        elif class_ == 'doctor':
                            self.run_action_repeated(self.forward_step_open_arm, Constants.FORWARD_TO_TARGET_COUNT)
                            self.run_action_repeated(self.move_left_1, Constants.SIDE_STEP_COUNT)
                        else:
                            self.run_action_repeated(self.forward_step_open_arm, Constants.FORWARD_TO_TARGET_COUNT)
                            self.run_action_repeated(self.move_right_1, Constants.SIDE_STEP_COUNT)
                        self.motion_manager.run_action(self.place_big_pink_block_action_name)
                        
                        print('endend')
                        self.running = False
                        common.loginfo('end')
                    else:
                        rospy.sleep(0.8)  

                if self.next_state == 'go_around':      
                    if self.enter_go_around(block_data):
                        self.current_state = 'go_around'
                        self.next_state = 'visual_patrol'
                        common.loginfo('exit visual_patrol ---> enter go_around')
                elif self.next_state == 'hurdles':
                    if self.enter_hurdles(side_data):
                        self.current_state = 'hurdles'
                        self.next_state = 'visual_patrol'
                        common.loginfo('exit visual_patrol ---> enter hurdles')
                elif self.next_state == 'climb_stairs':
                    if self.enter_climb_stairs(side_data):
                        self.current_state = 'climb_stairs'
                        self.next_state = 'visual_patrol'
                        common.loginfo('exit visual_patrol ---> enter climb_stairs')
                elif self.next_state == 'descend_stairs':
                    if self.enter_descend_stairs(side_data):
                        self.current_state = 'descend_stairs'
                        self.next_state = 'visual_patrol'
                        common.loginfo('exit visual_patrol ---> enter descend_stairs')

                elif self.next_state == 'crawl_left':
                    if self.enter_crawl_left(intersection_data):
                        self.current_state = 'crawl_left'
                        self.next_state = 'throw'
                        common.loginfo('exit visual_patrol ---> enter crawl_left')

                elif self.next_state == 'throw':
                    if self.enter_throw(block_data):
                        self.current_state = 'throw'
                        self.next_state = 'visual_patrol'
                        common.loginfo('exit crawl_left ---> enter throw')
            
                self.state_init(self.current_state, self.next_state)

                rospy.sleep(0.01)  
            else:
                rospy.sleep(0.01)
        
        self.init_action(self.head_pan_init, self.head_tilt_init)
        self.stop_srv_callback(None)
        rospy.signal_shutdown('shutdown')

if __name__ == "__main__":
    CompetitionNode('competition').run()
