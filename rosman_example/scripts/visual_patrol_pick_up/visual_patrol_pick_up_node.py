#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/07/04
# @author:aiden
# Line following and grasping(巡线抓取)
import time
import rospy
import signal
from rosman_sdk import misc, common
from rosman_example.color_common import Common
from rosman_example.visual_patrol import VisualPatrol
from rosman_example.approach_object import ApproachObject
from rosman_interfaces.srv import SetString
from rosman_interfaces.msg import ObjectsInfo, ColorDetect, ROI

class VisualPatrolPickUp(Common):
    # Check three ROIs in order, and immediately exit if a black line is detected(按顺序检测三个roi，如果检测到黑线立刻跳出)
    line_roi = [(5 / 12, 6 / 12, 1 / 4, 3 / 4),
                (6 / 12, 7 / 12, 1 / 4, 3 / 4),
                (7 / 12, 8 / 12, 1 / 4, 3 / 4)
                ]

    block_roi = [0, 2 / 3, 0, 1]
    intersection_roi = [100 / 480, 340 / 480, 0, 1]
    # Threshold values for target position in walking left phase(向左左行走状态下的目标位置阈值)
    enter_crawl_left_y = 290 / 480 # When the detected object's y-coordinate in the image is greater than this value, it enters this phrase(当检测到的标识像素坐标y值占图像的比例大于此值时进入此阶段)
    crawl_left_x_stop = 265 / 640  # The target range of the x-axis position for walking left (can be set in ApproachObject); when the robot reaches this range, it stops moving forward or backward)(左行走的x轴目标位置范围可在ApproachObject里设置)时停止前后移动
    crawl_left_y_stop = 290 / 480  # The target range of the y-axis position for walking left (can be set in ApproachObject); when the robot reaches this range, it stops moving horizontally(左行走的y轴目标位置(范围可在ApproachObject里设置)时停止横向移动)
    crawl_left_yaw_stop = 0        # The target range of the detected object's angle (can be set in ApproachObject); when the detected object's angle is within this range, the robot stops rotating(当检测到的标识角度在此值附近(范围可在ApproachObject里设置)时停止旋转移动)
    # Threshold values for target position in walking right phase(向左左行走状态下的目标位置阈值，原理同上)
    enter_crawl_right_y = 280 / 480
    crawl_right_x_stop = 390 / 640
    crawl_right_y_stop = 272 / 480
    crawl_right_yaw_stop = 0
    # Threshold values for target position in block placing phase(放块状态下的目标位置阈值)
    enter_place_block_y = 320 / 480
    place_block_x_stop = 320 / 640
    place_block_y_stop = 350 / 480
    place_block_yaw_stop = 0

    crawl_left_action_name = 'crawl_left'   # Name of the walking left action(左行走动作名称)
    crawl_right_action_name = 'crawl_right' # Name of the walking right action(右行走动作名称)
    place_block_action_name = 'place_block' # Name of the block placing action(放置方块动作名称)

    image_process_size = [160, 120]

    def __init__(self, name):
        rospy.init_node(name)       # Initialize ROS node(初始化ROS节点)
        self.name = name
        self.running = True         # Run flag(运行标志)
        self.count = 0              # Counter(计数器)
        self.objects_info = []      # Store recognition result(存储识别结果)
        self.current_state = "visual_patrol"    # Current state(当前状态)
        self.next_state = "crawl_left"          # Next state(下一状态)

        # Define state machine(定义状态机)
        self.state = {'visual_patrol':  [[500, 260], ['black', self.line_roi, self.image_process_size, self.set_visual_patrol_color], False],           # 巡线状态(line following state)
                      'crawl_left':   [[500, 260], ['green', self.block_roi, self.image_process_size, self.set_block_color], False],                    # 左爬行状态(left crawling state)
                      'crawl_right': [[500, 260], ['green', self.block_roi, self.image_process_size, self.set_block_color], False],                     # 右爬行状态(right crawling state)
                      'place_block':   [[500, 260], ['black', self.intersection_roi, self.image_process_size, self.set_intersection_color], False]}     # 放块状态(block placing state)
        
        self.head_pan_init = self.state[self.current_state][0][0]   # Initial value of the left-right servo(左右舵机的初始值)
        self.head_tilt_init = self.state[self.current_state][0][1]  # Initial value of the up-down servo(上下舵机的初始值)
        super().__init__(name, self.head_pan_init, self.head_tilt_init) # Initialize parent class(初始化父类)

        self.visual_patrol = VisualPatrol(self.gait_manager)        # Line following node(巡线节点)
        self.visual_patrol.update_go_gait(x_max=0.01)
        self.visual_patrol.update_turn_gait(x_max=0.01)
        self.approach_object = ApproachObject(self.gait_manager)    # Approaching node(逼近节点)
        signal.signal(signal.SIGINT, self.shutdown)

        # Subscribe to color recognition result(订阅颜色识别结果)
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)
        rospy.Service('~set_color', SetString, self.set_color_srv_callback)  # Set color(设置颜色)
        self.motion_manager.run_action('walk_ready')

        if rospy.get_param('~start', True):
            # Notify the color recognition to prepare, at this time only display the camera original image(通知颜色识别准备，此时只显示摄像头原画)
            self.enter_func(None)
            self.start_srv_callback(None)  # Start color recognition(开启颜色识别)
            common.loginfo('start crawl')

    # Close node(关闭节点)
    def shutdown(self, signum, frame):
        self.running = False
        common.loginfo('%s shutdown' % self.name)

    # Generate line following color recognition parameter(生成巡线颜色识别参数)
    def set_visual_patrol_color(self, color, roi, image_process_size):
        # Set line following color(设置巡线颜色)
        line_param = ColorDetect()
        line_param.color_name = color
        line_param.use_name = True
        line_param.detect_type = 'line'     # Detection type is line(检测类型为线条)
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
    # Generate block color recognition parameter(生成方块颜色识别参数)
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
    # Generate intersection color recognition parameter(生成交叉点颜色识别参数)
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
    # Set color service callback function(设置颜色服务回调函数)
    def set_color_srv_callback(self, msg):
        # Set color(设置颜色)
        block_param = self.set_block_color(msg.data)
        line_param = self.set_visual_patrol_color('black')
        intersection_param = self.set_intersection_color('black')
        
        self.detect_pub.publish([line_param, block_param, intersection_param])
        common.loginfo('%s set_color' % self.name)
        
        return [True, 'set_color']
    # Obtain color recognition result callback function(获取颜色识别结果回调函数)
    def get_color_callback(self, msg):
        # Obtain color recognition result(获取颜色识别结果)
        self.objects_info = msg.data
    # State initialization function(状态初始化函数)
    def state_init(self, current_state, next_state):
        # 不同阶段的初始化(initialization in different phrase)
        if self.state[current_state][2] == False:
            self.state[current_state][2] = True
            self.init_action(self.state[current_state][0][0], self.state[current_state][0][1])  # Head posture(头部姿态)
            param1 = self.state[current_state][1][3](self.state[current_state][1][0], self.state[current_state][1][1], self.state[current_state][1][2])
            param2 = self.state[next_state][1][3](self.state[next_state][1][0], self.state[next_state][1][1], self.state[next_state][1][2])
            self.detect_pub.publish([param1, param2])  # Color detection settings(颜色检测设置)
            common.loginfo('%s init' % current_state)
    # Enter left walking determination function(进入左行走判断函数)
    def enter_crawl_left(self, block_data):
        if block_data is not None:
            # If the y-coordinate of the block is greater than 1/4 of its height, part of it is in the ROI area(如果色块的y坐标大于高度的1/4,说明有一部分进入ROI区域)
            if block_data.y > block_data.height / 4:
                self.count += 1
                if self.count > 5:  # If the condition is satisfied continuously for more than 5 times, switch to the walking left phase(连续检测到满足条件,切换到左行走状态)
                    self.count = 0
                    self.gait_manager.disable()
                    self.motion_manager.run_action('hand_back')  # Move the hand back to prevent obstruction(手往后，防止遮挡)
                    return True
            else:
                self.count = 0
        else:
            self.count = 0
        return False
    # Enter the walking right judgment function(进入右行走判断函数)
    def enter_crawl_right(self, block_data):
        if block_data is not None:
            if block_data.y > block_data.height / 4:
                self.count += 1
                if self.count > 5:
                    self.count = 0
                    self.gait_manager.disable()
                    self.motion_manager.run_action('hand_back')  # Move the hand back to prevent obstruction(手往后，防止遮挡)
                    return True
            else:
                self.count = 0
        else:
            self.count = 0
        return False
    # Enter the crawling right judgment function(进入右爬行判断函数)
    def enter_place_block(self, line_data):
        if line_data is not None:
            if max(line_data.y, line_data.left_point[1], line_data.right_point[1]) > self.enter_place_block_y * line_data.height:
                self.count += 1
                if self.count > 2:
                    self.count = 0
                    self.gait_manager.stop()
                    common.loginfo('intersection detect')
                    return True
            else:
                self.count = 0
        else:
            self.count = 0
        return False
    # Exit the walking left judgment function(退出左行走判断函数)
    def exit_crawl_left(self, block_data):
        if block_data is not None:
            if self.approach_object.process(block_data.y, block_data.x, block_data.angle, 
                                            self.crawl_left_y_stop*block_data.height, self.crawl_left_x_stop*block_data.width, self.crawl_left_yaw_stop, block_data.width, block_data.height):
                self.gait_manager.disable()
                common.loginfo('crawl_left')
                self.motion_manager.run_action(self.crawl_left_action_name)
                self.visual_patrol.update_go_gait(arm_swap=0)
                self.visual_patrol.update_turn_gait(arm_swap=0)
                return True
        return False
    # Exit the walking right judgment function(退出右行走判断函数)
    def exit_crawl_right(self, block_data):
        if block_data is not None:
            if self.approach_object.process(block_data.y, block_data.x, block_data.angle, 
                                            self.crawl_right_y_stop*block_data.height, self.crawl_right_x_stop*block_data.width, self.crawl_right_yaw_stop, block_data.width, block_data.height):
                self.gait_manager.disable()
                common.loginfo('crawl_right')
                self.motion_manager.run_action(self.crawl_right_action_name)
                self.visual_patrol.update_go_gait(arm_swap=0)
                self.visual_patrol.update_turn_gait(arm_swap=0)
                return True
        return False
    # Exit the block placing judgment function(退出放块判断函数)
    def exit_place_block(self, line_data):
        if line_data is not None:
            if self.approach_object.process(max(line_data.y, line_data.left_point[1], line_data.right_point[1]), line_data.x, line_data.angle, 
                                            self.place_block_y_stop*line_data.height, self.place_block_x_stop*line_data.width, self.place_block_yaw_stop, line_data.width, line_data.height):
                self.gait_manager.disable()  # Close gait control(关闭步态控制)
                walking_param = self.gait_manager.get_gait_param()  # Get the current gait parameters(获取当前的步态参数)
                walking_param['body_height'] = 0.015                # Set the body height in meters(设置身体高度,单位米)
                walking_param['pelvis_offset'] = 7                  # Set the front and rear offset of the pelvis position in degrees(设置骨盆位置的前后偏移量,单位度)
                walking_param['step_height'] = 0.02                 # Set the step height in meters(设置步高,单位米)
                walking_param['hip_pitch_offset'] = 20              # Set the hip joint angle offset in degrees(设置髋关节角度偏移量,单位度)
                walking_param['z_swap_amplitude'] = 0.006           # Set the amplitude of the left and right foot height alternation in meters(设置左右足高度交替时的振幅,单位米)
                # Call set_step to set the gait parameters, including step length, step frequency, etc(调用set_step 设置步态参数,包括步长、步频等)
                self.gait_manager.set_step([500, 0.2, 0.035], 0.02, 0, 0, walking_param, 0, 3)           
                common.loginfo('place_block')
                self.motion_manager.run_action(self.place_block_action_name)
                return True
        return False
    # program main loop(程序主循环)
    def run(self):
        while self.running:
            if self.start:  # If the start is Ture, it indicates the program starts to run(如果start为True,表示程序开始运行)
                line_data = None
                block_data = None
                intersection_data = None   
                # obtain data of line, block, and cross points from the recognition result(从识别结果中提取出线、方块、交叉点的数据)
                for object_info in self.objects_info:
                    if object_info.type == 'line':
                        line_data = object_info
                    if object_info.type == 'circle':
                        block_data = object_info
                    if object_info.type == 'intersection':
                        intersection_data = object_info

                # The current processing is completed, and return to line following(当前阶段处理完成，回到巡线)
                if self.current_state == 'visual_patrol':
                     # If the current state is line following(如果当前状态是巡线)
                    if line_data is not None:
                        # If the line is detected(如果检测到线条)
                        # Perform line following(执行巡线过程)
                        self.visual_patrol.process(line_data.x, line_data.width)
                elif self.current_state == 'crawl_left':
                    # If the current state is right walking(如果当前状态是左行走)
                    if self.exit_crawl_left(block_data):
                        # If finishing walking to the left(如果完成左行走)
                        self.current_state = 'visual_patrol'
                        self.next_state = 'crawl_right'
                        self.state[self.current_state][2] = False
                    else:
                        time.sleep(0.8)  # Wait for the robot to stabilize(等机体平稳下来)
                elif self.current_state == 'crawl_right':
                    # The state of walking right(右行走状态)
                    if self.exit_crawl_right(block_data):
                        # If finishing walking to the right(如果完成右行走)
                        self.current_state = 'visual_patrol'
                        self.next_state = 'place_block'
                        self.state[self.current_state][2] = False
                    else:
                        time.sleep(0.8)
                elif self.current_state == 'place_block':
                    # The state of putting down the block(放块状态)
                    if self.exit_place_block(intersection_data):
                        self.running = False
                    else:
                        time.sleep(0.8)

                # # Determine state switching; whether to exit the line following and enter next phrase(状态切换判断，是否退出巡线，进入下一阶段)
                if self.next_state == 'crawl_left':
                    if self.enter_crawl_left(block_data):
                        self.current_state = 'crawl_left'
                        self.next_state = 'visual_patrol'
                elif self.next_state == 'crawl_right':
                    if self.enter_crawl_right(block_data):
                        self.current_state = 'crawl_right'
                        self.next_state = 'visual_patrol'
                elif self.next_state == 'place_block':
                    if self.enter_place_block(intersection_data):
                        self.current_state = 'place_block'
                        self.next_state = 'visual_patrol'

                self.state_init(self.current_state, self.next_state)
                
                time.sleep(0.01)  # Prevent the robot from running under no loading(防止空载)
            else:
                time.sleep(0.01)
        #The program is over, close node (程序结束,关闭节点)
        self.init_action(self.head_pan_init, self.head_tilt_init)
        self.stop_srv_callback(None)
        rospy.signal_shutdown('shutdown')

if __name__ == "__main__":
    VisualPatrolPickUp('visual_patrol_pick_up').run()
