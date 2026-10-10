#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/07/04
# @author:aiden
# Line following & up and down steps(巡线+上下阶梯)
import time
import rospy
import signal
from rosman_sdk import misc, common
from rosman_example.color_common import Common
from rosman_example.visual_patrol import VisualPatrol
from rosman_example.approach_object import ApproachObject
from rosman_interfaces.srv import SetString
from rosman_interfaces.msg import ObjectsInfo, ColorDetect, ROI

class ClimbStairsVisualPatrol(Common):
    # Check three ROIs in order, and immediately exit if a black line is detected(按顺序检测三个roi，如果检测到黑线立刻跳出)
    # y_min, y_max, x_min, and x_max represent respectively the proportion of the image occupied. The actual size is y_min*height(y_min, y_max, x_min, x_max分别表示占图像的比例, 即实际大小为y_min*height)
    line_roi = [(5 / 12, 6 / 12, 1 / 4, 3 / 4),
                (6 / 12, 7 / 12, 1 / 4, 3 / 4),
                (7 / 12, 8 / 12, 1 / 4, 3 / 4)
                ]

    # ROI of the steps(阶梯的roi)
    # y_min, y_max, x_min, and x_max represent respectively the proportions of the image(y_min, y_max, x_min, x_max分别表示占图像的比例)
    stairs_roi = [1 / 5, 1, 0, 1]

    # Action of up and down steps(上下阶梯动作)
    climb_stairs_action_name = 'climb_stairs'
    descend_stairs_action_name = 'descend_stairs'

    # When process the image, scale it to this resolution(图像处理时缩放到这个分辨率)
    image_process_size = [160, 120]

    enter_climb_stairs_y = 280 / 480  # When the detected object's y-coordinate in the image is greater than this value, it enters this phrase)(当检测到的标识像素坐标y值占图像的比例大于此值时进入此阶段
    climb_stairs_x_stop = 0.5  # When the detected object's x-coordinate in the image is within a certain range (specified in the ApproachObject function) based on the proportion of the image, the robot will stop moving forward or backward(当检测到的标识像素坐标x值占图像的比例在此值附近(范围可在ApproachObject里设置)时停止前后移动)
    climb_stairs_y_stop = 375 / 480  # When the detected object's y-coordinate in the image is within a certain range (specified in the ApproachObject function) based on the proportion of the image, the robot will stop moving left or right(当检测到的标识像素坐标y值占图像的比例在此值附近(范围可在ApproachObject里设置)时停止横向移动)
    climb_stairs_yaw_stop = 0  # When the detected object's orientation angle is within a certain range (specified in the ApproachObject function), the robot will stop rotating(当检测到的标识角度在此值附近(范围可在ApproachObject里设置)时停止旋转移动)

    enter_descend_stairs_y = 280/480
    descend_stairs_x_stop = 0.5
    descend_stairs_y_stop = 385 / 480
    descend_stairs_yaw_stop = 0

    def __init__(self, name):
        rospy.init_node(name)
        self.name = name
        self.running = True
        self.objects_info = []
        self.count = 0

        self.current_state = "visual_patrol"
        self.next_state = "climb_stairs"

        self.state = {'visual_patrol':  [[500, 260], ['black', self.line_roi, self.image_process_size, self.set_visual_patrol_color], False],
                      'climb_stairs':   [[500, 260], ['red', self.stairs_roi, self.image_process_size, self.set_stairs_color], False],
                      'descend_stairs': [[500, 260], ['red', self.stairs_roi, self.image_process_size, self.set_stairs_color], False]}
        
        self.head_pan_init = self.state[self.current_state][0][0]   # Initial value of the left-right servo(左右舵机的初始值)
        self.head_tilt_init = self.state[self.current_state][0][1]  # Initial value of the up-down servo(上下舵机的初始值)
        super().__init__(name, self.head_pan_init, self.head_tilt_init)

        self.visual_patrol = VisualPatrol(self.gait_manager)
        # Decrease the stride of line following, improving the stability of other flag detection(减小巡线步幅，提高其他标志检测稳定性)
        self.visual_patrol.update_go_gait(dsp=[300, 0.2, 0.02], x_max=0.01)
        self.visual_patrol.update_turn_gait(dsp=[400, 0.2, 0.02], x_max=0.01) 
        self.approach_object = ApproachObject(self.gait_manager)
        self.motion_manager.run_action('walk_ready')
        signal.signal(signal.SIGINT, self.shutdown)
        
        # Subscribe to color recognition result(订阅颜色识别结果)
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)
        rospy.Service('~set_color', SetString, self.set_color_srv_callback)  # Set color(设置颜色)

        if rospy.get_param('~start', True):
            # Notify the color recognition to prepare, at this time only display the camera original image(通知颜色识别准备，此时只显示摄像头原画)
            self.enter_func(None)
            self.start_srv_callback(None)  # Start color recognition(开启颜色识别)
            common.loginfo('start climb_stairs')

    def shutdown(self, signum, frame):
        self.running = False
        common.loginfo('%s shutdown' % self.name)

    def set_visual_patrol_color(self, color, roi, image_process_size):
        # Set the color of line to be followed(设置巡线颜色)
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

    def set_stairs_color(self, color, roi, image_process_size):
        # Set the color of the step flag(设置台阶标志颜色)
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

    def set_color_srv_callback(self, msg):
        # Set color(设置颜色)
        stairs_param = self.set_stairs_color(msg.data)
        line_param = self.set_visual_patrol_color('black')
        
        self.detect_pub.publish([line_param, stairs_param])
        common.loginfo('%s set_color' % self.name)
        
        return [True, 'set_color']

    def get_color_callback(self, msg):
        # Obtain color recognition result(获取颜色识别结果)
        self.objects_info = msg.data

    def state_init(self, current_state, next_state):
        # Initialization in different phrases(不同阶段的初始化)
        if self.state[current_state][2] == False:
            self.state[current_state][2] = True
            self.init_action(self.state[current_state][0][0], self.state[current_state][0][1])  # Head posture(头部姿态)
            param1 = self.state[current_state][1][3](self.state[current_state][1][0], self.state[current_state][1][1], self.state[current_state][1][2])
            param2 = self.state[next_state][1][3](self.state[next_state][1][0], self.state[next_state][1][1], self.state[next_state][1][2])
            self.detect_pub.publish([param1, param2])  # Color detection settings(颜色检测设置)

    def exit_climb_stairs(self, stairs_data):
        # Up steps processing(上阶梯处理)
        if stairs_data is not None:
            if self.approach_object.process(max(stairs_data.y, stairs_data.left_point[1], stairs_data.right_point[1]), stairs_data.x, stairs_data.angle, 
                                        self.climb_stairs_y_stop*stairs_data.height, self.climb_stairs_x_stop*stairs_data.width, self.climb_stairs_yaw_stop, stairs_data.width, stairs_data.height):
                self.gait_manager.disable()  # Close gait control(关闭步态控制)
                common.loginfo('climb_stairs')
                self.motion_manager.run_action(self.climb_stairs_action_name)  # Execute up-steps action(执行上台阶动作)
                return True
        return False

    def exit_descend_stairs(self, stairs_data):
        # Down steps processing(下阶梯处理)
        if stairs_data is not None:
            if self.approach_object.process(max(stairs_data.y, stairs_data.left_point[1], stairs_data.right_point[1]), stairs_data.x, stairs_data.angle, 
                                        self.descend_stairs_y_stop*stairs_data.height, self.descend_stairs_x_stop*stairs_data.width, self.descend_stairs_yaw_stop, stairs_data.width, stairs_data.height):
                self.gait_manager.disable()  # Close gait control(关闭步态控制)
                common.loginfo('descend_stairs')
                self.motion_manager.run_action(self.descend_stairs_action_name)  # Execute down-steps action(执行下台阶动作)
                return True
        return False

    def enter_climb_stairs(self, stairs_data):
        if stairs_data is not None:
            if max(stairs_data.y, stairs_data.left_point[1], stairs_data.right_point[1]) > self.enter_climb_stairs_y * stairs_data.height:
                self.count += 1
                if self.count > 5:
                    self.count = 0
                    self.gait_manager.disable()
                    self.approach_object.update_approach_stop_value(20, 30, 6)
                    self.motion_manager.run_action('hand_back')  # Move the hand back to prevent obstruction(手往后，防止遮挡)
                    time.sleep(0.5)
                    return True
            else:
                self.count = 0
        return False

    def enter_descend_stairs(self, stairs_data):
        if stairs_data is not None:
            if max(stairs_data.y, stairs_data.left_point[1], stairs_data.right_point[1]) > self.enter_descend_stairs_y * stairs_data.height:
                self.count += 1
                if self.count > 5:
                    self.count = 0
                    self.gait_manager.disable()
                    self.approach_object.update_approach_stop_value(20, 30, 6)
                    self.motion_manager.run_action('hand_back')  # Move the hand back to prevent obstruction(手往后，防止遮挡)
                    time.sleep(0.5)
                    return True
            else:
                self.count = 0
        return False

    def run(self):
        while self.running:
            if self.start:
                # Obtain recognition result(获取识别结果)
                line_data = None
                stairs_data = None
                for object_info in self.objects_info:
                    if object_info.type == 'line':
                        line_data = object_info
                    if object_info.type == 'side':
                        stairs_data = object_info

                # The current processing is completed, and return to line following(当前阶段处理完成，回到巡线)
                if self.current_state == 'visual_patrol':
                    if line_data is not None:
                        self.visual_patrol.process(line_data.x, line_data.width)
                elif self.current_state == 'climb_stairs':
                    if self.exit_climb_stairs(stairs_data):
                        self.current_state = 'visual_patrol'
                        self.next_state = 'descend_stairs'
                        self.state[self.current_state][2] = False
                    else:
                        time.sleep(0.8)  # Wait for the robot to stabilize(等机体平稳下来)
                elif self.current_state == 'descend_stairs':
                    if self.exit_descend_stairs(stairs_data):
                        self.current_state = 'visual_patrol'
                        self.next_state = 'climb_stairs'
                        self.state[self.current_state][2] = False
                    else:
                        time.sleep(0.8)

                # Whether to exit the line following to enter the next phrase(是否退出巡线，进入下一阶段)
                if self.next_state == 'climb_stairs':
                    if self.enter_climb_stairs(stairs_data):
                        self.current_state = 'climb_stairs'
                        self.next_state = 'visual_patrol'
                elif self.next_state == 'descend_stairs':
                    if self.enter_descend_stairs(stairs_data):
                        self.current_state = 'descend_stairs'
                        self.next_state = 'visual_patrol'

                self.state_init(self.current_state, self.next_state)
                
                time.sleep(0.01)  # Prevent the robot from running without load(防止空载)
            else:
                time.sleep(0.01)
        
        self.init_action(self.head_pan_init, self.head_tilt_init)
        self.stop_srv_callback(None)
        rospy.signal_shutdown('shutdown')

if __name__ == "__main__":
    ClimbStairsVisualPatrol('climb_stairs_visual_patrol').run()
