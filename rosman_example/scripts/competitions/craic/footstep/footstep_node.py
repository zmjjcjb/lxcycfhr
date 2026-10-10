#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/07/13
# @author:aiden
# Irregular terrain locomotion (不规则地面运动)
import time
import rospy
import math
import signal
from rosman_sdk import common
from rosman_kinematics.gait_manager import GaitManager
from rosman_example.color_common import Common
from rosman_interfaces.srv import SetString
from rosman_interfaces.msg import ObjectsInfo, ColorDetect, ROI

class FootStepNode(Common):
    # Detect three ROIs in sequence, exit immediately if a black line is detected (按顺序检测三个roi，如果检测到黑线立刻跳出)
    # y_min, y_max, x_min, x_max represent proportions of the image; actual size = y_min * height (y_min, y_max, x_min, x_max分别表示占图像的比例, 即实际大小为y_min*height)
    line_roi = [(4 / 12, 6 / 12, 0, 1 ),
                (7 / 12, 9 / 12, 0, 1 ),
                (10 / 12, 12 / 12, 0, 1 )
                ]

    stairs_roi = [0, 1, 0.2, 0.8]
    # Name of the required motion (所需动作的名称)
    climb_stairs_action_name = 'climb_stairs_competition_fast0'
    descend_stairs_action_name = 'descend_stairs_fast1'
    back = 'back_step_fast'
    go = 'forward_one_step_fast'

    # Resize to this resolution during image processing, not recommended to change (图像处理时缩放到这个分辨率， 不建议修改)
    image_process_size = [160, 120]

    enter_climb_stairs_y = 280 / 480  # When the detected marker’s y-coordinate (as a ratio) exceeds this, slow down and prepare to climb stairs (当检测到的标识像素坐标y值占图像的比例大于此值时进入此阶段，进入上台阶，减慢速度)
    climb_stairs_y_stop = 100 / 480  # Start climbing when the y ratio reaches this threshold (当检测到的标识像素坐标y值占图像的比例在此值附近开始上台阶)
    descend_stairs_y_stop = 320 / 480 # Descend stairs(下台阶)

    def __init__(self, name):
        rospy.init_node(name)
        self.name = name
        self.gait_manager = GaitManager()
        self.go_gait_param = self.gait_manager.get_gait_param()
        self.go_gait_param['body_height'] = 0.025
        self.go_gait_param['step_height'] = 0.02
        self.go_gait_param['hip_pitch_offset'] = 15
        self.go_gait_param['z_swap_amplitude'] = 0.006
        self.go_gait_param['init_y_offset'] = -0.005
        self.go_dsp = [400, 0.2, 0.023]
        self.go_arm_swap = 30
        self.count = 0
        self.running = True
        self.line_data = None
        self.side_data = None
        self.current_state = "walk"      #Current state (当前状态)
        #State list (状态列表)
        self.state = {'walk':   [[500, 300], ['white', self.line_roi, self.image_process_size, self.set_visual_patrol_color], False],
                      'up_yellow_climb_stairs':   [[500, 300], ['yellow', self.stairs_roi, self.image_process_size, self.set_stairs_color], False],
                      'up_climb_stairs':   [[500, 300], ['blue', self.stairs_roi, self.image_process_size, self.set_stairs_color], False],
                      'down_descend_stairs':   [[500, 300], ['blue', self.stairs_roi, self.image_process_size, self.set_stairs_color], False],
                      'visual_patrol': [[500, 320], ['white', self.line_roi, self.image_process_size, self.set_visual_patrol_color], False]}
        
        self.head_pan_init = self.state[self.current_state][0][0]   # Initial value of head servo rotating left and right (左右舵机的初始值)
        self.head_tilt_init = self.state[self.current_state][0][1]  # Initial value of head servo rotating up and down(上下舵机的初始值)
        super().__init__(name, self.head_pan_init, self.head_tilt_init)

        signal.signal(signal.SIGINT, self.shutdown)
        
        # Subscribe to color recognition results (订阅颜色识别结果)
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)
        self.motion_manager.run_action('walk_ready')

        if rospy.get_param('~start', True):
            # Notify color recognition is ready; only raw camera feed is displayed at this time (通知颜色识别准备，此时只显示摄像头原画)
            self.enter_func(None)
            self.start_srv_callback(None)
            common.loginfo('start footstep')

    def shutdown(self, signum, frame):
        self.running = False
        common.loginfo('%s shutdown' % self.name)

    def set_visual_patrol_color(self, color, roi, image_process_size):
        # Set the line-following color (设置巡线颜色)
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
        # Set the stair marker color (设置台阶标志颜色)
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

    def get_color_callback(self, msg):
        # Get recognition results (获取识别结果)
        self.line_data = None
        self.side_data = None
        for object_info in msg.data:
            if object_info.type == 'line':
                self.line_data = object_info
            if object_info.type == 'side':
                self.side_data = object_info

    def state_init(self, current_state):
        # Initialization for different phases (不同阶段的初始化)
        if self.state[current_state][2] == False:
            self.state[current_state][2] = True
            self.init_action(self.state[current_state][0][0], self.state[current_state][0][1])  # Head state(头部姿态)
            param1 = self.state[current_state][1][3](self.state[current_state][1][0], self.state[current_state][1][1], self.state[current_state][1][2])
            self.detect_pub.publish([param1])  # Set color detection(颜色检测设置)
            common.loginfo(current_state + ' init')

    def state_init_walk(self, current_state):
        # Initialization for different phases (不同阶段的初始化)
        if self.state[current_state][2] == False:
            self.state[current_state][2] = True
            self.init_action(self.state[current_state][0][0], self.state[current_state][0][1])  # Head state(头部姿态)
            param1 = self.state[current_state][1][3](self.state[current_state][1][0], self.state[current_state][1][1], self.state[current_state][1][2])
            param2 = self.state['up_yellow_climb_stairs'][1][3](self.state['up_yellow_climb_stairs'][1][0], self.state['up_yellow_climb_stairs'][1][1], self.state['up_yellow_climb_stairs'][1][2])
            self.detect_pub.publish([param1, param2])  # Set color detection(颜色检测设置)
            common.loginfo(current_state + ' init')
     
    #Enter stair-climbing phase (进入上台阶)
    def enter_climb_stairs(self, stairs_data):
        if stairs_data is not None:
            print('enter_climb_stairs', min(stairs_data.y, stairs_data.left_point[1], stairs_data.right_point[1]), self.enter_climb_stairs_y * stairs_data.height)
            if min(stairs_data.y, stairs_data.left_point[1], stairs_data.right_point[1]) > self.enter_climb_stairs_y * stairs_data.height:
                self.go_gait_param['step_height'] = 0.01
                self.gait_manager.set_step(self.go_dsp, 0.005, 0, 0, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=3)
                self.gait_manager.stop()
                time.sleep(1)
                self.go_gait_param['step_height'] = 0.02
                return True
            elif min(stairs_data.y, stairs_data.left_point[1], stairs_data.right_point[1]) > self.enter_climb_stairs_y * stairs_data.height - 60:
                self.gait_manager.set_step(self.go_dsp, 0.005, 0, 0, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0)
            else:
                self.count = 0
        return False
    
    def exit_climb_stairs(self, stairs_data):
        # Stair climbing handling (上阶梯处理)
        if stairs_data is None:  
            self.gait_manager.set_step(self.go_dsp, 0.005, 0, 0, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0)
        if stairs_data is not None:
            print("exit_climb_stairs",stairs_data.angle, 15, stairs_data.y, self.climb_stairs_y_stop*stairs_data.height)
            if abs(stairs_data.angle) >= 15:
                yaw = math.copysign(5, 0 - stairs_data.angle)
                self.gait_manager.set_step(self.go_dsp, 0, 0, yaw, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0)
            elif stairs_data.y > self.climb_stairs_y_stop*stairs_data.height:
                self.gait_manager.disable()  # Disable gait control (关闭步态控制)
                common.loginfo('climb_stairs_competition')
                self.motion_manager.run_action(self.climb_stairs_action_name)
                time.sleep(0.8)
                self.motion_manager.run_action(self.go)
                time.sleep(0.8)
                self.motion_manager.run_action(self.climb_stairs_action_name)
                time.sleep(0.8)
                self.motion_manager.run_action(self.go)
                time.sleep(0.8)
                self.motion_manager.run_action(self.climb_stairs_action_name)
                time.sleep(0.8)
                return True
            else:
                self.gait_manager.set_step(self.go_dsp, 0.005, 0, 0, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0)
                
        return False

    def exit_descend_stairs(self, stairs_data):
        # Stair descending handling (下阶梯处理)
        if stairs_data is None:  
            self.gait_manager.set_step(self.go_dsp, 0.005, 0, 0, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0)
        if stairs_data is not None:
            print('exit_descend_stairs', stairs_data.angle, 10, stairs_data.y, self.descend_stairs_y_stop*stairs_data.height)
            if abs(stairs_data.angle) >= 10:
                yaw = math.copysign(5, 0 - stairs_data.angle )
                self.gait_manager.set_step(self.go_dsp, 0, 0, yaw, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0)
            elif stairs_data.y > self.descend_stairs_y_stop*stairs_data.height:
                self.gait_manager.disable()  # Disable gait control (关闭步态控制)
                common.loginfo('climb_stairs')
                self.motion_manager.run_action(self.descend_stairs_action_name)
                time.sleep(0.8)
                self.motion_manager.run_action(self.back)
                self.motion_manager.run_action(self.back)
                self.motion_manager.run_action(self.back)
                time.sleep(0.8)
                self.motion_manager.run_action(self.descend_stairs_action_name)
                time.sleep(0.8)
                self.motion_manager.run_action(self.back)
                self.motion_manager.run_action(self.back)
                self.motion_manager.run_action(self.back)
                time.sleep(0.8)
                self.motion_manager.run_action(self.descend_stairs_action_name)
                time.sleep(0.8)
                return True
            else:
                self.gait_manager.set_step(self.go_dsp, 0.005, 0, 0, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0)

        return False

    def visual_patrol(self, line_data):
        go_dsp = [250, 0.2, 0.023]
        if line_data is not None:
            if 180 < line_data.x < 360 or line_data.x < 180 and 80 > line_data.angle > 40:
                yaw = -6
                x = 0.013
                print('right')
            elif 540 > line_data.x > 360 or line_data.x > 540 and line_data.angle < 40:
                yaw = 6
                x = 0.013
            else:
                yaw = 0
                x = 0.016
            self.gait_manager.set_step(go_dsp, x, 0, yaw, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0)
        else:
            self.gait_manager.set_step(go_dsp, 0.016, 0, 0, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0)


    def run(self):
        while self.running:
            if self.start:
                # Current phase completed (当前阶段处理完成)
                line_data = self.line_data
                side_data = self.side_data
                if self.current_state == 'walk':
                    if line_data is not None and side_data is None:
                        self.visual_patrol(line_data)
                    elif line_data is None and side_data is None:
                        self.gait_manager.set_step(self.go_dsp, 0.02, 0, 0, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0)
                    if self.enter_climb_stairs(side_data):
                        self.current_state = 'up_climb_stairs'
                        self.state[self.current_state][2] = False  # Re-initialize current phase (重新初始化当前阶段)
                        common.loginfo('exit yellow_climb_stairs ---> enter while_climb_stairs')
                    else:
                        time.sleep(0.4)
                
                elif self.current_state == 'up_climb_stairs':
                    if self.exit_climb_stairs(side_data):
                        self.current_state = 'down_descend_stairs'
                        self.state[self.current_state][2] = False  # Re-initialize current phase (重新初始化当前阶段)
                        common.loginfo('exit yellow_climb_stairs ---> enter while_climb_stairs')
                    else:
                        time.sleep(0.8)
                elif self.current_state == 'down_descend_stairs':
                    if self.exit_descend_stairs(side_data):
                        self.current_state = 'visual_patrol'
                        self.state[self.current_state][2] = False
                        common.loginfo('exit red_descend_stairs ---> enter while_descend_stairs')
                    else:
                        time.sleep(0.8)
                elif self.current_state == 'visual_patrol':
                    self.visual_patrol(line_data)
                if self.current_state == 'walk':
                    self.state_init_walk(self.current_state)
                else:
                    self.state_init(self.current_state)
                time.sleep(0.01)  # Prevent idle load (防止空载)
            else:
                time.sleep(0.01)
        
        self.init_action(self.head_pan_init, self.head_tilt_init)
        self.stop_srv_callback(None)
        rospy.signal_shutdown('shutdown')

if __name__ == "__main__":
    FootStepNode('footstep').run()
