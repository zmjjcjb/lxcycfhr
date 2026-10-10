#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/07/13
# @author:aiden
# Obstacle avoidance (绕障碍)
import time
import rospy
import signal
from rosman_sdk import pid, misc, common
from rosman_kinematics.gait_manager import GaitManager
from rosman_example.color_common import Common
from rosman_interfaces.srv import SetString
from rosman_interfaces.msg import ObjectsInfo, ColorDetect, ROI


class ObstacleNode(Common):
    # Check three ROIs in order, break if black line is detected (按顺序检测三个roi，如果检测到黑线立刻跳出)
    # y_min, y_max, x_min, x_max represent image ratios (y_min * height) (y_min, y_max, x_min, x_max分别表示占图像的比例, 即实际大小为y_min*height)
    line_roi = [(4 / 12, 6 / 12, 0, 1 ),
                (7 / 12, 9 / 12, 0, 1 ),
                (10 / 12, 12 / 12, 0, 1 )
                ]

    column_roi = [0, 4 / 5, 0, 1]
    hurdles_roi = [1 / 5, 1, 0, 1]
    ball_roi = [0, 1, 0, 1]
    # Action names required (所需动作的名称)
    hurdles_action_name = 'hurdles'
    climb_stairs_action_name = 'climb_stairs_competition_2'
    descend_stairs_action_name = 'descend_stairs'
    back = 'go_back'
    go = 'forward_step'

    # Resize images for processing; not recommended to change (图像处理时缩放到这个分辨率， 不建议修改)
    image_process_size = [160, 120]

    def __init__(self, name):
        rospy.init_node(name)
        self.name = name
        self.current_color_1 = None
        self.current_color_2 = None
        self.current_color_3 = None
        self.gait_manager = GaitManager()
        self.go_gait_param = self.gait_manager.get_gait_param()
        self.go_gait_param['body_height'] = 0.025
        self.go_gait_param['step_height'] = 0.02
        self.go_gait_param['hip_pitch_offset'] = 15
        self.go_gait_param['z_swap_amplitude'] = 0.006
        self.go_gait_param['init_roll_offset'] = 0
        self.go_gait_param['init_y_offset'] = -0.005
        self.go_dsp = [250, 0.2, 0.023]
        self.go_arm_swap = 30
        self.line_data = None
        self.column_data = None
        self.turn = 8
        self.running = True
        self.current_state = "look"
        self.init_column = {}
        self.state = {
            'look': [[500, 500], ['red', self.column_roi, self.image_process_size, self.set_column_color], False],
            'walk': [[500, 400], ['white', self.line_roi, self.image_process_size, self.set_visual_patrol_color],
                     False],
            'red': [[500, 320], ['red', self.column_roi, self.image_process_size, self.set_column_color], False],
            'yellow': [[500, 320], ['yellow', self.column_roi, self.image_process_size, self.set_column_color], False],
            'blue': [[500, 320], ['blue', self.column_roi, self.image_process_size, self.set_column_color], False],
            }
        self.head_pan_init = 500  # Initial value for servo rotating left and right(左右舵机的初始值)
        self.head_tilt_init = 500  # Initial value for servo rotating up and down(上下舵机的初始值)
        super().__init__(name, self.head_pan_init, self.head_tilt_init)

        self.calib_config = common.get_yaml_data('/home/ubuntu/ros_ws/src/rosman_example/config/calib.yaml')

        signal.signal(signal.SIGINT, self.shutdown)

        # Subscribe to color detection result (订阅颜色识别结果)
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)
        self.motion_manager.run_action('walk_ready')
        
        self.state_init_look(self.current_state)
        if rospy.get_param('~start', True):
            # Notify color detection is ready; only original camera image shown (通知颜色识别准备，此时只显示摄像头原画)
            self.enter_func(None)
            self.start_srv_callback(None)
            common.loginfo('start ombination')

    def shutdown(self, signum, frame):
        self.running = False
        common.loginfo('%s shutdown' % self.name)

    def set_visual_patrol_color(self, color, roi, image_process_size):
        # Set line-following color (设置巡线颜色)
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

    def set_column_color(self, color, roi, image_process_size):
        # Set column mark color (设置柱子标志颜色)
        stairs_param = ColorDetect()
        stairs_param.color_name = color
        stairs_param.detect_type = 'side'
        stairs_param.use_name = True
        stairs_param.image_process_size = image_process_size
        stairs_param.roi.y_min = int(roi[0] * image_process_size[1])
        stairs_param.roi.y_max = int(roi[1] * image_process_size[1])
        stairs_param.roi.x_min = int(roi[2] * image_process_size[0])
        stairs_param.roi.x_max = int(roi[3] * image_process_size[0])
        stairs_param.min_area = 10 * 10
        stairs_param.max_area = image_process_size[0] * image_process_size[1]

        return stairs_param

    def get_color_callback(self, msg):
        #  Get color detection result (获取颜色识别结果)
        self.line_data = None
        self.column_data = None
        for color in msg.data:
            if color.type == 'line':
                self.line_data = color
            if color.type == 'side':
                self.column_data = color
            if self.current_state == 'look':
                if color.type == 'side':
                    self.init_column[color.label] = [color.x, color.y]
        if len(self.init_column) >= 2 and self.current_state == 'look':
            keys = sorted(self.init_column, key=lambda k: self.init_column[k][1], reverse=True)
            if len(keys) == 3:
                self.current_color_1, self.current_color_2, self.current_color_3 = keys
            else:
                self.current_color_1, self.current_color_2 = keys
                self.current_color_3 = ['red', 'blue', 'yellow']
                self.current_color_3.remove(self.current_color_1)
                self.current_color_3.remove(self.current_color_2)
                self.current_color_3 = self.current_color_3[0]
            if self.init_column[self.current_color_1][0] < 320:
                self.turn = -self.turn
            self.current_state = 'walk'

    def state_init_look(self, current_state):
        # Initialization for each stage (不同阶段的初始化)
        if self.state[current_state][2] == False:
            self.state[current_state][2] = True
            self.init_action(self.state[current_state][0][0], self.state[current_state][0][1])  # Head state(头部姿态)
            param2 = self.state['red'][1][3](self.state['red'][1][0], self.state['red'][1][1], self.state['red'][1][2])
            param3 = self.state['yellow'][1][3](self.state['yellow'][1][0], self.state['yellow'][1][1],
                                                self.state['yellow'][1][2])
            param4 = self.state['blue'][1][3](self.state['blue'][1][0], self.state['blue'][1][1],
                                              self.state['blue'][1][2])
            self.detect_pub.publish([param2, param3, param4])  # Set color detection(颜色检测设置)
            common.loginfo(current_state + ' init')

    def state_init_walk(self, current_state):
        # Initialization for each stage (不同阶段的初始化)
        if self.state[current_state][2] == False:
            self.state[current_state][2] = True
            self.init_action(self.state[current_state][0][0], self.state[current_state][0][1])  # Head state(头部姿态)
            param1 = self.state[current_state][1][3](self.state[current_state][1][0], self.state[current_state][1][1],
                                                     self.state[current_state][1][2])
            param2 = self.state['red'][1][3](self.state['red'][1][0], self.state['red'][1][1], self.state['red'][1][2])
            param3 = self.state['yellow'][1][3](self.state['yellow'][1][0], self.state['yellow'][1][1],
                                                self.state['yellow'][1][2])
            param4 = self.state['blue'][1][3](self.state['blue'][1][0], self.state['blue'][1][1],
                                              self.state['blue'][1][2])
            self.detect_pub.publish([param1, param2, param3, param4])  # Set color detection(颜色检测设置)
            common.loginfo(current_state + ' init')

    # Main loop function (主循环函数)
    def run(self):
        while self.running:
            if self.start:
                # If there is a recognition result, perform line-following control (如果有识别结果,巡线控制)
                column_data = self.column_data
                line_data = self.line_data

                # print(self.current_color_1, self.current_color_2, self.current_color_3)
                if self.current_state == 'walk':
                    if line_data is not None:
                        if 180 < line_data.x < 360 or line_data.x < 180 and 80 > line_data.angle > 40:
                            yaw = -6
                            x = 0.013
                            print('right')
                        elif 540 > line_data.x > 360 or line_data.x> 540 and line_data.angle < 40:
                            yaw = 6
                            x = 0.013
                            print('left')
                        else:
                            print('go')
                            yaw = 0
                            x = 0.016
                        # print(line_data.x, line_data.angle)
                        self.gait_manager.set_step(self.go_dsp, x, 0, yaw, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0)
                    elif column_data is not None:
                        if self.current_color_1 == column_data.label or self.current_color_3 == column_data.label:
                            self.gait_manager.set_step(self.go_dsp, 0.02, 0, self.turn, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0)
                        elif self.current_color_2 == column_data.label:
                            self.gait_manager.set_step(self.go_dsp, 0.02, 0, -self.turn, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0)
                    elif line_data is None and column_data is None:
                        self.gait_manager.set_step(self.go_dsp, 0.02, 0, 0, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0)

                time.sleep(0.01)
                if self.current_state == 'walk':
                    self.state_init_walk(self.current_state)
            else:
                time.sleep(0.01)

        # Action before exit(退出前动作)
        self.stop_srv_callback(None)
        rospy.signal_shutdown('shutdown')

if __name__ == "__main__":
    ObstacleNode('obstacle').run()
