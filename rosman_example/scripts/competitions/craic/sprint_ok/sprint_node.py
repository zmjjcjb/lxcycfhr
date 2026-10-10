#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/07/03
# @author:aiden
# Sprint (短跑)
import time
import rospy
import signal
from rosman_kinematics.gait_manager import GaitManager
from std_msgs.msg import String
from rosman_sdk import common
from rosman_example.color_common import Common
from rosman_interfaces.srv import SetString
from rosman_interfaces.msg import ObjectsInfo, ColorDetect

class SprintNode(Common):
    # Detect three ROIs in sequence, break immediately if black line is detected (按顺序检测三个roi，如果检测到黑线立刻跳出)
    # y_min, y_max, x_min, x_max represent image ratios, actual size is y_min * height (y_min, y_max, x_min, x_max分别表示占图像的比例, 即实际大小为y_min*height)
    line_roi = [(4 / 12, 6 / 12, 0, 1 ),
                (7 / 12, 9 / 12, 0, 1 ),
                (10 / 12, 12 / 12, 0, 1 )
                ]

    # Resize image to this resolution for processing (图像处理时缩放到这个分辨率)
    image_process_size = [160, 120]

    def __init__(self, name):
        rospy.init_node(name)
        self.name = name
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
        self.running = True
        
        self.current_state = "walk"

        self.state = {'walk':   [[500, 330], ['white', self.line_roi, self.image_process_size, self.set_visual_patrol_color], False],
                      }
        super().__init__(name, self.state[self.current_state][0][0], self.state[self.current_state][0][1])

        # Set shutdown handler (设置退出处理函数)
        signal.signal(signal.SIGINT, self.shutdown)

        # Subscribe to color detection results (订阅颜色识别结果)
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)

        self.motion_manager.run_action('walk_ready')    # Walking pose action (行走姿势动作)
        # If auto-start parameter is true, start the node (如果自动开始参数为真,启动节点)
        self.enter_func(None)
        self.start_srv_callback(None)  # Enable color detection (开启颜色识别)
        param = self.state[self.current_state][1][3](self.state[self.current_state][1][0],
                                                     self.state[self.current_state][1][1],
                                                     self.state[self.current_state][1][2])
        self.detect_pub.publish([param])  # Set color detection parameters (颜色检测设置)
        common.loginfo('start track %s lane' % self.state[self.current_state][1][0])

    # Node shutdown callback function (节点关闭回调函数)
    def shutdown(self, signum, frame):
        with self.lock:
            self.motion_manager.run_action('stand')
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

    def get_color_callback(self, msg):
        # Get color detection result (获取颜色识别结果)
        self.line_data = None
        for object_info in msg.data:
            if object_info.type == 'line':
                self.line_data = object_info

    # Main loop function (主循环函数)
    def run(self):
        while self.running:
            if self.start:
                line_data = self.line_data
                if self.current_state == "walk":
                    # Strategy: determine walking direction based on position and angle of the line (策略：根据看到的线条位置和角度来决定行走方向)
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
                    else:
                        print('miss go')
                        self.gait_manager.set_step(self.go_dsp, 0.016, 0, 0, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0)
            else:
                time.sleep(0.01)
        # Action before shutdown (退出前动作)
        self.stop_srv_callback(None)
        rospy.signal_shutdown('shutdown')

if __name__ == "__main__":
    SprintNode('sprint').run()
