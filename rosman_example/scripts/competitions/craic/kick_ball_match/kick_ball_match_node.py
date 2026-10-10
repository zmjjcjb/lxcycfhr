#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/07/11
# @author:aiden
# Auto shooting(自动踢球)
import time
import rospy
import signal
from std_msgs.msg import Float64, String
from rosman_sdk import pid, misc, common
from rosman_kinematics.gait_manager import GaitManager
from rosman_example.color_common import Common
from rosman_example.pid_track import PIDTrack
from rosman_example.approach_object import ApproachObject
from rosman_interfaces.srv import SetString
from rosman_interfaces.msg import ObjectsInfo, ColorDetect, ColorsDetect

class KickBallNode(Common):
    # Action names for left and right kick (左右踢球的动作名)
    left_shot_action_name = 'new_left_shot'
    right_shot_action_name = 'new_right_shot'
    forward_step = 'forward_one_step'  #Move one step forward (前进一步)
    back_step = 'back_step'        #Move one step back (后退一步)
    move_left = 'move_left'        #Move left (左移)
    move_right = 'move_right'      #Move right (右移)
    

    door_roi = [0, 1, 0.1, 0.9]
    ball_roi = [0, 1, 0, 1] 

    # Image processing size (图像处理大小)
    image_process_size = [160, 120]

    door_x_stop = 250/640
    ball_y_stop = 350/480
    ball_x_stop = 420/640
    
    kick_y_stop = 320/480
    kick_x_stop = 280/640

    def __init__(self, name):
        rospy.init_node(name)
        self.name = name
        self.gait_manager = GaitManager()
        self.go_gait_param = self.gait_manager.get_gait_param()
        self.go_gait_param['body_height'] = 0.025
        self.go_gait_param['step_height'] = 0.02
        self.go_gait_param['hip_pitch_offset'] = 12
        self.go_gait_param['z_swap_amplitude'] = 0.006
        self.go_gait_param['init_roll_offset'] = -3
        self.go_dsp = [400, 0.2, 0.03]
        self.go_arm_swap = 30

        self.door_dsp = [300, 0.2, 0.025]

        self.running = True
        # Store detection result(存储检测结果)
        self.objects_info = []

        # Initialize head position (初始化头部位置)
        self.head_pan_init = 500  # Initial value for head servo rotating left and right (左右舵机的初始值)
        self.head_tilt_init = 300 # Initial value for head servo rotating up and down (上下舵机的初始值)
        self.head_time_stamp = rospy.get_time()
        # Initialize parent class (初始化父类)
        super().__init__(name, self.head_pan_init, self.head_tilt_init)
        self.calib_config = common.get_yaml_data('/home/ubuntu/ros_ws/src/rosman_example/config/calib.yaml')
        # Initialize approach control (初始化接近控制)
        self.approach_object = ApproachObject(self.gait_manager, step_mode=0)
        self.approach_object.update_gait(dsp=[400, 0.2, 0.025])
        self.approach_object.update_gait(walking_param = self.go_gait_param)
        self.approach_object.update_stop_count(1)
        self.approach_object.update_gait_range(x_range=[-0.013, 0.013])
        self.approach_object.update_approach_stop_value(30, 0, 3)
        signal.signal(signal.SIGINT, self.shutdown)

        self.current_state = "ball"    #Current state (当前状态)
        #State list (状态列表)
        self.state = {'door': [[500, 400], ['red', self.door_roi, self.image_process_size, self.set_door_color], False],
                      'ball': [[500, 400], ['blue', self.ball_roi, self.image_process_size, self.set_ball_color], False],
                      'kick': [[500, 300], ['blue', self.ball_roi, self.image_process_size, self.set_ball_color], False]}

        # Subscribe to color detection results (订阅颜色识别结果)
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)
        rospy.Service('~set_color', SetString, self.set_color_srv_callback)  # 设置颜色
        # Initialize standing pose (初始化站姿)
        self.motion_manager.run_action('walk_ready')
        # Whether start autonomously(是否自动开始)
        if rospy.get_param('~start', True):
            # Notify color detection is ready, only original camera image is displayed now (通知颜色识别准备，此时只显示摄像头原画)
            self.enter_func(None)
            self.start_srv_callback(None)
            common.loginfo('start ombination')
    # Shutdown callback (退出回调)
    def shutdown(self, signum, frame):
        with self.lock:
            self.running = False
            common.loginfo('%s shutdown' % self.name)
    
    def set_ball_color(self, color, roi, image_process_size):
        # Set tracking color (设置追踪颜色)
        ball_param = ColorDetect()
        ball_param.color_name = color
        ball_param.detect_type = 'circle'
        ball_param.use_name = True
        ball_param.image_process_size = self.image_process_size
        ball_param.roi.y_min = int(roi[0] * image_process_size[1])
        ball_param.roi.y_max = int(roi[1] * image_process_size[1])
        ball_param.roi.x_min = int(roi[2] * image_process_size[0])
        ball_param.roi.x_max = int(roi[3] * image_process_size[0])
        ball_param.min_area = 20
        ball_param.max_area = self.image_process_size[0]*self.image_process_size[1]

        return ball_param
    
    def set_door_color(self, color, roi, image_process_size):
        # Set door color (设置门颜色)
        door_param = ColorDetect()
        door_param.color_name = color
        door_param.detect_type = 'rect'
        door_param.use_name = True
        door_param.image_process_size = self.image_process_size
        door_param.roi.y_min = int(roi[0] * image_process_size[1])
        door_param.roi.y_max = int(roi[1] * image_process_size[1])
        door_param.roi.x_min = int(roi[2] * image_process_size[0])
        door_param.roi.x_max = int(roi[3] * image_process_size[0])
        door_param.min_area = 20
        door_param.max_area = self.image_process_size[0]*self.image_process_size[1]

        return door_param
    
    def set_color_srv_callback(self, msg):
        # Set color (设置颜色)
        door_param = self.set_stairs_color(self.state['door'][1][0])
        ball_param = self.set_visual_patrol_color(self.state['ball'][1][0])

        self.detect_pub.publish([door_param, ball_param])
        common.loginfo('%s set_color' % self.name)
        
        return [True, 'set_color']
    

    def state_init(self, current_state):
        # Initialization for different phases (不同阶段的初始化)
        if self.state[current_state][2] == False:
            self.state[current_state][2] = True
            self.init_action(self.state[current_state][0][0], self.state[current_state][0][1])  # Head state(头部姿态)
            param1 = self.state[current_state][1][3](self.state[current_state][1][0], self.state[current_state][1][1], self.state[current_state][1][2])
            param2 = self.state['ball'][1][3](self.state['ball'][1][0], self.state['ball'][1][1], self.state['ball'][1][2])
            self.detect_pub.publish([param1, param2])  # Set color detection (颜色检测设置)
            common.loginfo(current_state + ' init')

    def get_color_callback(self, msg):
        # Get color detection results (获取颜色识别结果)
        self.objects_info = msg.data

    # Body tracking (躯体追踪)
    def body_track_process(self, ball_data):                                                
        # Adjust left/right according to head servo position (左右根据头部左右舵机位置值进行调整)
        if ball_data is not None:
            print('body track', ball_data.y, self.ball_y_stop*ball_data.height)
            if ball_data.y < self.ball_y_stop * ball_data.height:  
                self.gait_manager.set_step(self.go_dsp, 0.02, 0, 0, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0)
            else:
                return True        
        return False  

    def exit_door(self, door_data, ball_data):
        if door_data is not None and ball_data is not None:
            print('exit door', door_data.x, 140,  ball_data.x, 280)
            if door_data.x > 140:
                self.gait_manager.set_step(self.go_dsp, 0, 0, -4, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0)
            elif ball_data.x -280 > 10:
                self.gait_manager.set_step(self.go_dsp, 0, -0.01, 0, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0)
            elif ball_data.x -280 < -10:
                self.gait_manager.set_step(self.go_dsp, 0, 0.01, 0, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0)
            else:
                time.sleep(1)
                return True 
        elif ball_data is None:
            self.gait_manager.set_step(self.go_dsp, -0.01, 0, 0, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0)
        return False
    
    def kick_ball(self, ball_data):
        if ball_data is not None:
            print('kick', ball_data.x, self.kick_x_stop*ball_data.width, ball_data.y, self.kick_y_stop*ball_data.height)
            if  ball_data.y < self.kick_y_stop * ball_data.height:   
                self.gait_manager.set_step(self.go_dsp, 0.008, 0, 0, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0) 
            elif  ball_data.x - self.kick_x_stop * ball_data.width < -10:
                self.gait_manager.set_step(self.go_dsp, 0, 0.005, 0, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0)
            elif ball_data.x - self.kick_x_stop * ball_data.width > 10:
                self.gait_manager.set_step(self.go_dsp, 0, -0.005, 0, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0)   
            else:
                self.gait_manager.disable()
                self.motion_manager.run_action(self.left_shot_action_name)
                self.running = False
            time.sleep(0.1)

    def run(self):
        while self.running:
            # Determine state(状态判断)
            if self.start:
                if self.current_state == "door":
                    door_data = None
                    ball_data = None
                    for object_info in self.objects_info:
                        if object_info.type == 'rect':
                            door_data = object_info
                        if object_info.type == 'circle':
                            ball_data = object_info
                    if self.exit_door(door_data, ball_data):
                        self.current_state = "kick"
                
                elif self.current_state == "kick":
                    ball_data = None
                    for object_info in self.objects_info:
                        if object_info.type == 'circle':
                            ball_data = object_info
                    self.kick_ball(ball_data)

                elif self.current_state == "ball":
                    ball_data = None 
                    # Search for ball in detected objects (在检测到的对象中查找球)
                    for object_info in self.objects_info:
                        if object_info.type == 'circle':
                            ball_data = object_info
                    # If a ball is detected, perform body tracking (如果识别到球就进行躯体追踪)
                    if ball_data is not None:
                        if self.body_track_process(ball_data):
                            self.current_state = "door"
                
                    time.sleep(0.01)

                self.state_init(self.current_state)
            else:
                time.sleep(0.01)

        self.stop_srv_callback(None)
        rospy.signal_shutdown('shutdown')

if __name__ == "__main__":
    KickBallNode('kick_ball').run()
