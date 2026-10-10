#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/07/04
# @author:aiden
# Low clearance passage(巡线低空穿越)
import time
import rospy
import signal
from rosman_sdk import misc, common
from rosman_example.color_common import Common
from rosman_example.visual_patrol import VisualPatrol
from rosman_interfaces.msg import ObjectsInfo, ColorDetect, ROI

class CrawlUnderVisualPatrol(Common):
    # Check three ROIs in order, and immediately exit if a black line is detected(按顺序检测三个roi，如果检测到黑线立刻跳出)
    # y_min, y_max, x_min, and x_max represent respectively the proportion of the image occupied. The actual size is y_min*height(y_min, y_max, x_min, x_max分别表示占图像的比例, 即实际大小为y_min*height)
    line_roi = [(5 / 12, 6 / 12, 1 / 4, 3 / 4),
                (6 / 12, 7 / 12, 1 / 4, 3 / 4),
                (7 / 12, 8 / 12, 1 / 4, 3 / 4)
                ]

    # y_min, y_max, x_min, and x_max represent respectively the proportions of the image(y_min, y_max, x_min, x_max分别表示占图像的比例)
    door_roi = [1 / 5, 1, 0, 1]

    # Set a threshold through a proportion, which indicating when the door position is below 240/480 = 0.5 (that is 50%) of the image height(这里通过一个比例来设置阈值,表示当门的位置低于图像高度的240/480 = 0.5,)
    # The door is considered low enough, and the robot can enter the low-altitude state(即50%时,就认为门足够低,可以进入低空状态了。)
    enter_crawl_under_y = 240 / 480

    # When process the image, scale it to this resolution(图像处理时缩放到这个分辨率)
    image_process_size = [160, 120]
    # Initialize node(初始化节点)
    def __init__(self, name):
        rospy.init_node(name)
        self.name = name
        self.running = True
        self.count_miss = 0
        self.objects_info = []
        self.head_raise_stamp = 0
        self.body_raise_stamp = 0
        self.head_state = 'drop'
        # Define two states in state machine(定义状态机中两个状态)
        self.current_state = "visual_patrol"    # Line following state(巡线状态)
        self.next_state = "crawl_under"         # The state after recognizing the door frame(识别到门框后的状态)
        
        # Define state machine(定义状态机)
        # Include two states "visual_patrol" and "crawl_under"(包括两个状态visual_patrol和crawl_under)
        # Each state includes:(每个状态包含:)
        #   - Head position [pan,tilt](头部位置[pan,tilt])
        #   - Color recognition parameter[color, ROI region, image size, and function for setting color parameters](颜色识别参数[颜色,ROI区域,图像大小,设置颜色的参数函数])
        #   -  The flag for determine whether the state is completed(是否完成该状态的标志)
        self.state = {'visual_patrol':  [[500, 260], ['black', self.line_roi, self.image_process_size, self.set_visual_patrol_color], False],
                      'crawl_under':   [[500, 260], ['green', self.door_roi, self.image_process_size, self.set_door_color], False]}
       
        self.head_pan_init = self.state[self.current_state][0][0]   # Initial value of the left-right servo(左右舵机的初始值)
        self.head_tilt_init = self.state[self.current_state][0][1]  # Initial value of the up-down servo(上下舵机的初始值)
        super().__init__(name, self.head_pan_init, self.head_tilt_init)
        # Create line following class(创建巡线类)
        self.visual_patrol = VisualPatrol(self.gait_manager)
        self.visual_patrol.update_go_gait(dsp=[400, 0.2, 0.025], x_max=0.01)
        self.visual_patrol.update_turn_gait(dsp=[400, 0.2, 0.025], x_max=0.01)
        # Press "Ctrl+C" to exit(Ctrl+C 退出)
        signal.signal(signal.SIGINT, self.shutdown)

        # Subscribe color recognition result(订阅颜色识别结果)
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)

        self.motion_manager.run_action('walk_ready')     # Enter standing up state(进入站立状态)

        if rospy.get_param('~start', True):
            self.enter_func(None)
            self.start_srv_callback(None)  # Enable color recognition(开启颜色识别)
            common.loginfo('start crawl_under')
    # "Ctrl+C" callback function(Ctrl+C 回调函数)
    def shutdown(self, signum, frame):
        self.running = False
        common.loginfo('%s shutdown' % self.name)
    # Set the parameter for the color of the line to be followed(设置巡线颜色的参数)
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
    # Set the parameter of door color(设置门颜色的参数)
    def set_door_color(self, color, roi, image_process_size):
        # Set door flag color(设置门标志颜色)
        door_param = ColorDetect()
        door_param.color_name = color
        door_param.detect_type = 'side'
        door_param.use_name = True
        door_param.image_process_size = image_process_size
        door_param.roi.y_min = int(roi[0] * image_process_size[1])
        door_param.roi.y_max = int(roi[1] * image_process_size[1])
        door_param.roi.x_min = int(roi[2] * image_process_size[0])
        door_param.roi.x_max = int(roi[3] * image_process_size[0])
        door_param.min_area = 10 * 20
        door_param.max_area = image_process_size[0] * image_process_size[1]

        return door_param

    # The callback function that has received color recognition result(接收到颜色识别结果的回调函数)
    def get_color_callback(self, msg):
        # Obtain color recognition result(获取颜色识别结果)
        self.objects_info = msg.data
    # Initialize state(状态初始化)
    def state_init(self, current_state, next_state):
        # If it is the first time to run the current state, perform initialization(如果是第一次运行当前状态,进行初始化)
        if self.state[current_state][2] == False:
            self.state[current_state][2] = True
            # Return the head to the initial position(头部位置还原)
            self.init_action(self.state[current_state][0][0], self.state[current_state][0][1])  # Head posture(头部姿态)
            param1 = self.state[current_state][1][3](self.state[current_state][1][0], self.state[current_state][1][1], self.state[current_state][1][2])
            param2 = self.state[next_state][1][3](self.state[next_state][1][0], self.state[next_state][1][1], self.state[next_state][1][2])
            self.detect_pub.publish([param1, param2])  # Color detection settings(颜色检测设置)
    # The determination for entering the low-altitude state(进入低空状态的判断)
    def enter_crawl_under(self, door_data):
        if rospy.get_time() > self.head_raise_stamp:  # Look at the door without raising the head to confirm the distance(不停抬头看门用来确定距离)
            # Periodic head lifting action(周期抬头动作)
            if self.head_state == 'drop':
                self.motion_manager.set_servos_position(100, [[23, 500], [24, 500]])
                self.head_raise_stamp = rospy.get_time() + 1.5
                self.head_state = 'raise'   # Raise head(抬头)
            else:
                self.motion_manager.set_servos_position(100, [[23, self.state[self.current_state][0][0]],
                                                              [24, self.state[self.current_state][0][1]]])
                self.head_raise_stamp = rospy.get_time() + 1.5
                self.head_state = 'drop'    # Lower head(放下头)
         # If the door is low enough(如果门的位置已经很低了)
        if door_data is not None:
            # door_data.height indicates the door's pixel height. The judgement condition is:(这里door_data.height表示门的像素高度。所以判断条件就是:)
            # The lower edge of the door < threshold ratio * the door pixel height(门的下边沿 < 阈值比例 * 门的像素高度(当门低于其自身高度的一半时,就会进入低空状态。)
            # The robot enters the low-altitude state when the door is less than half its own height)
            if max(door_data.y, door_data.left_point[1], door_data.right_point[1]) < self.enter_crawl_under_y * door_data.height:
                # Switch to low posture(低姿态切换)
                self.gait_manager.stop()
                self.gait_manager.set_body_height(0.06, 1)  # Set body height to 0.06m(设置身体高度为0.06m)
                time.sleep(0.5)
                walking_param = self.gait_manager.get_gait_param()
                walking_param['body_height'] = 0.06         # Set the gait body height to 0.06m(设置步态身体高度为0.06m)
                # Update forward gait parameters(更新正向步态参数)
                self.visual_patrol.update_go_gait(walking_param=walking_param)
                # Update turning gait parameters(更新转向步态参数)
                self.visual_patrol.update_turn_gait(walking_param=walking_param)
                self.motion_manager.set_servos_position(100, [[23, self.state[self.current_state][0][0]],
                                                              [24, self.state[self.current_state][0][1]]])
                common.loginfo('crawl_under')
                self.head_state = 'drop'
                self.body_raise_stamp = rospy.get_time() + 7    # Set the duration of low-altitude to 7 seconds(设置7秒低空持续时间)
                return True
        return False
    # Determination for exiting low-altitude state(退出低空状态的判断)
    def exit_crawl_under(self, line_data):
        if line_data is not None:
            if rospy.get_time() < self.body_raise_stamp:  # The duration for low-altitude walking(低姿态持续行走的时间)
                self.visual_patrol.process(line_data.x, line_data.width)
            else:
                # Switch to normal line following state(切换正常巡线姿态)
                self.gait_manager.stop()
                self.gait_manager.set_body_height(0.015, 1)
                time.sleep(0.5)
                walking_param = self.gait_manager.get_gait_param()
                walking_param['body_height'] = 0.015
                self.visual_patrol.update_go_gait(walking_param=walking_param)
                self.visual_patrol.update_turn_gait(walking_param=walking_param)
                return True
        return False

    def run(self):
        while self.running:
            if self.state:
                # Obtain recognition result(获取识别结果)
                line_data = None
                door_data = None
                for object_info in self.objects_info:
                    if object_info.type == 'line':
                        if self.head_state == 'drop':
                            line_data = object_info
                    if object_info.type == 'side':
                        if self.head_state == 'raise':
                            door_data = object_info
                
                # Different phrase processing(不同阶段处理)
                if self.current_state == 'visual_patrol':
                    if line_data is not None:
                        self.visual_patrol.process(line_data.x, line_data.width)
                elif self.current_state == 'crawl_under':
                    if self.exit_crawl_under(line_data):
                        self.current_state = 'visual_patrol'
                        self.next_state = 'crawl_under'
                        self.state[self.current_state][2] = False
                 # Process based on the next state(根据下一个状态进行处理)
                if self.next_state == 'crawl_under':
                    if self.enter_crawl_under(door_data):
                        self.current_state = 'crawl_under'
                        self.next_state = 'visual_patrol'
                        self.state[self.current_state][2] = False
                # Initialize state(状态初始化)
                self.state_init(self.current_state, self.next_state)

                time.sleep(0.01)
            else:
                time.sleep(0.01)
        
        self.init_action(self.head_pan_init, self.head_tilt_init)
        self.stop_srv_callback(None)        # Stop running(停止运行)
        rospy.signal_shutdown('shutdown')

if __name__ == "__main__":
    CrawlUnderVisualPatrol('crawl_under_visual_patrol').run()
