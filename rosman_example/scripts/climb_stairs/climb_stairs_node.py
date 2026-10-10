#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/07/10
# @author:aiden
# Up and down steps(上阶梯or下阶梯)
import time
import rospy
import signal
from rosman_sdk import common
from rosman_example.color_common import Common
from rosman_example.approach_object import ApproachObject
from rosman_interfaces.srv import SetString
from rosman_interfaces.msg import ObjectsInfo, ColorDetect

class ClimbStairsNode(Common):
    # ROI of the steps(阶梯的roi)
    # (y_min, y_max, x_min, and x_max represent respectively the proportions of the image(y_min, y_max, x_min, x_max分别表示占图像的比例)
    stairs_roi = [1/5, 1, 0, 1]

    # Action of up and down steps(上下阶梯动作)
    climb_stairs_action_name = 'climb_stairs'       # Up steps(上台阶)
    descend_stairs_action_name = 'descend_stairs'   # Down steps(下台阶)

    # When process the image, scale it to this resolution(图像处理时缩放到这个分辨率)
    image_process_size = [160, 120]

    # Stop conditions for up steps in the x direction(上阶梯x方向的停止条件)
    # Stop when the x distance to the target position is less than 385/480 of the image width(x距离目标位置小于图片宽度的385/480时停止)
    climb_stairs_x_stop = 385/480  
    # Stop conditions for up steps in the y direction(上阶梯y方向的停止条件)
    # Stop when the y distance to the target position is less than half of the image height(y距离目标位置小于图片高度的一半时停止)
    climb_stairs_y_stop = 0.5
    # Stop conditions for up steps in the yaw direction(上阶梯航向角停止条件)
    # Allow for a certain amount of yaw error(允许一定的航向角误差)
    climb_stairs_yaw_stop = 0


    # Stop conditions for down steps in the x direction(下阶梯x方向的停止条件)
    # Stop when the x distance to the target position is less than 385/480 of the image width(x距离目标位置小于图片宽度的385/480时停止)
    descend_stairs_x_stop = 385/480
    # Stop conditions for down steps in the y direction(下阶梯y方向的停止条件)
    # Stop when the y distance to the target position is less than half of the image height(y距离目标位置小于图片高度的一半时停止)
    descend_stairs_y_stop = 0.5
    # Stop conditions for down steps in the yaw direction(上阶梯航向角停止条件)
    # Allow for a certain amount of yaw error(允许一定的航向角误差)
    descend_stairs_yaw_stop = 0 

    def __init__(self, name):
        rospy.init_node(name)       # Initialize ROS node(初始化ROS节点)
        self.name = name
        self.running = True
        self.objects_info = []
        self.current_state = 'climb_stairs' # Current state(当前状态)

        self.state = {'climb_stairs':   [[500, 260], ['red', self.stairs_roi, self.image_process_size, self.set_stairs_color], False],
                      'descend_stairs': [[500, 260], ['red', self.stairs_roi, self.image_process_size, self.set_stairs_color], False]}
        # Initialize the head posture(初始化头部姿态)
        self.head_pan_init = self.state[self.current_state][0][0]   # Initial value of the left-right servo(左右舵机的初始值)
        self.head_tilt_init = self.state[self.current_state][0][1]  # Initial value of the up-down servo(上下舵机的初始值)
        super().__init__(name, self.head_pan_init, self.head_tilt_init)
        # Initialize related nodes and modules(初始化相关节点和模块)
        self.approach_object = ApproachObject(self.gait_manager)

        # Update the judgment conditions for approaching and stopping(更新逼近停止的判断条件)
        # Distance error allowed in the x direction, in units of pixels(x方向允许的距离误差,像素为单位)
        # Distance error allowed in the y direction, in units of pixels(y方向允许的距离误差,像素为单位)
        # Error allowed in the yaw angle, in units of degrees(航向角允许的误差,度为单位)
        self.approach_object.update_approach_stop_value(40, 50, 6)
        # The first parameter 40 represents the tolerance range in the x direction of the target position, in units of pixels. That is, the robot is considered to have approached the target position when it is within 50 pixels in the x direction(第一个参数40表示目标位置x方向的容差范围,单位是像素。也就是机器人x方向距离目标位置在50像素之内就认为逼近完成。)
        # The second parameter 50 represents the tolerance range in the y direction of the target position. Similarly, it is considered to have approached the target position when it is within 60 pixels in the y direction(第二个参数50表示目标位置y方向的容差范围。同理,在60像素之内认为y方向逼近完成。)
        # The third parameter 6 represents the tolerance range of the target position's yaw angle, in degrees. It is considered to have approached the target position when the yaw angle error is within 7 degrees(第三个参数6表示目标位置的航向角容差范围,单位是度。在7度角度误差之内就认为航向角逼近完成。)

        self.motion_manager.run_action('hand_back')  # Move the hand back to prevent obstruction(手往后，防止遮挡)
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
        stairs_param.min_area = 10 * 20
        stairs_param.max_area = image_process_size[0] * image_process_size[1]

        return stairs_param

    def set_color_srv_callback(self, msg):
        # Set color(设置颜色)
        stairs_param = self.set_stairs_color(msg.data)
        self.detect_pub.publish([stairs_param])
        common.loginfo('%s set_color' % self.name)
        
        return [True, 'set_color']

    def get_color_callback(self, msg):
        # Obtain color recognition result(获取颜色识别结果)
        self.objects_info = msg.data

    def climb_stairs_process(self, stairs_data):
        # Up steps processing(上阶梯处理)
        if self.approach_object.process(max(stairs_data.y, stairs_data.left_point[1], stairs_data.right_point[1]), stairs_data.x, stairs_data.angle, 
                                        self.climb_stairs_x_stop*stairs_data.height, self.climb_stairs_y_stop*stairs_data.width, self.climb_stairs_yaw_stop, stairs_data.width, stairs_data.height):
            self.gait_manager.disable()  # Close gait control(关闭步态控制)
            common.loginfo('climb_stairs')
            self.motion_manager.run_action(self.climb_stairs_action_name)  # Execute up-steps action(执行上台阶动作)
            return True
        else:
            return False

    def descend_stairs_process(self, stairs_data):
        # Down steps processing(下阶梯处理)
        if self.approach_object.process(max(stairs_data.y, stairs_data.left_point[1], stairs_data.right_point[1]), stairs_data.x, stairs_data.angle, 
                                        self.descend_stairs_x_stop*stairs_data.height, self.descend_stairs_y_stop*stairs_data.width, self.descend_stairs_yaw_stop, stairs_data.width, stairs_data.height):
            self.gait_manager.disable()  # Close gait control(关闭步态控制)
            common.loginfo('descend_stairs')
            self.motion_manager.run_action(self.descend_stairs_action_name)  # Execute down-steps action(执行下台阶动作)
            return True
        else:
            return False

    def run(self):
        while self.running:
            if self.start:
                # Set color recognition parameter(设置颜色识别参数)
                if self.state[self.current_state][2] == False:
                    self.state[self.current_state][2] = True
                    self.init_action(self.state[self.current_state][0][0], self.state[self.current_state][0][1])  # Head posture(头部姿态)
                    param = self.state[self.current_state][1][3](self.state[self.current_state][1][0],
                                                                 self.state[self.current_state][1][1],
                                                                 self.state[self.current_state][1][2])
                    self.detect_pub.publish([param])  # Color detection settings(颜色检测设置)

                # Obtain recognition result(获取识别结果)
                stairs_data = None
                for object_info in self.objects_info:
                    if object_info.type == 'side':
                        stairs_data = object_info
                # Process corresponding up and down steps logic(处理对应的上下阶梯逻辑)
                if stairs_data is not None and self.current_state == 'climb_stairs':
                    if self.climb_stairs_process(stairs_data):
                        self.running = False
                    else:
                        time.sleep(0.8)

                if stairs_data is not None and self.current_state == 'descend_stairs':
                    if self.descend_stairs_process(stairs_data):
                        self.running = False
                    else:
                        time.sleep(0.8)
            else:
                time.sleep(0.01)
        # Exit processing(退出处理)
        self.init_action(self.head_pan_init, self.head_tilt_init)
        self.stop_srv_callback(None)
        rospy.signal_shutdown('shutdown')

if __name__ == "__main__":
    ClimbStairsNode('climb_stairs').run()
