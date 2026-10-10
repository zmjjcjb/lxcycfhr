#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/07/03
# @author:aiden
# Visual line following node(视觉巡线节点)
import time
import rospy
import signal
from std_msgs.msg import String
from rosman_sdk import common
from rosman_example.color_common import Common
from rosman_example.visual_patrol import VisualPatrol
from rosman_interfaces.srv import SetString
from rosman_interfaces.msg import ObjectsInfo, ColorDetect

class VisualPatrolNode(Common):
    # Check three ROIs in order, and immediately exit if a black line is detected(按顺序检测三个roi，如果检测到黑线立刻跳出)
    # y_min, y_max, x_min, and x_max represent respectively the proportion of the image occupied. The actual size is y_min*height(y_min, y_max, x_min, x_max分别表示占图像的比例, 即实际大小为y_min*height)
    # 顺序 = 优先级：color_detection 是按顺序第一个命中就 break，所以排在前面的那排
    # 才是真正驱动转向的那一排。
    #
    # 原来远排(5/12~6/12)排第一，实测导致"机器人和白线平行往前走、不纠偏"：
    # 盯得越远，横向偏移越看不出来 —— 平行偏移只在近处那排明显，而近处那排永远轮不到被读。
    # 直到偏出远排窄框(1/4~3/4)才 fall through 到近排，那时偏差已经 300px 左右了。
    # 所以改成近排优先：脚跟前那段线才是"我偏了多少"的真信号，远排只当兜底/前瞻。
    line_roi = [(7 / 12, 8 / 12, 0, 1),
                (6 / 12, 7 / 12, 0, 1),
                (5 / 12, 6 / 12, 1 / 4, 3 / 4)
                ]
    # When process the image, scale it to this resolution(图像处理时缩放到这个分辨率)
    image_process_size = [160, 120]

    def __init__(self, name):
        rospy.init_node(name)
        self.name = name
        self.running = True
        self.objects_info = []
        # Initialize head servo(初始化头部位置)
        self.head_pan_init = 500   # Initial value of the left-right servo(左右舵机的初始值)
        self.head_tilt_init = 260  # Initial value of the up-down servo(上下舵机的初始值)
        super().__init__(name, self.head_pan_init, self.head_tilt_init)
        # Create line following control instance(创建巡线控制实例)
        self.visual_patrol = VisualPatrol(self.gait_manager)
        # Set exit processing function(设置退出处理函数)
        signal.signal(signal.SIGINT, self.shutdown)

        # Subscribe to color recognition result(订阅颜色识别结果)
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)
        # Initialize color setting service(初始化设置颜色服务)
        rospy.Service('~set_color', SetString, self.set_color_srv_callback)  
        self.motion_manager.run_action('walk_ready')     # Play ready posture action(播放准备姿势动作)
        # If the autonomous start parameter is ture, enable node(如果自动开始参数为真,启动节点)
        if rospy.get_param('~start', True):
            # Notify the color recognition to prepare, at this time only display the camera original image(通知颜色识别准备，此时只显示摄像头原画)
            target_color = rospy.get_param('~color', 'black')  # Set to recognize black(设置识别黑色)
            self.enter_func(None)
            self.set_color_srv_callback(String(target_color))  
            self.start_srv_callback(None)  # Enable color recognition(开启颜色识别)
            common.loginfo('start track %s lane' % target_color)

    # Node closing callback function(节点关闭回调函数)
    def shutdown(self, signum, frame):
        with self.lock:
            self.motion_manager.run_action('stand')
            self.running = False 
            common.loginfo('%s shutdown' % self.name)
            
    # Set color service callback function(设置颜色服务回调函数)
    def set_color_srv_callback(self, msg):
        # Generate color recognition parameter(生成颜色识别参数)
        param = ColorDetect()
        param.color_name = msg.data
        param.use_name = True
        param.detect_type = 'line'  # Specify the detection type to line(指定了检测类型为线条"line")
        param.image_process_size = self.image_process_size
        # Set ROI paramete(设置ROI参数r)
        param.line_roi.up.y_min = int(self.line_roi[0][0] * self.image_process_size[1])
        param.line_roi.up.y_max = int(self.line_roi[0][1] * self.image_process_size[1])
        param.line_roi.up.x_min = int(self.line_roi[0][2] * self.image_process_size[0])
        param.line_roi.up.x_max = int(self.line_roi[0][3] * self.image_process_size[0])

        param.line_roi.center.y_min = int(self.line_roi[1][0] * self.image_process_size[1])
        param.line_roi.center.y_max = int(self.line_roi[1][1] * self.image_process_size[1])
        param.line_roi.center.x_min = int(self.line_roi[1][2] * self.image_process_size[0])
        param.line_roi.center.x_max = int(self.line_roi[1][3] * self.image_process_size[0])

        param.line_roi.down.y_min = int(self.line_roi[2][0] * self.image_process_size[1])
        param.line_roi.down.y_max = int(self.line_roi[2][1] * self.image_process_size[1])
        param.line_roi.down.x_min = int(self.line_roi[2][2] * self.image_process_size[0])
        param.line_roi.down.x_max = int(self.line_roi[2][3] * self.image_process_size[0])
        # Area filtering parameter(面积过滤参数)
        param.min_area = 1
        param.max_area = self.image_process_size[0] * self.image_process_size[1]
        # Publish color recognition parameter(发布颜色识别参数)
        self.detect_pub.publish([param])
        common.loginfo('%s set_color' % self.name)
        return [True, 'set_color']
    
    # Publish color recognition parameter(发布颜色识别参数)
    def get_color_callback(self, msg):
        # Obtain color recognition result(获取颜色识别结果)
        self.objects_info = msg.data

    # Main loop function(主循环函数)
    def run(self):
        while self.running:
            if self.start:
                # Obtain recognition result(获取识别结果)
                line_data = None
                # stairs_data = None
                for object_info in self.objects_info:
                    if object_info.type == 'line':
                        line_data = object_info
                # If there is recognition result, control line following(如果有识别结果,巡线控制)
                if line_data is not None:
                    # 第三个参数 y 只用于追踪日志（反推是哪一排 ROI 命中的）
                    self.visual_patrol.process(line_data.x, line_data.width, line_data.y)
                else:
                    # 丢线必须主动接管：步态是 step_num=0 的连续模式，什么都不发就会
                    # 沿用上一条(往往已饱和的)转向指令继续前冲，越偏越远
                    self.visual_patrol.on_no_line()
                time.sleep(0.01)
            else:
                time.sleep(0.01)
        # Action before exiting(退出前动作)
        self.init_action(self.head_pan_init, self.head_tilt_init)
        self.stop_srv_callback(None)
        rospy.signal_shutdown('shutdown')

if __name__ == "__main__":
    VisualPatrolNode('visual_patrol').run()
