#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/09/27
# @author:aiden

import rospy
import signal
import time
from rosman_sdk import misc, common
from rosman_example.color_common import Common
from rosman_example.visual_patrol import VisualPatrol
from rosman_example.approach_object import ApproachObject
from rosman_interfaces.srv import SetString
from rosman_interfaces.msg import ObjectsInfo, ColorDetect, ROI


class CompetitionNode(Common):
    # 按顺序检测三个roi，如果检测到黑线立刻跳出
    # y_min, y_max, x_min, x_max分别表示占图像的比例, 即实际大小为y_min*height
    line_roi = [(5 / 12, 6 / 12, 1 / 4, 3 / 4),
                (6 / 12, 7 / 12, 1 / 4, 3 / 4),
                (7 / 12, 8 / 12, 1 / 4, 3 / 4) 
                ]

    #stairs_roi = [1 / 5, 1, 0, 1]
    stairs_roi = [0, 1, 0.3, 0.7]
    hurdles_roi = [0 , 1, 0.3, 0.7]
    block_roi = [0, 270 / 480, 0, 1]
    intersection_roi = [100 / 480, 300 / 480, 0, 1]
    
    # 所需动作的名称
    hurdles_action_name = 'hurdles_leng'
    climb_stairs_action_name = 'climb_stairs'
    descend_stairs_action_name = 'descend_stairs'
    crawl_left_action_name = 'crawl_left_xinan'   # 左行走动作名称
    crawl_right_action_name = 'crawl_right_2025' # 右行走动作名称
    place_block_action_name = 'place_block' # 放置方块动作名称
    forward_step = 'forward_one_step_1'   #前进一步
    back_step = 'back_step_1'             #后退一步
    move_left = 'move_left_1'             #左移
    move_right = 'move_right_1'           #右移
    turn_right = 'turn_right_30'
    turn_left = 'turn_left_30'
    forward_step_1 = 'forward_step'
    # 图像处理时缩放到这个分辨率， 不建议修改
    image_process_size = [160, 120]
    # 上台阶状态下的目标位置阈值
    enter_climb_stairs_y = 240 / 480  # 当检测到的标识像素坐标y值占图像的比例大于此值时进入此阶段
    climb_stairs_x_stop = 0.5  # 当检测到的标识像素坐标x值占图像的比例在此值附近(范围可在ApproachObject里设置)时停止前后移动
    climb_stairs_y_stop = 350 / 480  # 当检测到的标识像素坐标y值占图像的比例在此值附近(范围可在ApproachObject里设置)时停止横向移动
    climb_stairs_yaw_stop = 5  # 当检测到的标识角度在此值附近(范围可在ApproachObject里设置)时停止旋转移动
    # 下台阶状态下的目标位置阈值
    enter_descend_stairs_y = 200/480
    descend_stairs_x_stop = 0.45
    descend_stairs_y_stop = 350 / 480
    descend_stairs_yaw_stop = 0
    # 跨栏状态下的目标位置阈值
    enter_hurdles_y = 272 / 480 #280
    hurdles_x_stop = 0.5
    hurdles_y_stop = 310 / 480
    hurdles_yaw_stop = 0
    # 向左抓取状态下的目标位置阈值
    enter_crawl_left_y = 130 / 480 # 当检测到的标识像素坐标y值占图像的比例大于此值时进入此阶段
    #enter_crawl_left_y = 150 / 480
    crawl_left_x_stop = 195 / 640  # 左行走的x轴目标位置范围可在ApproachObject里设置)时停止前后移动
    #crawl_left_x_stop = 206 / 640
    crawl_left_y_stop = 225 / 480  # 左行走的y轴目标位置(范围可在ApproachObject里设置)时停止横向移动
    crawl_left_yaw_stop = 0        # 当检测到的标识角度在此值附近(范围可在ApproachObject里设置)时停止旋转移动
    # 向右抓取状态下的目标位置阈值，原理同上
    enter_crawl_right_y = 140 / 480
    crawl_right_x_stop = 390 / 640 #爪子偏左减小，偏右增大
    crawl_right_y_stop = 255 / 480
    crawl_right_yaw_stop = 0
    # 放块状态下的目标位置阈值
    #enter_place_block_y = 135 / 480
    enter_place_block_y = 350 / 480
    place_block_x_stop = 320 / 640
    place_block_y_stop = 450 / 480
    place_block_yaw_stop = 0
  
    #是否完成第一次调整标志
    hurdles_approach_signal = False
    climb_stairs_approach_signal = False
    descend_stairs_approach_signal = False
    descend_stairs_approach_signal = True
    place_block_approach_signal = True

    #是否多次检测到上台阶标志
    exist_stairs_count = 0
    exist_right_block_count = 0
    line_data_count = 0
    first_descend_stairs_data = 349
    
    def __init__(self, name):
        rospy.init_node(name)
        self.calib_config = common.get_yaml_data('/home/ubuntu/ros_ws/src/rosman_example/config/calib.yaml')
        self.name = name
        self.count = 0  
        self.running = True
        self.slow = True
        self.forward_signal = True      #左抓取向前走
        self.objects_info = []
        self.delay_time = 0
        self.current_state = "visual_patrol"            # 当前状态
        self.next_state = "hurdles"                # 下一状态

        self.state = {'visual_patrol':  [[500, 260], ['black', self.line_roi, self.image_process_size, self.set_visual_patrol_color], False],        # 巡线
                      'climb_stairs':   [[500, 260], ['red', self.stairs_roi, self.image_process_size, self.set_stairs_color], False],               # 上台阶
                      'descend_stairs': [[500, 260], ['red', self.stairs_roi, self.image_process_size, self.set_stairs_color], False],               # 下台阶
                      'hurdles': [[500, 260], ['blue', self.hurdles_roi, self.image_process_size, self.set_hurdles_color], False],                   # 跨栏
                      'crawl_left':   [[500, 260], ['green', self.block_roi, self.image_process_size, self.set_block_color], False],                    # 左抓取状态
                      'crawl_right': [[500, 260], ['green', self.block_roi, self.image_process_size, self.set_block_color], False],                     # 右抓取
                      'place_block':   [[500, 260], ['black', self.intersection_roi, self.image_process_size, self.set_intersection_color], False]}     # 放置  
        
        self.head_pan_init = self.state[self.current_state][0][0]   # 左右舵机的初始值
        self.head_tilt_init = self.state[self.current_state][0][1]  # 上下舵机的初始值
        super().__init__(name, self.head_pan_init, self.head_tilt_init)

        self.approach_object = ApproachObject(self.gait_manager)
        self.visual_patrol = VisualPatrol(self.gait_manager)
        # 减小巡线步幅，提高其他标志检测稳定性
        self.visual_patrol.update_go_gait( x_max=0.02)
        self.visual_patrol.update_turn_gait( x_max=0.02)

        signal.signal(signal.SIGINT, self.shutdown)
        
        # 订阅颜色识别结果
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)
        rospy.Service('~set_color', SetString, self.set_color_srv_callback)  # 设置颜色
        self.motion_manager.run_action('walk_ready')

        if rospy.get_param('~start', True):
            # 通知颜色识别准备，此时只显示摄像头原画
            self.enter_func(None)
            self.start_srv_callback(None)
            common.loginfo('start ombination')

    def shutdown(self, signum, frame):
        self.running = False
        common.loginfo('%s shutdown' % self.name)

    def set_visual_patrol_color(self, color, roi, image_process_size):
        # 设置巡线颜色
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

    # 生成方块颜色识别参数
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

    # 生成交叉点颜色识别参数 
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
        # 设置台阶标志颜色
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
        # 设置台阶标志颜色
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
        # 设置颜色
        block_param = self.set_block_color(self.state['crawl_left'][1][0])
        stairs_param = self.set_stairs_color(self.state['climb_stairs'][1][0])
        line_param = self.set_visual_patrol_color(self.state['visual_patrol'][1][0])
        hurdles_param = self.set_hurdles_color(self.state['hurdles'][1][0])
        intersection_param = self.set_intersection_color(self.state['place_block'][1][0])

        self.detect_pub.publish([line_param, stairs_param, hurdles_param, intersection_param, block_param])
        common.loginfo('%s set_color' % self.name)
        
        return [True, 'set_color']

    def get_color_callback(self, msg):
        # 获取颜色识别结果
        self.objects_info = msg.data

    def state_init(self, current_state, next_state):
        # 不同阶段的初始化
        if self.state[current_state][2] == False:
            self.state[current_state][2] = True
            self.init_action(self.state[current_state][0][0], self.state[current_state][0][1])  # 头部姿态
            param1 = self.state[current_state][1][3](self.state[current_state][1][0], self.state[current_state][1][1], self.state[current_state][1][2])
            param2 = self.state[next_state][1][3](self.state[next_state][1][0], self.state[next_state][1][1], self.state[next_state][1][2])
            self.detect_pub.publish([param1, param2])  # 颜色检测设置
            common.loginfo(current_state + ' init')
    #退出上台阶
    def exit_climb_stairs(self, stairs_data):
        # 上阶梯处理
        if stairs_data is not None:
            print("state:exit_climb_stairs ","stairs_data.y:",stairs_data.y)
            if self.approach_object.process(max(stairs_data.y, stairs_data.left_point[1], stairs_data.right_point[1]), stairs_data.x, stairs_data.angle, 
                                        self.climb_stairs_y_stop*stairs_data.height, self.climb_stairs_x_stop*stairs_data.width, self.climb_stairs_yaw_stop, stairs_data.width, stairs_data.height):
                self.gait_manager.disable()  # 关闭步态控制
                common.loginfo('climb_stairs')
                #判断当前状态是否还需要向前走一步
                if stairs_data.y < 350:
                  self.motion_manager.run_action(self.forward_step)
                
                self.motion_manager.run_action(self.climb_stairs_action_name)  # 执行上台阶动作
                #rospy.sleep(0.5)
                self.visual_patrol.update_go_gait(dsp=[350, 0.4, 0.02], x_max=0.01)
                self.visual_patrol.update_turn_gait(dsp=[380, 0.4, 0.02], x_max=0.01)                
                self.motion_manager.run_action('walk_ready')
                #rospy.sleep(0.5)
                return True
        
        if self.climb_stairs_approach_signal:
            self.gait_manager.disable()  # 关闭步态控制
            common.loginfo('climb_stairs')
            #判断当前状态是否还需要向前走一步
            #if stairs_data.y < 350:
              #self.motion_manager.run_action(self.forward_step)
            
            self.motion_manager.run_action(self.climb_stairs_action_name)  # 执行上台阶动作
            #rospy.sleep(0.5)
            self.visual_patrol.update_go_gait(dsp=[380, 0.4, 0.02], x_max=0.01)
            self.visual_patrol.update_turn_gait(dsp=[380, 0.4, 0.02], x_max=0.01)                
            self.motion_manager.run_action('walk_ready')
            #rospy.sleep(0.5)
            return True
        
        self.climb_stairs_approach_signal = True
        
        return False
    #退出下台阶
    def exit_descend_stairs(self, stairs_data):
        # 下阶梯处理
        if stairs_data is not None:
            self.first_descend_stairs_data = stairs_data.y
            print("state:exit_descend_stairs ","stairs_data.y:",stairs_data.y)
            if self.descend_stairs_approach_signal and self.approach_object.process(max(stairs_data.y, stairs_data.left_point[1], stairs_data.right_point[1]), stairs_data.x + self.calib_config['center_x_offset'], stairs_data.angle, 
                                        self.descend_stairs_y_stop*stairs_data.height, self.descend_stairs_x_stop*stairs_data.width, self.descend_stairs_yaw_stop, stairs_data.width, stairs_data.height):
                self.descend_stairs_approach_signal = False
                self.gait_manager.disable()  # 关闭步态控制
                common.loginfo('descend_stairs')
                #判断当前状态是否还需要向前走一步
            if stairs_data.y < 300:
                self.motion_manager.run_action(self.forward_step_1)
                print("look here:  ", stairs_data.y)
                return False
        
        if self.first_descend_stairs_data < 350:
            self.motion_manager.run_action(self.forward_step_1)
            print('forward one step')
        #当没有数据的时候执行下台阶动作
        self.gait_manager.disable()  # 关闭步态控制
        common.loginfo('descend_stairs')
        print('connot see stairs')
        self.motion_manager.run_action(self.forward_step)
        self.motion_manager.run_action(self.descend_stairs_action_name)  # 执行下台阶动作
        rospy.sleep(0.5)
        self.visual_patrol.update_go_gait(dsp=[350, 0.4, 0.02], x_max=0.02)
        self.visual_patrol.update_turn_gait(dsp=[380, 0.4, 0.02], x_max=0.02)
        self.slow = True
        self.motion_manager.run_action('walk_ready')
        rospy.sleep(0.5)
        return True
        
    #退出跨栏
    def exit_hurdles(self, hurdles_data):
        # 跨栏处理
        if hurdles_data is not None:
            print("state:exit_hurdles ","hurdles_data.y:",hurdles_data.y)
            #rospy.sleep(2)
            
            if self.approach_object.process(max(hurdles_data.y, hurdles_data.left_point[1], hurdles_data.right_point[1]), hurdles_data.x + self.calib_config['center_x_offset'], hurdles_data.angle, 
                                   self.hurdles_y_stop*hurdles_data.height, self.hurdles_x_stop*hurdles_data.width, self.hurdles_yaw_stop, hurdles_data.width, hurdles_data.height):
                self.gait_manager.disable()
                common.loginfo('hurdles')
                if hurdles_data.y < 280:
                    print('look_here hurdles_data.y:',hurdles_data.y," the_first_step")
                    self.motion_manager.run_action(self.forward_step_1)
                    return False

                if hurdles_data.y < 375: 
                    print('look_here hurdles_data.y:',hurdles_data.y," the_second_step")
                    self.motion_manager.run_action(self.forward_step)
                
                self.motion_manager.run_action(self.hurdles_action_name)
                #rospy.sleep(0.5)
                self.visual_patrol.update_go_gait(dsp=[350, 0.4, 0.02], x_max=0.02)
                self.visual_patrol.update_turn_gait( x_max=0.02)
                self.slow = False
                self.motion_manager.run_action('walk_ready')
                #rospy.sleep(0.5)
                return True
            
        if self.hurdles_approach_signal:
          self.gait_manager.disable()
          common.loginfo('hurdles')     
          self.motion_manager.run_action(self.hurdles_action_name)
          #rospy.sleep(0.5)
          self.visual_patrol.update_go_gait(dsp=[350, 0.4, 0.02], x_max=0.02)
          self.visual_patrol.update_turn_gait( x_max=0.02)
          self.slow = False
          self.motion_manager.run_action('walk_ready')
          #rospy.sleep(0.5)
          return True
            
        self.hurdles_approach_signal = True
                        
        return False
    #进入上台阶
    def enter_climb_stairs(self, stairs_data):
        if stairs_data is not None:
            print("state:enter_climb_stairs ","stairs_data.y:",stairs_data.y)
            self.exist_stairs_count += 1
            if stairs_data.y > self.stairs_roi[0] * stairs_data.height and self.exist_stairs_count > 3:
                self.visual_patrol.update_go_gait(dsp=[400, 0.4, 0.02], x_max=0.01)
                self.visual_patrol.update_turn_gait( x_max=0.01)
                self.slow = False
            if max(stairs_data.y, stairs_data.left_point[1], stairs_data.right_point[1]) > self.enter_climb_stairs_y * stairs_data.height:
                self.count += 1
                if self.count > 5:  # 主线程比较快，颜色检测回调慢一点，需要连续检测来排除滞后干扰
                    self.count = 0
                    self.gait_manager.disable()  
                    self.approach_object.update_approach_stop_value(20, 30, 6)  # 设置靠近目标停止的条件，分别为y, x, angle误差
                    self.motion_manager.run_action('hand_back')  # 手往后，防止遮挡
                    rospy.sleep(0.5)
                    return True
            else:
                self.count = 0
        return False
    #进入下台阶
    def enter_descend_stairs(self, stairs_data):
        if stairs_data is not None:
            print("state:enter_descend_stairs ","stairs_data.y:",stairs_data.y)
            if max(stairs_data.y, stairs_data.left_point[1], stairs_data.right_point[1]) > self.enter_descend_stairs_y * stairs_data.height:
                self.count += 1
                if self.count > 5:
                    self.count = 0
                    self.gait_manager.disable()
                    self.approach_object.update_approach_stop_value(20, 30, 6)
                    self.motion_manager.run_action('hand_back')  # 手往后，防止遮挡
                    rospy.sleep(0.5)
                    return True
            else:
                self.count = 0
        return False
    #进入跨栏
    def enter_hurdles(self, hurdles_data):
        if hurdles_data is not None:
            print("state:enter_hurdles ","hurdles_data.y:",hurdles_data.y)
            if hurdles_data.y > self.hurdles_roi[0] * hurdles_data.height and self.slow:
                self.visual_patrol.update_go_gait(dsp=[400, 0.4, 0.02], x_max=0.01)
                self.visual_patrol.update_turn_gait( x_max=0.01)
                self.slow = False            
            if max(hurdles_data.y, hurdles_data.left_point[1], hurdles_data.right_point[1]) > self.enter_hurdles_y * hurdles_data.height:
                self.count += 1
                if self.count > 5:
                    self.count = 0
                    self.gait_manager.disable()
                    self.approach_object.update_approach_stop_value(x_approach_value=30, yaw_approach_value=5)
                    self.motion_manager.run_action('hand_back')  # 手往后，防止遮挡
                    return True
            else:
                self.count = 0
        return False
    # 进入左抓取判断函数
    def enter_crawl_left(self, block_data):
        if block_data is not None:
            print("state:enter_crawl_left"," block_data.x:",block_data.x," block_data.y:",block_data.y)
            if block_data.y > self.block_roi[0] * block_data.height and self.slow:
                self.visual_patrol.update_go_gait(dsp=[400, 0.4, 0.02], x_max=0.01)
                self.visual_patrol.update_turn_gait(dsp=[400, 0.4, 0.02], x_max=0.01)
                self.slow = False
            # 如果色块的y坐标大于高度的1/4,说明有一部分进入ROI区域
            #if block_data.y > block_data.height / 4:
            if block_data.y > self.enter_crawl_left_y * 480:
                self.count += 1
                if self.count > 100:  # 连续检测到满足条件,切换到左行走状态
                    self.count = 0
                    self.gait_manager.disable()
                    return True
            else:
                self.count = 0
        else:
            self.count = 0
        return False
    # 进入右抓取判断函数
    def enter_crawl_right(self, block_data):
        if block_data is not None:
            self.exist_right_block_count += 1
            print("state:enter_crawl_right"," block_data.x:",block_data.x," block_data.y:",block_data.y)
            if block_data.y > self.block_roi[0] * block_data.height and self.exist_right_block_count > 10:
                self.visual_patrol.update_go_gait(dsp=[400, 0.4, 0.02], x_max=0.01)
                self.visual_patrol.update_turn_gait(dsp=[400, 0.4, 0.02], x_max=0.01)
                self.slow = False
            #if block_data.y > block_data.height / 4:
            if block_data.y > self.enter_crawl_right_y * 480:
                self.count += 1
                if self.count > 100:
                    self.count = 0
                    self.gait_manager.disable()
                    return True
            else:
                self.count = 0
        else:
            self.count = 0
        return False
    # 进入放置判断函数
    def enter_place_block(self, line_data):
        if line_data is not None and abs(line_data.angle) < 10:
            self.line_data_count += 1
            self.visual_patrol.update_go_gait(dsp=[420, 0.4, 0.02], x_max=0.01)
            print("state:enter_place_block ","line_data.y:",line_data.y)
            print((line_data.right_point[0] - line_data.left_point[0]))
            if line_data.y > 150 and self.line_data_count > 10:
                return True
            print('look_here: line_data.angle = ', abs(line_data.angle))
        return False
    # 退出左抓取判断函数
    def exit_crawl_left(self, block_data):
        if self.forward_signal:

            self.forward_signal = False
        if block_data is not None:
            print("state:exit_crawl_left"," block_data.x:",block_data.x," block_data.y:",block_data.y)

            if block_data.y - 245 > 5:
                self.motion_manager.run_action(self.back_step)
                return False
            
            if block_data.y - 245 < -10:
                self.motion_manager.run_action(self.forward_step_1)
                return False
            
            if block_data.x - 300 > 5:
                self.motion_manager.run_action(self.move_right)
                return False
            
            if block_data.x - 300 < -5: 
                self.motion_manager.run_action(self.move_left)
                return False
            
            if block_data.y >= self.crawl_left_y_stop*block_data.height: 
                time.sleep(0.5)
                if block_data.y >= self.crawl_left_y_stop*block_data.height:
                    self.motion_manager.run_action('hand_back')
                    self.motion_manager.run_action(self.crawl_left_action_name)
                    self.visual_patrol.update_go_gait(arm_swap=0)
                    self.visual_patrol.update_turn_gait(arm_swap=0)
                    self.visual_patrol.update_go_gait(dsp=[350, 0.4, 0.02], x_max=0.02)
                    self.visual_patrol.update_turn_gait(dsp=[380, 0.4, 0.02], x_max=0.02)
                    self.slow = False                    
                    return True
                                            
            time.sleep(0.01)

        if block_data is None:
            self.motion_manager.run_action(self.back_step)
            return False

        return 

    # 退出右抓取判断函数
    def exit_crawl_right(self, block_data):
        if block_data is not None:
            print("state:exit_crawl_right"," block_data.x:",block_data.x,"block_data.y:",block_data.y)

            if block_data.y - 240 < -10:
                self.motion_manager.run_action(self.forward_step_1)
                return False
            
            if block_data.x - 410 > 5:
                self.motion_manager.run_action(self.move_right)
                return False
                    
            if block_data.x - 410 < -5: 
                self.motion_manager.run_action(self.move_left)
                return False
            
            if block_data.y - 240 > 5:
                self.motion_manager.run_action(self.back_step)
                return False
                        
            time.sleep(0.5)
            
            self.motion_manager.run_action(self.crawl_right_action_name)
            self.visual_patrol.update_go_gait(arm_swap=0)
            self.visual_patrol.update_turn_gait(arm_swap=0)
            self.visual_patrol.update_go_gait(dsp=[420, 0.4, 0.02], x_max=0.02)
            self.visual_patrol.update_turn_gait(dsp=[420, 0.4, 0.02], x_max=0.02)
            self.slow = True 
            self.delay_time = time.time() #+ 19.5        #抓完色块延时后再识别横线
            return True
        
        if block_data is None:
            self.motion_manager.run_action(self.back_step)      

        return False
    # 退出放块判断函数
    def exit_place_block(self, line_data):
        if line_data is not None:
            print("state:exit_place_block ","line_data.y:",line_data.y)
            if self.place_block_approach_signal and self.approach_object.process(max(line_data.y, line_data.left_point[1], line_data.right_point[1]), line_data.x, line_data.angle, 
                                            self.place_block_y_stop*line_data.height, self.place_block_x_stop*line_data.width, self.place_block_yaw_stop, line_data.width, line_data.height):
                self.place_block_approach_signal = False
                print('approach_finish')
            
            if line_data.y < 296:
                print('look_here:', line_data.y)
                self.motion_manager.run_action(self.forward_step_1)
                return False
            self.gait_manager.disable()  # 关闭步态控制
            self.motion_manager.run_action(self.hurdles_action_name)
            self.motion_manager.run_action(self.forward_step_1)
            common.loginfo('place_block')
            for i in range(2):
                self.motion_manager.run_action(self.turn_right)
            self.motion_manager.run_action(self.place_block_action_name)
            return True
        
        if self.place_block_approach_signal:
            self.motion_manager.run_action(self.hurdles_action_name)
            self.motion_manager.run_action(self.forward_step_1)
            for i in range(2):
                self.motion_manager.run_action(self.turn_right)
            common.loginfo('place_block')
            self.motion_manager.run_action(self.place_block_Fme)
            return True
                
        self.place_block_approach_signal = True
                
        return False

    def start_forward(self, step_count):
        walking_param = self.gait_manager.get_gait_param()  
        walking_param['body_height'] = 0.015                
        walking_param['pelvis_offset'] = 5                  # 设置骨盆位置的前后偏移量,单位度
        walking_param['step_height'] = 0.02                 # 设置步高,单位米
        walking_param['hip_pitch_offset'] = 15              # 设置髋关节角度偏移量,单位度
        walking_param['init_x_offset'] = -0.02
        walking_param['z_swap_amplitude'] = 0.006           # 设置左右足高度交替时的振幅,单位米
        # 调用set_step 设置步态参数,包括步长、步频等
        self.gait_manager.set_step([420, 0.4, 0.02], 0.02, 0, 0, None, 0.5, step_count)

    def run(self):
        while self.running:
            if self.start:
                # 获取识别结果
                line_data = None
                side_data = None
                block_data = None
                intersection_data = None               
                for object_info in self.objects_info:
                    if object_info.type == 'line':
                        line_data = object_info
                    if object_info.type == 'side':
                        side_data = object_info
                    if object_info.type == 'circle':
                        block_data = object_info
                    if object_info.type == 'intersection':
                        intersection_data = object_info
                    #print(object_info)
                # 当前阶段处理完成，回到巡线
                
                if self.current_state == 'visual_patrol' and line_data is None:
                    print('connot find line!!!!!!!')
                    self.motion_manager.run_action(self.turn_right)

                if self.current_state == 'visual_patrol':
                    if line_data is not None:
                        self.visual_patrol.process(line_data.x, line_data.width)
                elif self.current_state == 'hurdles':
                    if self.exit_hurdles(side_data):
                        self.current_state = 'visual_patrol'
                        self.next_state = 'climb_stairs'
                        self.state[self.current_state][2] = False  # 重新初始化当前阶段
                        common.loginfo('exit hurdles ---> enter visual_patrol')
                    else:
                        rospy.sleep(0.8)

                if self.current_state == 'visual_patrol':
                    if line_data is not None:
                        self.visual_patrol.process(line_data.x, line_data.width)
                elif self.current_state == 'climb_stairs':
                    if self.exit_climb_stairs(side_data):
                        self.current_state = 'visual_patrol'
                        self.next_state = 'descend_stairs'
                        self.state[self.current_state][2] = False  # 重新初始化当前阶段
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
                    # 如果当前状态是左抓取
                    if self.exit_crawl_left(block_data):
                        # 如果完成左抓取
                        self.current_state = 'visual_patrol'
                        self.next_state = 'crawl_right'
                        self.state[self.current_state][2] = False
                    else:
                        rospy.sleep(0.8)  # 等机体平稳下来
                        
                elif self.current_state == 'crawl_right':
                    # 右抓取状态
                    if self.exit_crawl_right(block_data):
                        self.current_state = 'visual_patrol'
                        self.next_state = 'place_block'
                        self.state[self.current_state][2] = False
                    else:
                        rospy.sleep(0.8)      

                elif self.current_state == 'place_block':
                    if self.exit_place_block(intersection_data):
                        self.running = False
                        common.loginfo('exit place_block ---> enter visual_patrol')
                    else:
                        rospy.sleep(0.8)

                # 是否退出巡线，进入下一阶段
                if self.next_state == 'hurdles':
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
                    if self.enter_crawl_left(block_data):
                        self.current_state = 'crawl_left'
                        self.next_state = 'visual_patrol'
                        common.loginfo('exit visual_patrol ---> enter crawl_left')
                elif self.next_state == 'crawl_right':
                    if self.enter_crawl_right(block_data):
                        self.current_state = 'crawl_right'
                        self.next_state = 'visual_patrol'
                        common.loginfo('exit visual_patrol ---> enter crawl_right')
                elif self.next_state == 'place_block':
                    #print(time.time(),self.delay_time)
                    if time.time() > self.delay_time :
                        if self.enter_place_block(intersection_data) :
                            self.current_state = 'place_block'
                            self.next_state = 'visual_patrol'

                self.state_init(self.current_state, self.next_state)

                rospy.sleep(0.01)  # 防止空载
            else:
                rospy.sleep(0.01)
        
        self.init_action(self.head_pan_init, self.head_tilt_init)
        self.stop_srv_callback(None)
        rospy.signal_shutdown('shutdown')

if __name__ == "__main__":
    CompetitionNode('competition').run()
