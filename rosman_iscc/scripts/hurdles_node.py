#!/usr/bin/env python3
# encoding: utf-8
# @Author: Dejin Xuan
# @Date: 2026.4.1 
# 跨栏节点
import math
import signal
import time

import rospy
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger, TriggerResponse

from rosman_example.color_common import Common
from rosman_example.visual_patrol import VisualPatrol
from rosman_interfaces.msg import ColorDetect, ObjectsInfo
from rosman_sdk import common


class HurdlesTracker(Common):
    """
    跨栏流程对齐 mission.py：
    激活后同时检测巡线 + 蓝色栅栏侧视；保持 VisualPatrol 巡线对齐；
    看到栅栏后放慢步态；当栅栏 y 超过配置阈值并连续多帧满足时执行跨栏动作。
    """

    def __init__(self, name):
        self.head_pan_init = rospy.get_param('/servo/head_pan_init', 500)
        self.head_tilt_look_down = rospy.get_param('/servo/head_tilt_look_down', 280)

        rospy.init_node(name)
        self.name = name
        self.running = True
        self.active = False

        super().__init__(name, self.head_pan_init, self.head_tilt_look_down)
        signal.signal(signal.SIGINT, self.shutdown)

        self.visual_patrol = VisualPatrol(self.gait_manager)

        self.current_state = 'idle'
        self.task_completed = False
        self.enter_counter = 0
        self.objects_info = []
        self.gait_switch_off_reported = False
        self.slow = True

        process_size = rospy.get_param('/image_processing/process_size', [160, 120])
        self.image_width = int(process_size[0]) if len(process_size) > 0 else 160
        self.image_height = int(process_size[1]) if len(process_size) > 1 else 120
        self.image_process_size = [self.image_width, self.image_height]

        self.load_hurdles_config()
        self.line_roi = self.load_line_roi()
        self.steering_gain = float(rospy.get_param('/tasks/line_following/steering_gain', 2.0))
        self.patrol_error_floor_px = float(rospy.get_param('/tasks/line_following/patrol_error_floor_px', 12))
        self.line_center_x_offset = self._load_line_center_x_offset()
        self.head_settle_sec = float(rospy.get_param('/tasks/line_following/head_settle_sec', 0.12))
        self.apply_line_following_gait_profile()

        self.setup_ros_interface()
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)

        self.motion_manager.run_action(self.walk_ready_action)

        common.loginfo('Hurdles tracker initialized (enter/低头在 /hurdles/start 时执行)')

    def load_hurdles_config(self):
        cfg = rospy.get_param('/tasks/hurdles', {})
        self.hurdles_color = cfg.get('color', 'blue')
        self.hurdles_roi = cfg.get('roi', [1.0 / 5.0, 1.0, 0.15, 0.85])
        self.enter_y_threshold = float(cfg.get('enter_y_threshold', 320.0))
        self.y_close_confirm_frames = int(cfg.get('y_close_confirm_frames', 6))
        self.slow_gait_x_max = float(cfg.get('slow_gait_x_max', 0.01))
        self.hurdles_min_area = int(cfg.get('min_area', 10))
        self.y_log_throttle_sec = float(cfg.get('y_log_throttle_sec', 0.2))
        m = cfg.get('motion', {})
        self.forward_steps_before_hurdles = int(m.get('forward_steps_before_hurdles', 1))
        self.forward_step_action = m.get('forward_step_action', 'forward_one_step')
        self.move_left_steps_before_hurdles = int(m.get('move_left_steps_before_hurdles', 0))
        self.move_right_steps_before_hurdles = int(m.get('move_right_steps_before_hurdles', 0))
        self.move_left_action = m.get('move_left_action', 'move_left')
        self.move_right_action = m.get('move_right_action', 'move_right')
        self.back_step_steps_when_lost = int(m.get('back_step_steps_when_lost', 1))
        self.back_step_action = m.get('back_step_action', 'back_step_1')
        self.lost_side_frames_before_back_step = int(m.get('lost_side_frames_before_back_step', 50))
        self.back_step_cooldown_sec = float(m.get('back_step_cooldown_sec', 1.0))
        # 其余动作组名（与 ActionGroups 内名称一致）
        self.hand_back_action = str(m.get('hand_back_action', 'hand_back'))
        self.hurdles_action_name = str(m.get('hurdles_action', 'hurdles'))
        self.walk_ready_action = str(m.get('walk_ready_action', 'walk_ready'))
        self.stand_action = str(m.get('stand_action', 'stand'))
        # 固定步分数日志中的阶段名（可自定义，便于终端 grep）
        self.fixed_step_log_phase_before = str(m.get('fixed_step_log_phase_before', 'hurdles_before'))
        self.fixed_step_log_phase_lost_back = str(m.get('fixed_step_log_phase_lost_back', 'hurdles_lost_back_step'))
        self._last_y_log_time = 0.0
        self._coord_frame_logged = False
        self._lost_side_counter = 0
        self._last_back_step_time = 0.0

    def _load_line_center_x_offset(self):
        try:
            calib = common.get_yaml_data('/home/ubuntu/ros_ws/src/rosman_example/config/calib.yaml')
            return float(calib.get('center_x_offset', 0))
        except Exception:
            return 0.0

    def patrol_corrected_x(self, line_data):
        width = float(line_data.width)
        patrol_center = width / 2.0 + self.line_center_x_offset
        err = float(line_data.x) - patrol_center
        scaled = err * self.steering_gain
        floor = self.patrol_error_floor_px
        if floor > 0.0 and scaled != 0.0 and abs(scaled) < floor:
            scaled = math.copysign(floor, scaled)
        cx = patrol_center + scaled
        return max(0.0, min(width - 1.0, cx)), width

    def reload_line_patrol_params(self):
        self.steering_gain = float(rospy.get_param('/tasks/line_following/steering_gain', 2.0))
        self.patrol_error_floor_px = float(rospy.get_param('/tasks/line_following/patrol_error_floor_px', 12))
        self.line_center_x_offset = self._load_line_center_x_offset()

    def parse_ratio(self, value):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            s = value.strip()
            if '/' in s:
                left, right = s.split('/', 1)
                return float(left.strip()) / float(right.strip())
            return float(s)
        raise ValueError(f'invalid roi ratio value: {value}')

    def load_line_roi(self):
        line_roi = rospy.get_param('/tasks/line_following/line_roi', None)
        if not (isinstance(line_roi, list) and len(line_roi) == 3):
            raise ValueError(
                'invalid /tasks/line_following/line_roi for hurdles+line mode; '
                'expected 3 segments [[y_min,y_max,x_min,x_max], x3]'
            )
        parsed = []
        for idx, roi in enumerate(line_roi):
            if not (isinstance(roi, list) and len(roi) == 4):
                raise ValueError(f'invalid line_roi[{idx}]')
            vals = [self.parse_ratio(v) for v in roi]
            y_min, y_max, x_min, x_max = vals
            if not (0.0 <= y_min < y_max <= 1.0 and 0.0 <= x_min < x_max <= 1.0):
                raise ValueError(f'invalid line_roi[{idx}] range: {vals}')
            parsed.append(vals)
        return parsed

    def apply_direct_gait_param_updates(self, cfg):
        base_updates = cfg.get('walking_param_updates', {})
        go_updates = cfg.get('go_walking_param_updates', {})
        turn_updates = cfg.get('turn_walking_param_updates', {})
        if not isinstance(base_updates, dict):
            base_updates = {}
        if not isinstance(go_updates, dict):
            go_updates = {}
        if not isinstance(turn_updates, dict):
            turn_updates = {}
        for k, v in base_updates.items():
            self.visual_patrol.go_gait_param[k] = v
            self.visual_patrol.turn_gait_param[k] = v
        for k, v in go_updates.items():
            self.visual_patrol.go_gait_param[k] = v
        for k, v in turn_updates.items():
            self.visual_patrol.turn_gait_param[k] = v

    def apply_line_following_gait_profile(self):
        cfg = rospy.get_param('/tasks/line_following/gait', {})
        profile = rospy.get_param('/tasks/line_following/gait_profile', 'default')
        profiles = rospy.get_param('/tasks/line_following/gait_profiles', {})
        if isinstance(profiles, dict) and profile in profiles and isinstance(profiles[profile], dict):
            cfg.update(profiles[profile])
        self.apply_direct_gait_param_updates(cfg)
        go_dsp = cfg.get('go_dsp', None)
        turn_dsp = cfg.get('turn_dsp', None)
        go_x_max = cfg.get('go_x_max', None)
        turn_x_max = cfg.get('turn_x_max', None)
        go_arm_swap = cfg.get('go_arm_swap', None)
        turn_arm_swap = cfg.get('turn_arm_swap', None)
        self.visual_patrol.update_go_gait(
            dsp=go_dsp if isinstance(go_dsp, list) and len(go_dsp) == 3 else None,
            x_max=go_x_max,
            arm_swap=go_arm_swap,
        )
        self.visual_patrol.update_turn_gait(
            dsp=turn_dsp if isinstance(turn_dsp, list) and len(turn_dsp) == 3 else None,
            x_max=turn_x_max,
            arm_swap=turn_arm_swap,
        )

    def setup_ros_interface(self):
        self.status_pub = rospy.Publisher('/hurdles/status', String, queue_size=1)
        self.active_pub = rospy.Publisher('/hurdles/active', Bool, queue_size=1)
        self.progress_pub = rospy.Publisher('/hurdles/progress', String, queue_size=1)
        self.start_service = rospy.Service('/hurdles/start', Trigger, self.start_service_callback)
        self.stop_service = rospy.Service('/hurdles/stop', Trigger, self.stop_service_callback)
        self.reset_service = rospy.Service('/hurdles/reset', Trigger, self.reset_service_callback)
        self.control_sub = rospy.Subscriber('/hurdles/control', String, self.control_callback)

    def is_gait_switch_on(self, default=True):
        v = rospy.get_param('/gait_switch', default)
        if isinstance(v, bool):
            return v
        return str(v).lower() in ('true', '1', 'yes', 'on')

    def publish_status(self, status):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

        active_msg = Bool()
        active_msg.data = self.active
        self.active_pub.publish(active_msg)

        progress_msg = String()
        progress_msg.data = (
            f'State: {self.current_state}, Completed: {self.task_completed}, '
            f'Y_counter: {self.enter_counter}, slow: {self.slow}'
        )
        self.progress_pub.publish(progress_msg)

    def set_servo(self, pan, tilt):
        try:
            self.motion_manager.set_servos_position(200, [[23, pan], [24, tilt]])
        except Exception as e:
            rospy.logerr(f'Servo control error: {e}')

    def build_line_detect_param(self):
        param = ColorDetect()
        param.color_name = rospy.get_param('/tasks/line_following/color', 'white')
        param.use_name = True
        param.detect_type = 'line'
        param.image_process_size = self.image_process_size
        w = int(self.image_width)
        h = int(self.image_height)
        param.line_roi.up.y_min = int(self.line_roi[0][0] * h)
        param.line_roi.up.y_max = int(self.line_roi[0][1] * h)
        param.line_roi.up.x_min = int(self.line_roi[0][2] * w)
        param.line_roi.up.x_max = int(self.line_roi[0][3] * w)
        param.line_roi.center.y_min = int(self.line_roi[1][0] * h)
        param.line_roi.center.y_max = int(self.line_roi[1][1] * h)
        param.line_roi.center.x_min = int(self.line_roi[1][2] * w)
        param.line_roi.center.x_max = int(self.line_roi[1][3] * w)
        param.line_roi.down.y_min = int(self.line_roi[2][0] * h)
        param.line_roi.down.y_max = int(self.line_roi[2][1] * h)
        param.line_roi.down.x_min = int(self.line_roi[2][2] * w)
        param.line_roi.down.x_max = int(self.line_roi[2][3] * w)
        param.min_area = int(rospy.get_param('/tasks/line_following/min_line_area', 1))
        param.max_area = int(rospy.get_param('/tasks/line_following/max_line_area', w * h))
        return param

    def build_hurdles_detect_param(self):
        param = ColorDetect()
        param.color_name = self.hurdles_color
        param.detect_type = 'side'
        param.use_name = True
        param.image_process_size = self.image_process_size
        param.roi.y_min = int(self.hurdles_roi[0] * self.image_height)
        param.roi.y_max = int(self.hurdles_roi[1] * self.image_height)
        param.roi.x_min = int(self.hurdles_roi[2] * self.image_width)
        param.roi.x_max = int(self.hurdles_roi[3] * self.image_width)
        param.min_area = self.hurdles_min_area
        param.max_area = self.image_width * self.image_height
        return param

    def configure_hurdles_detection(self):
        self.detect_pub.publish([self.build_line_detect_param(), self.build_hurdles_detect_param()])
        common.loginfo('Hurdles: published line + side(blue) ColorDetect (mission-style)')

    def get_color_callback(self, msg):
        self.objects_info = msg.data

    def get_line_data(self):
        for obj in self.objects_info:
            if obj.type == 'line':
                return obj
        return None

    def get_side_data(self):
        for obj in self.objects_info:
            if obj.type == 'side' and obj.label == self.hurdles_color:
                return obj
        return None

    def y_threshold_pixels(self, side_data):
        frame_h = float(getattr(side_data, 'height', self.image_height)) or float(self.image_height)
        thr = float(self.enter_y_threshold)
        if thr <= 1.0:
            return thr * frame_h
        return thr

    def side_y_past_execute_threshold(self, side_data):
        y_now = max(side_data.y, side_data.left_point[1], side_data.right_point[1])
        y_thr = self.y_threshold_pixels(side_data)
        frame_h = float(getattr(side_data, 'height', self.image_height)) or float(self.image_height)
        now = rospy.get_time()
        if now - self._last_y_log_time >= self.y_log_throttle_sec:
            rospy.loginfo('hurdles y=%.1f', y_now)
            self._last_y_log_time = now
        if not self._coord_frame_logged and frame_h > 0:
            self._coord_frame_logged = True
        return y_now > y_thr

    def apply_slow_gait_if_needed(self, side_data):
        """mission enter_hurdles：栅栏进入 ROI 下半区后收紧直行/转向步幅。"""
        h = float(getattr(side_data, 'height', self.image_height)) or float(self.image_height)
        roi_y0 = float(self.hurdles_roi[0]) * h
        if side_data.y > roi_y0 and self.slow:
            self.apply_slow_gait(reason='roi_seen')

    def apply_slow_gait(self, reason='approach'):
        if not self.slow or self.slow_gait_x_max <= 0.0:
            return
        self.visual_patrol.update_go_gait(x_max=self.slow_gait_x_max)
        self.visual_patrol.update_turn_gait(x_max=self.slow_gait_x_max)
        self.slow = False
        common.loginfo(
            'Hurdles: slow approach (x_max=%s reason=%s)' % (self.slow_gait_x_max, reason)
        )

    def execute_hurdles(self):
        if self.task_completed:
            return
        if not self.is_gait_switch_on():
            self.gait_manager.stop()
            if not self.gait_switch_off_reported:
                self.publish_status('gait_switch_off')
                rospy.logwarn('gait_switch_off已设置为false，步态发布关闭')
                self.gait_switch_off_reported = True
        else:
            self.gait_switch_off_reported = False
        try:
            self.publish_status('executing_hurdles')
            self.gait_manager.stop()
            time.sleep(0.2)
            self.gait_manager.disable()
            left_steps = max(0, self.move_left_steps_before_hurdles)
            for i in range(left_steps):
                rospy.loginfo(
                    '%d/%d %s',
                    i + 1,
                    left_steps,
                    self.fixed_step_log_phase_before,
                )
                self.motion_manager.run_action(self.move_left_action)
            right_steps = max(0, self.move_right_steps_before_hurdles)
            for i in range(right_steps):
                rospy.loginfo(
                    '%d/%d %s',
                    i + 1,
                    right_steps,
                    self.fixed_step_log_phase_before,
                )
                self.motion_manager.run_action(self.move_right_action)
            forward_steps = max(0, self.forward_steps_before_hurdles)
            for i in range(forward_steps):
                rospy.loginfo(
                    '%d/%d %s',
                    i + 1,
                    forward_steps,
                    self.fixed_step_log_phase_before,
                )
                self.motion_manager.run_action(self.forward_step_action)
            self.motion_manager.run_action(self.hand_back_action)
            rospy.sleep(0.5)
            self.motion_manager.run_action(self.hurdles_action_name)
            rospy.sleep(0.5)
            self.motion_manager.run_action(self.walk_ready_action)
            self.task_completed = True
            self.current_state = 'done'
            self.active = False
            self.slow = True
            self.publish_status('completed')
            common.loginfo('Hurdles action executed')
        except Exception as e:
            rospy.logerr(f'Hurdles action error: {e}')
            self.publish_status('error')

    def start_service_callback(self, req):
        self.load_hurdles_config()
        self.reload_line_patrol_params()
        self.active = True
        self.task_completed = False
        self.enter_counter = 0
        self.slow = True
        self._coord_frame_logged = False
        self._lost_side_counter = 0
        self._last_back_step_time = 0.0
        self.current_state = 'detecting'
        self.enter_func(None)
        self.configure_hurdles_detection()
        self.apply_line_following_gait_profile()
        self.apply_slow_gait(reason='task_started')
        self.start_srv_callback(None)
        self.set_servo(self.head_pan_init, self.head_tilt_look_down)
        if self.head_settle_sec > 0:
            rospy.sleep(self.head_settle_sec)
        self.publish_status('started')
        return TriggerResponse(success=True, message='Hurdles task started')

    def stop_service_callback(self, req):
        self.active = False
        self.gait_manager.stop()
        self.slow = True
        self.stop_srv_callback(None)
        self.publish_status('stopped')
        return TriggerResponse(success=True, message='Hurdles task stopped')

    def reset_service_callback(self, req):
        self.active = False
        self.gait_manager.stop()
        self.current_state = 'idle'
        self.enter_counter = 0
        self.slow = True
        self.task_completed = False
        self.stop_srv_callback(None)
        self.publish_status('reset')
        return TriggerResponse(success=True, message='Hurdles task reset')

    def control_callback(self, msg):
        command = msg.data.lower()
        if command == 'start':
            self.start_service_callback(None)
        elif command == 'stop':
            self.stop_service_callback(None)
        elif command == 'reset':
            self.reset_service_callback(None)

    def shutdown(self, signum, frame):
        with self.lock:
            self.motion_manager.run_action(self.stand_action)
            self.running = False
            self.active = False
            self.stop_srv_callback(None)
            self.publish_status('shutdown')
            common.loginfo('%s shutdown' % self.name)

    def run(self):
        self.publish_status('ready')
        while self.running and not rospy.is_shutdown():
            try:
                if not self.active:
                    rospy.sleep(0.05)
                    continue

                gait_on = self.is_gait_switch_on()
                if not gait_on:
                    self.gait_manager.stop()
                    if not self.gait_switch_off_reported:
                        self.publish_status('gait_switch_off')
                        rospy.logwarn('gait_switch_off已设置为false，步态发布关闭')
                        self.gait_switch_off_reported = True
                else:
                    self.gait_switch_off_reported = False

                line_data = self.get_line_data()
                side_data = self.get_side_data()

                if side_data is None:
                    self.enter_counter = 0
                    self._lost_side_counter += 1
                else:
                    self._lost_side_counter = 0

                if line_data is not None and gait_on:
                    corrected_x, width = self.patrol_corrected_x(line_data)
                    self.visual_patrol.process(corrected_x, width)

                if side_data is not None:
                    # 步态关闭时也继续打印蓝栏 y，便于手动标定阈值
                    y_past = self.side_y_past_execute_threshold(side_data)
                    if gait_on:
                        self.apply_slow_gait_if_needed(side_data)
                        if y_past:
                            self.enter_counter += 1
                        else:
                            self.enter_counter = 0

                        if self.enter_counter >= self.y_close_confirm_frames:
                            self.enter_counter = 0
                            self.current_state = 'ready'
                            self.execute_hurdles()
                            rospy.sleep(0.02)
                            continue
                    else:
                        self.enter_counter = 0
                else:
                    # 如果长时间检测不到蓝栏，执行一次后退动作，避免“贴着栏找不到”或误入空区
                    now = rospy.get_time()
                    if (
                        gait_on
                        and self.back_step_steps_when_lost > 0
                        and self._lost_side_counter >= self.lost_side_frames_before_back_step
                        and (now - self._last_back_step_time) >= self.back_step_cooldown_sec
                    ):
                        try:
                            self.publish_status('lost_fence_back_step')
                            self.gait_manager.stop()
                            time.sleep(0.2)
                            self.gait_manager.disable()
                            back_steps = max(0, self.back_step_steps_when_lost)
                            for i in range(back_steps):
                                rospy.loginfo(
                                    '%d/%d %s',
                                    i + 1,
                                    back_steps,
                                    self.fixed_step_log_phase_lost_back,
                                )
                                self.motion_manager.run_action(self.back_step_action)
                            rospy.sleep(0.2)
                            self.motion_manager.run_action(self.walk_ready_action)
                            self._last_back_step_time = now
                            self._lost_side_counter = 0
                        except Exception as e:
                            rospy.logwarn(f'back_step when lost failed: {e}')

                    if line_data is None:
                        self.publish_status('searching')
                    else:
                        self.publish_status('following_no_fence')

                if line_data is not None and side_data is not None:
                    self.publish_status('approaching')
                elif line_data is not None:
                    self.publish_status('following')

                rospy.sleep(0.02)
            except KeyboardInterrupt:
                break
            except Exception as e:
                rospy.logerr(f'Error in hurdles loop: {e}')
                rospy.sleep(0.1)

        self.gait_manager.stop()
        self.set_servo(self.head_pan_init, self.head_tilt_look_down)
        self.stop_srv_callback(None)
        self.publish_status('stopped')
        rospy.signal_shutdown('shutdown')


if __name__ == '__main__':
    try:
        HurdlesTracker('hurdles_tracker').run()
    except Exception as e:
        rospy.logerr(f'Hurdles tracker initialization error: {e}')
