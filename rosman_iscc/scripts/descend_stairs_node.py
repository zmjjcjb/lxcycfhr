#!/usr/bin/env python3
# encoding: utf-8
# @Author: Dejin Xuan
# @Date: 2026.4.2
# 下台阶节点
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


class DescendStairsTracker(Common):
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
        self.slow = True
        self.gait_switch_off_reported = False

        process_size = rospy.get_param('/image_processing/process_size', [160, 120])
        self.image_width = int(process_size[0]) if len(process_size) > 0 else 160
        self.image_height = int(process_size[1]) if len(process_size) > 1 else 120
        self.image_process_size = [self.image_width, self.image_height]

        self.load_config()
        self.line_roi = self.load_line_roi()
        self.steering_gain = float(rospy.get_param('/tasks/line_following/steering_gain', 2.0))
        self.patrol_error_floor_px = float(rospy.get_param('/tasks/line_following/patrol_error_floor_px', 12))
        self.line_center_x_offset = self._load_line_center_x_offset()
        self.head_settle_sec = float(rospy.get_param('/tasks/line_following/head_settle_sec', 0.12))
        self.apply_line_following_gait_profile()

        self.setup_ros_interface()
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)

        self.motion_manager.run_action(self.walk_ready_action)
        common.loginfo('Descend stairs tracker init (enter/低头在 /descend_stairs/start 时执行)')

    def load_config(self):
        cfg = rospy.get_param('/tasks/descend_stairs', {})
        self.stairs_color = cfg.get('color', 'green')
        self.stairs_roi = cfg.get('roi', [0.2, 1.0, 0.0, 1.0])
        self.enter_y_threshold = float(cfg.get('enter_y_threshold', 310))
        self.y_close_confirm_frames = int(cfg.get('y_close_confirm_frames', 2))
        self.use_max_y_for_execute = bool(cfg.get('use_max_y', True))
        self.slow_gait_x_max = float(cfg.get('slow_gait_x_max', 0.01))
        self.stairs_min_area = int(cfg.get('min_area', 10))
        self.y_log_throttle_sec = float(cfg.get('y_log_throttle_sec', 0.2))
        motion = cfg.get('motion', {})
        self.forward_step_action = motion.get('forward_step_action', 'forward_one_step')
        self.move_left_action = motion.get('move_left_action', 'move_left')
        self.move_right_action = motion.get('move_right_action', 'move_right')
        self.move_left_steps_before_descend = int(motion.get('move_left_steps_before_descend', 0))
        self.move_right_steps_before_descend = int(motion.get('move_right_steps_before_descend', 0))
        self.descend_forward_steps = int(motion.get('descend_forward_steps', 3))
        self.descend_action_name = str(motion.get('descend_action', 'descend_stairs'))
        self.walk_ready_action = str(motion.get('walk_ready_action', 'walk_ready'))
        self.stand_action = str(motion.get('stand_action', 'stand'))
        self.fixed_step_log_phase_before = str(motion.get('fixed_step_log_phase_before', 'descend_before'))
        self._last_y_log_time = 0.0

    def parse_ratio(self, value):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            s = value.strip()
            if '/' in s:
                left, right = s.split('/', 1)
                return float(left.strip()) / float(right.strip())
            return float(s)
        raise ValueError('invalid roi ratio value: %s' % value)

    def load_line_roi(self):
        line_roi = rospy.get_param('/tasks/line_following/line_roi', None)
        if not (isinstance(line_roi, list) and len(line_roi) == 3):
            raise ValueError('invalid /tasks/line_following/line_roi for descend+line mode')
        parsed = []
        for idx, roi in enumerate(line_roi):
            if not (isinstance(roi, list) and len(roi) == 4):
                raise ValueError('invalid line_roi[%d]' % idx)
            vals = [self.parse_ratio(v) for v in roi]
            y_min, y_max, x_min, x_max = vals
            if not (0.0 <= y_min < y_max <= 1.0 and 0.0 <= x_min < x_max <= 1.0):
                raise ValueError('invalid line_roi[%d] range: %s' % (idx, vals))
            parsed.append(vals)
        return parsed

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
        if self.patrol_error_floor_px > 0.0 and scaled != 0.0 and abs(scaled) < self.patrol_error_floor_px:
            scaled = math.copysign(self.patrol_error_floor_px, scaled)
        cx = patrol_center + scaled
        return max(0.0, min(width - 1.0, cx)), width

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
        self.visual_patrol.update_go_gait(
            dsp=cfg.get('go_dsp') if isinstance(cfg.get('go_dsp'), list) and len(cfg.get('go_dsp')) == 3 else None,
            x_max=cfg.get('go_x_max'),
            arm_swap=cfg.get('go_arm_swap'),
        )
        self.visual_patrol.update_turn_gait(
            dsp=cfg.get('turn_dsp') if isinstance(cfg.get('turn_dsp'), list) and len(cfg.get('turn_dsp')) == 3 else None,
            x_max=cfg.get('turn_x_max'),
            arm_swap=cfg.get('turn_arm_swap'),
        )

    def setup_ros_interface(self):
        self.status_pub = rospy.Publisher('/descend_stairs/status', String, queue_size=1)
        self.active_pub = rospy.Publisher('/descend_stairs/active', Bool, queue_size=1)
        self.progress_pub = rospy.Publisher('/descend_stairs/progress', String, queue_size=1)
        self.start_srv = rospy.Service('/descend_stairs/start', Trigger, self.start_callback)
        self.stop_srv = rospy.Service('/descend_stairs/stop', Trigger, self.stop_callback)
        self.reset_srv = rospy.Service('/descend_stairs/reset', Trigger, self.reset_callback)

    def is_gait_switch_on(self, default=True):
        v = rospy.get_param('/gait_switch', default)
        if isinstance(v, bool):
            return v
        return str(v).lower() in ('true', '1', 'yes', 'on')

    def publish_status(self, status):
        msg = String(); msg.data = status; self.status_pub.publish(msg)
        am = Bool(); am.data = self.active; self.active_pub.publish(am)
        pm = String(); pm.data = f'State:{self.current_state} counter:{self.enter_counter}'; self.progress_pub.publish(pm)

    def set_servo(self, pan, tilt):
        try:
            self.motion_manager.set_servos_position(200, [[23, pan], [24, tilt]])
        except Exception as e:
            rospy.logerr('Servo control error: %s' % e)

    def build_line_detect_param(self):
        param = ColorDetect()
        param.color_name = rospy.get_param('/tasks/line_following/color', 'white')
        param.use_name = True
        param.detect_type = 'line'
        param.image_process_size = self.image_process_size
        w, h = int(self.image_width), int(self.image_height)
        param.line_roi.up.y_min = int(self.line_roi[0][0] * h); param.line_roi.up.y_max = int(self.line_roi[0][1] * h)
        param.line_roi.up.x_min = int(self.line_roi[0][2] * w); param.line_roi.up.x_max = int(self.line_roi[0][3] * w)
        param.line_roi.center.y_min = int(self.line_roi[1][0] * h); param.line_roi.center.y_max = int(self.line_roi[1][1] * h)
        param.line_roi.center.x_min = int(self.line_roi[1][2] * w); param.line_roi.center.x_max = int(self.line_roi[1][3] * w)
        param.line_roi.down.y_min = int(self.line_roi[2][0] * h); param.line_roi.down.y_max = int(self.line_roi[2][1] * h)
        param.line_roi.down.x_min = int(self.line_roi[2][2] * w); param.line_roi.down.x_max = int(self.line_roi[2][3] * w)
        param.min_area = int(rospy.get_param('/tasks/line_following/min_line_area', 1))
        param.max_area = int(rospy.get_param('/tasks/line_following/max_line_area', w * h))
        return param

    def build_side_detect_param(self):
        param = ColorDetect()
        param.color_name = self.stairs_color
        param.detect_type = 'side'
        param.use_name = True
        param.image_process_size = self.image_process_size
        param.roi.y_min = int(self.stairs_roi[0] * self.image_height)
        param.roi.y_max = int(self.stairs_roi[1] * self.image_height)
        param.roi.x_min = int(self.stairs_roi[2] * self.image_width)
        param.roi.x_max = int(self.stairs_roi[3] * self.image_width)
        param.min_area = self.stairs_min_area
        param.max_area = self.image_width * self.image_height
        return param

    def configure_detection(self):
        self.detect_pub.publish([self.build_line_detect_param(), self.build_side_detect_param()])
        common.loginfo(f'Descend stairs: published line + side({self.stairs_color})')

    def get_color_callback(self, msg):
        self.objects_info = msg.data

    def get_line_data(self):
        for obj in self.objects_info:
            if obj.type == 'line':
                return obj
        return None

    def get_side_data(self):
        for obj in self.objects_info:
            if obj.type == 'side' and obj.label == self.stairs_color:
                return obj
        return None

    def y_threshold_pixels(self, side_data):
        frame_h = float(getattr(side_data, 'height', self.image_height)) or float(self.image_height)
        return float(self.enter_y_threshold) * frame_h if float(self.enter_y_threshold) <= 1.0 else float(self.enter_y_threshold)

    def execute_y_now(self, side_data):
        if self.use_max_y_for_execute:
            return max(side_data.y, side_data.left_point[1], side_data.right_point[1])
        return float(side_data.y)

    def y_past_execute_threshold(self, side_data):
        y_now = self.execute_y_now(side_data)
        y_thr = self.y_threshold_pixels(side_data)
        now = rospy.get_time()
        if now - self._last_y_log_time >= self.y_log_throttle_sec:
            rospy.loginfo('descend y=%.1f', y_now)
            self._last_y_log_time = now
        return y_now > y_thr

    def apply_slow_gait_if_needed(self, side_data):
        h = float(getattr(side_data, 'height', self.image_height)) or float(self.image_height)
        if side_data.y > float(self.stairs_roi[0]) * h and self.slow:
            self.apply_slow_gait(reason='roi_seen')

    def apply_slow_gait(self, reason='approach'):
        if not self.slow or self.slow_gait_x_max <= 0.0:
            return
        self.visual_patrol.update_go_gait(x_max=self.slow_gait_x_max)
        self.visual_patrol.update_turn_gait(x_max=self.slow_gait_x_max)
        self.slow = False
        common.loginfo(
            'Descend stairs: slow approach (x_max=%s reason=%s)' % (self.slow_gait_x_max, reason)
        )

    def execute_descend(self):
        try:
            self.publish_status('executing_descend')
            self.gait_manager.stop()
            time.sleep(0.2)
            self.gait_manager.disable()
            left_before = max(0, self.move_left_steps_before_descend)
            for i in range(left_before):
                rospy.loginfo(
                    '%d/%d %s',
                    i + 1,
                    left_before,
                    self.fixed_step_log_phase_before,
                )
                self.motion_manager.run_action(self.move_left_action)
            right_before = max(0, self.move_right_steps_before_descend)
            for i in range(right_before):
                rospy.loginfo(
                    '%d/%d %s',
                    i + 1,
                    right_before,
                    self.fixed_step_log_phase_before,
                )
                self.motion_manager.run_action(self.move_right_action)
            forward_before = max(0, self.descend_forward_steps)
            for i in range(forward_before):
                rospy.loginfo(
                    '%d/%d %s',
                    i + 1,
                    forward_before,
                    self.fixed_step_log_phase_before,
                )
                self.motion_manager.run_action(self.forward_step_action)
            self.motion_manager.run_action(self.descend_action_name)
            rospy.sleep(0.5)
            self.motion_manager.run_action(self.walk_ready_action)
            rospy.sleep(0.5)
            self.task_completed = True
            self.current_state = 'done'
            self.active = False
            self.slow = True
            self.publish_status('completed_descend')
            common.loginfo('Descend stairs executed (split node)')
        except Exception as e:
            rospy.logerr('Descend stairs error: %s' % e)
            self.publish_status('error')

    def start_callback(self, req):
        self.load_config()
        self.active = True
        self.task_completed = False
        self.enter_counter = 0
        self.slow = True
        self.current_state = 'detecting'
        self.enter_func(None)
        self.configure_detection()
        self.apply_line_following_gait_profile()
        self.apply_slow_gait(reason='task_started')
        self.start_srv_callback(None)
        self.set_servo(self.head_pan_init, self.head_tilt_look_down)
        if self.head_settle_sec > 0:
            rospy.sleep(self.head_settle_sec)
        self.publish_status('started_descend')
        return TriggerResponse(success=True, message='descend stairs started')

    def stop_callback(self, req):
        self.active = False
        self.gait_manager.stop()
        self.stop_srv_callback(None)
        self.publish_status('stopped')
        return TriggerResponse(success=True, message='descend stairs stopped')

    def reset_callback(self, req):
        self.active = False
        self.gait_manager.stop()
        self.current_state = 'idle'
        self.enter_counter = 0
        self.slow = True
        self.task_completed = False
        self.stop_srv_callback(None)
        self.publish_status('reset')
        return TriggerResponse(success=True, message='descend stairs reset')

    def shutdown(self, signum, frame):
        with self.lock:
            self.motion_manager.run_action(self.stand_action)
            self.running = False
            self.active = False
            self.stop_srv_callback(None)
            self.publish_status('shutdown')

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
                if line_data is not None and gait_on:
                    corrected_x, width = self.patrol_corrected_x(line_data)
                    self.visual_patrol.process(corrected_x, width)
                if side_data is not None:
                    y_past = self.y_past_execute_threshold(side_data)
                    if gait_on:
                        self.apply_slow_gait_if_needed(side_data)
                        if y_past:
                            self.enter_counter += 1
                        else:
                            self.enter_counter = 0
                        if self.enter_counter >= self.y_close_confirm_frames:
                            self.enter_counter = 0
                            self.current_state = 'ready'
                            self.execute_descend()
                            rospy.sleep(0.02)
                            continue
                        self.publish_status('approaching')
                    else:
                        self.enter_counter = 0
                elif line_data is not None and gait_on:
                    self.publish_status('following')
                elif gait_on:
                    self.publish_status('searching')
                rospy.sleep(0.02)
            except Exception as e:
                rospy.logerr('Error in descend loop: %s' % e)
                rospy.sleep(0.1)
        self.gait_manager.stop()
        self.set_servo(self.head_pan_init, self.head_tilt_look_down)
        self.stop_srv_callback(None)
        self.publish_status('stopped')
        rospy.signal_shutdown('shutdown')


if __name__ == '__main__':
    try:
        DescendStairsTracker('descend_stairs_tracker').run()
    except Exception as e:
        rospy.logerr('Descend stairs tracker init error: %s' % e)
