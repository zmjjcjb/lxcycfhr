#!/usr/bin/env python3
# encoding: utf-8
# @Author: Dejin Xuan
# @Date: 2026.4.6
# 绿色目标对齐与抓取节点
import signal
import time

import rospy
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger, TriggerResponse

from rosman_example.color_common import Common
from rosman_interfaces.msg import ColorDetect, ObjectsInfo
from rosman_sdk import common


class GreenPickNode(Common):
    def __init__(self, name):
        self.head_pan_init = rospy.get_param('/servo/head_pan_init', 500)
        self.head_tilt_look_down = rospy.get_param('/servo/head_tilt_look_down', 280)

        rospy.init_node(name)
        self.name = name
        self.running = True
        self.active = False
        self.phase = 'idle'
        self.objects_info = []

        super().__init__(name, self.head_pan_init, self.head_tilt_look_down)
        signal.signal(signal.SIGINT, self.shutdown)

        self._start_time = 0.0
        self._last_log_time = 0.0
        self._fine_steps = 0
        self._lost_frames = 0
        self.gait_switch_off_reported = False

        self.load_config()
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.object_callback)

        self.status_pub = rospy.Publisher('/green_pick/status', String, queue_size=1)
        self.active_pub = rospy.Publisher('/green_pick/active', Bool, queue_size=1)
        self.start_service = rospy.Service('/green_pick/start', Trigger, self.start_service_callback)
        self.stop_service = rospy.Service('/green_pick/stop', Trigger, self.stop_service_callback)
        self.reset_service = rospy.Service('/green_pick/reset', Trigger, self.reset_service_callback)

        common.loginfo('green_pick initialized')

    def load_config(self):
        cfg = rospy.get_param('/tasks/green_pick', {})
        align = cfg.get('alignment', {})
        motion = cfg.get('motion', {})
        fine = cfg.get('fine_actions', {})
        pick = cfg.get('pick', {})

        self.enabled = bool(cfg.get('enabled', True))
        self.placeholder_message = str(cfg.get('placeholder_message', '进入green_pick阶段'))
        self.color = str(cfg.get('color', 'green_pick'))
        self.detect_type = str(cfg.get('detect_type', 'side'))
        self.roi = cfg.get('roi', [0.2, 1.0, 0.15, 0.85])
        self.min_area = int(cfg.get('min_area', 10))

        self.process_size = rospy.get_param('/image_processing/process_size', [160, 120])
        self.image_width = int(self.process_size[0]) if len(self.process_size) > 0 else 160
        self.image_height = int(self.process_size[1]) if len(self.process_size) > 1 else 120
        self.image_process_size = [self.image_width, self.image_height]

        self.target_x_pixel = float(align.get('target_x_pixel', 0.5))
        self.target_y_pixel = float(align.get('target_y_pixel', 300))
        self.target_x_offset = float(align.get('target_x_offset', 0.0))
        self.target_y_offset = float(align.get('target_y_offset', 0.0))
        self.x_tolerance = float(align.get('x_tolerance', 20))
        self.y_tolerance = float(align.get('y_tolerance', 20))
        self.x_band_px = float(align.get('x_band_px', 40))
        self.y_band_px = float(align.get('y_band_px', 40))
        self.fine_tune_max_steps = int(align.get('fine_tune_max_steps', 12))

        self.coarse_step_distance = float(motion.get('coarse_step_distance', 0.006))
        self.coarse_turn_yaw = int(motion.get('coarse_turn_yaw', 8))
        self.coarse_fine_yaw = int(motion.get('coarse_fine_yaw', 4))
        self.coarse_dsp = motion.get('coarse_dsp', [320, 0.25, 0.023])
        self.coarse_arm_swap = int(motion.get('coarse_arm_swap', 25))
        self.coarse_walking_param = self.gait_manager.get_gait_param()
        wp_updates = motion.get('coarse_walking_param_updates', {})
        if isinstance(wp_updates, dict):
            self.coarse_walking_param.update(wp_updates)

        self.action_forward = str(fine.get('forward', 'forward_one_step'))
        self.action_back = str(fine.get('back', 'back_step'))
        self.action_move_left = str(fine.get('move_left', 'move_left'))
        self.action_move_right = str(fine.get('move_right', 'move_right'))

        self.pick_action_name = str(pick.get('action_name', 'pick_green_iscc'))
        self.pick_wait_sec = float(pick.get('wait_sec_after_pick', 1.0))
        self.walk_ready_after_pick = bool(pick.get('walk_ready_after_pick', True))

        self.align_timeout_sec = float(cfg.get('align_timeout_sec', 30.0))
        self.lost_frames_to_search = int(cfg.get('lost_frames_to_search', 10))
        self.log_throttle_sec = float(cfg.get('align_log_throttle_sec', 0.2))

    def object_callback(self, msg):
        self.objects_info = msg.data

    def clear_object_tracking(self):
        self.objects_info = []

    def parse_ratio(self, value):
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip()
        if '/' in s:
            a, b = s.split('/', 1)
            return float(a.strip()) / float(b.strip())
        return float(s)

    def is_gait_switch_on(self, default=True):
        v = rospy.get_param('/gait_switch', default)
        if isinstance(v, bool):
            return v
        return str(v).lower() in ('true', '1', 'yes', 'on')

    def build_green_param(self):
        p = ColorDetect()
        p.color_name = self.color
        p.use_name = True
        p.detect_type = self.detect_type
        p.image_process_size = self.image_process_size
        p.roi.y_min = int(self.parse_ratio(self.roi[0]) * self.image_height)
        p.roi.y_max = int(self.parse_ratio(self.roi[1]) * self.image_height)
        p.roi.x_min = int(self.parse_ratio(self.roi[2]) * self.image_width)
        p.roi.x_max = int(self.parse_ratio(self.roi[3]) * self.image_width)
        p.min_area = self.min_area
        p.max_area = self.image_width * self.image_height
        return p

    def get_best_green(self):
        best = None
        best_y = -1
        for obj in self.objects_info:
            if getattr(obj, 'label', '') == self.color and getattr(obj, 'type', '') == self.detect_type:
                yv = float(getattr(obj, 'y', 0))
                if yv > best_y:
                    best_y = yv
                    best = obj
        return best

    def target_x_px(self, obj):
        frame_w = int(getattr(obj, 'width', self.image_width)) or self.image_width
        tx = self.target_x_pixel * frame_w if self.target_x_pixel <= 1.0 else self.target_x_pixel
        return float(tx + self.target_x_offset)

    def target_y_px(self, obj):
        frame_h = int(getattr(obj, 'height', self.image_height)) or self.image_height
        ty = self.target_y_pixel * frame_h if self.target_y_pixel <= 1.0 else self.target_y_pixel
        return float(ty + self.target_y_offset)

    def in_tolerance(self, ex, ey):
        return abs(ex) <= self.x_tolerance and abs(ey) <= self.y_tolerance

    def inside_bands(self, ex, ey):
        return abs(ex) <= self.x_band_px and abs(ey) <= self.y_band_px

    def apply_coarse_step(self, ex, ey):
        x_step = 0.0
        yaw = 0
        if abs(ey) > self.y_band_px:
            x_step = self.coarse_step_distance * 0.7 if ey < 0 else -self.coarse_step_distance * 0.5
        if abs(ex) > self.x_band_px:
            yaw = -self.coarse_turn_yaw if ex > 0 else self.coarse_turn_yaw
            if abs(ex) <= self.x_band_px * 1.4:
                yaw = -self.coarse_fine_yaw if ex > 0 else self.coarse_fine_yaw
        if x_step == 0.0 and yaw == 0:
            self.gait_manager.stop()
            return False
        self.gait_manager.set_step(
            self.coarse_dsp,
            float(x_step),
            0.0,
            int(yaw),
            self.coarse_walking_param,
            arm_swap=self.coarse_arm_swap,
            step_num=1,
        )
        return True

    def run_fine_action_once(self, ex, ey):
        if not self.is_gait_switch_on():
            return False
        choose_x = abs(ex) >= abs(ey)
        action_name = None
        axis = None
        if choose_x and abs(ex) > self.x_tolerance:
            action_name = self.action_move_right if ex > 0 else self.action_move_left
            axis = 'x'
        elif abs(ey) > self.y_tolerance:
            action_name = self.action_back if ey > 0 else self.action_forward
            axis = 'y'
        elif abs(ex) > self.x_tolerance:
            action_name = self.action_move_right if ex > 0 else self.action_move_left
            axis = 'x'
        if not action_name:
            return False
        rospy.loginfo('green_pick fine_tune: axis=%s action=%s ex=%.1f ey=%.1f', axis, action_name, ex, ey)
        self.motion_manager.run_action(action_name)
        self._fine_steps += 1
        return True

    def execute_pick(self):
        self.phase = 'picking'
        self.publish_status('picking')
        self.gait_manager.stop()
        time.sleep(0.2)
        self.gait_manager.disable()
        self.motion_manager.run_action(self.pick_action_name)
        if self.pick_wait_sec > 0:
            time.sleep(self.pick_wait_sec)
        if self.walk_ready_after_pick:
            self.motion_manager.run_action('walk_ready')
        self.phase = 'done'
        self.active = False
        try:
            self.stop_srv_callback(None)
        except Exception:
            pass
        self.publish_status('completed_pick')
        return True

    def publish_status(self, status):
        sm = String()
        sm.data = status
        self.status_pub.publish(sm)
        am = Bool()
        am.data = self.active
        self.active_pub.publish(am)

    def start_service_callback(self, req):
        self.load_config()
        if not self.enabled:
            self.publish_status('disabled')
            return TriggerResponse(success=False, message='green_pick disabled by config')
        self.active = True
        self.phase = 'aligning'
        self._start_time = rospy.get_time()
        self._last_log_time = 0.0
        self._fine_steps = 0
        self._lost_frames = 0
        self.clear_object_tracking()
        rospy.loginfo(self.placeholder_message)
        try:
            self.enter_func(None)
            detect_params = [self.build_green_param()]
            self.detect_pub.publish(detect_params)
            self.start_srv_callback(None)
        except Exception as e:
            self.active = False
            self.publish_status('waiting_color_detection')
            rospy.logwarn('green_pick color_detection unavailable: %s', e)
            return TriggerResponse(success=False, message='color_detection unavailable')
        self.publish_status('started')
        return TriggerResponse(success=True, message='green_pick started')

    def stop_service_callback(self, req):
        self.active = False
        self.phase = 'idle'
        self.clear_object_tracking()
        self.gait_manager.stop()
        try:
            self.stop_srv_callback(None)
        except Exception:
            pass
        self.publish_status('stopped')
        return TriggerResponse(success=True, message='green_pick stopped')

    def reset_service_callback(self, req):
        self.active = False
        self.phase = 'idle'
        self._fine_steps = 0
        self._lost_frames = 0
        self.clear_object_tracking()
        self.gait_manager.stop()
        try:
            self.stop_srv_callback(None)
        except Exception:
            pass
        self.publish_status('reset')
        return TriggerResponse(success=True, message='green_pick reset')

    def shutdown(self, signum, frame):
        self.running = False
        self.active = False
        self.phase = 'idle'
        self.clear_object_tracking()
        self.gait_manager.stop()
        try:
            self.stop_srv_callback(None)
        except Exception:
            pass
        self.publish_status('shutdown')
        common.loginfo('%s shutdown' % self.name)

    def run(self):
        self.publish_status('ready')
        rate_sleep = 0.02
        while self.running and not rospy.is_shutdown():
            if not self.active:
                time.sleep(rate_sleep)
                continue
            now = rospy.get_time()
            gait_on = self.is_gait_switch_on()
            if not gait_on:
                if not self.gait_switch_off_reported:
                    self.publish_status('gait_switch_off')
                    rospy.logwarn('gait_switch_off已设置为false，闭环步态发布关闭（动作仍按条件执行，仅打印坐标）')
                    self.gait_switch_off_reported = True
                self.gait_manager.stop()
            else:
                self.gait_switch_off_reported = False

            if self.align_timeout_sec > 0 and (now - self._start_time) > self.align_timeout_sec:
                if self.phase in ('aligning', 'coarse_align', 'fine_align'):
                    self.active = False
                    self.gait_manager.stop()
                    self.publish_status('align_timeout')
                    rospy.logwarn('green_pick align timeout')
                    continue
            if self.phase in ('aligning', 'coarse_align', 'fine_align'):
                obj = self.get_best_green()
                if obj is None:
                    self._lost_frames += 1
                    if self._lost_frames >= self.lost_frames_to_search:
                        self.gait_manager.stop()
                        self.publish_status('searching')
                    time.sleep(rate_sleep)
                    continue
                self._lost_frames = 0
                cx = float(getattr(obj, 'x', 0))
                cy = float(getattr(obj, 'y', 0))
                tx = self.target_x_px(obj)
                ty = self.target_y_px(obj)
                ex = cx - tx
                ey = cy - ty
                if now - self._last_log_time >= self.log_throttle_sec:
                    rospy.loginfo('green x=%.1f y=%.1f', cx, cy)
                    self._last_log_time = now

                if self.in_tolerance(ex, ey):
                    self.execute_pick()
                    continue

                if not gait_on:
                    time.sleep(rate_sleep)
                    continue

                if self.inside_bands(ex, ey):
                    self.phase = 'fine_align'
                    self.gait_manager.stop()
                    if self._fine_steps >= self.fine_tune_max_steps:
                        self.active = False
                        self.publish_status('fine_tune_exceeded')
                        rospy.logwarn('green_pick fine tune exceeded max steps')
                        continue
                    self.run_fine_action_once(ex, ey)
                else:
                    self.phase = 'coarse_align'
                    self.apply_coarse_step(ex, ey)
            time.sleep(rate_sleep)


if __name__ == '__main__':
    try:
        GreenPickNode('green_pick').run()
    except Exception as e:
        rospy.logerr('green_pick init error: %s', e)
