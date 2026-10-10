#!/usr/bin/env python3
# encoding: utf-8
# @Author: Dejin Xuan
# @Date: 2026.4.3
# 橙线区域B 阶段：左右橙线阈值循迹（可独立启停，供 A 结束后衔接）
import os
import signal
import sys
import time

import rospy
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger, TriggerResponse

from rosman_example.color_common import Common
from rosman_interfaces.msg import ColorDetect, ObjectsInfo
from rosman_sdk import common
try:
    from rosman_iscc.shared_rqt_result_view import (
        acquire_shared_rqt_result_view,
        release_shared_rqt_result_view,
    )
except ModuleNotFoundError:
    _PKG_SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
    if _PKG_SRC_DIR not in sys.path:
        sys.path.insert(0, _PKG_SRC_DIR)
    from rosman_iscc.shared_rqt_result_view import (
        acquire_shared_rqt_result_view,
        release_shared_rqt_result_view,
    )


class OrangeLineFollow(Common):
    def __init__(self, name):
        self.head_pan_init = rospy.get_param('/servo/head_pan_init', 500)
        self.head_tilt_look_down = rospy.get_param('/servo/head_tilt_look_down', 280)

        rospy.init_node(name)
        self.name = name
        self.running = True
        self.active = False
        self.objects_info = []
        self._both_lost_since = None
        self._holding_lost = False
        self.gait_switch_off_reported = False
        self._last_status = ''
        self._last_status_time = 0.0
        self._last_control_log_time = 0.0
        self._last_purple_y_log_time = 0.0
        self._last_green_y_log_time = 0.0
        self.rqt_owner_id = 'orange_line_follow:%s:%d' % (self.name, os.getpid())

        super().__init__(name, self.head_pan_init, self.head_tilt_look_down)
        signal.signal(signal.SIGINT, self.shutdown)

        self.process_size = rospy.get_param('/image_processing/process_size', [160, 120])
        self.image_width = int(self.process_size[0]) if len(self.process_size) > 0 else 160
        self.image_height = int(self.process_size[1]) if len(self.process_size) > 1 else 120
        self.image_process_size = [self.image_width, self.image_height]
        self.head_settle_sec = float(rospy.get_param('/tasks/line_following/head_settle_sec', 0.12))
        self.result_image_topic = rospy.get_param(
            '/tasks/line_following/result_image_topic',
            '/color_detection/image_result',
        )
        self.auto_open_rqt_result = rospy.get_param('/tasks/line_following/auto_open_rqt_result', True)

        self.load_config()
        self.setup_ros_interface()
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)

        self.motion_manager.run_action('walk_ready')
        common.loginfo('Orange line follow initialized (enter 在 /orange_line_follow/start 时执行)')

    def load_config(self):
        cfg = rospy.get_param('/tasks/orange_line_following', {})
        self.orange_color = cfg.get('color', 'orange')
        # B 阶段 ROI 仅读 orange_line_following，与 A 解耦（缺省为较宽的走廊搜索区）
        self.left_roi = cfg.get('left_roi', [0.25, 1.0, 0.0, 0.48])
        self.right_roi = cfg.get('right_roi', [0.25, 1.0, 0.52, 1.0])
        self.min_area = int(cfg.get('min_area', 10))

        # gait_manager 的 rotation_angle 合法范围是 [-10, 10]，这里做硬保护。
        self.rotation_yaw_limit = 10.0

        # 左右线 x 阈值用全图 ObjectInfo.x；<=1 按图像宽度比例解释，否则按像素解释。
        self.left_adjust_x_threshold = float(cfg.get('left_adjust_x_threshold', 0.2))
        self.right_adjust_x_threshold = float(cfg.get('right_adjust_x_threshold', 0.8))
        self.go_linear_x = float(cfg.get('go_linear_x', 0.015))
        self.side_adjust_linear_x = float(cfg.get('side_adjust_linear_x', 0.005))
        self.side_adjust_linear_y = abs(float(cfg.get('side_adjust_linear_y', 0.004)))
        self.horizontal_guard_angle_threshold_deg = abs(
            float(cfg.get('horizontal_guard_angle_threshold_deg', 30.0))
        )
        self.horizontal_guard_yaw = abs(self.clamp_yaw(cfg.get('horizontal_guard_yaw', 10)))
        self.horizontal_guard_linear_x = 0.0
        self.angle_adjust_yaw_threshold_deg = abs(float(cfg.get('angle_adjust_yaw_threshold_deg', 45.0)))
        self.angle_adjust_linear_x = float(cfg.get('angle_adjust_linear_x', 0.002))
        self.angle_adjust_yaw = abs(self.clamp_yaw(cfg.get('angle_adjust_yaw', 8)))

        self.lost_both_grace_sec = float(cfg.get('lost_both_grace_sec', 0.6))
        self.lost_both_stop_sec = float(cfg.get('lost_both_stop_sec', 3.0))

        self.go_dsp = cfg.get('go_dsp', [320, 0.25, 0.022])
        self.turn_dsp = cfg.get('turn_dsp', [320, 0.25, 0.022])
        if not isinstance(self.go_dsp, (list, tuple)) or len(self.go_dsp) < 3:
            self.go_dsp = [320, 0.25, 0.022]
        if not isinstance(self.turn_dsp, (list, tuple)) or len(self.turn_dsp) < 3:
            self.turn_dsp = [320, 0.25, 0.022]
        self.side_adjust_dsp = cfg.get('side_adjust_dsp', [320, 0.25, 0.022])
        if not isinstance(self.side_adjust_dsp, (list, tuple)) or len(self.side_adjust_dsp) < 3:
            self.side_adjust_dsp = [320, 0.25, 0.022]
        self.go_arm_swap = int(cfg.get('go_arm_swap', 30))
        self.turn_arm_swap = int(cfg.get('turn_arm_swap', 30))
        self.side_adjust_arm_swap = int(cfg.get('side_adjust_arm_swap', 30))
        self.go_walking_param = self.gait_manager.get_gait_param()
        self.turn_walking_param = self.gait_manager.get_gait_param()
        self.side_adjust_walking_param = self.gait_manager.get_gait_param()
        go_updates = cfg.get('go_walking_param_updates', {})
        turn_updates = cfg.get('turn_walking_param_updates', {})
        side_updates = cfg.get('side_adjust_walking_param_updates', {})
        if isinstance(go_updates, dict):
            self.go_walking_param.update(go_updates)
        if isinstance(turn_updates, dict):
            self.turn_walking_param.update(turn_updates)
        if isinstance(side_updates, dict):
            self.side_adjust_walking_param.update(side_updates)

        self.status_throttle_sec = float(cfg.get('status_throttle_sec', 0.25))
        self.log_xy_throttle_sec = float(cfg.get('log_xy_throttle_sec', 0.2))
        sb = rospy.get_param('/tasks/sandbag_throw', {})
        self.purple_y_log_throttle_sec = float(sb.get('y_log_throttle_sec', 0.2))
        self.sandbag_purple_color = str(sb.get('color', 'purple'))
        self.sandbag_detect_type = str(sb.get('detect_type', 'side'))
        gg = rospy.get_param('/tasks/green_pick_gate', {})
        self.green_gate_enabled = bool(gg.get('enabled', True))
        self.green_color = str(gg.get('color', 'green_pick'))
        self.green_detect_type = str(gg.get('detect_type', 'side'))
        self.green_roi = gg.get('roi', [0.2, 1.0, 0.15, 0.85])
        self.green_min_area = int(gg.get('min_area', 10))
        self.green_y_log_throttle_sec = float(gg.get('y_log_throttle_sec', 0.2))

    def log_purple_y_for_tune(self):
        """识别到紫色时打印中心 x/y；在 gait_switch 分支之前调用，gait_switch=false 时也会打印。"""
        sb = rospy.get_param('/tasks/sandbag_throw', {})
        if not bool(sb.get('enabled', False)):
            return
        if not bool(rospy.get_param('/iscc/sandbag_purple_watch_active', True)):
            return
        color = str(sb.get('color', self.sandbag_purple_color))
        dtype = str(sb.get('detect_type', self.sandbag_detect_type))
        purple = None
        best_y = -1
        for obj in self.objects_info:
            if getattr(obj, 'type', '') == dtype and getattr(obj, 'label', '') == color:
                yv = int(getattr(obj, 'y', 0))
                if yv > best_y:
                    best_y = yv
                    purple = obj
        if purple is None:
            return
        throttle = float(sb.get('y_log_throttle_sec', self.purple_y_log_throttle_sec))
        now = rospy.get_time()
        if throttle > 0.0 and (now - self._last_purple_y_log_time) < throttle:
            return
        self._last_purple_y_log_time = now
        pcx = int(getattr(purple, 'x', 0))
        pcy = int(getattr(purple, 'y', 0))
        rospy.loginfo(
            'purple x=%d y=%d',
            pcx,
            pcy,
        )

    def log_green_y_for_tune(self):
        if not self.green_gate_enabled:
            return
        if not bool(rospy.get_param('/iscc/green_watch_active', False)):
            return
        green = None
        best_y = -1
        for obj in self.objects_info:
            if getattr(obj, 'type', '') == self.green_detect_type and getattr(obj, 'label', '') == self.green_color:
                yv = int(getattr(obj, 'y', 0))
                if yv > best_y:
                    best_y = yv
                    green = obj
        if green is None:
            return
        now = rospy.get_time()
        if self.green_y_log_throttle_sec > 0.0 and (now - self._last_green_y_log_time) < self.green_y_log_throttle_sec:
            return
        self._last_green_y_log_time = now
        gcx = int(getattr(green, 'x', 0))
        gcy = int(getattr(green, 'y', 0))
        rospy.loginfo(
            'green x=%d y=%d',
            gcx,
            gcy,
        )

    def setup_ros_interface(self):
        self.status_pub = rospy.Publisher('/orange_line_follow/status', String, queue_size=1)
        self.active_pub = rospy.Publisher('/orange_line_follow/active', Bool, queue_size=1)
        self.start_service = rospy.Service('/orange_line_follow/start', Trigger, self.start_service_callback)
        self.stop_service = rospy.Service('/orange_line_follow/stop', Trigger, self.stop_service_callback)
        self.reset_service = rospy.Service('/orange_line_follow/reset', Trigger, self.reset_service_callback)

    def publish_status(self, status, force=False):
        now = rospy.get_time()
        if (
            not force
            and status == self._last_status
            and (now - self._last_status_time) < self.status_throttle_sec
        ):
            return
        self._last_status = status
        self._last_status_time = now
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        active_msg = Bool()
        active_msg.data = self.active
        self.active_pub.publish(active_msg)

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

    def try_enter_color_detection(self):
        try:
            self.enter_func(None)
            return True
        except Exception as e:
            rospy.logwarn('orange_line_follow enter unavailable: %s', e)
            return False

    def try_start_color_detection(self):
        try:
            self.start_srv_callback(None)
            return True
        except Exception as e:
            rospy.logwarn('orange_line_follow start color_detection failed: %s', e)
            return False

    def try_stop_color_detection(self):
        try:
            self.stop_srv_callback(None)
            return True
        except Exception as e:
            rospy.logwarn('orange_line_follow stop color_detection failed: %s', e)
            return False

    def start_rqt_result_view(self):
        if not self.auto_open_rqt_result:
            return
        try:
            acquire_shared_rqt_result_view(self.rqt_owner_id, self.result_image_topic)
        except Exception as e:
            rospy.logwarn(f'failed to open rqt_image_view: {e}')

    def stop_rqt_result_view(self):
        if not self.auto_open_rqt_result:
            return
        try:
            release_shared_rqt_result_view(self.rqt_owner_id)
        except Exception as e:
            rospy.logwarn(f'failed to release rqt_image_view: {e}')

    def build_orange_side_param(self, roi_ratio):
        param = ColorDetect()
        param.color_name = self.orange_color
        param.use_name = True
        param.detect_type = 'side'
        param.image_process_size = self.image_process_size
        param.roi.y_min = int(self.parse_ratio(roi_ratio[0]) * self.image_height)
        param.roi.y_max = int(self.parse_ratio(roi_ratio[1]) * self.image_height)
        param.roi.x_min = int(self.parse_ratio(roi_ratio[2]) * self.image_width)
        param.roi.x_max = int(self.parse_ratio(roi_ratio[3]) * self.image_width)
        param.min_area = self.min_area
        param.max_area = self.image_width * self.image_height
        return param

    def build_purple_watch_param(self, sb):
        """首段 B 且 sandbag_purple_watch_active 时追加，与主控门控共用 tasks/sandbag_throw。"""
        roi = sb.get('roi', [0.2, 1.0, 0.15, 0.85])
        color = str(sb.get('color', 'purple'))
        detect_type = str(sb.get('detect_type', 'side'))
        min_a = int(sb.get('min_area', 10))
        param = ColorDetect()
        param.color_name = color
        param.use_name = True
        param.detect_type = detect_type
        param.image_process_size = self.image_process_size
        param.roi.y_min = int(self.parse_ratio(roi[0]) * self.image_height)
        param.roi.y_max = int(self.parse_ratio(roi[1]) * self.image_height)
        param.roi.x_min = int(self.parse_ratio(roi[2]) * self.image_width)
        param.roi.x_max = int(self.parse_ratio(roi[3]) * self.image_width)
        param.min_area = min_a
        param.max_area = self.image_width * self.image_height
        return param

    def build_green_watch_param(self):
        param = ColorDetect()
        param.color_name = self.green_color
        param.use_name = True
        param.detect_type = self.green_detect_type
        param.image_process_size = self.image_process_size
        param.roi.y_min = int(self.parse_ratio(self.green_roi[0]) * self.image_height)
        param.roi.y_max = int(self.parse_ratio(self.green_roi[1]) * self.image_height)
        param.roi.x_min = int(self.parse_ratio(self.green_roi[2]) * self.image_width)
        param.roi.x_max = int(self.parse_ratio(self.green_roi[3]) * self.image_width)
        param.min_area = self.green_min_area
        param.max_area = self.image_width * self.image_height
        return param

    def configure_detection(self):
        left_param = self.build_orange_side_param(self.left_roi)
        right_param = self.build_orange_side_param(self.right_roi)
        params = [left_param, right_param]
        sb = rospy.get_param('/tasks/sandbag_throw', {})
        watch = bool(rospy.get_param('/iscc/sandbag_purple_watch_active', True))
        green_watch = bool(rospy.get_param('/iscc/green_watch_active', False))
        if bool(sb.get('enabled', False)) and watch and not green_watch:
            params.append(self.build_purple_watch_param(sb))
            common.loginfo('orange_line_follow: orange x2 + purple watch ROI published')
        elif self.green_gate_enabled and green_watch:
            params.append(self.build_green_watch_param())
            common.loginfo('orange_line_follow: orange x2 + green watch ROI published')
        else:
            common.loginfo('orange_line_follow: orange dual side ROI published (no purple watch)')
        self.detect_pub.publish(params)

    def get_color_callback(self, msg):
        self.objects_info = msg.data

    def get_orange_side_objects(self):
        orange = []
        for obj in self.objects_info:
            if getattr(obj, 'type', '') == 'side' and getattr(obj, 'label', '') == self.orange_color:
                orange.append(obj)
        return orange

    def _pixel_frame_width(self):
        for obj in self.objects_info:
            w = int(getattr(obj, 'width', 0) or 0)
            if w > 0:
                return w
        return int(self.image_width)

    def split_left_right(self):
        left_obj = None
        right_obj = None
        mid = self._pixel_frame_width() / 2.0
        for obj in self.get_orange_side_objects():
            x = float(getattr(obj, 'x', 0))
            if x < mid:
                if left_obj is None or x < float(getattr(left_obj, 'x', 0)):
                    left_obj = obj
            else:
                if right_obj is None or x > float(getattr(right_obj, 'x', 0)):
                    right_obj = obj
        return left_obj, right_obj

    def is_gait_switch_on(self, default=True):
        v = rospy.get_param('/gait_switch', default)
        if isinstance(v, bool):
            return v
        return str(v).lower() in ('true', '1', 'yes', 'on')

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def clamp_yaw(self, value):
        return int(round(self.clamp(float(value), -self.rotation_yaw_limit, self.rotation_yaw_limit)))

    def normalize_line_angle(self, angle):
        """归一化到 [-90, 90]；0 近水平，绝对值越大越接近竖直。"""
        try:
            return ((float(angle) + 90.0) % 180.0) - 90.0
        except Exception:
            return 0.0

    def get_side_angle(self, obj):
        if obj is None:
            return None
        return self.normalize_line_angle(getattr(obj, 'angle', 0.0))

    def get_visible_line_angles(self, left_obj, right_obj):
        return {
            'left': float(self.get_side_angle(left_obj)) if left_obj is not None else None,
            'right': float(self.get_side_angle(right_obj)) if right_obj is not None else None,
        }

    def select_danger_line(self, left_obj, right_obj, threshold_deg):
        angles = self.get_visible_line_angles(left_obj, right_obj)
        candidates = []
        if angles['left'] is not None and abs(angles['left']) <= threshold_deg:
            candidates.append(('left', angles['left']))
        if angles['right'] is not None and abs(angles['right']) <= threshold_deg:
            candidates.append(('right', angles['right']))
        if not candidates:
            return None
        label, angle = min(candidates, key=lambda item: abs(item[1]))
        return {
            'label': label,
            'angle': float(angle),
            'left_angle': angles['left'],
            'right_angle': angles['right'],
        }

    def resolve_x_threshold_px(self, threshold, frame_width):
        value = float(threshold)
        if value <= 1.0:
            value *= float(frame_width)
        return self.clamp(value, 0.0, float(frame_width))

    def compute_control(self, left_obj, right_obj):
        frame_width = float(self._pixel_frame_width())
        left_ok = left_obj is not None
        right_ok = right_obj is not None
        left_x = float(getattr(left_obj, 'x', 0.0)) if left_ok else None
        right_x = float(getattr(right_obj, 'x', 0.0)) if right_ok else None
        angle_info = self.get_visible_line_angles(left_obj, right_obj)

        # 优先级：极危险横线保护 -> 角度近原地调整 -> x 阈值侧移纠偏。
        guard_line = self.select_danger_line(
            left_obj,
            right_obj,
            self.horizontal_guard_angle_threshold_deg,
        )
        if guard_line is not None:
            guard_yaw = -self.horizontal_guard_yaw if guard_line['label'] == 'left' else self.horizontal_guard_yaw
            mode = 'guard_right' if guard_yaw < 0 else 'guard_left'
            return {
                'mode': mode,
                'target_yaw': float(guard_yaw),
                'linear_x': float(self.horizontal_guard_linear_x),
                'linear_y': 0.0,
                'motion_profile': 'turn',
                'step_num': 0,
                'frame_width': frame_width,
                'left_x': left_x,
                'right_x': right_x,
                'left_angle': guard_line.get('left_angle'),
                'right_angle': guard_line.get('right_angle'),
                'trigger_label': guard_line['label'],
                'trigger_angle': guard_line['angle'],
            }

        if not left_ok and not right_ok:
            return {
                'mode': 'both_lost',
                'target_yaw': 0.0,
                'linear_x': float(self.go_linear_x),
                'linear_y': 0.0,
                'motion_profile': 'go',
                'step_num': 0,
                'frame_width': frame_width,
            }

        angle_adjust_line = self.select_danger_line(
            left_obj,
            right_obj,
            self.angle_adjust_yaw_threshold_deg,
        )
        if angle_adjust_line is not None:
            angle_adjust_yaw = -self.angle_adjust_yaw if angle_adjust_line['label'] == 'left' else self.angle_adjust_yaw
            mode = 'angle_adjust_right' if angle_adjust_yaw < 0 else 'angle_adjust_left'
            return {
                'mode': mode,
                'target_yaw': float(angle_adjust_yaw),
                'linear_x': float(self.angle_adjust_linear_x),
                'linear_y': 0.0,
                'motion_profile': 'turn',
                'step_num': 0,
                'frame_width': frame_width,
                'left_x': left_x,
                'right_x': right_x,
                'left_angle': angle_adjust_line.get('left_angle'),
                'right_angle': angle_adjust_line.get('right_angle'),
            }

        left_threshold_x = self.resolve_x_threshold_px(self.left_adjust_x_threshold, frame_width)
        right_threshold_x = self.resolve_x_threshold_px(self.right_adjust_x_threshold, frame_width)
        left_over = (left_x - left_threshold_x) if left_ok else None
        right_over = (right_threshold_x - right_x) if right_ok else None
        left_violation = left_over is not None and left_over > 0.0
        right_violation = right_over is not None and right_over > 0.0
        side_adjust_direction = None
        if left_violation and right_violation:
            compare_eps = 1e-6
            if float(left_over) > float(right_over) + compare_eps:
                side_adjust_direction = 'right'
            elif float(right_over) > float(left_over) + compare_eps:
                side_adjust_direction = 'left'
            elif left_x is not None and right_x is not None:
                lane_center = (float(left_x) + float(right_x)) / 2.0
                frame_center = frame_width / 2.0
                if lane_center < frame_center:
                    side_adjust_direction = 'left'
                elif lane_center > frame_center:
                    side_adjust_direction = 'right'
        elif left_violation:
            side_adjust_direction = 'right'
        elif right_violation:
            side_adjust_direction = 'left'

        if side_adjust_direction == 'right':
            return {
                'mode': 'side_adjust_right',
                'target_yaw': 0.0,
                'linear_x': float(self.side_adjust_linear_x),
                'linear_y': float(-abs(self.side_adjust_linear_y)),
                'motion_profile': 'side_adjust',
                'step_num': 1,
                'frame_width': frame_width,
                'left_x': left_x,
                'right_x': right_x,
                'left_angle': angle_info['left'],
                'right_angle': angle_info['right'],
            }

        if side_adjust_direction == 'left':
            return {
                'mode': 'side_adjust_left',
                'target_yaw': 0.0,
                'linear_x': float(self.side_adjust_linear_x),
                'linear_y': float(abs(self.side_adjust_linear_y)),
                'motion_profile': 'side_adjust',
                'step_num': 1,
                'frame_width': frame_width,
                'left_x': left_x,
                'right_x': right_x,
                'left_angle': angle_info['left'],
                'right_angle': angle_info['right'],
            }

        return {
            'mode': 'forward',
            'target_yaw': 0.0,
            'linear_x': float(self.go_linear_x),
            'linear_y': 0.0,
            'motion_profile': 'go',
            'step_num': 0,
            'frame_width': frame_width,
            'left_x': left_x,
            'right_x': right_x,
            'left_angle': angle_info['left'],
            'right_angle': angle_info['right'],
        }

    def log_control_state(self, control, gait_on=True):
        now = rospy.get_time()
        mode = control.get('mode', 'unknown')
        throttle = self.log_xy_throttle_sec
        if (now - self._last_control_log_time) < throttle:
            return
        self._last_control_log_time = now
        cmd_x = float(control.get('linear_x', 0.0))
        cmd_y = float(control.get('linear_y', 0.0))
        cmd_yaw = float(control.get('target_yaw', 0.0))
        step_num = int(control.get('step_num', 0))
        if mode in ('guard_left', 'guard_right'):
            trigger_label = str(control.get('trigger_label', '')).upper()
            trigger_angle = float(control.get('trigger_angle', 0.0))
            rospy.logwarn(
                'orange_line_follow %s %s line_yaw=%.1f cmd_x=%.4f cmd_y=%.4f cmd_yaw=%.1f step_num=%d',
                mode,
                trigger_label,
                trigger_angle,
                cmd_x,
                cmd_y,
                cmd_yaw,
                step_num,
            )
            return
        left_x = control.get('left_x')
        right_x = control.get('right_x')
        left_angle = control.get('left_angle')
        right_angle = control.get('right_angle')
        if left_x is not None and right_x is not None:
            rospy.loginfo(
                'orange_line_follow %s L x=%.1f yaw=%.1f R x=%.1f yaw=%.1f cmd_x=%.4f cmd_y=%.4f cmd_yaw=%.1f step_num=%d',
                mode,
                left_x,
                float(left_angle if left_angle is not None else 0.0),
                right_x,
                float(right_angle if right_angle is not None else 0.0),
                cmd_x,
                cmd_y,
                cmd_yaw,
                step_num,
            )
            return
        if left_x is not None:
            rospy.loginfo(
                'orange_line_follow %s L x=%.1f yaw=%.1f cmd_x=%.4f cmd_y=%.4f cmd_yaw=%.1f step_num=%d',
                mode,
                left_x,
                float(left_angle if left_angle is not None else 0.0),
                cmd_x,
                cmd_y,
                cmd_yaw,
                step_num,
            )
            return
        if right_x is not None:
            rospy.loginfo(
                'orange_line_follow %s R x=%.1f yaw=%.1f cmd_x=%.4f cmd_y=%.4f cmd_yaw=%.1f step_num=%d',
                mode,
                right_x,
                float(right_angle if right_angle is not None else 0.0),
                cmd_x,
                cmd_y,
                cmd_yaw,
                step_num,
            )

    def start_service_callback(self, req):
        self.start_rqt_result_view()
        self.load_config()
        self._both_lost_since = None
        self._holding_lost = False
        self.gait_switch_off_reported = False
        self._last_control_log_time = 0.0
        self._last_purple_y_log_time = 0.0
        self._last_green_y_log_time = 0.0
        self.active = True
        self.try_enter_color_detection()
        self.configure_detection()
        if not self.try_start_color_detection():
            self.active = False
            self.stop_rqt_result_view()
            self.publish_status('waiting_color_detection', force=True)
            return TriggerResponse(success=False, message='color_detection/start unavailable')
        self.motion_manager.run_action('walk_ready')
        if self.head_settle_sec > 0:
            rospy.sleep(self.head_settle_sec)
        self.publish_status('started', force=True)
        return TriggerResponse(success=True, message='orange line follow started')

    def stop_service_callback(self, req):
        self.active = False
        self.gait_manager.stop()
        self.try_stop_color_detection()
        self.stop_rqt_result_view()
        self.publish_status('stopped', force=True)
        return TriggerResponse(success=True, message='orange line follow stopped')

    def reset_service_callback(self, req):
        self.active = False
        self._both_lost_since = None
        self._holding_lost = False
        self.gait_switch_off_reported = False
        self._last_control_log_time = 0.0
        self.gait_manager.stop()
        self.try_stop_color_detection()
        self.stop_rqt_result_view()
        self.publish_status('reset', force=True)
        return TriggerResponse(success=True, message='orange line follow reset')

    def apply_motion(self, linear_x, linear_y, yaw_int, motion_profile='go', step_num=0):
        if motion_profile == 'turn':
            dsp = self.turn_dsp
            walking = self.turn_walking_param
            arm_swap = self.turn_arm_swap
        elif motion_profile == 'side_adjust':
            dsp = self.side_adjust_dsp
            walking = self.side_adjust_walking_param
            arm_swap = self.side_adjust_arm_swap
        else:
            dsp = self.go_dsp
            walking = self.go_walking_param
            arm_swap = self.go_arm_swap
        yaw_int = self.clamp_yaw(yaw_int)
        step_num = int(step_num)
        if step_num > 0 and (self.gait_manager.state == 'walking' or self.gait_manager.is_walking):
            self.gait_manager.stop()
            wait_deadline = time.time() + 0.2
            while self.gait_manager.is_walking and time.time() < wait_deadline:
                time.sleep(0.01)
        self.gait_manager.set_step(
            dsp,
            float(linear_x),
            float(linear_y),
            int(yaw_int),
            walking,
            arm_swap=arm_swap,
            step_num=step_num,
        )

    def run(self):
        self.publish_status('ready', force=True)
        rate_sleep = 0.02
        while self.running and not rospy.is_shutdown():
            if not self.active:
                time.sleep(rate_sleep)
                continue

            left_obj, right_obj = self.split_left_right()
            left_ok = left_obj is not None
            right_ok = right_obj is not None
            now = rospy.get_time()
            self.log_purple_y_for_tune()
            self.log_green_y_for_tune()
            gait_on = self.is_gait_switch_on()
            if not gait_on:
                if not self.gait_switch_off_reported:
                    self.gait_manager.stop()
                    self.publish_status('gait_switch_off')
                    rospy.logwarn('gait_switch_off已设置为false，步态发布关闭（仅提示一次）')
                    self.gait_switch_off_reported = True
            else:
                self.gait_switch_off_reported = False

            if not left_ok and not right_ok:
                if self._both_lost_since is None:
                    self._both_lost_since = now
                lost_dt = now - self._both_lost_since
                if lost_dt < self.lost_both_grace_sec:
                    if gait_on:
                        self.apply_motion(self.go_linear_x, 0.0, 0, 'go')
                    self.publish_status('lost_both_grace')
                elif lost_dt < self.lost_both_stop_sec:
                    if gait_on:
                        self.apply_motion(self.go_linear_x, 0.0, 0, 'go')
                    self.publish_status('lost_both_straight')
                else:
                    if gait_on and not self._holding_lost:
                        self.gait_manager.stop()
                        self._holding_lost = True
                    self.publish_status('lost_both_stopped')
                time.sleep(rate_sleep)
                continue

            self._both_lost_since = None
            self._holding_lost = False

            control = self.compute_control(left_obj, right_obj)
            self.log_control_state(control, gait_on=gait_on)
            yaw_i = self.clamp_yaw(control['target_yaw'])
            lin_x = control['linear_x']
            lin_y = control.get('linear_y', 0.0)
            motion_profile = control.get('motion_profile', 'go')
            step_num = int(control.get('step_num', 0))
            if gait_on:
                self.apply_motion(lin_x, lin_y, yaw_i, motion_profile, step_num)

            self.publish_status(control['mode'])

            time.sleep(rate_sleep)

        self.gait_manager.stop()
        self.stop_rqt_result_view()
        self.try_stop_color_detection()
        self.motion_manager.run_action('stand')
        self.publish_status('shutdown', force=True)
        rospy.signal_shutdown('shutdown')

    def shutdown(self, signum, frame):
        with self.lock:
            self.running = False
            self.active = False
            self.gait_manager.stop()
            self.stop_rqt_result_view()
            self.try_stop_color_detection()
            self.motion_manager.run_action('stand')
            self.publish_status('shutdown', force=True)
            common.loginfo('%s shutdown' % self.name)


if __name__ == '__main__':
    try:
        OrangeLineFollow('orange_line_follow').run()
    except Exception as e:
        rospy.logerr('Orange line follow init error: %s', e)
