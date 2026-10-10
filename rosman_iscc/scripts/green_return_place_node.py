#!/usr/bin/env python3
# encoding: utf-8
# @Author: Dejin Xuan
# @Date: 2026.4.6
# 绿块抓取后：右转回程找双橙线，循迹到终点 Tag 后放块。
import os
import signal
import sys
import time

import rospy
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

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

try:
    from apriltag_ros.msg import AprilTagDetectionArray
except Exception:
    AprilTagDetectionArray = None


class GreenReturnPlaceNode(Common):
    def __init__(self, name):
        self.head_pan_init = rospy.get_param('/servo/head_pan_init', 500)
        self.head_tilt_look_down = rospy.get_param('/servo/head_tilt_look_down', 280)

        rospy.init_node(name)
        self.name = name
        self.running = True
        self.active = False
        self.phase = 'idle'
        self.objects_info = []
        self.tag_center = None
        self.tag_id = -1
        self._tag_seen_time = 0.0
        self._tag_visible_streak = 0
        self._tag_approach_ready_streak = 0
        self._tag_align_stable_count = 0
        self._tag_sample_seq = 0
        self._last_tag_align_eval_seq = 0
        self._phase_start_time = 0.0
        self._line_stable_count = 0
        self._lane_stable_count = 0
        self._lane_right_only_streak = 0
        self._lost_frames = 0
        self._post_pick_blind_steps_done = 0
        self._blind_steps_done = 0
        self._last_log_time = 0.0
        self._last_tag_log_time = 0.0
        self._last_lane_log_time = 0.0
        self._last_terminal_log_time = 0.0
        self._orange_line_status = ''
        self._orange_line_status_time = 0.0
        self._corridor_lost_since = 0.0
        self.gait_switch_off_reported = False
        self.rqt_owner_id = 'green_return_place:%s:%d' % (self.name, os.getpid())

        super().__init__(name, self.head_pan_init, self.head_tilt_look_down)
        signal.signal(signal.SIGINT, self.shutdown)

        self.process_size = rospy.get_param('/image_processing/process_size', [160, 120])
        self.image_width = int(self.process_size[0]) if len(self.process_size) > 0 else 160
        self.image_height = int(self.process_size[1]) if len(self.process_size) > 1 else 120
        self.image_process_size = [self.image_width, self.image_height]
        self.result_image_topic = rospy.get_param(
            '/tasks/line_following/result_image_topic',
            '/color_detection/image_result',
        )
        self.auto_open_rqt_result = rospy.get_param('/tasks/line_following/auto_open_rqt_result', True)

        self.load_config()
        self.setup_ros_interface()
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.object_callback)
        rospy.Subscriber('/orange_line_follow/status', String, self.orange_line_status_callback)
        if self.tag_trigger_enabled and AprilTagDetectionArray is not None:
            rospy.Subscriber(self.tag_topic, AprilTagDetectionArray, self.tag_callback)
        elif self.tag_trigger_enabled:
            rospy.logwarn('apriltag_ros.msg unavailable, green_return_place tag trigger disabled')
            self.tag_trigger_enabled = False

        self.motion_manager.run_action('walk_ready')
        common.loginfo('green_return_place initialized')

    def load_config(self):
        cfg = rospy.get_param('/tasks/green_return_place', {})
        if not isinstance(cfg, dict):
            cfg = {}

        self.enabled = bool(cfg.get('enabled', True))
        self.orange_color = str(cfg.get('color', 'orange'))

        line_cfg = cfg.get('line_align', {})
        if not isinstance(line_cfg, dict):
            line_cfg = {}
        self.align_color = str(line_cfg.get('align_color', self.orange_color))
        self.align_detect_type = str(line_cfg.get('align_detect_type', 'side'))
        self.align_roi = line_cfg.get('align_roi', [0.0, 0.8, 0.0, 1.0])
        self.line_min_area = int(line_cfg.get('min_area', 10))
        self.line_angle_target = float(line_cfg.get('angle_target', 0.0))
        self.line_angle_tolerance = float(line_cfg.get('angle_tolerance', 10.0))
        self.line_stable_frames = max(1, int(line_cfg.get('stable_frames', 3)))
        self.line_timeout_sec = float(line_cfg.get('timeout_sec', 20.0))
        self.line_lost_frames = int(line_cfg.get('lost_frames_to_search', 8))
        self.line_right_turn_yaw = int(line_cfg.get('right_turn_yaw', -10))
        self.line_left_backoff_yaw = int(line_cfg.get('left_backoff_yaw', 5))
        self.line_right_turn_x_step = float(line_cfg.get('right_turn_x_step', 0.004))
        self.line_search_x_step = float(line_cfg.get('search_x_step', self.line_right_turn_x_step))
        self.line_rotate_dsp = line_cfg.get('rotate_dsp', [320, 0.25, 0.023])
        self.line_rotate_arm_swap = int(line_cfg.get('rotate_arm_swap', 25))
        self.line_rotate_walking_param = self.gait_manager.get_gait_param()
        line_wp_updates = line_cfg.get('rotate_walking_param_updates', {})
        if isinstance(line_wp_updates, dict):
            self.line_rotate_walking_param.update(line_wp_updates)

        blind_cfg = cfg.get('blind_turn', {})
        if not isinstance(blind_cfg, dict):
            blind_cfg = {}
        self.blind_steps = max(0, int(blind_cfg.get('steps', 5)))
        self.blind_turn_yaw = int(blind_cfg.get('yaw', -10))
        self.blind_turn_x_step = float(blind_cfg.get('x_step', 0.010))
        self.blind_turn_dsp = blind_cfg.get('dsp', self.line_rotate_dsp)
        self.blind_turn_arm_swap = int(blind_cfg.get('arm_swap', self.line_rotate_arm_swap))
        self.blind_turn_walking_param = self.gait_manager.get_gait_param()
        blind_wp_updates = blind_cfg.get('walking_param_updates', {})
        if isinstance(blind_wp_updates, dict):
            self.blind_turn_walking_param.update(blind_wp_updates)

        post_pick_blind_cfg = cfg.get('post_pick_blind_right', {})
        if not isinstance(post_pick_blind_cfg, dict):
            post_pick_blind_cfg = {}
        self.post_pick_blind_enabled = bool(post_pick_blind_cfg.get('enabled', True))
        self.post_pick_blind_steps = max(0, int(post_pick_blind_cfg.get('steps', self.blind_steps)))
        self.post_pick_blind_yaw = int(post_pick_blind_cfg.get('yaw', self.blind_turn_yaw))
        self.post_pick_blind_x_step = float(post_pick_blind_cfg.get('x_step', self.blind_turn_x_step))
        self.post_pick_blind_dsp = post_pick_blind_cfg.get('dsp', self.blind_turn_dsp)
        self.post_pick_blind_arm_swap = int(post_pick_blind_cfg.get('arm_swap', self.blind_turn_arm_swap))
        self.post_pick_blind_walking_param = self.gait_manager.get_gait_param()
        post_pick_blind_wp_updates = post_pick_blind_cfg.get('walking_param_updates', blind_wp_updates)
        if isinstance(post_pick_blind_wp_updates, dict):
            self.post_pick_blind_walking_param.update(post_pick_blind_wp_updates)

        lane_cfg = cfg.get('lane_search', {})
        if not isinstance(lane_cfg, dict):
            lane_cfg = {}
        self.lane_left_color = str(lane_cfg.get('left_color', self.orange_color))
        self.lane_left_detect_type = str(lane_cfg.get('left_detect_type', 'side'))
        self.lane_left_roi = lane_cfg.get('left_roi', [0.4, 0.8, 0.0, 0.4])
        self.lane_right_color = str(lane_cfg.get('right_color', self.orange_color))
        self.lane_right_detect_type = str(lane_cfg.get('right_detect_type', 'side'))
        self.lane_right_roi = lane_cfg.get('right_roi', [0.4, 0.8, 0.6, 1.0])
        self.lane_min_area = int(lane_cfg.get('min_area', 10))
        self.lane_turn_yaw = int(lane_cfg.get('turn_yaw', -10))
        self.lane_turn_x_step = float(lane_cfg.get('turn_x_step', 0.004))
        self.lane_stable_frames = max(1, int(lane_cfg.get('stable_frames', 3)))
        self.lane_timeout_sec = float(lane_cfg.get('timeout_sec', 30.0))
        self.lane_lost_frames = int(lane_cfg.get('lost_frames_to_search', 8))
        self.lane_min_width = float(lane_cfg.get('min_lane_width_px', 40))
        self.lane_max_width = float(lane_cfg.get('max_lane_width_px', 600))
        self.lane_require_order = bool(lane_cfg.get('require_left_right_order', True))
        self.lane_rotate_dsp = lane_cfg.get('rotate_dsp', self.line_rotate_dsp)
        self.lane_rotate_arm_swap = int(lane_cfg.get('rotate_arm_swap', self.line_rotate_arm_swap))
        self.lane_rotate_walking_param = self.gait_manager.get_gait_param()
        lane_wp_updates = lane_cfg.get('rotate_walking_param_updates', {})
        if isinstance(lane_wp_updates, dict):
            self.lane_rotate_walking_param.update(lane_wp_updates)

        corridor_cfg = cfg.get('corridor_follow', {})
        if not isinstance(corridor_cfg, dict):
            corridor_cfg = {}
        self.orange_line_start_service = str(corridor_cfg.get('start_service', '/orange_line_follow/start'))
        self.orange_line_stop_service = str(corridor_cfg.get('stop_service', '/orange_line_follow/stop'))
        self.corridor_lost_statuses = set(corridor_cfg.get('lost_statuses', ['lost_both_straight', 'lost_both_stopped']))
        self.corridor_lost_min_sec = float(corridor_cfg.get('lost_status_min_sec', 0.4))
        self.corridor_timeout_sec = float(corridor_cfg.get('timeout_sec', 0.0))

        tag_cfg = cfg.get('tag_trigger', {})
        if not isinstance(tag_cfg, dict):
            tag_cfg = {}
        self.tag_trigger_enabled = bool(tag_cfg.get('enabled', True))
        self.tag_topic = str(tag_cfg.get('topic', '/tag_detections'))
        self.tag_target_id = int(tag_cfg.get('target_id', -1))
        self.tag_roi = tag_cfg.get('roi', [0.0, 1.0, 0.0, 1.0])
        self.tag_lost_timeout_sec = float(tag_cfg.get('lost_timeout_sec', 1.0))
        self.tag_frame_width = max(1, int(tag_cfg.get('frame_width', 640)))
        self.tag_frame_height = max(1, int(tag_cfg.get('frame_height', 480)))
        self.tag_log_throttle_sec = float(tag_cfg.get('log_throttle_sec', 0.2))

        tag_approach_cfg = cfg.get('tag_approach', {})
        if not isinstance(tag_approach_cfg, dict):
            tag_approach_cfg = {}
        self.tag_approach_target_x_pixel = float(tag_approach_cfg.get('target_x_pixel', 0.5))
        self.tag_approach_switch_visible_stable_frames = max(
            1,
            int(tag_approach_cfg.get('switch_visible_stable_frames', 2)),
        )
        self.tag_approach_x_band_px = float(tag_approach_cfg.get('x_band_px', 40))
        self.tag_approach_coarse_turn_yaw = int(tag_approach_cfg.get('coarse_turn_yaw', 6))
        self.tag_approach_fine_turn_yaw = int(tag_approach_cfg.get('fine_turn_yaw', 3))
        self.tag_approach_forward_x_step = float(tag_approach_cfg.get('forward_x_step', 0.008))
        self.tag_approach_pre_align_y_threshold = float(tag_approach_cfg.get('pre_align_y_threshold', 180))
        self.tag_approach_pre_align_stable_frames = 2
        self.tag_approach_dsp = tag_approach_cfg.get('dsp', [350, 0.35, 0.023])
        self.tag_approach_arm_swap = int(tag_approach_cfg.get('arm_swap', 25))
        self.tag_approach_walking_param = self.gait_manager.get_gait_param()
        tag_approach_wp_updates = tag_approach_cfg.get('walking_param_updates', {})
        if isinstance(tag_approach_wp_updates, dict):
            self.tag_approach_walking_param.update(tag_approach_wp_updates)

        tag_align_cfg = cfg.get('tag_align', {})
        if not isinstance(tag_align_cfg, dict):
            tag_align_cfg = {}
        self.tag_align_target_x_pixel = float(tag_align_cfg.get('target_x_pixel', 0.5))
        self.tag_align_target_y_pixel = float(tag_align_cfg.get('target_y_pixel', 250))
        self.tag_align_x_tolerance = float(tag_align_cfg.get('x_tolerance', 20))
        self.tag_align_y_tolerance = float(tag_align_cfg.get('y_tolerance', 20))
        self.tag_align_stable_frames = max(1, int(tag_align_cfg.get('stable_frames', 2)))
        self.tag_align_step_distance = float(tag_align_cfg.get('coarse_step_distance', 0.006))
        self.tag_align_side_step_distance = float(tag_align_cfg.get('coarse_side_step_distance', 0.004))
        self.tag_align_dsp = tag_align_cfg.get('dsp', [350, 0.35, 0.023])
        self.tag_align_arm_swap = int(tag_align_cfg.get('arm_swap', 25))
        self.tag_align_lost_timeout_sec = float(
            tag_align_cfg.get('lost_timeout_sec', self.tag_lost_timeout_sec)
        )
        self.tag_align_walking_param = self.gait_manager.get_gait_param()
        tag_align_wp_updates = tag_align_cfg.get('walking_param_updates', {})
        if isinstance(tag_align_wp_updates, dict):
            self.tag_align_walking_param.update(tag_align_wp_updates)

        orange_cfg = rospy.get_param('/tasks/orange_line_following', {})
        if not isinstance(orange_cfg, dict):
            orange_cfg = {}

        terminal_cfg = cfg.get('terminal_scan', {})
        if not isinstance(terminal_cfg, dict):
            terminal_cfg = {}
        default_slow_x = min(0.008, float(orange_cfg.get('go_linear_x', 0.02)) * 0.4)
        self.terminal_x_step = float(terminal_cfg.get('x_step', default_slow_x))
        self.terminal_yaw = int(terminal_cfg.get('yaw', 0))
        self.terminal_dsp = terminal_cfg.get('dsp', [320, 0.25, 0.023])
        self.terminal_arm_swap = int(terminal_cfg.get('arm_swap', 25))
        self.terminal_timeout_sec = float(terminal_cfg.get('timeout_sec', 30.0))
        self.terminal_log_throttle_sec = float(terminal_cfg.get('log_throttle_sec', 0.2))
        self.terminal_reacquire_lane_stable_frames = max(
            1,
            int(terminal_cfg.get('reacquire_lane_stable_frames', self.lane_stable_frames)),
        )
        self.terminal_walking_param = self.gait_manager.get_gait_param()
        terminal_wp_updates = terminal_cfg.get('walking_param_updates', {})
        if isinstance(terminal_wp_updates, dict):
            self.terminal_walking_param.update(terminal_wp_updates)

        place_cfg = cfg.get('place', {})
        if not isinstance(place_cfg, dict):
            place_cfg = {}
        self.place_action_name = str(place_cfg.get('action_name', 'place_block'))
        self.place_wait_sec = float(place_cfg.get('wait_sec_after_place', 0.5))
        self.walk_ready_after_place = bool(place_cfg.get('walk_ready_after_place', True))

        self.log_throttle_sec = float(cfg.get('log_throttle_sec', 0.2))

    def setup_ros_interface(self):
        self.status_pub = rospy.Publisher('/green_return_place/status', String, queue_size=1)
        self.active_pub = rospy.Publisher('/green_return_place/active', Bool, queue_size=1)
        self.start_service = rospy.Service('/green_return_place/start', Trigger, self.start_service_callback)
        self.stop_service = rospy.Service('/green_return_place/stop', Trigger, self.stop_service_callback)
        self.reset_service = rospy.Service('/green_return_place/reset', Trigger, self.reset_service_callback)

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

    def fixed_step_motion_allowed(self):
        if self.phase == 'post_pick_blind_right':
            return self._post_pick_blind_steps_done < self.post_pick_blind_steps
        if self.phase == 'blind_turn_right':
            return self._blind_steps_done < self.blind_steps
        return False

    def call_service(self, service_name, timeout=5.0):
        try:
            rospy.wait_for_service(service_name, timeout=timeout)
            proxy = rospy.ServiceProxy(service_name, Trigger)
            resp = proxy(TriggerRequest())
            return bool(resp.success)
        except Exception as e:
            rospy.logwarn('green_return_place service call failed: %s (%s)', service_name, e)
            return False

    def publish_status(self, status):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        active_msg = Bool()
        active_msg.data = self.active
        self.active_pub.publish(active_msg)

    def object_callback(self, msg):
        self.objects_info = msg.data

    def orange_line_status_callback(self, msg):
        self._orange_line_status = str(msg.data)
        self._orange_line_status_time = rospy.get_time()

    def tag_roi_px(self):
        y_min = int(self.parse_ratio(self.tag_roi[0]) * self.tag_frame_height)
        y_max = int(self.parse_ratio(self.tag_roi[1]) * self.tag_frame_height)
        x_min = int(self.parse_ratio(self.tag_roi[2]) * self.tag_frame_width)
        x_max = int(self.parse_ratio(self.tag_roi[3]) * self.tag_frame_width)
        return y_min, y_max, x_min, x_max

    def resolve_target_pixel(self, value, frame_size):
        target = float(value)
        if target <= 1.0:
            target *= float(frame_size)
        return float(target)

    def tag_approach_target_x_px(self):
        return self.resolve_target_pixel(self.tag_approach_target_x_pixel, self.tag_frame_width)

    def tag_pre_align_y_threshold_px(self):
        return self.resolve_target_pixel(self.tag_approach_pre_align_y_threshold, self.tag_frame_height)

    def tag_align_target_x_px(self):
        return self.resolve_target_pixel(self.tag_align_target_x_pixel, self.tag_frame_width)

    def tag_align_target_y_px(self):
        return self.resolve_target_pixel(self.tag_align_target_y_pixel, self.tag_frame_height)

    def extract_tag_center(self, detection):
        corners = list(getattr(detection, 'corners', []))
        if len(corners) >= 4:
            return (
                sum(float(c.x) for c in corners[:4]) / 4.0,
                sum(float(c.y) for c in corners[:4]) / 4.0,
            )
        centre = getattr(detection, 'centre', None)
        if centre is None:
            centre = getattr(detection, 'center', None)
        if centre is not None:
            return float(getattr(centre, 'x', 0.0)), float(getattr(centre, 'y', 0.0))
        return None

    def tag_callback(self, msg):
        now = rospy.get_time()
        y_min, y_max, x_min, x_max = self.tag_roi_px()
        best = None
        best_id = -1
        best_y = -1.0
        for detection in getattr(msg, 'detections', []):
            ids = list(getattr(detection, 'id', []))
            if self.tag_target_id >= 0 and self.tag_target_id not in ids:
                continue
            center = self.extract_tag_center(detection)
            if center is None:
                continue
            cx, cy = center
            if cx < x_min or cx > x_max or cy < y_min or cy > y_max:
                continue
            if cy > best_y:
                best = center
                best_id = ids[0] if len(ids) > 0 else -1
                best_y = cy
        if best is None:
            self._tag_visible_streak = 0
            self._tag_approach_ready_streak = 0
            self._tag_align_stable_count = 0
            return
        self._tag_sample_seq += 1
        self.tag_center = best
        self.tag_id = best_id
        self._tag_seen_time = now
        tx, ty = best
        self._tag_visible_streak += 1
        ready_y = ty >= self.tag_pre_align_y_threshold_px()
        if ready_y:
            self._tag_approach_ready_streak += 1
        else:
            self._tag_approach_ready_streak = 0
        if self.tag_log_throttle_sec <= 0.0 or (now - self._last_tag_log_time) >= self.tag_log_throttle_sec:
            rospy.loginfo(
                'green_return_tag: id=%s x=%.1f y=%.1f visible=%d/%d y_ready=%s ready=%d/%d threshold=%.1f phase=%s',
                str(best_id),
                tx,
                ty,
                self._tag_visible_streak,
                self.tag_approach_switch_visible_stable_frames,
                str(ready_y),
                self._tag_approach_ready_streak,
                self.tag_approach_pre_align_stable_frames,
                self.tag_pre_align_y_threshold_px(),
                self.phase,
            )
            self._last_tag_log_time = now

    def has_recent_tag(self, now, timeout_sec=None):
        if self.tag_center is None or self._tag_seen_time <= 0.0:
            return False
        lost_timeout_sec = self.tag_lost_timeout_sec if timeout_sec is None else float(timeout_sec)
        return lost_timeout_sec <= 0.0 or (now - self._tag_seen_time) <= lost_timeout_sec

    def should_enter_tag_approach(self, now):
        if not self.tag_trigger_enabled or not self.has_recent_tag(now):
            return False
        return self._tag_visible_streak >= self.tag_approach_switch_visible_stable_frames

    def should_enter_tag_align(self, now):
        if not self.tag_trigger_enabled or not self.has_recent_tag(now, self.tag_align_lost_timeout_sec):
            return False
        return self._tag_approach_ready_streak >= self.tag_approach_pre_align_stable_frames

    def tag_aligned_to_place(self, now):
        if not self.tag_trigger_enabled or not self.has_recent_tag(now, self.tag_align_lost_timeout_sec):
            return False
        return self._tag_align_stable_count >= self.tag_align_stable_frames

    def clear_object_tracking(self):
        self.objects_info = []

    def reset_tag_phase_counters(self, reset_eval_seq=False):
        self._tag_visible_streak = 0
        self._tag_approach_ready_streak = 0
        self._tag_align_stable_count = 0
        if reset_eval_seq:
            self._last_tag_align_eval_seq = 0

    def clear_tag_tracking(self):
        self.tag_center = None
        self.tag_id = -1
        self._tag_seen_time = 0.0
        self._tag_sample_seq = 0
        self.reset_tag_phase_counters(reset_eval_seq=True)

    def build_side_param(self, color, detect_type, roi_ratio, min_area):
        param = ColorDetect()
        param.color_name = color
        param.use_name = True
        param.detect_type = detect_type
        param.image_process_size = self.image_process_size
        param.roi.y_min = int(self.parse_ratio(roi_ratio[0]) * self.image_height)
        param.roi.y_max = int(self.parse_ratio(roi_ratio[1]) * self.image_height)
        param.roi.x_min = int(self.parse_ratio(roi_ratio[2]) * self.image_width)
        param.roi.x_max = int(self.parse_ratio(roi_ratio[3]) * self.image_width)
        param.min_area = min_area
        param.max_area = self.image_width * self.image_height
        return param

    def build_detection_params_for_phase(self, phase=None):
        phase = phase or self.phase
        if phase in ('post_pick_blind_right', 'line_align_right'):
            return [self.build_side_param(self.align_color, self.align_detect_type, self.align_roi, self.line_min_area)]
        if phase == 'lane_search':
            return [
                self.build_side_param(
                    self.lane_left_color,
                    self.lane_left_detect_type,
                    self.lane_left_roi,
                    self.lane_min_area,
                ),
                self.build_side_param(
                    self.lane_right_color,
                    self.lane_right_detect_type,
                    self.lane_right_roi,
                    self.lane_min_area,
                ),
            ]
        if phase in ('terminal_scan', 'tag_approach', 'tag_align'):
            return [
                self.build_side_param(
                    self.lane_left_color,
                    self.lane_left_detect_type,
                    self.lane_left_roi,
                    self.lane_min_area,
                ),
                self.build_side_param(
                    self.lane_right_color,
                    self.lane_right_detect_type,
                    self.lane_right_roi,
                    self.lane_min_area,
                ),
            ]
        return []

    def publish_detection_for_phase(self, phase=None):
        phase = phase or self.phase
        params = self.build_detection_params_for_phase(phase)
        self.detect_pub.publish(params)
        rospy.loginfo('green_return_place detect phase=%s roi_count=%d', phase, len(params))
        return params

    def try_enter_color_detection(self):
        try:
            self.enter_func(None)
            return True
        except Exception as e:
            rospy.logwarn('green_return_place enter color_detection unavailable: %s', e)
            return False

    def try_start_color_detection(self):
        try:
            self.start_srv_callback(None)
            return True
        except Exception as e:
            rospy.logwarn('green_return_place start color_detection failed: %s', e)
            return False

    def try_stop_color_detection(self):
        try:
            self.stop_srv_callback(None)
            return True
        except Exception as e:
            rospy.logwarn('green_return_place stop color_detection failed: %s', e)
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

    def side_y_max(self, obj, fallback_y=0):
        try:
            y0 = float(getattr(obj, 'y', fallback_y))
            lp = getattr(obj, 'left_point', None)
            rp = getattr(obj, 'right_point', None)
            ys = [y0]
            if isinstance(lp, (list, tuple)) and len(lp) >= 2:
                ys.append(float(lp[1]))
            if isinstance(rp, (list, tuple)) and len(rp) >= 2:
                ys.append(float(rp[1]))
            return int(max(ys))
        except Exception:
            return int(getattr(obj, 'y', fallback_y) or fallback_y)

    def line_span_x(self, obj):
        lp = getattr(obj, 'left_point', None)
        rp = getattr(obj, 'right_point', None)
        if isinstance(lp, (list, tuple)) and len(lp) >= 1 and isinstance(rp, (list, tuple)) and len(rp) >= 1:
            return abs(float(rp[0]) - float(lp[0]))
        return 0.0

    def get_best_align_line(self):
        best = None
        best_span = -1.0
        best_y = -1
        for obj in self.objects_info:
            if getattr(obj, 'label', '') != self.align_color:
                continue
            if getattr(obj, 'type', '') != self.align_detect_type:
                continue
            span_x = self.line_span_x(obj)
            yv = self.side_y_max(obj, fallback_y=int(getattr(obj, 'y', 0)))
            if span_x > best_span or (abs(span_x - best_span) < 1e-6 and yv > best_y):
                best_span = span_x
                best_y = yv
                best = obj
        return best

    def pixel_frame_width(self):
        for obj in self.objects_info:
            w = int(getattr(obj, 'width', 0) or 0)
            if w > 0:
                return w
        return int(self.image_width)

    def get_best_lane_side(self, color, detect_type, side):
        best = None
        best_y = -1
        mid_x = self.pixel_frame_width() * 0.5
        for obj in self.objects_info:
            if getattr(obj, 'label', '') != color:
                continue
            if getattr(obj, 'type', '') != detect_type:
                continue
            ox = float(getattr(obj, 'x', 0))
            if side == 'left' and ox >= mid_x:
                continue
            if side == 'right' and ox <= mid_x:
                continue
            oy = self.side_y_max(obj, fallback_y=int(getattr(obj, 'y', 0)))
            if oy > best_y:
                best_y = oy
                best = obj
        return best

    def evaluate_lane_state(self):
        left_obj = self.get_best_lane_side(self.lane_left_color, self.lane_left_detect_type, 'left')
        right_obj = self.get_best_lane_side(self.lane_right_color, self.lane_right_detect_type, 'right')
        if left_obj is None or right_obj is None:
            return {
                'left_obj': left_obj,
                'right_obj': right_obj,
                'lane_ok': False,
                'lane_w': 0.0,
                'order_ok': False,
            }
        lx = float(getattr(left_obj, 'x', 0))
        rx = float(getattr(right_obj, 'x', 0))
        lane_w = rx - lx
        order_ok = (lx < rx) if self.lane_require_order else True
        lane_ok = order_ok and (self.lane_min_width <= lane_w <= self.lane_max_width)
        return {
            'left_obj': left_obj,
            'right_obj': right_obj,
            'lane_ok': lane_ok,
            'lane_w': lane_w,
            'order_ok': order_ok,
        }

    def enter_phase(self, phase, status=None, detection=True):
        self.phase = phase
        self._phase_start_time = rospy.get_time()
        self._lost_frames = 0
        if phase != 'lane_search':
            self._lane_right_only_streak = 0
        if detection:
            self.publish_detection_for_phase(phase)
        if status:
            self.publish_status(status)

    def switch_to_blind_turn(self):
        self.gait_manager.stop()
        self._blind_steps_done = 0
        self.enter_phase('blind_turn_right', 'blind_turn_right', detection=False)

    def switch_to_line_align(self):
        self.gait_manager.stop()
        self.enter_phase('line_align_right', 'line_align_right', detection=True)

    def switch_to_lane_search(self):
        self.gait_manager.stop()
        self._lane_stable_count = 0
        self._lane_right_only_streak = 0
        self.clear_object_tracking()
        self.enter_phase('lane_search', 'lane_search', detection=True)

    def switch_to_corridor_follow(self):
        self.gait_manager.stop()
        self.try_stop_color_detection()
        self.clear_tag_tracking()
        self._lane_right_only_streak = 0
        self._orange_line_status = ''
        self._orange_line_status_time = 0.0
        self._corridor_lost_since = 0.0
        ok = self.call_service(self.orange_line_start_service, timeout=5.0)
        if not ok:
            self.active = False
            self.phase = 'done'
            self.publish_status('corridor_follow_start_failed')
            return
        self.phase = 'corridor_follow'
        self._phase_start_time = rospy.get_time()
        self.publish_status('corridor_follow_started')
        rospy.loginfo('green_return_place corridor_follow: orange_line_follow started')

    def switch_to_terminal_scan(self, status='terminal_scan', reason='corridor_lost', reset_tag_tracking=False):
        self.call_service(self.orange_line_stop_service, timeout=2.0)
        self.gait_manager.stop()
        self.try_stop_color_detection()
        self.clear_object_tracking()
        if reset_tag_tracking:
            self.clear_tag_tracking()
        else:
            self.reset_tag_phase_counters(reset_eval_seq=False)
        self._lane_stable_count = 0
        self._lane_right_only_streak = 0
        self.phase = 'terminal_scan'
        self._phase_start_time = rospy.get_time()
        self._corridor_lost_since = 0.0
        self.try_enter_color_detection()
        self.publish_detection_for_phase('terminal_scan')
        if not self.try_start_color_detection():
            rospy.logwarn('green_return_place terminal_scan: color_detection restart failed, fallback to tag-only scan')
        self.publish_status(status)
        rospy.loginfo(
            'green_return_place terminal_scan: reason=%s slow_x=%.4f yaw=%d',
            reason,
            self.terminal_x_step,
            self.terminal_yaw,
        )

    def switch_to_tag_approach(self):
        self.gait_manager.stop()
        self._tag_align_stable_count = 0
        self._last_log_time = 0.0
        self.enter_phase('tag_approach', 'tag_approach', detection=True)

    def switch_to_tag_align(self):
        self.gait_manager.stop()
        self._tag_align_stable_count = 0
        self._last_tag_align_eval_seq = 0
        self._last_log_time = 0.0
        self.enter_phase('tag_align', 'tag_align', detection=True)

    def compute_tag_approach_control(self, ex):
        x_step = float(self.tag_approach_forward_x_step)
        yaw = 0
        if abs(ex) > self.tag_approach_x_band_px:
            yaw_mag = abs(self.tag_approach_coarse_turn_yaw)
            if abs(ex) <= self.tag_approach_x_band_px * 1.4:
                yaw_mag = abs(self.tag_approach_fine_turn_yaw)
            yaw = -yaw_mag if ex > 0 else yaw_mag
        return x_step, int(yaw)

    def apply_tag_approach_step(self, ex):
        x_step, yaw = self.compute_tag_approach_control(ex)
        if abs(x_step) <= 1e-6 and int(yaw) == 0:
            self.gait_manager.stop()
            return False
        self.gait_manager.set_step(
            self.tag_approach_dsp,
            float(x_step),
            0.0,
            int(yaw),
            self.tag_approach_walking_param,
            arm_swap=self.tag_approach_arm_swap,
            step_num=0,
        )
        return True

    def compute_tag_align_step(self, ex, ey):
        x_step = 0.0
        y_step = 0.0
        if abs(ey) > self.tag_align_y_tolerance:
            if ey < 0.0:
                x_step = self.tag_align_step_distance * 0.7
            else:
                x_step = -self.tag_align_step_distance * 0.5
        if abs(ex) > self.tag_align_x_tolerance:
            if ex > 0.0:
                y_step = -abs(self.tag_align_side_step_distance)
            else:
                y_step = abs(self.tag_align_side_step_distance)
        return float(x_step), float(y_step)

    def apply_tag_align_step(self, ex, ey):
        x_step, y_step = self.compute_tag_align_step(ex, ey)
        if abs(x_step) <= 1e-6 and abs(y_step) <= 1e-6:
            self.gait_manager.stop()
            return False
        self.gait_manager.set_step(
            self.tag_align_dsp,
            float(x_step),
            float(y_step),
            0,
            self.tag_align_walking_param,
            arm_swap=self.tag_align_arm_swap,
            step_num=1,
        )
        return True

    def execute_place(self):
        self.phase = 'placing'
        self.publish_status('placing')
        self.call_service(self.orange_line_stop_service, timeout=2.0)
        self.try_stop_color_detection()
        self.gait_manager.stop()
        time.sleep(0.2)
        self.gait_manager.disable()
        rospy.loginfo('green_return_place: run place action %s', self.place_action_name)
        self.motion_manager.run_action(self.place_action_name)
        if self.place_wait_sec > 0:
            time.sleep(self.place_wait_sec)
        if self.walk_ready_after_place:
            self.motion_manager.run_action('walk_ready')
        self.active = False
        self.phase = 'done'
        self.publish_status('completed_place')
        return True

    def reset_runtime(self):
        self._phase_start_time = rospy.get_time()
        self._line_stable_count = 0
        self._lane_stable_count = 0
        self._lane_right_only_streak = 0
        self._lost_frames = 0
        self._post_pick_blind_steps_done = 0
        self._blind_steps_done = 0
        self._last_log_time = 0.0
        self._last_tag_log_time = 0.0
        self._last_lane_log_time = 0.0
        self._last_terminal_log_time = 0.0
        self._orange_line_status = ''
        self._orange_line_status_time = 0.0
        self._corridor_lost_since = 0.0
        self.gait_switch_off_reported = False
        self.clear_object_tracking()
        self.clear_tag_tracking()

    def start_service_callback(self, req):
        self.load_config()
        if not self.enabled:
            self.publish_status('disabled')
            return TriggerResponse(success=False, message='green_return_place disabled by config')
        if self.tag_trigger_enabled and AprilTagDetectionArray is None:
            self.publish_status('tag_unavailable')
            return TriggerResponse(success=False, message='apriltag_ros unavailable')
        self.start_rqt_result_view()
        self.active = True
        self.reset_runtime()
        initial_phase = 'line_align_right'
        if self.post_pick_blind_enabled and self.post_pick_blind_steps > 0:
            initial_phase = 'post_pick_blind_right'
        self.phase = initial_phase
        self._phase_start_time = rospy.get_time()
        self.try_enter_color_detection()
        self.publish_detection_for_phase(initial_phase)
        if not self.try_start_color_detection():
            self.active = False
            self.stop_rqt_result_view()
            self.publish_status('waiting_color_detection')
            return TriggerResponse(success=False, message='color_detection/start unavailable')
        self.publish_status('started')
        self.publish_status(initial_phase)
        return TriggerResponse(success=True, message='green_return_place started')

    def stop_service_callback(self, req):
        self.active = False
        self.phase = 'idle'
        self.reset_runtime()
        self.gait_manager.stop()
        self.call_service(self.orange_line_stop_service, timeout=1.0)
        self.try_stop_color_detection()
        self.stop_rqt_result_view()
        self.publish_status('stopped')
        return TriggerResponse(success=True, message='green_return_place stopped')

    def reset_service_callback(self, req):
        self.active = False
        self.phase = 'idle'
        self.reset_runtime()
        self.gait_manager.stop()
        self.call_service(self.orange_line_stop_service, timeout=1.0)
        self.try_stop_color_detection()
        self.stop_rqt_result_view()
        self.publish_status('reset')
        return TriggerResponse(success=True, message='green_return_place reset')

    def check_phase_timeout(self, now):
        if self.phase == 'line_align_right' and self.line_timeout_sec > 0:
            if now - self._phase_start_time > self.line_timeout_sec:
                self.active = False
                self.gait_manager.stop()
                self.try_stop_color_detection()
                self.publish_status('line_align_timeout')
                rospy.logwarn('green_return_place line_align timeout')
                return True
        if self.phase == 'lane_search' and self.lane_timeout_sec > 0:
            if now - self._phase_start_time > self.lane_timeout_sec:
                self.active = False
                self.gait_manager.stop()
                self.try_stop_color_detection()
                self.publish_status('lane_search_timeout')
                rospy.logwarn('green_return_place lane_search timeout')
                return True
        if self.phase == 'corridor_follow' and self.corridor_timeout_sec > 0:
            if now - self._phase_start_time > self.corridor_timeout_sec:
                self.active = False
                self.call_service(self.orange_line_stop_service, timeout=2.0)
                self.publish_status('corridor_follow_timeout')
                rospy.logwarn('green_return_place corridor_follow timeout')
                return True
        if self.phase in ('terminal_scan', 'tag_approach', 'tag_align') and self.terminal_timeout_sec > 0:
            if now - self._phase_start_time > self.terminal_timeout_sec:
                self.active = False
                self.gait_manager.stop()
                self.publish_status('%s_timeout' % self.phase)
                rospy.logwarn('green_return_place %s timeout', self.phase)
                return True
        return False

    def shutdown(self, signum, frame):
        self.running = False
        self.active = False
        self.phase = 'idle'
        self.gait_manager.stop()
        self.stop_rqt_result_view()
        self.call_service(self.orange_line_stop_service, timeout=1.0)
        self.try_stop_color_detection()
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
            fixed_step_allowed = self.fixed_step_motion_allowed()
            if not gait_on and not fixed_step_allowed:
                if not self.gait_switch_off_reported:
                    self.publish_status('gait_switch_off')
                    rospy.logwarn('green_return_place gait_switch=false, closed-loop gait disabled (fixed-step/action allowed)')
                    self.gait_switch_off_reported = True
                self.gait_manager.stop()
            else:
                self.gait_switch_off_reported = False

            if self.check_phase_timeout(now):
                time.sleep(rate_sleep)
                continue

            if self.phase == 'post_pick_blind_right':
                if self._post_pick_blind_steps_done >= self.post_pick_blind_steps:
                    self.switch_to_line_align()
                    time.sleep(rate_sleep)
                    continue
                next_step = self._post_pick_blind_steps_done + 1
                rospy.loginfo(
                    'green_return_post_pick_blind_right: %d/%d cmd_x_step=%.4f cmd_yaw=%d gait_switch=%s',
                    next_step,
                    self.post_pick_blind_steps,
                    self.post_pick_blind_x_step,
                    self.post_pick_blind_yaw,
                    str(gait_on),
                )
                self.gait_manager.set_step(
                    self.post_pick_blind_dsp,
                    float(self.post_pick_blind_x_step),
                    0.0,
                    int(self.post_pick_blind_yaw),
                    self.post_pick_blind_walking_param,
                    arm_swap=self.post_pick_blind_arm_swap,
                    step_num=1,
                )
                self._post_pick_blind_steps_done = next_step

            elif self.phase == 'line_align_right':
                line_obj = self.get_best_align_line()
                if line_obj is None:
                    self._lost_frames += 1
                    self._line_stable_count = 0
                    if now - self._last_log_time >= self.log_throttle_sec:
                        rospy.loginfo(
                            'green_return_line_align: mode=search_right lost=%d/%d cmd_x_step=%.4f cmd_yaw=%d gait_switch=%s',
                            self._lost_frames,
                            self.line_lost_frames,
                            self.line_search_x_step,
                            self.line_right_turn_yaw,
                            str(gait_on),
                        )
                        self._last_log_time = now
                    if gait_on:
                        self.gait_manager.set_step(
                            self.line_rotate_dsp,
                            float(self.line_search_x_step),
                            0.0,
                            int(self.line_right_turn_yaw),
                            self.line_rotate_walking_param,
                            arm_swap=self.line_rotate_arm_swap,
                            step_num=1,
                        )
                    time.sleep(rate_sleep)
                    continue
                self._lost_frames = 0
                ang = float(getattr(line_obj, 'angle', 0.0))
                aerr = ang - self.line_angle_target
                if abs(aerr) <= self.line_angle_tolerance:
                    self._line_stable_count += 1
                else:
                    self._line_stable_count = 0
                yaw = 0
                x_step = 0.0
                if abs(aerr) > self.line_angle_tolerance:
                    yaw = self.line_right_turn_yaw if aerr > 0 else self.line_left_backoff_yaw
                    x_step = self.line_right_turn_x_step if aerr > 0 else 0.0
                if now - self._last_log_time >= self.log_throttle_sec:
                    rospy.loginfo(
                        'green_return_line_align: angle=%.1f target=%.1f err=%.1f stable=%d/%d cmd_x_step=%.4f cmd_yaw=%d gait_switch=%s',
                        ang,
                        self.line_angle_target,
                        aerr,
                        self._line_stable_count,
                        self.line_stable_frames,
                        x_step,
                        int(yaw),
                        str(gait_on),
                    )
                    self._last_log_time = now
                if self._line_stable_count >= self.line_stable_frames:
                    self.switch_to_blind_turn()
                    time.sleep(rate_sleep)
                    continue
                if gait_on:
                    if abs(x_step) > 1e-6 or int(yaw) != 0:
                        self.gait_manager.set_step(
                            self.line_rotate_dsp,
                            float(x_step),
                            0.0,
                            int(yaw),
                            self.line_rotate_walking_param,
                            arm_swap=self.line_rotate_arm_swap,
                            step_num=1,
                        )
                    else:
                        self.gait_manager.stop()

            elif self.phase == 'blind_turn_right':
                if self._blind_steps_done >= self.blind_steps:
                    self.switch_to_lane_search()
                    time.sleep(rate_sleep)
                    continue
                next_step = self._blind_steps_done + 1
                rospy.loginfo(
                    'green_return_blind_turn: %d/%d cmd_x_step=%.4f cmd_yaw=%d gait_switch=%s',
                    next_step,
                    self.blind_steps,
                    self.blind_turn_x_step,
                    self.blind_turn_yaw,
                    str(gait_on),
                )
                self.gait_manager.set_step(
                    self.blind_turn_dsp,
                    float(self.blind_turn_x_step),
                    0.0,
                    int(self.blind_turn_yaw),
                    self.blind_turn_walking_param,
                    arm_swap=self.blind_turn_arm_swap,
                    step_num=1,
                )
                self._blind_steps_done = next_step

            elif self.phase == 'lane_search':
                lane_state = self.evaluate_lane_state()
                left_obj = lane_state['left_obj']
                right_obj = lane_state['right_obj']
                cmd_x_step = self.lane_turn_x_step
                cmd_yaw = int(self.lane_turn_yaw)
                mode = 'search_right'
                left_x = float(getattr(left_obj, 'x', -1.0)) if left_obj is not None else -1.0
                right_x = float(getattr(right_obj, 'x', -1.0)) if right_obj is not None else -1.0
                if left_obj is None and right_obj is None:
                    self._lost_frames += 1
                    self._lane_stable_count = 0
                    self._lane_right_only_streak = 0
                    lane_ok = False
                    lane_w = 0.0
                elif left_obj is None and right_obj is not None:
                    self._lost_frames += 1
                    self._lane_stable_count = 0
                    self._lane_right_only_streak += 1
                    lane_ok = False
                    lane_w = 0.0
                    if self._lane_right_only_streak >= 2:
                        mode = 'right_only_recover_left'
                        cmd_x_step = 0.0
                        cmd_yaw = abs(int(self.lane_turn_yaw))
                elif left_obj is not None and right_obj is None:
                    self._lost_frames += 1
                    self._lane_stable_count = 0
                    self._lane_right_only_streak = 0
                    lane_ok = False
                    lane_w = 0.0
                else:
                    self._lost_frames = 0
                    self._lane_right_only_streak = 0
                    lane_ok = bool(lane_state['lane_ok'])
                    lane_w = float(lane_state['lane_w'])
                    if lane_ok:
                        self._lane_stable_count += 1
                        mode = 'lane_ok'
                    else:
                        self._lane_stable_count = 0
                        mode = 'lane_bad'
                if now - self._last_lane_log_time >= self.log_throttle_sec:
                    rospy.loginfo(
                        'green_return_lane_search: mode=%s lane_ok=%s stable=%d/%d left_x=%.1f right_x=%.1f lane_w=%.1f range=[%.1f,%.1f] lost=%d/%d right_only=%d/2 cmd_x_step=%.4f cmd_yaw=%d gait_switch=%s',
                        mode,
                        str(lane_ok),
                        self._lane_stable_count,
                        self.lane_stable_frames,
                        left_x,
                        right_x,
                        lane_w,
                        self.lane_min_width,
                        self.lane_max_width,
                        self._lost_frames,
                        self.lane_lost_frames,
                        self._lane_right_only_streak,
                        cmd_x_step,
                        cmd_yaw,
                        str(gait_on),
                    )
                    self._last_lane_log_time = now
                if self._lane_stable_count >= self.lane_stable_frames:
                    self.switch_to_corridor_follow()
                    time.sleep(rate_sleep)
                    continue
                if gait_on:
                    self.gait_manager.set_step(
                        self.lane_rotate_dsp,
                        float(cmd_x_step),
                        0.0,
                        int(cmd_yaw),
                        self.lane_rotate_walking_param,
                        arm_swap=self.lane_rotate_arm_swap,
                        step_num=1,
                    )

            elif self.phase == 'corridor_follow':
                status = self._orange_line_status
                if status in self.corridor_lost_statuses:
                    if self._corridor_lost_since <= 0.0:
                        self._corridor_lost_since = now
                    if now - self._corridor_lost_since >= self.corridor_lost_min_sec:
                        rospy.loginfo(
                            'green_return_place: orange_line status=%s for %.2fs -> terminal_scan',
                            status,
                            now - self._corridor_lost_since,
                        )
                        self.switch_to_terminal_scan()
                else:
                    self._corridor_lost_since = 0.0

            elif self.phase == 'terminal_scan':
                lane_state = self.evaluate_lane_state()
                left_obj = lane_state['left_obj']
                right_obj = lane_state['right_obj']
                lane_ok = bool(lane_state['lane_ok'])
                lane_w = float(lane_state['lane_w'])
                left_x = float(getattr(left_obj, 'x', -1.0)) if left_obj is not None else -1.0
                right_x = float(getattr(right_obj, 'x', -1.0)) if right_obj is not None else -1.0
                if lane_ok:
                    self._lane_stable_count += 1
                else:
                    self._lane_stable_count = 0
                if now - self._last_terminal_log_time >= self.terminal_log_throttle_sec:
                    age = (now - self._tag_seen_time) if self._tag_seen_time > 0.0 else -1.0
                    rospy.loginfo(
                        'green_return_terminal_scan: lane_ok=%s stable=%d/%d left_x=%.1f right_x=%.1f lane_w=%.1f range=[%.1f,%.1f] tag_age=%.2f tag_xy=%s cmd_x_step=%.4f cmd_yaw=%d gait_switch=%s',
                        str(lane_ok),
                        self._lane_stable_count,
                        self.terminal_reacquire_lane_stable_frames,
                        left_x,
                        right_x,
                        lane_w,
                        self.lane_min_width,
                        self.lane_max_width,
                        age,
                        str(self.tag_center),
                        self.terminal_x_step,
                        self.terminal_yaw,
                        str(gait_on),
                    )
                    self._last_terminal_log_time = now
                if self.should_enter_tag_approach(now):
                    rospy.loginfo(
                        'green_return_terminal_scan: tag visible stable=%d/%d tag_xy=%s -> tag_approach',
                        self._tag_visible_streak,
                        self.tag_approach_switch_visible_stable_frames,
                        str(self.tag_center),
                    )
                    self.switch_to_tag_approach()
                    time.sleep(rate_sleep)
                    continue
                if self._lane_stable_count >= self.terminal_reacquire_lane_stable_frames:
                    rospy.loginfo(
                        'green_return_terminal_scan: lane recovered stable=%d/%d -> corridor_follow',
                        self._lane_stable_count,
                        self.terminal_reacquire_lane_stable_frames,
                    )
                    self.switch_to_corridor_follow()
                    time.sleep(rate_sleep)
                    continue
                if gait_on:
                    self.gait_manager.set_step(
                        self.terminal_dsp,
                        float(self.terminal_x_step),
                        0.0,
                        int(self.terminal_yaw),
                        self.terminal_walking_param,
                        arm_swap=self.terminal_arm_swap,
                        step_num=0,
                    )

            elif self.phase == 'tag_approach':
                if not self.has_recent_tag(now):
                    lost_age = (now - self._tag_seen_time) if self._tag_seen_time > 0.0 else -1.0
                    rospy.loginfo(
                        'green_return_tag_approach: lost_age=%.2f -> terminal_scan',
                        lost_age,
                    )
                    self.switch_to_terminal_scan(
                        status='tag_lost_back_to_terminal_scan',
                        reason='tag_approach_lost',
                        reset_tag_tracking=True,
                    )
                    time.sleep(rate_sleep)
                    continue
                tcx, tcy = self.tag_center
                if self.should_enter_tag_align(now):
                    rospy.loginfo(
                        'green_return_tag_approach: id=%d x=%.1f y=%.1f ready=%d/%d threshold=%.1f -> tag_align',
                        int(self.tag_id),
                        float(tcx),
                        float(tcy),
                        self._tag_approach_ready_streak,
                        self.tag_approach_pre_align_stable_frames,
                        self.tag_pre_align_y_threshold_px(),
                    )
                    self.switch_to_tag_align()
                    time.sleep(rate_sleep)
                    continue
                tx = self.tag_approach_target_x_px()
                ex = float(tcx) - float(tx)
                cmd_x_step, cmd_yaw = self.compute_tag_approach_control(ex)
                if now - self._last_log_time >= self.log_throttle_sec:
                    rospy.loginfo(
                        'green_return_tag_approach: id=%d x=%.1f y=%.1f ex=%.1f visible=%d/%d ready=%d/%d pre_align_y=%.1f cmd_x_step=%.4f cmd_yaw=%d gait_switch=%s',
                        int(self.tag_id),
                        float(tcx),
                        float(tcy),
                        ex,
                        self._tag_visible_streak,
                        self.tag_approach_switch_visible_stable_frames,
                        self._tag_approach_ready_streak,
                        self.tag_approach_pre_align_stable_frames,
                        self.tag_pre_align_y_threshold_px(),
                        cmd_x_step,
                        cmd_yaw,
                        str(gait_on),
                    )
                    self._last_log_time = now
                if not gait_on:
                    time.sleep(rate_sleep)
                    continue
                self.apply_tag_approach_step(ex)

            elif self.phase == 'tag_align':
                if not self.has_recent_tag(now, self.tag_align_lost_timeout_sec):
                    lost_age = (now - self._tag_seen_time) if self._tag_seen_time > 0.0 else -1.0
                    rospy.loginfo(
                        'green_return_tag_align: lost_age=%.2f -> terminal_scan',
                        lost_age,
                    )
                    self.switch_to_terminal_scan(
                        status='tag_lost_back_to_terminal_scan',
                        reason='tag_align_lost',
                        reset_tag_tracking=True,
                    )
                    time.sleep(rate_sleep)
                    continue
                tcx, tcy = self.tag_center
                tx = self.tag_align_target_x_px()
                ty = self.tag_align_target_y_px()
                ex = float(tcx) - float(tx)
                ey = float(tcy) - float(ty)
                if self._tag_sample_seq != self._last_tag_align_eval_seq:
                    if abs(ex) <= self.tag_align_x_tolerance and abs(ey) <= self.tag_align_y_tolerance:
                        self._tag_align_stable_count += 1
                    else:
                        self._tag_align_stable_count = 0
                    self._last_tag_align_eval_seq = self._tag_sample_seq
                cmd_x_step, cmd_y_step = self.compute_tag_align_step(ex, ey)
                if now - self._last_log_time >= self.log_throttle_sec:
                    rospy.loginfo(
                        'green_return_tag_align: id=%d x=%.1f y=%.1f ex=%.1f ey=%.1f stable=%d/%d cmd_x_step=%.4f cmd_y_step=%.4f gait_switch=%s',
                        int(self.tag_id),
                        float(tcx),
                        float(tcy),
                        ex,
                        ey,
                        self._tag_align_stable_count,
                        self.tag_align_stable_frames,
                        cmd_x_step,
                        cmd_y_step,
                        str(gait_on),
                    )
                    self._last_log_time = now
                if self.tag_aligned_to_place(now):
                    self.gait_manager.stop()
                    self.publish_status('tag_aligned')
                    self.execute_place()
                    time.sleep(rate_sleep)
                    continue
                if not gait_on:
                    time.sleep(rate_sleep)
                    continue
                self.apply_tag_align_step(ex, ey)

            time.sleep(rate_sleep)

        self.gait_manager.stop()
        self.stop_rqt_result_view()
        self.call_service(self.orange_line_stop_service, timeout=1.0)
        self.try_stop_color_detection()


if __name__ == '__main__':
    try:
        GreenReturnPlaceNode('green_return_place').run()
    except Exception as e:
        rospy.logerr('green_return_place initialization error: %s', e)
