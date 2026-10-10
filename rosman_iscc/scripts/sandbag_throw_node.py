#!/usr/bin/env python3
# encoding: utf-8
# @Author: Dejin Xuan
# @Date: 2026.4.5
# ISCC 拾取与投掷沙包节点：紫色目标对齐 + pick + 右线角度对齐 + Tag对齐 + throw + 左线回轨
import signal
import time

import rospy
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger, TriggerResponse

from rosman_example.color_common import Common
from rosman_interfaces.msg import ColorDetect, ObjectsInfo
from rosman_sdk import common
try:
    from apriltag_ros.msg import AprilTagDetectionArray
except Exception:
    AprilTagDetectionArray = None


class SandbagThrowNode(Common):
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
        self._tag_seen_time = 0.0
        self._last_tag_msg_time = 0.0
        self.tag_sub = None

        super().__init__(name, self.head_pan_init, self.head_tilt_look_down)
        signal.signal(signal.SIGINT, self.shutdown)

        self.process_size = rospy.get_param('/image_processing/process_size', [160, 120])
        self.image_width = int(self.process_size[0]) if len(self.process_size) > 0 else 160
        self.image_height = int(self.process_size[1]) if len(self.process_size) > 1 else 120
        self.image_process_size = [self.image_width, self.image_height]

        self._fine_steps = 0
        self._start_time = 0.0
        self._last_log_time = 0.0
        self._last_line_log_time = 0.0
        self._last_tag_log_time = 0.0
        self._last_tag_detect_log_time = 0.0
        self._last_rejoin_log_time = 0.0
        self._picked_once = False
        self._line_stable_count = 0
        self._tag_stable_count = 0
        self._tag_switch_visible_streak = 0
        self._rejoin_stable_count = 0
        self._rejoin_left_only_streak = 0
        self._rejoin_green_handoff_count = 0
        self._post_pick_forward_steps_done = 0
        self._post_pick_blind_steps_done = 0
        self._rejoin_blind_steps_done = 0
        self._tag_seen_once_after_pick = False
        self._line_parallel_tag_unlocked = False
        self._last_line_adjust_start_monotonic = 0.0
        self._last_tag_adjust_start_monotonic = 0.0
        self.gait_switch_off_reported = False

        self.load_config()
        self.setup_ros_interface()
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.object_callback)

        self.motion_manager.run_action('walk_ready')
        common.loginfo('sandbag_throw initialized')

    def load_config(self):
        cfg = rospy.get_param('/tasks/sandbag_throw', {})
        align = cfg.get('alignment', {})
        motion = cfg.get('motion', {})
        fine_actions = cfg.get('fine_actions', {})
        pick = cfg.get('pick', {})

        self.enabled = bool(cfg.get('enabled', True))
        self.color = str(cfg.get('color', 'purple'))
        self.detect_type = str(cfg.get('detect_type', 'side'))
        self.roi = cfg.get('roi', [0.2, 1.0, 0.15, 0.85])
        self.min_area = int(cfg.get('min_area', 10))

        # 目标点（像素与比例两种写法）
        self.target_x_pixel = float(align.get('target_x_pixel', 0.5))
        self.target_y_pixel = float(align.get('target_y_pixel', 300))
        self.target_x_offset = float(align.get('target_x_offset', 0.0))
        self.target_y_offset = float(align.get('target_y_offset', 0.0))

        # 最终到位容差
        self.x_tolerance = float(align.get('x_tolerance', 20))
        self.y_tolerance = float(align.get('y_tolerance', 20))
        # 粗调/精调分隔阈值（按轴独立）
        self.x_band_px = float(align.get('x_band_px', 40))
        self.y_band_px = float(align.get('y_band_px', 40))
        self.fine_tune_max_steps = int(align.get('fine_tune_max_steps', 8))

        # 粗调步态参数
        self.coarse_step_distance = float(motion.get('coarse_step_distance', 0.008))
        self.coarse_turn_yaw = int(motion.get('coarse_turn_yaw', 4))
        self.coarse_fine_yaw = int(motion.get('coarse_fine_yaw', 2))
        self.coarse_dsp = motion.get('coarse_dsp', [320, 0.25, 0.023])
        self.coarse_arm_swap = int(motion.get('coarse_arm_swap', 25))
        self.coarse_walking_param = self.gait_manager.get_gait_param()
        wp_updates = motion.get('coarse_walking_param_updates', {})
        if isinstance(wp_updates, dict):
            self.coarse_walking_param.update(wp_updates)

        # 精调动作组（名称由配置决定）
        self.action_forward = str(fine_actions.get('forward', 'forward_one_step'))
        self.action_back = str(fine_actions.get('back', 'back_step_1'))
        self.action_move_left = str(fine_actions.get('move_left', 'move_left_1'))
        self.action_move_right = str(fine_actions.get('move_right', 'move_right'))

        # pick
        self.pick_action_name = str(pick.get('action_name', 'pick'))
        self.pick_wait_sec = float(pick.get('wait_sec_after_pick', 3.0))
        self.walk_ready_after_pick = bool(pick.get('walk_ready_after_pick', True))

        # pick 后先按固定步态向前走若干步，再进入盲右转/找线。
        post_pick_forward = cfg.get('post_pick_forward', {})
        self.post_pick_forward_enabled = bool(post_pick_forward.get('enabled', False))
        self.post_pick_forward_steps = int(post_pick_forward.get('steps', 0))
        self.post_pick_forward_yaw = int(post_pick_forward.get('yaw', 0))
        self.post_pick_forward_x_step = float(post_pick_forward.get('x_step', 0.01))
        self.post_pick_forward_y_step = float(post_pick_forward.get('y_step', 0.0))
        self.post_pick_forward_dsp = post_pick_forward.get('dsp', self.coarse_dsp)
        self.post_pick_forward_arm_swap = int(post_pick_forward.get('arm_swap', self.coarse_arm_swap))
        self.post_pick_forward_walking_param = self.gait_manager.get_gait_param()
        post_pick_forward_wp_updates = post_pick_forward.get('walking_param_updates', {})
        if isinstance(post_pick_forward_wp_updates, dict):
            self.post_pick_forward_walking_param.update(post_pick_forward_wp_updates)

        # pick 后先盲右转，避免刚抓完时横线/旧 tag 姿态造成误判。
        post_pick_blind = cfg.get('post_pick_blind_right', {})
        self.post_pick_blind_enabled = bool(post_pick_blind.get('enabled', True))
        self.post_pick_blind_steps = int(post_pick_blind.get('steps', 0))
        self.post_pick_blind_yaw = int(post_pick_blind.get('yaw', -10))
        self.post_pick_blind_x_step = float(post_pick_blind.get('x_step', 0.002))
        self.post_pick_blind_y_step = float(post_pick_blind.get('y_step', 0.0))

        # 捡后对视野中的目标橙线做角度对齐。
        line_cfg = cfg.get('line_align', {})
        self.line_align_enabled = bool(line_cfg.get('enabled', True))
        self.align_color = str(line_cfg.get('align_color', 'orange'))
        self.align_detect_type = str(line_cfg.get('align_detect_type', 'side'))
        self.align_roi = line_cfg.get('align_roi', [0.0, 0.8, 0.0, 1.0])
        self.line_min_area = int(line_cfg.get('min_area', 10))
        self.line_angle_target = float(line_cfg.get('angle_target', 0.0))
        self.line_angle_tolerance = float(line_cfg.get('angle_tolerance', 8.0))
        self.line_stable_frames = int(line_cfg.get('stable_frames', 3))
        self.line_align_timeout_sec = float(line_cfg.get('timeout_sec', 8.0))
        self.line_lost_frames = int(line_cfg.get('lost_frames_to_search', 8))
        self.line_right_turn_yaw = int(line_cfg.get('right_turn_yaw', -5))
        self.line_left_backoff_yaw = int(line_cfg.get('left_backoff_yaw', 3))
        # 仅右转找线时给微前进，左回调时固定 x_step=0
        self.line_right_turn_x_step = float(line_cfg.get('right_turn_x_step', 0.003))
        self.search_forward_x_step = float(cfg.get('search_forward_x_step', self.line_right_turn_x_step))
        self.line_rotate_dsp = line_cfg.get('rotate_dsp', self.coarse_dsp)
        self.line_rotate_arm_swap = int(line_cfg.get('rotate_arm_swap', self.coarse_arm_swap))
        self.line_min_adjust_interval_sec = max(0.0, float(line_cfg.get('min_adjust_interval_sec', 0.6)))
        self.line_rotate_walking_param = self.gait_manager.get_gait_param()
        line_wp_updates = line_cfg.get('rotate_walking_param_updates', {})
        if isinstance(line_wp_updates, dict):
            self.line_rotate_walking_param.update(line_wp_updates)

        # Tag 对齐
        tag_cfg = cfg.get('tag_align', {})
        self.tag_align_enabled = bool(tag_cfg.get('enabled', True))
        self.tag_topic = str(tag_cfg.get('topic', '/tag_detections'))
        self.tag_target_id = int(tag_cfg.get('target_id', -1))
        self.tag_target_x_pixel = float(tag_cfg.get('target_x_pixel', 0.5))
        self.tag_target_y_pixel = float(tag_cfg.get('target_y_pixel', 300))
        self.tag_frame_width = int(tag_cfg.get('frame_width', 640))
        self.tag_frame_height = int(tag_cfg.get('frame_height', 480))
        # Tag 检测 ROI（比例）：[y_min, y_max, x_min, x_max]；默认全视野
        self.tag_roi = tag_cfg.get('roi', [0.0, 1.0, 0.0, 1.0])
        self.tag_x_tolerance = float(tag_cfg.get('x_tolerance', 20))
        self.tag_y_tolerance = float(tag_cfg.get('y_tolerance', 20))
        self.tag_stable_frames = int(tag_cfg.get('stable_frames', 3))
        self.tag_switch_stable_frames = int(tag_cfg.get('switch_stable_frames', 1))
        self.tag_timeout_sec = float(tag_cfg.get('timeout_sec', 10.0))
        self.tag_lost_timeout_sec = float(tag_cfg.get('lost_timeout_sec', 1.0))
        self.tag_step_distance = float(tag_cfg.get('coarse_step_distance', self.coarse_step_distance))
        self.tag_side_step_distance = float(tag_cfg.get('coarse_side_step_distance', self.tag_step_distance))
        self.tag_dsp = tag_cfg.get('coarse_dsp', self.coarse_dsp)
        self.tag_arm_swap = int(tag_cfg.get('coarse_arm_swap', self.coarse_arm_swap))
        self.tag_min_adjust_interval_sec = max(0.0, float(tag_cfg.get('min_adjust_interval_sec', 0.6)))
        self.tag_walking_param = self.gait_manager.get_gait_param()
        tag_wp_updates = tag_cfg.get('coarse_walking_param_updates', {})
        if isinstance(tag_wp_updates, dict):
            self.tag_walking_param.update(tag_wp_updates)

        # throw
        throw_cfg = cfg.get('throw', {})
        self.throw_enabled = bool(throw_cfg.get('enabled', True))
        self.throw_action_name = str(throw_cfg.get('action_name', 'throw'))
        self.throw_wait_sec = float(throw_cfg.get('wait_sec_after_throw', 1.0))
        self.walk_ready_after_throw = bool(throw_cfg.get('walk_ready_after_throw', True))

        # 投掷后左转回轨（双线可接管判据）
        rejoin_cfg = cfg.get('rejoin_left', {})
        self.rejoin_enabled = bool(rejoin_cfg.get('enabled', True))
        self.rejoin_left_color = str(rejoin_cfg.get('left_color', 'orange'))
        self.rejoin_left_detect_type = str(rejoin_cfg.get('left_detect_type', 'side'))
        self.rejoin_left_roi = rejoin_cfg.get('left_roi', [0.4, 1.0, 0.0, 0.48])
        self.rejoin_right_color = str(rejoin_cfg.get('right_color', 'orange'))
        self.rejoin_right_detect_type = str(rejoin_cfg.get('right_detect_type', 'side'))
        self.rejoin_right_roi = rejoin_cfg.get('right_roi', [0.4, 1.0, 0.52, 1.0])
        self.rejoin_min_area = int(rejoin_cfg.get('min_area', 10))
        self.rejoin_left_turn_yaw = int(rejoin_cfg.get('left_turn_yaw', 4))
        self.rejoin_left_turn_x_step = float(rejoin_cfg.get('left_turn_x_step', 0.0))
        self.rejoin_blind_turn_steps = int(rejoin_cfg.get('blind_turn_steps', 6))
        self.rejoin_blind_turn_yaw = int(rejoin_cfg.get('blind_turn_yaw', self.rejoin_left_turn_yaw))
        self.rejoin_blind_turn_x_step = float(rejoin_cfg.get('blind_turn_x_step', 0.0))
        self.rejoin_stable_frames = int(rejoin_cfg.get('stable_frames', 3))
        self.rejoin_timeout_sec = float(rejoin_cfg.get('timeout_sec', 10.0))
        self.rejoin_lost_frames = int(rejoin_cfg.get('lost_frames_to_search', 8))
        self.rejoin_min_lane_width = float(rejoin_cfg.get('min_lane_width_px', 120))
        self.rejoin_max_lane_width = float(rejoin_cfg.get('max_lane_width_px', 520))
        self.rejoin_require_order = bool(rejoin_cfg.get('require_left_right_order', True))
        self.rejoin_log_throttle_sec = float(rejoin_cfg.get('log_throttle_sec', 0.2))
        self.rejoin_rotate_dsp = rejoin_cfg.get('rotate_dsp', self.line_rotate_dsp)
        self.rejoin_rotate_arm_swap = int(rejoin_cfg.get('rotate_arm_swap', self.line_rotate_arm_swap))
        self.rejoin_rotate_walking_param = self.gait_manager.get_gait_param()
        rejoin_wp_updates = rejoin_cfg.get('rotate_walking_param_updates', {})
        if isinstance(rejoin_wp_updates, dict):
            self.rejoin_rotate_walking_param.update(rejoin_wp_updates)
        self.rejoin_green_handoff_enabled = bool(rejoin_cfg.get('green_handoff_enabled', True))
        self.rejoin_green_color = str(rejoin_cfg.get('green_color', 'green_pick'))
        self.rejoin_green_detect_type = str(rejoin_cfg.get('green_detect_type', 'side'))
        self.rejoin_green_roi = rejoin_cfg.get('green_roi', [0, 0.8, 0.15, 0.85])
        self.rejoin_green_min_area = int(rejoin_cfg.get('green_min_area', 5))
        self.rejoin_green_min_x = float(rejoin_cfg.get('green_min_x', 0.5))
        self.rejoin_green_stable_frames = int(rejoin_cfg.get('green_stable_frames', 1))

        # 安全与日志
        self.align_timeout_sec = float(cfg.get('align_timeout_sec', 20.0))
        self.lost_frames_to_search = int(cfg.get('lost_frames_to_search', 10))
        self.log_throttle_sec = float(cfg.get('align_log_throttle_sec', 0.2))
        self._lost_frames = 0

    def tag_roi_px(self):
        y_min = int(self.parse_ratio(self.tag_roi[0]) * self.tag_frame_height)
        y_max = int(self.parse_ratio(self.tag_roi[1]) * self.tag_frame_height)
        x_min = int(self.parse_ratio(self.tag_roi[2]) * self.tag_frame_width)
        x_max = int(self.parse_ratio(self.tag_roi[3]) * self.tag_frame_width)
        return y_min, y_max, x_min, x_max

    def setup_ros_interface(self):
        self.status_pub = rospy.Publisher('/sandbag_throw/status', String, queue_size=1)
        self.active_pub = rospy.Publisher('/sandbag_throw/active', Bool, queue_size=1)
        self.start_service = rospy.Service('/sandbag_throw/start', Trigger, self.start_service_callback)
        self.stop_service = rospy.Service('/sandbag_throw/stop', Trigger, self.stop_service_callback)
        self.reset_service = rospy.Service('/sandbag_throw/reset', Trigger, self.reset_service_callback)

    def is_gait_switch_on(self, default=True):
        v = rospy.get_param('/gait_switch', default)
        if isinstance(v, bool):
            return v
        return str(v).lower() in ('true', '1', 'yes', 'on')

    def fixed_step_motion_allowed(self):
        if self.phase in ('post_pick_forward', 'post_pick_blind_right'):
            return True
        if self.phase == 'rejoin_left' and self._rejoin_blind_steps_done < self.rejoin_blind_turn_steps:
            return True
        return False

    def reset_post_pick_tag_gate(self):
        self._line_parallel_tag_unlocked = False
        self._last_line_adjust_start_monotonic = 0.0
        self._last_tag_adjust_start_monotonic = 0.0

    def is_line_align_tag_detection_unlocked(self):
        return (not self.line_align_enabled) or self._line_parallel_tag_unlocked

    def get_adjust_interval_state(self, phase_name):
        if phase_name == 'line_align_right':
            return self.line_min_adjust_interval_sec, self._last_line_adjust_start_monotonic
        if phase_name == 'tag_align':
            return self.tag_min_adjust_interval_sec, self._last_tag_adjust_start_monotonic
        return 0.0, 0.0

    def note_adjust_command_start(self, phase_name, started_at=None):
        if started_at is None:
            started_at = time.monotonic()
        if phase_name == 'line_align_right':
            self._last_line_adjust_start_monotonic = float(started_at)
        elif phase_name == 'tag_align':
            self._last_tag_adjust_start_monotonic = float(started_at)

    def wait_for_adjust_interval(self, phase_name):
        min_interval_sec, last_started_at = self.get_adjust_interval_state(phase_name)
        if min_interval_sec <= 0.0 or last_started_at <= 0.0:
            return True
        deadline = float(last_started_at) + float(min_interval_sec)
        while self.running and self.active and self.phase == phase_name:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return True
            time.sleep(min(remaining, 0.02))
        return False

    def unlock_tag_detection_after_parallel(self, angle):
        if self._line_parallel_tag_unlocked:
            return self.tag_sub is not None
        self._line_parallel_tag_unlocked = True
        tag_detection_enabled = self.sync_tag_detection('line_align_right')
        self.publish_status('line_parallel_ready')
        rospy.loginfo(
            'line parallel ready, tag detection %s: angle=%.1f tolerance=%.1f',
            'enabled' if tag_detection_enabled else 'skipped',
            float(angle),
            float(self.line_angle_tolerance),
        )
        return tag_detection_enabled

    def object_callback(self, msg):
        self.objects_info = msg.data

    def clear_object_tracking(self):
        self.objects_info = []

    def needs_tag_detection(self, phase_name=None):
        phase = phase_name or self.phase
        if phase == 'tag_align':
            return True
        if phase == 'line_align_right':
            return self.is_line_align_tag_detection_unlocked()
        return False

    def clear_tag_tracking(self, reset_seen_once=True):
        self._tag_stable_count = 0
        self._tag_switch_visible_streak = 0
        if reset_seen_once:
            self._tag_seen_once_after_pick = False
        self.tag_center = None
        self._tag_seen_time = 0.0
        self._last_tag_msg_time = 0.0

    def enable_tag_detection(self, phase_name=None):
        phase = phase_name or self.phase
        if not self.tag_align_enabled:
            self.disable_tag_detection(reset_tracking=True)
            return False
        if not self.needs_tag_detection(phase):
            self.disable_tag_detection(reset_tracking=True)
            return False
        if AprilTagDetectionArray is None:
            rospy.logwarn('apriltag_ros.msg unavailable, tag_align will be skipped')
            self.tag_align_enabled = False
            self.disable_tag_detection(reset_tracking=True)
            return False
        if self.tag_sub is None:
            self.clear_tag_tracking(reset_seen_once=False)
            self.tag_sub = rospy.Subscriber(self.tag_topic, AprilTagDetectionArray, self.tag_callback)
            rospy.loginfo('sandbag_throw tag detection enabled: phase=%s topic=%s', phase, self.tag_topic)
        return True

    def disable_tag_detection(self, reset_tracking=True):
        if self.tag_sub is not None:
            try:
                self.tag_sub.unregister()
            except Exception as e:
                rospy.logwarn('sandbag_throw stop tag subscriber failed: %s', e)
            self.tag_sub = None
            rospy.loginfo('sandbag_throw tag detection disabled: phase=%s', self.phase)
        if reset_tracking:
            self.clear_tag_tracking()

    def sync_tag_detection(self, phase_name=None, reset_tracking=True):
        phase = phase_name or self.phase
        if self.needs_tag_detection(phase):
            return self.enable_tag_detection(phase)
        self.disable_tag_detection(reset_tracking=reset_tracking)
        return False

    def tag_callback(self, msg):
        if not self.active or not self.needs_tag_detection():
            return
        now = rospy.get_time()
        self._last_tag_msg_time = now
        best = None
        best_id = None
        y_min, y_max, x_min, x_max = self.tag_roi_px()
        for det in msg.detections:
            ids = list(getattr(det, 'id', []))
            if self.tag_target_id >= 0 and self.tag_target_id not in ids:
                continue
            corners = list(getattr(det, 'corners', []))
            if len(corners) >= 4:
                cx = sum(float(c.x) for c in corners[:4]) / 4.0
                cy = sum(float(c.y) for c in corners[:4]) / 4.0
                # 只保留 ROI 内 Tag（先给全视野 [0,1,0,1]）
                if cx < x_min or cx > x_max or cy < y_min or cy > y_max:
                    continue
                best = (cx, cy)
                best_id = ids[0] if len(ids) > 0 else -1
                break
        if best is not None:
            self.tag_center = best
            self._tag_seen_time = now
            self._tag_switch_visible_streak += 1
            if self.log_throttle_sec <= 0 or (self._tag_seen_time - self._last_tag_detect_log_time) >= self.log_throttle_sec:
                rospy.loginfo('tag x=%.1f y=%.1f', best[0], best[1])
                self._last_tag_detect_log_time = self._tag_seen_time
        else:
            self._tag_switch_visible_streak = 0

    def publish_status(self, status):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        active_msg = Bool()
        active_msg.data = self.active
        self.active_pub.publish(active_msg)

    def check_phase_timeout(self, now):
        if self.phase == 'line_align_right' and self.line_align_timeout_sec > 0:
            if (now - self._start_time) > self.line_align_timeout_sec:
                self.active = False
                self.gait_manager.stop()
                self.publish_status('line_align_timeout')
                rospy.logwarn('sandbag_throw line align timeout')
                return True
        if self.phase == 'tag_align' and self.tag_timeout_sec > 0:
            if (now - self._start_time) > self.tag_timeout_sec:
                self.active = False
                self.gait_manager.stop()
                self.publish_status('tag_align_timeout')
                rospy.logwarn('sandbag_throw tag align timeout')
                return True
        if self.phase == 'rejoin_left' and self.rejoin_timeout_sec > 0:
            if (now - self._start_time) > self.rejoin_timeout_sec:
                self.active = False
                self.gait_manager.stop()
                self.publish_status('rejoin_timeout')
                rospy.logwarn('sandbag_throw rejoin timeout')
                return True
        return False

    def parse_ratio(self, value):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            s = value.strip()
            if '/' in s:
                left, right = s.split('/', 1)
                return float(left.strip()) / float(right.strip())
            return float(s)
        raise ValueError('invalid ratio value: %s' % value)

    def build_purple_param(self):
        param = ColorDetect()
        param.color_name = self.color
        param.use_name = True
        param.detect_type = self.detect_type
        param.image_process_size = self.image_process_size
        param.roi.y_min = int(self.parse_ratio(self.roi[0]) * self.image_height)
        param.roi.y_max = int(self.parse_ratio(self.roi[1]) * self.image_height)
        param.roi.x_min = int(self.parse_ratio(self.roi[2]) * self.image_width)
        param.roi.x_max = int(self.parse_ratio(self.roi[3]) * self.image_width)
        param.min_area = self.min_area
        param.max_area = self.image_width * self.image_height
        return param

    def build_align_line_param(self):
        param = ColorDetect()
        param.color_name = self.align_color
        param.use_name = True
        param.detect_type = self.align_detect_type
        param.image_process_size = self.image_process_size
        param.roi.y_min = int(self.parse_ratio(self.align_roi[0]) * self.image_height)
        param.roi.y_max = int(self.parse_ratio(self.align_roi[1]) * self.image_height)
        param.roi.x_min = int(self.parse_ratio(self.align_roi[2]) * self.image_width)
        param.roi.x_max = int(self.parse_ratio(self.align_roi[3]) * self.image_width)
        param.min_area = self.line_min_area
        param.max_area = self.image_width * self.image_height
        return param

    def build_rejoin_left_param(self):
        param = ColorDetect()
        param.color_name = self.rejoin_left_color
        param.use_name = True
        param.detect_type = self.rejoin_left_detect_type
        param.image_process_size = self.image_process_size
        param.roi.y_min = int(self.parse_ratio(self.rejoin_left_roi[0]) * self.image_height)
        param.roi.y_max = int(self.parse_ratio(self.rejoin_left_roi[1]) * self.image_height)
        param.roi.x_min = int(self.parse_ratio(self.rejoin_left_roi[2]) * self.image_width)
        param.roi.x_max = int(self.parse_ratio(self.rejoin_left_roi[3]) * self.image_width)
        param.min_area = self.rejoin_min_area
        param.max_area = self.image_width * self.image_height
        return param

    def build_rejoin_right_param(self):
        param = ColorDetect()
        param.color_name = self.rejoin_right_color
        param.use_name = True
        param.detect_type = self.rejoin_right_detect_type
        param.image_process_size = self.image_process_size
        param.roi.y_min = int(self.parse_ratio(self.rejoin_right_roi[0]) * self.image_height)
        param.roi.y_max = int(self.parse_ratio(self.rejoin_right_roi[1]) * self.image_height)
        param.roi.x_min = int(self.parse_ratio(self.rejoin_right_roi[2]) * self.image_width)
        param.roi.x_max = int(self.parse_ratio(self.rejoin_right_roi[3]) * self.image_width)
        param.min_area = self.rejoin_min_area
        param.max_area = self.image_width * self.image_height
        return param

    def build_rejoin_green_param(self):
        param = ColorDetect()
        param.color_name = self.rejoin_green_color
        param.use_name = True
        param.detect_type = self.rejoin_green_detect_type
        param.image_process_size = self.image_process_size
        param.roi.y_min = int(self.parse_ratio(self.rejoin_green_roi[0]) * self.image_height)
        param.roi.y_max = int(self.parse_ratio(self.rejoin_green_roi[1]) * self.image_height)
        param.roi.x_min = int(self.parse_ratio(self.rejoin_green_roi[2]) * self.image_width)
        param.roi.x_max = int(self.parse_ratio(self.rejoin_green_roi[3]) * self.image_width)
        param.min_area = self.rejoin_green_min_area
        param.max_area = self.image_width * self.image_height
        return param

    def build_detection_params_for_phase(self, phase_name=None):
        phase = phase_name or self.phase
        if phase in ('align_pick', 'coarse_align', 'fine_align', 'picking'):
            return [self.build_purple_param()]
        if phase in ('post_pick_forward', 'post_pick_blind_right', 'line_align_right', 'tag_align', 'throwing'):
            if self.line_align_enabled:
                return [self.build_align_line_param()]
            return []
        if phase == 'rejoin_left':
            if self.rejoin_enabled:
                params = [self.build_rejoin_left_param(), self.build_rejoin_right_param()]
                if self.rejoin_green_handoff_enabled:
                    params.append(self.build_rejoin_green_param())
                return params
            return []
        return []

    def publish_detection_for_phase(self, phase_name=None):
        phase = phase_name or self.phase
        detect_params = self.build_detection_params_for_phase(phase)
        self.detect_pub.publish(detect_params)
        rospy.loginfo('sandbag_throw detect phase=%s roi_count=%d', phase, len(detect_params))
        return detect_params

    def try_enter_color_detection(self):
        try:
            self.enter_func(None)
            return True
        except Exception as e:
            rospy.logwarn('sandbag_throw enter color_detection unavailable: %s', e)
            return False

    def try_start_color_detection(self):
        try:
            self.start_srv_callback(None)
            return True
        except Exception as e:
            rospy.logwarn('sandbag_throw start color_detection failed: %s', e)
            return False

    def try_stop_color_detection(self):
        try:
            self.stop_srv_callback(None)
            return True
        except Exception as e:
            rospy.logwarn('sandbag_throw stop color_detection failed: %s', e)
            return False

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

    def get_best_purple(self):
        best = None
        best_y = -1
        for obj in self.objects_info:
            if getattr(obj, 'label', '') == self.color and getattr(obj, 'type', '') == self.detect_type:
                yv = self.side_y_max(obj, fallback_y=int(getattr(obj, 'y', 0)))
                if yv > best_y:
                    best_y = yv
                    best = obj
        return best

    def align_line_span_x(self, obj):
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
            span_x = self.align_line_span_x(obj)
            yv = self.side_y_max(obj, fallback_y=int(getattr(obj, 'y', 0)))
            if span_x > best_span or (abs(span_x - best_span) < 1e-6 and yv > best_y):
                best_span = span_x
                best_y = yv
                best = obj
        return best

    def get_best_rejoin_side(self, color_name, detect_type, side):
        best = None
        best_y = -1
        mid_x = self.image_width * 0.5
        for obj in self.objects_info:
            if getattr(obj, 'label', '') != color_name:
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

    def get_best_rejoin_green(self):
        best = None
        best_y = -1
        for obj in self.objects_info:
            if getattr(obj, 'label', '') != self.rejoin_green_color:
                continue
            if getattr(obj, 'type', '') != self.rejoin_green_detect_type:
                continue
            oy = self.side_y_max(obj, fallback_y=int(getattr(obj, 'y', 0)))
            if oy > best_y:
                best_y = oy
                best = obj
        return best

    def rejoin_green_min_x_px(self, obj=None):
        frame_w = self.image_width
        if obj is not None:
            frame_w = int(getattr(obj, 'width', self.image_width)) or self.image_width
        x = float(self.rejoin_green_min_x)
        if x <= 1.0:
            return x * float(frame_w)
        return x

    def target_x_px(self, obj):
        frame_w = int(getattr(obj, 'width', self.image_width)) or self.image_width
        if self.target_x_pixel <= 1.0:
            tx = self.target_x_pixel * float(frame_w)
        else:
            tx = self.target_x_pixel
        tx += self.target_x_offset
        return float(tx)

    def target_y_px(self, obj):
        frame_h = int(getattr(obj, 'height', self.image_height)) or self.image_height
        if self.target_y_pixel <= 1.0:
            ty = self.target_y_pixel * float(frame_h)
        else:
            ty = self.target_y_pixel
        ty += self.target_y_offset
        return float(ty)

    def tag_target_x_px(self):
        if self.tag_target_x_pixel <= 1.0:
            return self.tag_target_x_pixel * float(self.tag_frame_width)
        return self.tag_target_x_pixel

    def tag_target_y_px(self):
        if self.tag_target_y_pixel <= 1.0:
            return self.tag_target_y_pixel * float(self.tag_frame_height)
        return self.tag_target_y_pixel

    def has_recent_tag(self, now):
        if self.tag_center is None or self._tag_seen_time <= 0.0:
            return False
        return self.tag_lost_timeout_sec <= 0.0 or (now - self._tag_seen_time) <= self.tag_lost_timeout_sec

    def clear_stale_tag_tracking(self):
        self.clear_tag_tracking()

    def should_switch_to_tag_align(self, now):
        if self.line_align_enabled and not self.is_line_align_tag_detection_unlocked():
            return False
        if not self.tag_align_enabled or not self.has_recent_tag(now):
            return False
        return self._tag_switch_visible_streak >= max(1, int(self.tag_switch_stable_frames))

    def switch_to_tag_align(self, reason):
        self.gait_manager.stop()
        self.phase = 'tag_align'
        self._start_time = rospy.get_time()
        self._tag_stable_count = 0
        self._tag_seen_once_after_pick = True
        self.sync_tag_detection('tag_align', reset_tracking=False)
        self.publish_detection_for_phase('tag_align')
        self.publish_status('switch_to_tag_align')
        rospy.loginfo(
            'switch_to_tag_align: reason=%s visible_streak=%d threshold=%d tag_xy=%s',
            reason,
            self._tag_switch_visible_streak,
            self.tag_switch_stable_frames,
            str(self.tag_center),
        )

    def enter_post_pick_search(self, reason):
        now = rospy.get_time()
        self._start_time = now
        self._lost_frames = 0
        self._line_stable_count = 0
        self._tag_stable_count = 0
        self.sync_tag_detection('line_align_right' if self.line_align_enabled else 'tag_align')
        if self.should_switch_to_tag_align(now):
            self.switch_to_tag_align(reason + '_tag_visible')
            return
        elif self.line_align_enabled:
            self.phase = 'line_align_right'
            self.publish_status('line_align_right')
        elif self.tag_align_enabled:
            self.phase = 'tag_align'
            self.publish_status('tag_align')
        elif self.throw_enabled:
            self.phase = 'throwing'
        else:
            self.phase = 'done'
            self.active = False
        self.publish_detection_for_phase(self.phase)

    def begin_post_pick_forward(self):
        self.phase = 'post_pick_forward'
        self._post_pick_forward_steps_done = 0
        self._last_line_log_time = 0.0
        self.sync_tag_detection('post_pick_forward')
        self.publish_detection_for_phase('post_pick_forward')
        self.publish_status('post_pick_forward')

    def begin_post_pick_blind_right(self):
        self.phase = 'post_pick_blind_right'
        self._post_pick_blind_steps_done = 0
        self._last_line_log_time = 0.0
        self.sync_tag_detection('post_pick_blind_right')
        self.publish_detection_for_phase('post_pick_blind_right')
        self.publish_status('post_pick_blind_right')

    def begin_post_pick_search(self, reason):
        self.enter_post_pick_search(reason)

    def in_tolerance(self, ex, ey):
        return abs(ex) <= self.x_tolerance and abs(ey) <= self.y_tolerance

    def inside_bands(self, ex, ey):
        return abs(ex) <= self.x_band_px and abs(ey) <= self.y_band_px

    def apply_coarse_step(self, ex, ey, y_band, x_band, step_distance, turn_yaw, fine_yaw, dsp, walking_param, arm_swap):
        x_step = 0.0
        yaw = 0

        # y 轴粗调：只在超过 y_band 时启用前后
        if abs(ey) > y_band:
            if ey < 0:  # 物体偏上（远）→ 前进
                x_step = step_distance * 0.7
            else:  # 物体偏下（近）→ 后退
                x_step = -step_distance * 0.5

        # x 轴粗调：只在超过 x_band 时启用 yaw
        if abs(ex) > x_band:
            yaw = -turn_yaw if ex > 0 else turn_yaw
            if abs(ex) <= x_band * 1.4:
                yaw = -fine_yaw if ex > 0 else fine_yaw

        if x_step == 0.0 and yaw == 0:
            self.gait_manager.stop()
            return False

        self.gait_manager.set_step(
            dsp,
            float(x_step),
            0.0,
            int(yaw),
            walking_param,
            arm_swap=arm_swap,
            step_num=1,
        )
        return True

    def run_fine_action_once(self, ex, ey):
        if not self.is_gait_switch_on():
            return False
        # 每次只动一轴，优先绝对误差更大的一轴
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
        rospy.loginfo('sandbag fine_tune: axis=%s action=%s ex=%.1f ey=%.1f', axis, action_name, ex, ey)
        self.motion_manager.run_action(action_name)
        self._fine_steps += 1
        return True

    def apply_search_forward_step(self, dsp, walking_param, arm_swap):
        x_step = float(self.search_forward_x_step)
        if abs(x_step) <= 1e-6:
            self.gait_manager.stop()
            return False
        self.gait_manager.set_step(
            dsp,
            x_step,
            0.0,
            0,
            walking_param,
            arm_swap=arm_swap,
            step_num=1,
        )
        return True

    def apply_tag_lateral_step(self, ex, ey):
        x_step = 0.0
        y_step = 0.0

        # y 轴误差控制前后：Tag 偏上(远)前进，偏下(近)后退
        if abs(ey) > 0.0:
            if ey < 0.0:
                x_step = self.tag_step_distance * 0.7
            else:
                x_step = -self.tag_step_distance * 0.5

        # x 轴误差控制横移：图像里目标偏右 -> 机器人右移（y_amplitude 为负）
        if abs(ex) > 0.0:
            if ex > 0.0:
                y_step = -abs(self.tag_side_step_distance)
            else:
                y_step = abs(self.tag_side_step_distance)

        if abs(x_step) <= 1e-6 and abs(y_step) <= 1e-6:
            self.gait_manager.stop()
            return False

        self.gait_manager.set_step(
            self.tag_dsp,
            float(x_step),
            float(y_step),
            0,
            self.tag_walking_param,
            arm_swap=self.tag_arm_swap,
            step_num=1,
        )
        return True

    def execute_pick(self):
        if self._picked_once:
            return True
        self.phase = 'picking'
        self.publish_status('picking')
        self.gait_manager.stop()
        time.sleep(0.2)
        self.gait_manager.disable()
        rospy.loginfo('开始捡沙包丢沙包')
        self.motion_manager.run_action(self.pick_action_name)
        if self.pick_wait_sec > 0:
            time.sleep(self.pick_wait_sec)
        if self.walk_ready_after_pick:
            self.motion_manager.run_action('walk_ready')
        self._picked_once = True
        self.publish_status('completed_pick')
        self._start_time = rospy.get_time()
        self._lost_frames = 0
        self._line_stable_count = 0
        self._tag_stable_count = 0
        self._post_pick_forward_steps_done = 0
        self._post_pick_blind_steps_done = 0
        self._tag_seen_once_after_pick = False
        self.reset_post_pick_tag_gate()
        if self.post_pick_forward_enabled and self.post_pick_forward_steps > 0:
            self.begin_post_pick_forward()
        elif self.post_pick_blind_enabled and self.post_pick_blind_steps > 0:
            self.begin_post_pick_blind_right()
        else:
            self.begin_post_pick_search('after_pick')
        return True

    def execute_throw(self):
        self.phase = 'throwing'
        self.sync_tag_detection('throwing')
        self.publish_status('throwing')
        self.gait_manager.stop()
        time.sleep(0.2)
        self.gait_manager.disable()
        self.motion_manager.run_action(self.throw_action_name)
        if self.throw_wait_sec > 0:
            time.sleep(self.throw_wait_sec)
        if self.walk_ready_after_throw:
            self.motion_manager.run_action('walk_ready')
        self.publish_status('completed_throw')
        self._start_time = rospy.get_time()
        self._lost_frames = 0
        self._rejoin_stable_count = 0
        self._rejoin_left_only_streak = 0
        self._rejoin_green_handoff_count = 0
        self._rejoin_blind_steps_done = 0
        if self.rejoin_enabled:
            self.phase = 'rejoin_left'
            self.sync_tag_detection('rejoin_left')
            self.publish_detection_for_phase('rejoin_left')
        else:
            self.phase = 'done'
            self.active = False
            self._rejoin_left_only_streak = 0
            self.sync_tag_detection('done')
            self.publish_detection_for_phase('done')
            self.try_stop_color_detection()
        return True

    def start_service_callback(self, req):
        self.load_config()
        if not self.enabled:
            self.publish_status('disabled')
            return TriggerResponse(success=False, message='sandbag_throw disabled by config')

        self.active = True
        self.phase = 'align_pick'
        self._start_time = rospy.get_time()
        self._last_log_time = 0.0
        self._last_line_log_time = 0.0
        self._last_tag_log_time = 0.0
        self._last_tag_detect_log_time = 0.0
        self._last_rejoin_log_time = 0.0
        self._fine_steps = 0
        self._lost_frames = 0
        self._picked_once = False
        self._line_stable_count = 0
        self._tag_stable_count = 0
        self._tag_switch_visible_streak = 0
        self._post_pick_forward_steps_done = 0
        self._post_pick_blind_steps_done = 0
        self._rejoin_stable_count = 0
        self._rejoin_left_only_streak = 0
        self._rejoin_green_handoff_count = 0
        self._rejoin_blind_steps_done = 0
        self._tag_seen_once_after_pick = False
        self.reset_post_pick_tag_gate()
        self.clear_object_tracking()
        self.clear_tag_tracking()
        self.sync_tag_detection('align_pick')

        self.try_enter_color_detection()
        self.publish_detection_for_phase('align_pick')
        if not self.try_start_color_detection():
            self.active = False
            self.publish_status('waiting_color_detection')
            return TriggerResponse(success=False, message='color_detection/start unavailable')

        self.publish_status('started')
        return TriggerResponse(success=True, message='sandbag_throw started')

    def stop_service_callback(self, req):
        self.active = False
        self.phase = 'idle'
        self._tag_seen_once_after_pick = False
        self._post_pick_forward_steps_done = 0
        self._rejoin_left_only_streak = 0
        self._rejoin_green_handoff_count = 0
        self.reset_post_pick_tag_gate()
        self.clear_object_tracking()
        self.sync_tag_detection('idle')
        self.gait_manager.stop()
        self.try_stop_color_detection()
        self.publish_status('stopped')
        return TriggerResponse(success=True, message='sandbag_throw stopped')

    def reset_service_callback(self, req):
        self.active = False
        self.phase = 'idle'
        self._fine_steps = 0
        self._lost_frames = 0
        self._picked_once = False
        self._post_pick_forward_steps_done = 0
        self._post_pick_blind_steps_done = 0
        self._tag_seen_once_after_pick = False
        self._rejoin_stable_count = 0
        self._rejoin_left_only_streak = 0
        self._rejoin_green_handoff_count = 0
        self._rejoin_blind_steps_done = 0
        self.reset_post_pick_tag_gate()
        self.clear_object_tracking()
        self.sync_tag_detection('idle')
        self.gait_manager.stop()
        self.try_stop_color_detection()
        self.publish_status('reset')
        return TriggerResponse(success=True, message='sandbag_throw reset')

    def shutdown(self, signum, frame):
        self.running = False
        self.active = False
        self.phase = 'idle'
        self._tag_seen_once_after_pick = False
        self._post_pick_forward_steps_done = 0
        self._rejoin_left_only_streak = 0
        self._rejoin_green_handoff_count = 0
        self.reset_post_pick_tag_gate()
        self.clear_object_tracking()
        self.sync_tag_detection('idle')
        self.gait_manager.stop()
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
                    rospy.logwarn('gait_switch_off已设置为false，闭环步态发布关闭（固定步/动作仍允许）')
                    self.gait_switch_off_reported = True
                self.gait_manager.stop()
            else:
                self.gait_switch_off_reported = False
            if self.align_timeout_sec > 0 and (now - self._start_time) > self.align_timeout_sec:
                self.active = False
                self.gait_manager.stop()
                self.publish_status('align_timeout')
                rospy.logwarn('sandbag_throw align timeout')
                continue
            if self.check_phase_timeout(now):
                continue

            if self.phase in ('align_pick', 'coarse_align', 'fine_align'):
                obj = self.get_best_purple()
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
                    rospy.loginfo(
                        'purple x=%.1f y=%.1f',
                        cx,
                        cy,
                    )
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
                        rospy.logwarn('sandbag_throw fine tune exceeded max steps')
                        continue
                    self.run_fine_action_once(ex, ey)
                else:
                    self.phase = 'coarse_align'
                    self.apply_coarse_step(
                        ex,
                        ey,
                        self.y_band_px,
                        self.x_band_px,
                        self.coarse_step_distance,
                        self.coarse_turn_yaw,
                        self.coarse_fine_yaw,
                        self.coarse_dsp,
                        self.coarse_walking_param,
                        self.coarse_arm_swap,
                    )
            elif self.phase == 'post_pick_forward':
                next_step = self._post_pick_forward_steps_done + 1
                rospy.loginfo(
                    '%d/%d sandbag_post_pick_forward cmd_x_step=%.4f cmd_y_step=%.4f cmd_yaw=%d gait_switch=%s',
                    next_step,
                    self.post_pick_forward_steps,
                    self.post_pick_forward_x_step,
                    self.post_pick_forward_y_step,
                    int(self.post_pick_forward_yaw),
                    str(gait_on),
                )
                self.gait_manager.set_step(
                    self.post_pick_forward_dsp,
                    float(self.post_pick_forward_x_step),
                    float(self.post_pick_forward_y_step),
                    int(self.post_pick_forward_yaw),
                    self.post_pick_forward_walking_param,
                    arm_swap=self.post_pick_forward_arm_swap,
                    step_num=1,
                )
                if not self.active or self.phase != 'post_pick_forward':
                    continue
                self._post_pick_forward_steps_done = next_step
                if self._post_pick_forward_steps_done >= self.post_pick_forward_steps:
                    self.gait_manager.stop()
                    if self.post_pick_blind_enabled and self.post_pick_blind_steps > 0:
                        self.begin_post_pick_blind_right()
                    else:
                        self.enter_post_pick_search('post_pick_forward_complete')
            elif self.phase == 'post_pick_blind_right':
                next_step = self._post_pick_blind_steps_done + 1
                rospy.loginfo(
                    '%d/%d sandbag_post_pick_blind_right cmd_x_step=%.4f cmd_y_step=%.4f cmd_yaw=%d gait_switch=%s',
                    next_step,
                    self.post_pick_blind_steps,
                    self.post_pick_blind_x_step,
                    self.post_pick_blind_y_step,
                    int(self.post_pick_blind_yaw),
                    str(gait_on),
                )
                self.gait_manager.set_step(
                    self.line_rotate_dsp,
                    float(self.post_pick_blind_x_step),
                    float(self.post_pick_blind_y_step),
                    int(self.post_pick_blind_yaw),
                    self.line_rotate_walking_param,
                    arm_swap=self.line_rotate_arm_swap,
                    step_num=1,
                )
                if not self.active or self.phase != 'post_pick_blind_right':
                    continue
                self._post_pick_blind_steps_done = next_step
                if self._post_pick_blind_steps_done >= self.post_pick_blind_steps:
                    self.gait_manager.stop()
                    self.enter_post_pick_search('post_pick_blind_complete')
            elif self.phase == 'line_align_right':
                # 橙线首次进入平行容差后才放行 Tag；放行之后若稳定看到 Tag，就切到 Tag x/y 对齐。
                if self.should_switch_to_tag_align(now):
                    self.switch_to_tag_align('tag_visible_during_line_align')
                    continue
                # 右转为主，过头允许小幅左回；目标是当前对齐线 angle 接近水平。Tag 不可见时才使用线对齐回退。
                line_obj = self.get_best_align_line()
                if line_obj is None:
                    self._lost_frames += 1
                    self._line_stable_count = 0
                    if now - self._last_line_log_time >= self.log_throttle_sec:
                        rospy.loginfo(
                            'sandbag_line_align: phase=%s mode=search_forward angle=none cmd_x_step=%.4f cmd_yaw=%d gait_switch=%s',
                            self.phase,
                            self.search_forward_x_step,
                            0,
                            str(gait_on),
                        )
                        self._last_line_log_time = now
                    if not gait_on:
                        time.sleep(rate_sleep)
                        continue
                    if not self.wait_for_adjust_interval('line_align_right'):
                        continue
                    cmd_started_at = time.monotonic()
                    motion_sent = self.apply_search_forward_step(
                        self.line_rotate_dsp,
                        self.line_rotate_walking_param,
                        self.line_rotate_arm_swap,
                    )
                    if motion_sent:
                        self.note_adjust_command_start('line_align_right', cmd_started_at)
                    continue
                self._lost_frames = 0
                ang = float(getattr(line_obj, 'angle', 0.0))
                aerr = ang - self.line_angle_target
                if abs(aerr) <= self.line_angle_tolerance:
                    self._line_stable_count += 1
                    self.unlock_tag_detection_after_parallel(ang)
                else:
                    self._line_stable_count = 0
                # 先按当前对齐线修角；进入容差后保持与该线平行并小步前进搜索 Tag。
                if abs(aerr) <= self.line_angle_tolerance:
                    yaw = 0
                    x_step = self.search_forward_x_step
                    mode = 'parallel_search'
                else:
                    yaw = self.line_right_turn_yaw if aerr > 0 else self.line_left_backoff_yaw
                    x_step = self.line_right_turn_x_step if aerr > 0 else 0.0
                    mode = 'angle_align'
                if now - self._last_line_log_time >= self.log_throttle_sec:
                    rospy.loginfo(
                        'sandbag_line_align: phase=%s mode=%s angle=%.1f cmd_x_step=%.4f cmd_yaw=%d gait_switch=%s',
                        self.phase,
                        mode,
                        ang,
                        x_step,
                        int(yaw),
                        str(gait_on),
                    )
                    self._last_line_log_time = now
                if not gait_on:
                    time.sleep(rate_sleep)
                    continue
                if not self.wait_for_adjust_interval('line_align_right'):
                    continue
                cmd_started_at = time.monotonic()
                self.gait_manager.set_step(
                    self.line_rotate_dsp,
                    float(x_step),
                    0.0,
                    int(yaw),
                    self.line_rotate_walking_param,
                    arm_swap=self.line_rotate_arm_swap,
                    step_num=1,
                )
                self.note_adjust_command_start('line_align_right', cmd_started_at)
            elif self.phase == 'tag_align':
                if not self.has_recent_tag(now):
                    lost_age = (now - self._tag_seen_time) if self._tag_seen_time > 0.0 else -1.0
                    self.clear_stale_tag_tracking()
                    if now - self._last_tag_log_time >= self.log_throttle_sec:
                        rospy.loginfo(
                            'sandbag_tag_lost: lost_age=%.2f line_align_enabled=%s fallback_x_step=%.4f',
                            lost_age,
                            str(self.line_align_enabled),
                            self.search_forward_x_step,
                        )
                        self._last_tag_log_time = now
                    if self.line_align_enabled:
                        self.gait_manager.stop()
                        self.phase = 'line_align_right'
                        self._start_time = now
                        self._lost_frames = 0
                        self._line_stable_count = 0
                        self.publish_detection_for_phase('line_align_right')
                        self.publish_status('tag_lost_back_to_line_align')
                        rospy.loginfo(
                            'tag_lost_back_to_line_align: lost_age=%.2f reason=stale_tag',
                            lost_age,
                        )
                    else:
                        self.publish_status('tag_lost_search_forward')
                        if gait_on:
                            self.apply_search_forward_step(
                                self.tag_dsp,
                                self.tag_walking_param,
                                self.tag_arm_swap,
                            )
                    continue
                tcx, tcy = self.tag_center
                tx = self.tag_target_x_px()
                ty = self.tag_target_y_px()
                ex = float(tcx) - float(tx)
                ey = float(tcy) - float(ty)
                tag_age = (now - self._tag_seen_time) if self._tag_seen_time > 0.0 else -1.0
                if abs(ex) <= self.tag_x_tolerance and abs(ey) <= self.tag_y_tolerance:
                    self._tag_stable_count += 1
                else:
                    self._tag_stable_count = 0
                if now - self._last_tag_log_time >= self.log_throttle_sec:
                    rospy.loginfo('tag x=%.1f y=%.1f', tcx, tcy)
                    self._last_tag_log_time = now
                if self._tag_stable_count >= self.tag_stable_frames:
                    self.gait_manager.stop()
                    self.publish_status('tag_aligned')
                    if self.throw_enabled:
                        self.execute_throw()
                    else:
                        self.phase = 'done'
                        self.active = False
                    continue

                if not gait_on:
                    time.sleep(rate_sleep)
                    continue

                if not self.wait_for_adjust_interval('tag_align'):
                    continue
                cmd_started_at = time.monotonic()
                motion_sent = self.apply_tag_lateral_step(ex, ey)
                if motion_sent:
                    self.note_adjust_command_start('tag_align', cmd_started_at)
            elif self.phase == 'throwing':
                self.execute_throw()
            elif self.phase == 'rejoin_left':
                # 阶段1：先写死左转若干步，防止刚转身时“横线”误判回轨成功
                if self._rejoin_blind_steps_done < self.rejoin_blind_turn_steps:
                    next_step = self._rejoin_blind_steps_done + 1
                    rospy.loginfo(
                        '%d/%d sandbag_rejoin_blind cmd_x_step=%.4f cmd_yaw=%d gait_switch=%s',
                        next_step,
                        self.rejoin_blind_turn_steps,
                        self.rejoin_blind_turn_x_step,
                        int(self.rejoin_blind_turn_yaw),
                        str(gait_on),
                    )
                    self.gait_manager.set_step(
                        self.rejoin_rotate_dsp,
                        float(self.rejoin_blind_turn_x_step),
                        0.0,
                        int(self.rejoin_blind_turn_yaw),
                        self.rejoin_rotate_walking_param,
                        arm_swap=self.rejoin_rotate_arm_swap,
                        step_num=1,
                    )
                    if not self.active or self.phase != 'rejoin_left':
                        continue
                    self._rejoin_blind_steps_done = next_step
                    if self._rejoin_blind_steps_done >= self.rejoin_blind_turn_steps:
                        self.gait_manager.stop()
                        self._start_time = rospy.get_time()
                        self._lost_frames = 0
                        self._rejoin_stable_count = 0
                        self._rejoin_left_only_streak = 0
                        self._rejoin_green_handoff_count = 0
                    continue
                left_obj = self.get_best_rejoin_side(self.rejoin_left_color, self.rejoin_left_detect_type, 'left')
                right_obj = self.get_best_rejoin_side(self.rejoin_right_color, self.rejoin_right_detect_type, 'right')
                green_obj = self.get_best_rejoin_green() if self.rejoin_green_handoff_enabled else None
                cmd_x_step = self.rejoin_left_turn_x_step
                cmd_yaw = int(self.rejoin_left_turn_yaw)
                mode = 'search_left'
                left_x = float(getattr(left_obj, 'x', -1.0)) if left_obj is not None else -1.0
                right_x = float(getattr(right_obj, 'x', -1.0)) if right_obj is not None else -1.0
                if left_obj is None and right_obj is None:
                    self._lost_frames += 1
                    if self._lost_frames >= self.rejoin_lost_frames:
                        self.publish_status('rejoin_searching')
                    self._rejoin_stable_count = 0
                    self._rejoin_left_only_streak = 0
                    lane_ok = False
                    order_ok = False
                    lane_w = 0.0
                elif left_obj is not None and right_obj is None:
                    self._lost_frames += 1
                    if self._lost_frames >= self.rejoin_lost_frames:
                        self.publish_status('rejoin_searching')
                    self._rejoin_stable_count = 0
                    self._rejoin_left_only_streak += 1
                    lane_ok = False
                    order_ok = False
                    lane_w = 0.0
                    if self._rejoin_left_only_streak >= 2:
                        mode = 'left_only_recover_right'
                        cmd_x_step = 0.0
                        cmd_yaw = -abs(int(self.rejoin_left_turn_yaw))
                elif left_obj is None and right_obj is not None:
                    self._lost_frames += 1
                    if self._lost_frames >= self.rejoin_lost_frames:
                        self.publish_status('rejoin_searching')
                    self._rejoin_stable_count = 0
                    self._rejoin_left_only_streak = 0
                    lane_ok = False
                    order_ok = False
                    lane_w = 0.0
                else:
                    self._lost_frames = 0
                    self._rejoin_left_only_streak = 0
                    lx = float(getattr(left_obj, 'x', 0))
                    rx = float(getattr(right_obj, 'x', 0))
                    lane_w = rx - lx
                    order_ok = (lx < rx) if self.rejoin_require_order else True
                    lane_ok = order_ok and (self.rejoin_min_lane_width <= lane_w <= self.rejoin_max_lane_width)
                    if lane_ok:
                        self._rejoin_stable_count += 1
                        mode = 'lane_ok'
                    else:
                        self._rejoin_stable_count = 0
                        mode = 'lane_bad'
                green_ok = False
                green_x = -1.0
                green_threshold = self.rejoin_green_min_x_px(green_obj) if green_obj is not None else self.rejoin_green_min_x_px()
                if green_obj is not None:
                    green_x = float(getattr(green_obj, 'x', 0))
                    green_ok = green_x >= green_threshold
                if green_ok:
                    self._rejoin_green_handoff_count += 1
                else:
                    self._rejoin_green_handoff_count = 0
                if now - self._last_rejoin_log_time >= self.rejoin_log_throttle_sec:
                    rospy.loginfo(
                        'sandbag_rejoin: mode=%s lane_ok=%s stable=%d/%d left_x=%.1f right_x=%.1f lane_w=%.1f range=[%.1f,%.1f] left_only=%d/2 green_ok=%s green_x=%.1f green_stable=%d/%d green_min_x=%.1f cmd_x_step=%.4f cmd_yaw=%d gait_switch=%s',
                        mode,
                        lane_ok,
                        self._rejoin_stable_count,
                        self.rejoin_stable_frames,
                        left_x,
                        right_x,
                        lane_w,
                        self.rejoin_min_lane_width,
                        self.rejoin_max_lane_width,
                        self._rejoin_left_only_streak,
                        str(green_ok),
                        green_x,
                        self._rejoin_green_handoff_count,
                        self.rejoin_green_stable_frames,
                        green_threshold,
                        cmd_x_step,
                        cmd_yaw,
                        str(gait_on),
                    )
                    self._last_rejoin_log_time = now
                if (
                    self._rejoin_stable_count >= self.rejoin_stable_frames
                    or self._rejoin_green_handoff_count >= self.rejoin_green_stable_frames
                ):
                    self.gait_manager.stop()
                    if self._rejoin_green_handoff_count >= self.rejoin_green_stable_frames:
                        rospy.loginfo(
                            'sandbag_rejoin complete by green handoff: green_x=%.1f threshold=%.1f stable=%d/%d',
                            green_x,
                            green_threshold,
                            self._rejoin_green_handoff_count,
                            self.rejoin_green_stable_frames,
                        )
                    self.publish_status('completed_rejoin')
                    self.phase = 'done'
                    self.active = False
                    self._rejoin_left_only_streak = 0
                    self.try_stop_color_detection()
                    continue
                if not gait_on:
                    time.sleep(rate_sleep)
                    continue
                self.gait_manager.set_step(
                    self.rejoin_rotate_dsp,
                    float(cmd_x_step),
                    0.0,
                    int(cmd_yaw),
                    self.rejoin_rotate_walking_param,
                    arm_swap=self.rejoin_rotate_arm_swap,
                    step_num=1,
                )
            else:
                time.sleep(rate_sleep)
                continue

            time.sleep(rate_sleep)


if __name__ == '__main__':
    try:
        SandbagThrowNode('sandbag_throw').run()
    except Exception as e:
        rospy.logerr('sandbag_throw initialization error: %s', e)
