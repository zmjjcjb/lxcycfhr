#!/usr/bin/env python3
# encoding: utf-8
# @Author: Dejin Xuan
# @Date: 2026.4.1 
# ISCC 主控节点
import os
import signal
import sys
import threading
import time

import rospy
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerRequest

from rosman_example.color_common import Common
from rosman_interfaces.msg import ObjectsInfo
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


class IsccController(Common):
    """
    ISCC 流程：巡线 →（蓝线稳定+阈值）跨栏 → 恢复巡线 →（台阶标志稳定+阈值）上台阶
    → 恢复巡线 →（台阶标志稳定+阈值）下台阶 → 橙 A → 橙 B；B 内可选紫色沙包 y 门限 → 丢沙包节点。
    → 绿 A → 绿 B；B 内可选绿色物块 y 门限 → 放绿块节点。
    """

    def __init__(self, name):
        self.head_pan_init = rospy.get_param('/servo/head_pan_init', 500)
        self.head_tilt_look_down = rospy.get_param('/servo/head_tilt_look_down', 280)

        rospy.init_node(name)
        self.name = name
        self.running = True
        self.competition_finished = False
        self.lock = threading.RLock()
        self._cleanup_done = False

        super().__init__(name, self.head_pan_init, self.head_tilt_look_down)
        signal.signal(signal.SIGINT, self.shutdown)

        self.current_state = 'line_following'
        self.iscc_phase = 'await_blue'
        self.task_completed = {
            'blue_object': False,
            'climb_stairs': False,
            'descend_stairs': False,
            'orange_guide': False,
        }
        self.progress_info = {
            'current_task': 'ISCC Start',
            'progress': 'Initializing...',
            'detected_objects': [],
            'completed_tasks': [],
        }

        self.objects_info = []
        self.blue_history = []
        self.wait_blue_clear = False
        self.wait_stairs_clear_descend = False
        self.climb_slow_latched = False
        self.descend_slow_latched = False
        self.orange_tag_history = []
        self.orange_tag_center = None
        self.orange_tag_id = -1
        self.orange_tag_seen_once = False
        self._orange_tag_seen_time = 0.0

        self.process_size = rospy.get_param('/image_processing/process_size', [160, 120])
        self.image_height = int(self.process_size[1]) if len(self.process_size) > 1 else 120
        self.hurdles_gate_enter_y_threshold = float(
            rospy.get_param('/tasks/hurdles/gate/enter_y_threshold', 300)
        )
        self.climb_stairs_gate_enter_y_threshold = float(
            rospy.get_param('/tasks/climb_stairs/gate/enter_y_threshold', 300)
        )
        self.descend_stairs_gate_enter_y_threshold = float(
            rospy.get_param('/tasks/descend_stairs/gate/enter_y_threshold', 300)
        )
        self.line_follow_default_gait_profile = str(rospy.get_param('/tasks/line_following/gait_profile', 'default'))
        self.await_orange_guide_gait_profile = str(
            rospy.get_param(
                '/tasks/line_following/await_orange_guide_gait_profile',
                self.line_follow_default_gait_profile,
            )
        )
        self.await_orange_guide_tag_seen_gait_profile = str(
            rospy.get_param(
                '/tasks/line_following/await_orange_guide_tag_seen_gait_profile',
                self.await_orange_guide_gait_profile,
            )
        )
        self.hurdles_slow_gait_x_max = float(
            rospy.get_param('/tasks/hurdles/slow_gait_x_max', 0.0)
        )
        self.climb_stairs_color = str(rospy.get_param('/tasks/climb_stairs/color', 'green'))
        self.climb_stairs_slow_gait_x_max = float(
            rospy.get_param('/tasks/climb_stairs/slow_gait_x_max', 0.0)
        )
        self.descend_stairs_color = str(rospy.get_param('/tasks/descend_stairs/color', 'green'))
        self.descend_stairs_slow_gait_x_max = float(
            rospy.get_param('/tasks/descend_stairs/slow_gait_x_max', 0.0)
        )
        self.post_climb_descend_gate_delay_sec = float(
            rospy.get_param('/tasks/climb_stairs/post_climb_descend_gate_delay_sec', 0.15)
        )
        self.stability_frames = int(rospy.get_param('/image_processing/detection/stability_frames', 3))
        self.position_tolerance = int(rospy.get_param('/image_processing/detection/position_tolerance', 50))
        self.orange_auto_start = bool(rospy.get_param('/tasks/orange_guide/auto_start_after_descend', True))
        tag_cfg = rospy.get_param('/tasks/orange_guide/tag_trigger', {})
        if not isinstance(tag_cfg, dict):
            tag_cfg = {}
        self.orange_tag_trigger_enabled = bool(tag_cfg.get('enabled', True))
        self.orange_tag_topic = str(tag_cfg.get('topic', '/tag_detections'))
        self.orange_tag_target_id = int(tag_cfg.get('target_id', -1))
        self.orange_tag_enter_y_threshold = float(tag_cfg.get('enter_y_threshold', 300))
        self.orange_tag_stable_frames = max(1, int(tag_cfg.get('stable_frames', 2)))
        self.orange_tag_lost_timeout_sec = float(tag_cfg.get('lost_timeout_sec', 0.5))
        self.orange_tag_frame_width = max(1, int(tag_cfg.get('frame_width', 640)))
        self.orange_tag_frame_height = max(1, int(tag_cfg.get('frame_height', 480)))
        self.orange_tag_roi = tag_cfg.get('roi', [0.0, 1.0, 0.0, 1.0])
        self.orange_tag_log_throttle_sec = float(tag_cfg.get('log_throttle_sec', 0.2))
        sb = rospy.get_param('/tasks/sandbag_throw', {})
        self.sandbag_throw_enabled = bool(sb.get('enabled', False))
        self.sandbag_purple_color = str(sb.get('color', 'purple'))
        self.sandbag_detect_type = str(sb.get('detect_type', 'side'))
        self.sandbag_enter_y_threshold = float(sb.get('enter_y_threshold', 300))
        self.sandbag_stability_frames = int(sb.get('stability_frames', self.stability_frames))
        self.sandbag_position_tolerance = int(sb.get('position_tolerance', self.position_tolerance))
        self.purple_sandbag_y_log_throttle = float(sb.get('y_log_throttle_sec', 0.2))
        self.purple_history = []
        self._last_purple_log_time = 0.0
        gg = rospy.get_param('/tasks/green_pick_gate', {})
        self.green_gate_enabled = bool(gg.get('enabled', True))
        self.green_color = str(gg.get('color', 'green_pick'))
        self.green_detect_type = str(gg.get('detect_type', 'side'))
        self.green_enter_y_threshold = float(gg.get('enter_y_threshold', 220))
        self.green_stability_frames = int(gg.get('stability_frames', self.stability_frames))
        self.green_position_tolerance = int(gg.get('position_tolerance', self.position_tolerance))
        self.green_y_log_throttle = float(gg.get('y_log_throttle_sec', 0.2))
        self.green_history = []
        self._last_green_log_time = 0.0

        self.y_log_throttle_sec = float(rospy.get_param('/tasks/hurdles/y_log_throttle_sec', 0.2))
        self._last_y_log_time = 0.0
        self._last_orange_xy_log_time = 0.0
        self._last_orange_tag_detect_log_time = 0.0
        self._last_seen_log_time = 0.0
        self._last_stairs_log_time = 0.0
        self._last_phase_log_time = 0.0
        self._last_descend_heartbeat_time = 0.0
        self._descend_gate_enable_time = 0.0
        self._descend_gate_delay_logged = False
        self._coord_frame_logged = False
        self.task_status_log_throttle_sec = float(rospy.get_param('/iscc/task_status_log_throttle_sec', 1.0))
        self._last_task_status = {}
        self._last_task_status_log_time = {}
        self._stop_service_timeout = float(rospy.get_param('/iscc/stop_service_timeout', 1.0))
        self._start_service_timeout = float(rospy.get_param('/iscc/start_service_timeout', 5.0))
        self._startup_run_gap_sec = float(rospy.get_param('/iscc/startup_run_gap_sec', 0.02))
        self.result_image_topic = rospy.get_param(
            '/tasks/line_following/result_image_topic',
            '/color_detection/image_result',
        )
        self.auto_open_rqt_result = bool(rospy.get_param('/tasks/line_following/auto_open_rqt_result', True))
        self.rqt_session_owner_id = 'iscc_controller_session:%s:%d' % (self.name, os.getpid())
        self._session_rqt_acquired = False
        self.setup_ros_communication()

        self.motion_manager.run_action('walk_ready')

        self.apply_start_stage()
        common.loginfo('ISCC controller initialized: line -> hurdles -> climb -> descend (start_stage=%s)' % self.start_stage)

    def apply_start_stage(self):
        self.start_stage = str(rospy.get_param('/iscc/start_stage', 'full')).strip().lower()
        # 允许用户写更短的别名
        if self.start_stage in ('full', 'default', 'all'):
            self.iscc_phase = 'await_blue'
        elif self.start_stage in ('hurdles', 'await_blue'):
            self.iscc_phase = 'await_blue'
        elif self.start_stage in ('climb', 'climb_stairs', 'await_stairs_climb', 'await_red_climb'):
            self.iscc_phase = 'await_stairs_climb'
        elif self.start_stage in ('descend', 'descend_stairs', 'await_stairs_descend', 'await_red_descend'):
            self.iscc_phase = 'await_stairs_descend'
        elif self.start_stage in ('orange', 'orange_guide', 'await_orange_guide'):
            self.iscc_phase = 'await_orange_guide'
        elif self.start_stage == 'sandbag':
            self.iscc_phase = 'orange_line_following'
        elif self.start_stage == 'green':
            self.iscc_phase = 'orange_line_following'
        elif self.start_stage in ('green_return_place', 'return'):
            self.iscc_phase = 'green_return_place'
        else:
            rospy.logwarn('unknown /iscc/start_stage=%s, fallback to full', self.start_stage)
            self.start_stage = 'full'
            self.iscc_phase = 'await_blue'

        # 清理前置阶段遗留的等待/历史，避免误卡
        self.wait_blue_clear = False
        self.wait_stairs_clear_descend = False
        self.climb_slow_latched = False
        self.descend_slow_latched = False
        self.blue_history = []
        self.clear_orange_tag_tracking(reset_seen_once=True)
        self.progress_info['progress'] = 'start_stage=%s phase=%s' % (self.start_stage, self.iscc_phase)

    def setup_ros_communication(self):
        # 仅订阅需驱动状态机的任务；巡线状态由 line_following_node 自行打日志
        rospy.Subscriber('/hurdles/status', String, lambda msg: self.task_status_callback('hurdles', msg))
        rospy.Subscriber('/climb_stairs/status', String, lambda msg: self.task_status_callback('climb_stairs', msg))
        rospy.Subscriber('/descend_stairs/status', String, lambda msg: self.task_status_callback('descend_stairs', msg))
        rospy.Subscriber('/orange_guide/status', String, lambda msg: self.task_status_callback('orange_guide', msg))
        rospy.Subscriber(
            '/orange_line_follow/status', String, lambda msg: self.task_status_callback('orange_line_follow', msg)
        )
        rospy.Subscriber(
            '/sandbag_throw/status', String, lambda msg: self.task_status_callback('sandbag_throw', msg)
        )
        rospy.Subscriber('/green_pick/status', String, lambda msg: self.task_status_callback('green_pick', msg))
        rospy.Subscriber(
            '/green_return_place/status', String, lambda msg: self.task_status_callback('green_return_place', msg)
        )
        self.object_sub = rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.object_callback)
        if self.orange_tag_trigger_enabled and AprilTagDetectionArray is not None:
            self.orange_tag_sub = rospy.Subscriber(
                self.orange_tag_topic,
                AprilTagDetectionArray,
                self.orange_tag_callback,
            )
        elif self.orange_tag_trigger_enabled:
            rospy.logwarn('apriltag_ros.msg unavailable, orange guide tag trigger disabled')
            self.orange_tag_trigger_enabled = False
        self.competition_status_pub = rospy.Publisher('/competition/status', String, queue_size=1)
        self.competition_progress_pub = rospy.Publisher('/competition/progress', String, queue_size=1)

    def object_callback(self, msg):
        self.objects_info = msg.data

    def task_status_callback(self, task_name, msg):
        status = msg.data
        now = rospy.get_time()
        prev = self._last_task_status.get(task_name)
        prev_t = self._last_task_status_log_time.get(task_name, 0.0)
        should_log = (status != prev) or ((now - prev_t) >= self.task_status_log_throttle_sec)
        if should_log:
            rospy.loginfo('[%s] status: %s', task_name, status)
            self._last_task_status[task_name] = status
            self._last_task_status_log_time[task_name] = now
        if task_name == 'hurdles' and status == 'completed':
            self.task_completed['blue_object'] = True
            if 'Blue Hurdles' not in self.progress_info['completed_tasks']:
                self.progress_info['completed_tasks'].append('Blue Hurdles')
        if task_name == 'climb_stairs':
            if status == 'completed_climb':
                self.task_completed['climb_stairs'] = True
                if 'Climb Stairs' not in self.progress_info['completed_tasks']:
                    self.progress_info['completed_tasks'].append('Climb Stairs')
        if task_name == 'descend_stairs':
            if status == 'completed_descend':
                self.task_completed['descend_stairs'] = True
                if 'Descend Stairs' not in self.progress_info['completed_tasks']:
                    self.progress_info['completed_tasks'].append('Descend Stairs')
        if task_name == 'orange_guide' and status == 'completed_orange_guide':
            self.task_completed['orange_guide'] = True
            if 'Orange Guide' not in self.progress_info['completed_tasks']:
                self.progress_info['completed_tasks'].append('Orange Guide')
        if task_name == 'orange_guide' and status == 'timeout':
            self.task_completed['orange_guide'] = True
            rospy.logwarn('orange_guide A阶段超时，主控结束等待')
        if task_name == 'sandbag_throw' and status == 'completed_rejoin':
            rospy.loginfo('sandbag_throw completed_rejoin -> start second orange_line_follow with green watch')
            rospy.set_param('/iscc/sandbag_purple_watch_active', False)
            rospy.set_param('/iscc/green_watch_active', True)
            self.call_service('/sandbag_throw/stop', timeout=self._stop_service_timeout)
            self.call_service('/orange_line_follow/start', timeout=self._start_service_timeout)
            self.current_state = 'orange_line_follow'
            self.iscc_phase = 'orange_line_following'
            self.publish_competition_status('orange_line_follow_restarted_for_green')
        if task_name == 'green_pick' and status == 'completed_pick':
            rospy.loginfo('green_pick completed_pick -> stop green_pick and start green_return_place')
            rospy.set_param('/iscc/sandbag_purple_watch_active', False)
            rospy.set_param('/iscc/green_watch_active', False)
            self.call_service('/green_pick/stop', timeout=self._stop_service_timeout)
            self.call_service('/orange_line_follow/stop', timeout=self._stop_service_timeout)
            ok = self.call_service('/green_return_place/start', timeout=self._start_service_timeout)
            if ok:
                self.current_state = 'green_return_place'
                self.iscc_phase = 'green_return_place'
                self.progress_info['current_task'] = 'Green Return Place'
                self.progress_info['progress'] = 'Return, search tag, place block'
                self.publish_competition_status('green_return_place_started')
            else:
                self.current_state = 'green_pick_completed'
                self.iscc_phase = 'green_pick_completed'
                self.progress_info['current_task'] = 'Green Pick'
                self.progress_info['progress'] = 'Green block picked; green_return_place start failed'
                self.publish_competition_status('green_return_place_start_failed')
        if task_name == 'green_return_place' and status == 'completed_place':
            rospy.loginfo('green_return_place completed_place -> finish ISCC')
            self.call_service('/green_return_place/stop', timeout=self._stop_service_timeout)
            self.call_service('/orange_line_follow/stop', timeout=self._stop_service_timeout)
            self.current_state = 'green_return_place'
            self.iscc_phase = 'finished'
            self.competition_finished = True
            self.progress_info['current_task'] = 'Green Return Place'
            self.progress_info['progress'] = 'Completed place block'
            if 'Green Return Place' not in self.progress_info['completed_tasks']:
                self.progress_info['completed_tasks'].append('Green Return Place')
            self.publish_competition_status('iscc_finished')
    def set_servo(self, pan, tilt):
        try:
            self.motion_manager.set_servos_position(200, [[23, pan], [24, tilt]])
        except Exception as e:
            rospy.logerr('Servo control error: %s', e)

    def publish_competition_status(self, status):
        status_msg = String()
        status_msg.data = status
        self.competition_status_pub.publish(status_msg)

        progress_msg = String()
        progress_msg.data = (
            f"Task: {self.progress_info['current_task']}, "
            f"Progress: {self.progress_info['progress']}, "
            f"Phase: {self.iscc_phase}, "
            f"Detected: {','.join(self.progress_info['detected_objects'])}"
        )
        self.competition_progress_pub.publish(progress_msg)

    def call_service(self, service_name, timeout=5.0):
        try:
            rospy.wait_for_service(service_name, timeout=timeout)
            proxy = rospy.ServiceProxy(service_name, Trigger)
            resp = proxy(TriggerRequest())
            return resp.success
        except Exception as e:
            rospy.logwarn('Service call failed: %s (%s)', service_name, e)
            return False

    def acquire_session_rqt_result_view(self):
        if not self.auto_open_rqt_result or self._session_rqt_acquired:
            return
        try:
            acquire_shared_rqt_result_view(self.rqt_session_owner_id, self.result_image_topic)
            self._session_rqt_acquired = True
        except Exception as e:
            rospy.logwarn('controller failed to acquire shared rqt_image_view: %s', e)

    def release_session_rqt_result_view(self):
        if not self.auto_open_rqt_result or not self._session_rqt_acquired:
            return
        try:
            release_shared_rqt_result_view(self.rqt_session_owner_id)
        except Exception as e:
            rospy.logwarn('controller failed to release shared rqt_image_view: %s', e)
        finally:
            self._session_rqt_acquired = False

    def get_blue_object(self):
        for obj in self.objects_info:
            if getattr(obj, 'label', '') == 'blue':
                return obj
        return None

    def get_stairs_object(self, color_name):
        # 选“最靠近”的台阶 side：取最大 y 的那条，避免列表里先出现远处小标志导致一直不过阈值
        color_name = str(color_name)
        best = None
        best_y = -1
        for obj in self.objects_info:
            if getattr(obj, 'type', '') == 'side' and getattr(obj, 'label', '') == color_name:
                yv = self.side_y_max(obj, fallback_y=0)
                if yv > best_y:
                    best_y = yv
                    best = obj
        return best

    def _pixel_frame_width(self):
        """与跨栏/台阶一致：/object/pixel_coords 中 x,y 为原图坐标，用 ObjectInfo.width 作幅宽。"""
        for obj in self.objects_info:
            w = int(getattr(obj, 'width', 0) or 0)
            if w > 0:
                return w
        return int(self.process_size[0]) if len(self.process_size) > 0 else 160

    def side_y_max(self, obj, fallback_y=0):
        """优先用 side 的端点最大 y，避免横线/斜线时中心 y 偏小导致迟迟不过阈值。"""
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

    def orange_tag_roi_px(self):
        roi = self.orange_tag_roi if isinstance(self.orange_tag_roi, list) and len(self.orange_tag_roi) == 4 else [0, 1, 0, 1]
        frame_h = max(1, int(self.orange_tag_frame_height))
        frame_w = max(1, int(self.orange_tag_frame_width))
        y_min = int(self.parse_ratio(roi[0]) * frame_h)
        y_max = int(self.parse_ratio(roi[1]) * frame_h)
        x_min = int(self.parse_ratio(roi[2]) * frame_w)
        x_max = int(self.parse_ratio(roi[3]) * frame_w)
        return y_min, y_max, x_min, x_max

    def extract_tag_center(self, detection):
        corners = list(getattr(detection, 'corners', []))
        if len(corners) >= 4:
            cx = sum(float(c.x) for c in corners[:4]) / 4.0
            cy = sum(float(c.y) for c in corners[:4]) / 4.0
            return cx, cy
        centre = getattr(detection, 'centre', None)
        if centre is None:
            centre = getattr(detection, 'center', None)
        if centre is not None:
            return float(getattr(centre, 'x', 0.0)), float(getattr(centre, 'y', 0.0))
        return None

    def select_best_orange_tag(self, detections):
        best = None
        best_y = -1.0
        y_min, y_max, x_min, x_max = self.orange_tag_roi_px()
        for detection in detections:
            ids = list(getattr(detection, 'id', []))
            if self.orange_tag_target_id >= 0 and self.orange_tag_target_id not in ids:
                continue
            center = self.extract_tag_center(detection)
            if center is None:
                continue
            cx, cy = center
            if cx < x_min or cx > x_max or cy < y_min or cy > y_max:
                continue
            if cy > best_y:
                best_y = cy
                best = (
                    ids[0] if len(ids) > 0 else -1,
                    center,
                )
        return best

    def orange_tag_callback(self, msg):
        best = self.select_best_orange_tag(getattr(msg, 'detections', []))
        if best is None:
            return
        self.orange_tag_id, self.orange_tag_center = best
        self._orange_tag_seen_time = rospy.get_time()
        first_seen_for_slowdown = False
        if self.iscc_phase == 'await_orange_guide' and not self.orange_tag_seen_once:
            self.orange_tag_seen_once = True
            first_seen_for_slowdown = True
            self.ensure_line_follow_gait_profile()
        if (
            self.iscc_phase == 'await_orange_guide'
            and (
                self.orange_tag_log_throttle_sec <= 0.0
                or (self._orange_tag_seen_time - self._last_orange_tag_detect_log_time) >= self.orange_tag_log_throttle_sec
            )
        ):
            tag_x, tag_y = self.orange_tag_center
            rospy.loginfo(
                'tag x=%.1f y=%.1f',
                float(tag_x),
                float(tag_y),
            )
            self._last_orange_tag_detect_log_time = self._orange_tag_seen_time
        if first_seen_for_slowdown:
            rospy.loginfo(
                'await_orange_guide: first tag seen -> switch line_follow gait to slow profile (%s)',
                self.await_orange_guide_tag_seen_gait_profile,
            )

    def clear_orange_tag_tracking(self, reset_seen_once=False):
        self.orange_tag_history = []
        self.orange_tag_center = None
        self.orange_tag_id = -1
        self._orange_tag_seen_time = 0.0
        if reset_seen_once:
            self.orange_tag_seen_once = False

    def orange_tag_enter_y_threshold_px(self):
        ty = float(self.orange_tag_enter_y_threshold)
        if ty <= 1.0:
            ty = float(ty) * float(max(1, self.orange_tag_frame_height))
            rospy.logwarn_once(
                'orange_guide.tag_trigger.enter_y_threshold <= 1, interpreted as ratio using frame height'
            )
        return int(ty)

    def debug_side_labels(self):
        labels = []
        for obj in self.objects_info:
            if getattr(obj, 'type', '') == 'side':
                labels.append(str(getattr(obj, 'label', '')))
        return labels

    def debug_detected_objects(self):
        """返回当前帧检测到的对象摘要，便于排查 await_stairs_descend 阶段是否真的识别到了目标。"""
        items = []
        for obj in self.objects_info:
            otype = str(getattr(obj, 'type', ''))
            label = str(getattr(obj, 'label', ''))
            x = int(getattr(obj, 'x', 0))
            y = int(getattr(obj, 'y', 0))
            area = int(getattr(obj, 'area', 0))
            items.append('%s/%s(x=%d,y=%d,area=%d)' % (otype, label, x, y, area))
        return items

    def is_blue_stable(self, center):
        self.blue_history.append(center)
        if len(self.blue_history) > self.stability_frames:
            self.blue_history.pop(0)
        if len(self.blue_history) < self.stability_frames:
            return False
        max_dev = max(abs(center[0] - p[0]) + abs(center[1] - p[1]) for p in self.blue_history)
        return max_dev < self.position_tolerance

    def is_orange_tag_stable(self, center):
        self.orange_tag_history.append(center)
        if len(self.orange_tag_history) > self.orange_tag_stable_frames:
            self.orange_tag_history.pop(0)
        if len(self.orange_tag_history) < self.orange_tag_stable_frames:
            return False
        max_dev = max(abs(center[0] - p[0]) + abs(center[1] - p[1]) for p in self.orange_tag_history)
        return max_dev < self.position_tolerance

    def get_purple_sandbag_object(self):
        best = None
        best_y = -1
        for obj in self.objects_info:
            if (
                getattr(obj, 'type', '') == self.sandbag_detect_type
                and getattr(obj, 'label', '') == self.sandbag_purple_color
            ):
                yv = float(getattr(obj, 'y', 0))
                if yv > best_y:
                    best_y = yv
                    best = obj
        return best

    def get_green_object(self):
        best = None
        best_y = -1
        for obj in self.objects_info:
            if (
                getattr(obj, 'type', '') == self.green_detect_type
                and getattr(obj, 'label', '') == self.green_color
            ):
                yv = float(getattr(obj, 'y', 0))
                if yv > best_y:
                    best_y = yv
                    best = obj
        return best

    def purple_enter_y_threshold_px(self, obj):
        """与其他门控一致：>1 直接当像素；<=1 按 ObjectInfo.height 的比例解释；判中心 y >= 阈值。"""
        frame_h = int(getattr(obj, 'height', self.image_height)) or self.image_height
        ty = float(self.sandbag_enter_y_threshold)
        if ty <= 1.0:
            ty = float(ty) * float(frame_h)
            rospy.logwarn_once(
                'sandbag_throw.enter_y_threshold <= 1, interpreted as ratio using frame height'
            )
        return int(ty)

    def is_purple_stable(self, center):
        self.purple_history.append(center)
        n = self.sandbag_stability_frames
        if len(self.purple_history) > n:
            self.purple_history.pop(0)
        if len(self.purple_history) < n:
            return False
        max_dev = max(abs(center[0] - p[0]) + abs(center[1] - p[1]) for p in self.purple_history)
        return max_dev < self.sandbag_position_tolerance

    def green_enter_y_threshold_px(self, obj):
        frame_h = int(getattr(obj, 'height', self.image_height)) or self.image_height
        ty = float(self.green_enter_y_threshold)
        if ty <= 1.0:
            ty = float(ty) * float(frame_h)
            rospy.logwarn_once('green_pick_gate.enter_y_threshold <= 1, interpreted as ratio using frame height')
        return int(ty)

    def is_green_stable(self, center):
        self.green_history.append(center)
        n = self.green_stability_frames
        if len(self.green_history) > n:
            self.green_history.pop(0)
        if len(self.green_history) < n:
            return False
        max_dev = max(abs(center[0] - p[0]) + abs(center[1] - p[1]) for p in self.green_history)
        return max_dev < self.green_position_tolerance

    def is_blue_past_hurdles_gate(self, y_val):
        frame_h = int(getattr(self.get_blue_object(), 'height', self.image_height)) or self.image_height
        gate_y = int(self.hurdles_gate_enter_y_threshold)
        if gate_y <= 1:
            gate_y = int(float(self.hurdles_gate_enter_y_threshold) * frame_h)
            rospy.logwarn_once('hurdles.gate.enter_y_threshold <= 1, interpreted as ratio using frame height')
        if not self._coord_frame_logged and frame_h > 0:
            rospy.loginfo(
                'Controller coord frame detected: height=%d, hurdles_gate_enter_y=%d',
                frame_h,
                gate_y,
            )
            self._coord_frame_logged = True
        now = rospy.get_time()
        if now - self._last_y_log_time >= self.y_log_throttle_sec:
            rospy.loginfo('blue_hurdles y=%d', y_val)
            self._last_y_log_time = now
        return y_val >= gate_y

    def is_stairs_past_gate(self, y_val, stairs_obj, configured_y):
        frame_h = int(getattr(stairs_obj, 'height', self.image_height)) or self.image_height
        ty = int(configured_y)
        if ty <= 1:
            ty = int(float(configured_y) * frame_h)
            rospy.logwarn_once('stairs gate enter_y_threshold <= 1, interpreted as ratio using frame height')
        now = rospy.get_time()
        if now - self._last_stairs_log_time >= self.y_log_throttle_sec:
            # 台阶 y 日志由各相位统一输出，避免重复刷屏
            self._last_stairs_log_time = now
        return y_val >= ty

    def start_line_following(self):
        self.ensure_line_follow_gait_profile()
        self.ensure_line_follow_roi_mode()
        self.ensure_line_follow_x_max_override()
        self.current_state = 'line_following'
        self.progress_info['current_task'] = 'Line Following'
        self.progress_info['progress'] = 'Following line...'
        self.publish_competition_status('line_following_started')
        return self.call_service('/line_follow/start', timeout=self._start_service_timeout)

    def stop_line_following(self):
        return self.call_service('/line_follow/stop', timeout=self._stop_service_timeout)

    def desired_line_follow_gait_profile(self):
        if self.iscc_phase == 'await_orange_guide':
            if self.orange_tag_seen_once:
                return self.await_orange_guide_tag_seen_gait_profile
            return self.await_orange_guide_gait_profile
        if self.wait_stairs_clear_descend and self.orange_auto_start:
            if self.orange_tag_seen_once:
                return self.await_orange_guide_tag_seen_gait_profile
            return self.await_orange_guide_gait_profile
        return self.line_follow_default_gait_profile

    def desired_line_follow_roi_mode(self):
        if self.iscc_phase == 'await_stairs_descend' and not self.wait_stairs_clear_descend:
            return 'down_only'
        return 'full'

    def desired_line_follow_x_max_override(self):
        if (
            self.iscc_phase == 'await_blue'
            and self.current_state == 'line_following'
            and self.get_blue_object() is not None
        ):
            return max(0.0, float(self.hurdles_slow_gait_x_max))
        if (
            self.iscc_phase == 'await_stairs_climb'
            and self.current_state == 'line_following'
        ):
            stairs = self.get_stairs_object(self.climb_stairs_color)
            if stairs is not None and not self.climb_slow_latched:
                self.climb_slow_latched = True
                rospy.loginfo('climb slow latched: color=%s', self.climb_stairs_color)
            if self.climb_slow_latched:
                return max(0.0, float(self.climb_stairs_slow_gait_x_max))
        if self.iscc_phase == 'await_stairs_descend' and not self.wait_stairs_clear_descend:
            stairs = self.get_stairs_object(self.descend_stairs_color)
            if stairs is not None and not self.descend_slow_latched:
                self.descend_slow_latched = True
                rospy.loginfo('descend slow latched: color=%s', self.descend_stairs_color)
            return max(0.0, float(self.descend_stairs_slow_gait_x_max))
        return 0.0

    def ensure_line_follow_gait_profile(self):
        desired = self.desired_line_follow_gait_profile()
        current = str(rospy.get_param('/tasks/line_following/gait_profile', self.line_follow_default_gait_profile))
        if current == desired:
            return
        rospy.set_param('/tasks/line_following/gait_profile', desired)
        rospy.loginfo(
            'line_follow gait profile -> %s (phase=%s wait_stairs_clear_descend=%s tag_seen_once=%s)',
            desired,
            self.iscc_phase,
            self.wait_stairs_clear_descend,
            self.orange_tag_seen_once,
        )

    def ensure_line_follow_roi_mode(self):
        desired = self.desired_line_follow_roi_mode()
        current = str(rospy.get_param('/iscc/line_follow_roi_mode', 'full')).strip().lower()
        if current == desired:
            return
        rospy.set_param('/iscc/line_follow_roi_mode', desired)
        rospy.loginfo(
            'line_follow roi mode -> %s (phase=%s wait_stairs_clear_descend=%s)',
            desired,
            self.iscc_phase,
            self.wait_stairs_clear_descend,
        )

    def ensure_line_follow_x_max_override(self):
        desired = float(self.desired_line_follow_x_max_override())
        current = float(rospy.get_param('/iscc/line_follow_x_max_override', 0.0))
        if abs(current - desired) < 1e-9:
            return
        rospy.set_param('/iscc/line_follow_x_max_override', desired)
        rospy.loginfo(
            'line_follow x_max override -> %.4f (phase=%s wait_stairs_clear_descend=%s)',
            desired,
            self.iscc_phase,
            self.wait_stairs_clear_descend,
        )

    def start_hurdles(self):
        self.current_state = 'hurdles'
        self.task_completed['blue_object'] = False
        self.progress_info['current_task'] = 'Hurdles'
        self.progress_info['progress'] = 'Executing hurdles...'
        self.stop_line_following()
        time.sleep(0.5)
        self.publish_competition_status('hurdles_started')
        return self.call_service('/hurdles/start', timeout=self._start_service_timeout)

    def start_stairs_climb(self):
        self.climb_slow_latched = False
        self.current_state = 'stairs_climb'
        self.task_completed['climb_stairs'] = False
        self.progress_info['current_task'] = 'Climb Stairs'
        self.progress_info['progress'] = 'Approaching stairs (climb)...'
        self.stop_line_following()
        time.sleep(0.5)
        self.publish_competition_status('stairs_climb_started')
        return self.call_service('/climb_stairs/start', timeout=self._start_service_timeout)

    def start_stairs_descend(self):
        self.descend_slow_latched = False
        self.current_state = 'stairs_descend'
        self.task_completed['descend_stairs'] = False
        self.progress_info['current_task'] = 'Descend Stairs'
        self.progress_info['progress'] = 'Approaching stairs (descend)...'
        self.stop_line_following()
        time.sleep(0.5)
        self.publish_competition_status('stairs_descend_started')
        return self.call_service('/descend_stairs/start', timeout=self._start_service_timeout)

    def start_orange_guide(self):
        """先停巡线再启动 A 阶段；仅当 /orange_guide/start 成功后才置 current_state，避免失败时卡在非 line_following 永不再检测 tag 入口。"""
        self.task_completed['orange_guide'] = False
        self.progress_info['current_task'] = 'Orange Guide'
        self.progress_info['progress'] = 'Single-line guide to dual-line stop'
        self.stop_line_following()
        time.sleep(0.5)
        ok = self.call_service('/orange_guide/start', timeout=self._start_service_timeout)
        if ok:
            self.current_state = 'orange_guide'
            self.publish_competition_status('orange_guide_started')
        else:
            rospy.logwarn('orange_guide/start failed, resume line_following in await_orange_guide')
            self.current_state = 'line_following'
            self.start_line_following()
        return ok

    def start_orange_line_follow(self):
        """A 结束后进入 B：巡橙色双线；失败则退回白线巡线。"""
        self.progress_info['current_task'] = 'Orange Line Follow'
        self.progress_info['progress'] = 'B phase: dual orange corridor'
        self.call_service('/orange_guide/stop', timeout=self._stop_service_timeout)
        time.sleep(0.4)
        ok = self.call_service('/orange_line_follow/start', timeout=self._start_service_timeout)
        if ok:
            self.current_state = 'orange_line_follow'
            self.iscc_phase = 'orange_line_following'
            self.publish_competition_status('orange_line_follow_started')
            rospy.loginfo('ISCC phase: orange_line_following (B started)')
        else:
            rospy.logwarn('orange_line_follow/start failed, resume white line following')
            self.current_state = 'line_following'
            self.iscc_phase = 'finished'
            self.competition_finished = True
            self.start_line_following()
        return ok

    def start_green_return_place_direct(self):
        self.progress_info['current_task'] = 'Green Return Place'
        self.progress_info['progress'] = 'Return, search tag, place block'
        ok = self.call_service('/green_return_place/start', timeout=self._start_service_timeout)
        if ok:
            self.current_state = 'green_return_place'
            self.iscc_phase = 'green_return_place'
            self.publish_competition_status('green_return_place_started')
            rospy.loginfo('ISCC phase: green_return_place (direct start)')
        else:
            rospy.logwarn('green_return_place/start failed (direct start)')
        return ok

    def configure_watch_flags_for_start_stage(self):
        purple_watch = True
        green_watch = False
        if self.start_stage == 'green':
            purple_watch = False
            green_watch = True
        elif self.start_stage in ('green_return_place', 'return'):
            purple_watch = False
            green_watch = False
        rospy.set_param('/iscc/sandbag_purple_watch_active', purple_watch)
        rospy.set_param('/iscc/green_watch_active', green_watch)
        rospy.loginfo(
            'start_stage watch flags -> purple=%s green=%s (start_stage=%s)',
            str(purple_watch),
            str(green_watch),
            self.start_stage,
        )

    def start_initial_stage(self):
        if self.start_stage in ('sandbag', 'green'):
            return self.start_orange_line_follow()
        if self.start_stage in ('green_return_place', 'return'):
            return self.start_green_return_place_direct()
        return self.start_line_following()

    def wait_for_orange_guide_completion(self):
        while self.running and not rospy.is_shutdown():
            if self.task_completed['orange_guide']:
                return
            time.sleep(0.1)

    def wait_for_hurdles_completion(self):
        while self.running and not rospy.is_shutdown():
            if self.task_completed['blue_object']:
                return
            time.sleep(0.1)

    def wait_for_climb_completion(self):
        while self.running and not rospy.is_shutdown():
            if self.task_completed['climb_stairs']:
                return
            time.sleep(0.1)

    def wait_for_descend_completion(self):
        while self.running and not rospy.is_shutdown():
            if self.task_completed['descend_stairs']:
                return
            time.sleep(0.1)

    def get_startup_sequence_desc(self):
        if self.start_stage == 'sandbag':
            return 'orange_line_follow (B) -> sandbag_throw -> green_pick -> green_return_place'
        if self.start_stage == 'green':
            return 'orange_line_follow (B) -> green_pick -> green_return_place'
        if self.start_stage in ('green_return_place', 'return'):
            return 'green_return_place -> orange_line_follow -> place_block'
        phase_to_pipeline = {
            'await_blue': 'line -> hurdles -> climb -> descend',
            'await_stairs_climb': 'line -> climb -> descend',
            'await_stairs_descend': 'line -> descend',
            'await_orange_guide': 'line -> orange_guide -> orange_line_follow',
            'orange_line_following': 'orange_line_follow (B) -> optional sandbag_throw',
            'sandbag_throw': 'sandbag_throw (after purple gate)',
            'green_pick': 'green_pick (align + pick)',
            'green_return_place': 'green_return_place -> orange_line_follow -> place_block',
        }
        return phase_to_pipeline.get(self.iscc_phase, 'line')

    def monitor_iscc_sequence(self):
        while self.running and not rospy.is_shutdown() and not self.competition_finished:
            try:
                if self.iscc_phase == 'await_blue':
                    if self.current_state != 'line_following':
                        time.sleep(0.05)
                        continue
                    self.ensure_line_follow_x_max_override()

                    if self.wait_blue_clear:
                        if self.get_blue_object() is None:
                            self.wait_blue_clear = False
                            self.blue_history = []
                            self.iscc_phase = 'await_stairs_climb'
                            rospy.loginfo('ISCC phase: await_stairs_climb')
                        time.sleep(0.02)
                        continue

                    blue_obj = self.get_blue_object()
                    if blue_obj is None:
                        self.blue_history = []
                        self.progress_info['detected_objects'] = []
                        time.sleep(0.02)
                        continue

                    cx = int(getattr(blue_obj, 'x', 0))
                    cy = int(getattr(blue_obj, 'y', 0))
                    now = rospy.get_time()
                    if now - self._last_seen_log_time >= self.y_log_throttle_sec:
                        rospy.loginfo('blue_hurdles y=%d', cy)
                        self._last_seen_log_time = now
                    self.progress_info['detected_objects'] = ['blue']

                    if self.is_blue_stable((cx, cy)) and self.is_blue_past_hurdles_gate(cy):
                        rospy.loginfo('blue_hurdles y=%d', cy)
                        if self.start_hurdles():
                            self.wait_for_hurdles_completion()
                            self.wait_blue_clear = True
                            self.blue_history = []
                            self.progress_info['current_task'] = 'Line Following'
                            self.progress_info['progress'] = 'Wait blue clear after hurdles'
                            self.publish_competition_status('line_following_resumed')
                            self.start_line_following()
                    time.sleep(0.02)
                    continue

                if self.iscc_phase == 'await_stairs_climb':
                    if self.current_state != 'line_following':
                        now = rospy.get_time()
                        if now - self._last_phase_log_time >= 1.0:
                            rospy.loginfo(
                                'await_stairs_climb: waiting line_following, current_state=%s',
                                self.current_state,
                            )
                            self._last_phase_log_time = now
                        time.sleep(0.05)
                        continue
                    self.ensure_line_follow_x_max_override()

                    stairs = self.get_stairs_object(self.climb_stairs_color)
                    if stairs is None:
                        self.progress_info['detected_objects'] = []
                        now = rospy.get_time()
                        if now - self._last_stairs_log_time >= self.y_log_throttle_sec:
                            rospy.loginfo(
                                'await_stairs_climb: no %s stairs side, side labels=%s',
                                self.climb_stairs_color,
                                self.debug_side_labels(),
                            )
                            self._last_stairs_log_time = now
                        time.sleep(0.02)
                        continue

                    scy = self.side_y_max(stairs, fallback_y=0)
                    now = rospy.get_time()
                    if now - self._last_seen_log_time >= self.y_log_throttle_sec:
                        rospy.loginfo('stairs_climb %s y=%d', self.climb_stairs_color, scy)
                        self._last_seen_log_time = now
                    self.progress_info['detected_objects'] = ['stairs_marker']

                    # 上下台阶主控切换逻辑保持一致：主要看台阶标志 y 过任务 gate 阈值
                    if self.is_stairs_past_gate(scy, stairs, self.climb_stairs_gate_enter_y_threshold):
                        rospy.loginfo('stairs_climb %s y=%d', self.climb_stairs_color, scy)
                        if self.start_stairs_climb():
                            self.wait_for_climb_completion()
                            self.iscc_phase = 'await_stairs_descend'
                            self.descend_slow_latched = False
                            self._descend_gate_enable_time = rospy.get_time() + max(
                                0.0, self.post_climb_descend_gate_delay_sec
                            )
                            self._descend_gate_delay_logged = False
                            rospy.loginfo(
                                'climb completed -> await_stairs_descend (gate_delay=%.2fs)',
                                self.post_climb_descend_gate_delay_sec,
                            )
                            self.progress_info['progress'] = 'Resumed after climb, seek descend cue'
                            self.start_line_following()
                    time.sleep(0.02)
                    continue

                if self.iscc_phase == 'await_stairs_descend':
                    now = rospy.get_time()
                    if now - self._last_descend_heartbeat_time >= 1.0:
                        rospy.loginfo(
                            'await_stairs_descend heartbeat: current_state=%s wait_clear=%s objects=%d',
                            self.current_state,
                            self.wait_stairs_clear_descend,
                            len(self.objects_info),
                        )
                        self._last_descend_heartbeat_time = now
                    if self.current_state != 'line_following':
                        if now - self._last_phase_log_time >= 1.0:
                            rospy.loginfo(
                                'await_stairs_descend: waiting line_following, current_state=%s',
                                self.current_state,
                            )
                            self._last_phase_log_time = now
                        time.sleep(0.05)
                        continue
                    self.ensure_line_follow_x_max_override()

                    if now < self._descend_gate_enable_time:
                        if not self._descend_gate_delay_logged:
                            rospy.loginfo(
                                'descend gate delayed after climb: remain=%.2fs',
                                self._descend_gate_enable_time - now,
                            )
                            self._descend_gate_delay_logged = True
                        time.sleep(0.02)
                        continue

                    if self.wait_stairs_clear_descend:
                        if self.get_stairs_object(self.descend_stairs_color) is None:
                            self.wait_stairs_clear_descend = False
                            self.climb_slow_latched = False
                            self.descend_slow_latched = False
                            self._descend_gate_enable_time = 0.0
                            self._descend_gate_delay_logged = False
                            if self.orange_auto_start:
                                self.iscc_phase = 'await_orange_guide'
                                self.clear_orange_tag_tracking(reset_seen_once=True)
                                rospy.loginfo('ISCC phase: await_orange_guide')
                                self.progress_info['progress'] = 'Seek AprilTag to start A guide'
                                # 下台阶完成后已在 start_stairs_descend 流程末启动巡线，勿重复 /line_follow/start
                            else:
                                self.iscc_phase = 'finished'
                                self.competition_finished = True
                                rospy.loginfo('ISCC phase: finished (after descend)')
                                self.progress_info['progress'] = 'ISCC sequence done'
                                self.publish_competition_status('iscc_finished')
                                self.start_line_following()
                        time.sleep(0.02)
                        continue

                    stairs = self.get_stairs_object(self.descend_stairs_color)
                    if stairs is None:
                        now = rospy.get_time()
                        if now - self._last_stairs_log_time >= self.y_log_throttle_sec:
                            rospy.loginfo(
                                'await_stairs_descend: no %s stairs side, side labels=%s, detected=%s',
                                self.descend_stairs_color,
                                self.debug_side_labels(),
                                self.debug_detected_objects(),
                            )
                            self._last_stairs_log_time = now
                        time.sleep(0.02)
                        continue

                    scy = self.side_y_max(stairs, fallback_y=0)
                    now = rospy.get_time()
                    if now - self._last_stairs_log_time >= self.y_log_throttle_sec:
                        rospy.loginfo('await_stairs_descend %s y=%d', self.descend_stairs_color, scy)
                        self._last_stairs_log_time = now
                    # 下台阶切换：看 y 过线
                    if self.is_stairs_past_gate(scy, stairs, self.descend_stairs_gate_enter_y_threshold):
                        rospy.loginfo('stairs_descend %s y=%d', self.descend_stairs_color, scy)
                        if self.start_stairs_descend():
                            self.wait_for_descend_completion()
                            self.wait_stairs_clear_descend = True
                            self._descend_gate_enable_time = 0.0
                            self._descend_gate_delay_logged = False
                            self.start_line_following()
                    time.sleep(0.02)
                    continue

                if self.iscc_phase == 'await_orange_guide':
                    if self.current_state != 'line_following':
                        time.sleep(0.05)
                        continue

                    if not self.orange_tag_trigger_enabled:
                        now = rospy.get_time()
                        if now - self._last_orange_xy_log_time >= self.orange_tag_log_throttle_sec:
                            rospy.logwarn('await_orange_guide: tag trigger disabled, no fallback trigger configured')
                            self._last_orange_xy_log_time = now
                        time.sleep(0.05)
                        continue

                    now = rospy.get_time()
                    if (
                        self._orange_tag_seen_time <= 0.0
                        or (self.orange_tag_lost_timeout_sec > 0.0
                            and (now - self._orange_tag_seen_time) > self.orange_tag_lost_timeout_sec)
                    ):
                        self.clear_orange_tag_tracking(reset_seen_once=False)
                        self.progress_info['detected_objects'] = []
                        time.sleep(0.02)
                        continue

                    tag_x, tag_y = self.orange_tag_center
                    tag_center = (float(tag_x), float(tag_y))
                    tag_threshold = self.orange_tag_enter_y_threshold_px()
                    tag_stable = self.is_orange_tag_stable(tag_center)
                    tag_y_ready = float(tag_y) >= float(tag_threshold)
                    self.progress_info['detected_objects'] = ['tag:%d' % int(self.orange_tag_id)]
                    if now - self._last_orange_xy_log_time >= self.orange_tag_log_throttle_sec:
                        rospy.loginfo(
                            'tag x=%.1f y=%.1f',
                            float(tag_x),
                            float(tag_y),
                        )
                        self._last_orange_xy_log_time = now
                    if tag_stable and tag_y_ready:
                        rospy.loginfo(
                            'AprilTag y gate passed, switching to orange_guide (id=%d x=%.1f y=%.1f threshold=%d)',
                            int(self.orange_tag_id),
                            float(tag_x),
                            float(tag_y),
                            tag_threshold,
                        )
                        if self.start_orange_guide():
                            self.wait_for_orange_guide_completion()
                            self.gait_manager.stop()
                            self.start_orange_line_follow()
                    time.sleep(0.02)
                    continue

                if self.iscc_phase == 'orange_line_following' and not bool(
                    rospy.get_param('/iscc/green_watch_active', False)
                ):
                    if not self.sandbag_throw_enabled:
                        time.sleep(0.1)
                        continue
                    if not bool(rospy.get_param('/iscc/sandbag_purple_watch_active', True)):
                        time.sleep(0.1)
                        continue
                    if self.current_state != 'orange_line_follow':
                        time.sleep(0.05)
                        continue

                    purple = self.get_purple_sandbag_object()
                    if purple is None:
                        self.purple_history = []
                        time.sleep(0.02)
                        continue

                    pcx = int(getattr(purple, 'x', 0))
                    pcy = int(getattr(purple, 'y', 0))
                    thr_px = self.purple_enter_y_threshold_px(purple)
                    stable = self.is_purple_stable((pcx, pcy))
                    y_ready = pcy >= thr_px
                    self.progress_info['detected_objects'] = [self.sandbag_purple_color]
                    now = rospy.get_time()
                    if now - self._last_purple_log_time >= self.purple_sandbag_y_log_throttle:
                        rospy.loginfo(
                            'purple x=%d y=%d',
                            pcx,
                            pcy,
                        )
                        self._last_purple_log_time = now

                    if stable and y_ready:
                        rospy.loginfo(
                            'Purple sandbag stable + y gate, switching to sandbag_throw (y=%d threshold=%d)',
                            pcy,
                            thr_px,
                        )
                        rospy.set_param('/iscc/sandbag_purple_watch_active', False)
                        self.purple_history = []
                        self.call_service('/orange_line_follow/stop', timeout=self._stop_service_timeout)
                        time.sleep(0.4)
                        ok = self.call_service('/sandbag_throw/start', timeout=self._start_service_timeout)
                        if ok:
                            self.current_state = 'sandbag_throw'
                            self.iscc_phase = 'sandbag_throw'
                            self.progress_info['current_task'] = 'Sandbag Throw'
                            self.progress_info['progress'] = 'Pick, align, throw, and rejoin'
                            self.publish_competition_status('sandbag_throw_started')
                        else:
                            rospy.logwarn(
                                'sandbag_throw/start failed, re-enable purple watch and resume orange_line_follow'
                            )
                            rospy.set_param('/iscc/sandbag_purple_watch_active', True)
                            self.call_service('/orange_line_follow/start', timeout=self._start_service_timeout)
                    time.sleep(0.02)
                    continue

                if self.iscc_phase == 'orange_line_following':
                    if not self.green_gate_enabled:
                        time.sleep(0.05)
                        continue
                    if not bool(rospy.get_param('/iscc/green_watch_active', False)):
                        time.sleep(0.05)
                        continue
                    if self.current_state != 'orange_line_follow':
                        time.sleep(0.05)
                        continue
                    green = self.get_green_object()
                    if green is None:
                        self.green_history = []
                        time.sleep(0.02)
                        continue
                    gcx = int(getattr(green, 'x', 0))
                    gcy = int(getattr(green, 'y', 0))
                    gthr = self.green_enter_y_threshold_px(green)
                    gstable = self.is_green_stable((gcx, gcy))
                    gy_ready = gcy >= gthr
                    now = rospy.get_time()
                    if now - self._last_green_log_time >= self.green_y_log_throttle:
                        rospy.loginfo(
                            'green x=%d y=%d',
                            gcx,
                            gcy,
                        )
                        self._last_green_log_time = now
                    if gstable and gy_ready:
                        rospy.loginfo(
                            'Green gate passed, switching to green_pick (y=%d threshold=%d)',
                            gcy,
                            gthr,
                        )
                        self.call_service('/orange_line_follow/stop', timeout=self._stop_service_timeout)
                        time.sleep(0.2)
                        ok = self.call_service('/green_pick/start', timeout=self._start_service_timeout)
                        if ok:
                            self.current_state = 'green_pick'
                            self.iscc_phase = 'green_pick'
                            self.progress_info['current_task'] = 'Green Pick'
                            self.progress_info['progress'] = 'Align + pick'
                            self.publish_competition_status('green_pick_started')
                    time.sleep(0.02)
                    continue

                time.sleep(0.05)
            except Exception as e:
                rospy.logerr('Monitor loop error: %s', e)
                time.sleep(0.1)

    def shutdown(self, signum, frame):
        with self.lock:
            self.cleanup(announce_status='shutdown')
            common.loginfo('%s shutdown' % self.name)

    def cleanup(self, announce_status=None):
        with self.lock:
            if self._cleanup_done:
                return
            self._cleanup_done = True
            self.running = False
            self.climb_slow_latched = False
            self.descend_slow_latched = False
            self.stop_line_following()
            self.call_service('/hurdles/stop', timeout=self._stop_service_timeout)
            self.call_service('/climb_stairs/stop', timeout=self._stop_service_timeout)
            self.call_service('/descend_stairs/stop', timeout=self._stop_service_timeout)
            self.call_service('/orange_guide/stop', timeout=self._stop_service_timeout)
            self.call_service('/orange_line_follow/stop', timeout=self._stop_service_timeout)
            self.call_service('/sandbag_throw/stop', timeout=self._stop_service_timeout)
            self.call_service('/green_pick/stop', timeout=self._stop_service_timeout)
            self.call_service('/green_return_place/stop', timeout=self._stop_service_timeout)
            self.motion_manager.run_action('stand')
            self.release_session_rqt_result_view()
            if announce_status:
                self.publish_competition_status(announce_status)

    def run(self):
        rospy.loginfo('Starting ISCC: %s (start_stage=%s)', self.get_startup_sequence_desc(), self.start_stage)
        self.configure_watch_flags_for_start_stage()
        self.stop_line_following()
        time.sleep(self._startup_run_gap_sec)
        self.call_service('/hurdles/stop', timeout=self._stop_service_timeout)
        self.call_service('/climb_stairs/stop', timeout=self._stop_service_timeout)
        self.call_service('/descend_stairs/stop', timeout=self._stop_service_timeout)
        self.call_service('/orange_guide/stop', timeout=self._stop_service_timeout)
        self.call_service('/orange_line_follow/stop', timeout=self._stop_service_timeout)
        self.call_service('/sandbag_throw/stop', timeout=self._stop_service_timeout)
        self.call_service('/green_pick/stop', timeout=self._stop_service_timeout)
        self.call_service('/green_return_place/stop', timeout=self._stop_service_timeout)
        self.acquire_session_rqt_result_view()

        if self.start_initial_stage():
            monitor_thread = threading.Thread(target=self.monitor_iscc_sequence)
            monitor_thread.daemon = True
            monitor_thread.start()
        else:
            self.release_session_rqt_result_view()

        while self.running and not rospy.is_shutdown():
            if self.competition_finished:
                rospy.loginfo('ISCC sequence completed')
                break
            time.sleep(0.1)

        self.cleanup()
        rospy.signal_shutdown('iscc finished')


if __name__ == '__main__':
    try:
        IsccController('iscc_controller').run()
    except Exception as e:
        rospy.logerr('ISCC controller initialization error: %s', e)
