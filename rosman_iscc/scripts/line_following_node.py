#!/usr/bin/env python3
# encoding: utf-8
# @Author: Dejin Xuan
# @Date: 2026.4.1   
# 巡线节点
import math
import os
import signal
import sys
import time
import rospy
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger, TriggerResponse
from rosman_sdk import common
from rosman_example.color_common import Common
from rosman_example.visual_patrol import VisualPatrol
from rosman_interfaces.msg import ObjectsInfo, ColorDetect
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


class LineFollowing(Common):
    def __init__(self, name):
        rospy.init_node(name)
        self.name = name
        self.running = True
        self.active = False
        self.objects_info = []

        self.head_pan_init = rospy.get_param('/servo/head_pan_init', 500)
        self.head_tilt_look_down = rospy.get_param('/servo/head_tilt_look_down', 280)
        super().__init__(name, self.head_pan_init, self.head_tilt_look_down)

        self.visual_patrol = VisualPatrol(self.gait_manager)
        self.image_process_size = rospy.get_param('/image_processing/process_size', [160, 120])
        self.target_color = rospy.get_param('/tasks/line_following/color', 'white')
        self.result_image_topic = rospy.get_param('/tasks/line_following/result_image_topic', '/color_detection/image_result')
        self.auto_open_rqt_result = rospy.get_param('/tasks/line_following/auto_open_rqt_result', True)
        self.steering_gain = float(rospy.get_param('/tasks/line_following/steering_gain', 2.0))
        self.patrol_error_floor_px = float(rospy.get_param('/tasks/line_following/patrol_error_floor_px', 12))
        self.line_center_x_offset = self._load_line_center_x_offset()
        self.head_settle_sec = float(rospy.get_param('/tasks/line_following/head_settle_sec', 0.12))
        self.rqt_owner_id = 'line_following:%s:%d' % (self.name, os.getpid())
        self.gait_switch_off_reported = False
        self.line_roi = self.load_line_roi()
        self.line_roi_mode = 'full'
        self.line_follow_x_max_override = 0.0
        self.current_gait_profile = ''
        self.apply_gait_profile()

        signal.signal(signal.SIGINT, self.shutdown)
        self.setup_ros_interface()

        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)

        self.motion_manager.run_action('walk_ready')
        common.loginfo('Line following initialized (enter/低头在 /line_follow/start 时执行)')

    def setup_ros_interface(self):
        self.status_pub = rospy.Publisher('/line_follow/status', String, queue_size=1)
        self.active_pub = rospy.Publisher('/line_follow/active', Bool, queue_size=1)
        self.start_service = rospy.Service('/line_follow/start', Trigger, self.start_service_callback)
        self.stop_service = rospy.Service('/line_follow/stop', Trigger, self.stop_service_callback)
        self.pause_service = rospy.Service('/line_follow/pause', Trigger, self.pause_service_callback)
        self.resume_service = rospy.Service('/line_follow/resume', Trigger, self.resume_service_callback)
        self.control_sub = rospy.Subscriber('/line_follow/control', String, self.control_callback)

    def is_gait_switch_on(self, default=True):
        v = rospy.get_param('/gait_switch', default)
        if isinstance(v, bool):
            return v
        return str(v).lower() in ('true', '1', 'yes', 'on')

    def load_line_roi(self):
        line_roi = rospy.get_param('/tasks/line_following/line_roi', None)
        if not (isinstance(line_roi, list) and len(line_roi) == 3):
            raise ValueError(
                "invalid /tasks/line_following/line_roi. expected 3 segments, "
                "format: [[y_min,y_max,x_min,x_max], ... x3], "
                "example: [[7.0/12.0,8.0/12.0,1.0/4.0,3.0/4.0], "
                "[8.0/12.0,9.0/12.0,1.0/4.0,3.0/4.0], "
                "[9.0/12.0,10.0/12.0,1.0/4.0,3.0/4.0]]"
            )

        parsed = []
        for idx, roi in enumerate(line_roi):
            if not (isinstance(roi, list) and len(roi) == 4):
                raise ValueError(
                    f"invalid /tasks/line_following/line_roi[{idx}] format. "
                    "expected [y_min,y_max,x_min,x_max]"
                )
            vals = [self.parse_ratio(v) for v in roi]
            y_min, y_max, x_min, x_max = vals
            if not (0.0 <= y_min < y_max <= 1.0 and 0.0 <= x_min < x_max <= 1.0):
                raise ValueError(
                    f"invalid /tasks/line_following/line_roi[{idx}] range: {vals}. "
                    "require 0<=y_min<y_max<=1 and 0<=x_min<x_max<=1"
                )
            parsed.append(vals)
        return parsed

    def _load_line_center_x_offset(self):
        try:
            calib = common.get_yaml_data('/home/ubuntu/ros_ws/src/rosman_example/config/calib.yaml')
            return float(calib.get('center_x_offset', 0))
        except Exception:
            return 0.0

    def load_line_roi_mode(self):
        mode = str(rospy.get_param('/iscc/line_follow_roi_mode', 'full')).strip().lower()
        if mode in ('full', 'down_only'):
            return mode
        rospy.logwarn_once('line_following: invalid /iscc/line_follow_roi_mode, fallback to full')
        return 'full'

    def effective_line_roi(self):
        if self.line_roi_mode == 'down_only':
            down_roi = list(self.line_roi[2])
            return [list(down_roi), list(down_roi), list(down_roi)], 'line_roi[2]x3'
        return [list(self.line_roi[0]), list(self.line_roi[1]), list(self.line_roi[2])], 'line_roi[0,1,2]'

    def load_line_follow_x_max_override(self):
        try:
            x_max = float(rospy.get_param('/iscc/line_follow_x_max_override', 0.0))
        except Exception:
            rospy.logwarn_once('line_following: invalid /iscc/line_follow_x_max_override, fallback to 0')
            return 0.0
        return x_max if x_max > 0.0 else 0.0

    def patrol_corrected_x(self, line_data):
        """与 VisualPatrol.process 使用同一中心（含 calib 偏移）；可选抬高小偏差以越过其内部 ~10px 死区。"""
        width = float(line_data.width)
        patrol_center = width / 2.0 + self.line_center_x_offset
        err = float(line_data.x) - patrol_center
        scaled = err * self.steering_gain
        floor = self.patrol_error_floor_px
        if floor > 0.0 and scaled != 0.0 and abs(scaled) < floor:
            scaled = math.copysign(floor, scaled)
        cx = patrol_center + scaled
        return max(0.0, min(width - 1.0, cx)), width

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

    def apply_gait_profile(self):
        cfg = rospy.get_param('/tasks/line_following/gait', {})
        profile = rospy.get_param('/tasks/line_following/gait_profile', 'default')
        profiles = rospy.get_param('/tasks/line_following/gait_profiles', {})
        if isinstance(profiles, dict) and profile in profiles and isinstance(profiles[profile], dict):
            cfg.update(profiles[profile])

        # 直接修改步态参数字典（覆盖 go/turn 的底层 gait param）
        # 支持:
        # - walking_param_updates: 同时作用于 go/turn
        # - go_walking_param_updates: 仅作用于 go
        # - turn_walking_param_updates: 仅作用于 turn
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
            arm_swap=go_arm_swap
        )
        self.visual_patrol.update_turn_gait(
            dsp=turn_dsp if isinstance(turn_dsp, list) and len(turn_dsp) == 3 else None,
            x_max=turn_x_max,
            arm_swap=turn_arm_swap,
        )
        self.line_follow_x_max_override = self.load_line_follow_x_max_override()
        if self.line_follow_x_max_override > 0.0:
            self.visual_patrol.update_go_gait(x_max=self.line_follow_x_max_override)
            self.visual_patrol.update_turn_gait(x_max=self.line_follow_x_max_override)
            common.loginfo(
                'line_following x_max override: %.4f'
                % self.line_follow_x_max_override
            )
        self.current_gait_profile = str(profile)
        common.loginfo(f'line_following gait profile: {profile}')

    def maybe_refresh_runtime_gait(self):
        desired_profile = str(rospy.get_param('/tasks/line_following/gait_profile', 'default'))
        desired_x_max_override = self.load_line_follow_x_max_override()
        if (
            desired_profile == self.current_gait_profile
            and abs(desired_x_max_override - self.line_follow_x_max_override) < 1e-9
        ):
            return
        common.loginfo(
            'line_following gait runtime refresh: profile %s -> %s, x_max_override %.4f -> %.4f'
            % (
                self.current_gait_profile,
                desired_profile,
                self.line_follow_x_max_override,
                desired_x_max_override,
            )
        )
        self.apply_gait_profile()
        
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

        # 直接写入 visual_patrol 的 gait param 字典（按键覆盖）
        for k, v in base_updates.items():
            self.visual_patrol.go_gait_param[k] = v
            self.visual_patrol.turn_gait_param[k] = v
        for k, v in go_updates.items():
            self.visual_patrol.go_gait_param[k] = v
        for k, v in turn_updates.items():
            self.visual_patrol.turn_gait_param[k] = v

    def set_servo(self, pan, tilt):
        try:
            self.motion_manager.set_servos_position(200, [[23, pan], [24, tilt]])
        except Exception as e:
            rospy.logerr(f"Servo control error: {e}")

    def publish_status(self, status):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        active_msg = Bool()
        active_msg.data = self.active
        self.active_pub.publish(active_msg)

    def safe_stop_gait(self, context):
        try:
            self.gait_manager.stop()
            return True, ''
        except Exception as e:
            rospy.logwarn('line_following gait stop failed during %s: %s', context, e)
            return False, str(e)

    def try_enter_color_detection(self):
        try:
            self.enter_func(None)
            return True
        except Exception as e:
            rospy.logwarn(f'color_detection enter unavailable: {e}')
            return False

    def try_start_color_detection(self):
        try:
            self.start_srv_callback(None)
            return True
        except Exception as e:
            rospy.logwarn(f'color_detection start unavailable: {e}')
            return False

    def try_stop_color_detection(self):
        try:
            self.stop_srv_callback(None)
            return True
        except Exception as e:
            rospy.logwarn(f'color_detection stop unavailable: {e}')
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

    def build_line_detect_param(self):
        param = ColorDetect()
        param.color_name = self.target_color
        param.use_name = True
        param.detect_type = 'line'
        param.image_process_size = self.image_process_size

        w = int(self.image_process_size[0])
        h = int(self.image_process_size[1])
        line_roi, _ = self.effective_line_roi()

        param.line_roi.up.y_min = int(line_roi[0][0] * h)
        param.line_roi.up.y_max = int(line_roi[0][1] * h)
        param.line_roi.up.x_min = int(line_roi[0][2] * w)
        param.line_roi.up.x_max = int(line_roi[0][3] * w)

        param.line_roi.center.y_min = int(line_roi[1][0] * h)
        param.line_roi.center.y_max = int(line_roi[1][1] * h)
        param.line_roi.center.x_min = int(line_roi[1][2] * w)
        param.line_roi.center.x_max = int(line_roi[1][3] * w)

        param.line_roi.down.y_min = int(line_roi[2][0] * h)
        param.line_roi.down.y_max = int(line_roi[2][1] * h)
        param.line_roi.down.x_min = int(line_roi[2][2] * w)
        param.line_roi.down.x_max = int(line_roi[2][3] * w)

        param.min_area = int(rospy.get_param('/tasks/line_following/min_line_area', 1))
        param.max_area = int(rospy.get_param('/tasks/line_following/max_line_area', w * h))
        return param

    def build_blue_side_detect_param(self):
        cfg = rospy.get_param('/tasks/hurdles', {})
        blue_color = cfg.get('color', 'blue')
        roi = cfg.get('roi', [0.2, 1.0, 0.15, 0.85])

        w = int(self.image_process_size[0])
        h = int(self.image_process_size[1])

        param = ColorDetect()
        param.color_name = blue_color
        param.use_name = True
        param.detect_type = 'side'
        param.image_process_size = self.image_process_size
        param.roi.y_min = int(self.parse_ratio(roi[0]) * h)
        param.roi.y_max = int(self.parse_ratio(roi[1]) * h)
        param.roi.x_min = int(self.parse_ratio(roi[2]) * w)
        param.roi.x_max = int(self.parse_ratio(roi[3]) * w)
        param.min_area = int(rospy.get_param('/tasks/hurdles/min_area', 200))
        param.max_area = w * h
        return param

    def build_stairs_side_detect_param(self):
        """与 tasks/climb_stairs 一致，供 ISCC 主控在巡线阶段识别台阶标记。"""
        cfg = rospy.get_param('/tasks/climb_stairs', {})
        color = cfg.get('color', 'green')
        roi = cfg.get('roi', [0.2, 1.0, 0.0, 1.0])
        w = int(self.image_process_size[0])
        h = int(self.image_process_size[1])
        param = ColorDetect()
        param.color_name = color
        param.use_name = True
        param.detect_type = 'side'
        param.image_process_size = self.image_process_size
        param.roi.y_min = int(self.parse_ratio(roi[0]) * h)
        param.roi.y_max = int(self.parse_ratio(roi[1]) * h)
        param.roi.x_min = int(self.parse_ratio(roi[2]) * w)
        param.roi.x_max = int(self.parse_ratio(roi[3]) * w)
        param.min_area = int(cfg.get('min_area', 200))
        param.max_area = w * h
        return param

    def configure_line_detection(self):
        line_param = self.build_line_detect_param()
        blue_side_param = self.build_blue_side_detect_param()
        params = [line_param, blue_side_param]
        _, roi_source = self.effective_line_roi()
        if rospy.get_param('/tasks/line_following/include_stairs_side', True):
            params.append(self.build_stairs_side_detect_param())
            common.loginfo(
                'line_following detect mode=%s roi=%s + blue-side + stairs-side'
                % (self.line_roi_mode, roi_source)
            )
        else:
            common.loginfo(
                'line_following detect mode=%s roi=%s + blue-side'
                % (self.line_roi_mode, roi_source)
            )
        self.detect_pub.publish(params)

    def get_color_callback(self, msg):
        self.objects_info = msg.data

    def start_service_callback(self, req):
        # 先开可视化，避免“跑起来很久才看到图”
        self.start_rqt_result_view()

        self.target_color = rospy.get_param('/tasks/line_following/color', 'white')
        self.steering_gain = float(rospy.get_param('/tasks/line_following/steering_gain', 2.0))
        self.patrol_error_floor_px = float(rospy.get_param('/tasks/line_following/patrol_error_floor_px', 12))
        self.line_center_x_offset = self._load_line_center_x_offset()
        self.line_roi_mode = self.load_line_roi_mode()
        self.apply_gait_profile()

        self.try_enter_color_detection()
        self.configure_line_detection()
        if not self.try_start_color_detection():
            self.publish_status("waiting_color_detection")
            self.stop_rqt_result_view()
            return TriggerResponse(success=False, message="color_detection/start unavailable")

        # 所有准备就绪后再低头，低头后立即进入巡线
        self.set_servo(self.head_pan_init, self.head_tilt_look_down)
        if self.head_settle_sec > 0:
            rospy.sleep(self.head_settle_sec)

        self.active = True
        self.running = True
        self.publish_status("started")
        return TriggerResponse(success=True, message="Line following started")

    def stop_service_callback(self, req):
        self.active = False
        ok, err = self.safe_stop_gait('stop_service')
        self.try_stop_color_detection()
        self.stop_rqt_result_view()
        self.publish_status("stopped")
        if ok:
            return TriggerResponse(success=True, message="Line following stopped")
        return TriggerResponse(success=False, message=f"Line following stop degraded: {err}")

    def pause_service_callback(self, req):
        self.active = False
        ok, err = self.safe_stop_gait('pause_service')
        self.publish_status("paused")
        if ok:
            return TriggerResponse(success=True, message="Line following paused")
        return TriggerResponse(success=False, message=f"Line following pause degraded: {err}")

    def resume_service_callback(self, req):
        self.target_color = rospy.get_param('/tasks/line_following/color', 'white')
        self.steering_gain = float(rospy.get_param('/tasks/line_following/steering_gain', 2.0))
        self.patrol_error_floor_px = float(rospy.get_param('/tasks/line_following/patrol_error_floor_px', 12))
        self.line_center_x_offset = self._load_line_center_x_offset()
        self.line_roi_mode = self.load_line_roi_mode()
        self.apply_gait_profile()
        self.try_enter_color_detection()
        self.active = True
        self.configure_line_detection()
        if not self.try_start_color_detection():
            self.publish_status("waiting_color_detection")
            return TriggerResponse(success=False, message="color_detection/start unavailable")
        self.publish_status("resumed")
        return TriggerResponse(success=True, message="Line following resumed")

    def control_callback(self, msg):
        command = msg.data.lower()
        if command == "start":
            self.start_service_callback(None)
        elif command in ("stop", "pause"):
            self.active = False
            self.safe_stop_gait('control_callback')
        elif command == "resume":
            self.resume_service_callback(None)

    def shutdown(self, signum, frame):
        with self.lock:
            self.motion_manager.run_action('stand')
            self.active = False
            self.running = False
            self.stop_rqt_result_view()
            self.safe_stop_gait('shutdown')
            self.try_stop_color_detection()
            self.publish_status("shutdown")
            common.loginfo('%s shutdown' % self.name)

    def run(self):
        self.publish_status("ready")
        while self.running and not rospy.is_shutdown():
            if not self.active:
                time.sleep(0.01)
                continue

            self.maybe_refresh_runtime_gait()

            if not self.is_gait_switch_on():
                self.safe_stop_gait('gait_switch_off')
                if not self.gait_switch_off_reported:
                    self.publish_status("gait_switch_off")
                    rospy.logwarn("gait_switch_off已设置为false，步态发布关闭")
                    self.gait_switch_off_reported = True
                time.sleep(0.05)
                continue
            self.gait_switch_off_reported = False

            line_data = None
            for object_info in self.objects_info:
                if object_info.type == 'line':
                    line_data = object_info
                    break

            if line_data is not None:
                corrected_x, width = self.patrol_corrected_x(line_data)
                self.visual_patrol.process(corrected_x, width)
                self.publish_status("following")
            else:
                self.publish_status("no_line")

            time.sleep(0.01)

        self.safe_stop_gait('run_exit')
        self.set_servo(self.head_pan_init, self.head_tilt_look_down)
        self.stop_rqt_result_view()
        self.try_stop_color_detection()
        rospy.signal_shutdown('shutdown')


if __name__ == '__main__':
    LineFollowing('line_following').run()
