#!/usr/bin/env python3
# encoding: utf-8
# @Author: Dejin Xuan
# @Date: 2026.4.3
# 橙线区域A阶段：停白线后持续前进并左偏航，直至左右橙线同时稳定出现若干帧（为 B 双线循迹占位）
import signal
import time

import rospy
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger, TriggerResponse

from rosman_example.color_common import Common
from rosman_interfaces.msg import ColorDetect, ObjectsInfo
from rosman_sdk import common


class OrangeGuide(Common):
    def __init__(self, name):
        self.head_pan_init = rospy.get_param('/servo/head_pan_init', 500)
        self.head_tilt_look_down = rospy.get_param('/servo/head_tilt_look_down', 280)

        rospy.init_node(name)
        self.name = name
        self.running = True
        self.active = False
        self.objects_info = []
        self.current_state = 'idle'
        self.task_completed = False
        self._last_right_y_log_time = 0.0
        self._last_detect_config_time = 0.0

        super().__init__(name, self.head_pan_init, self.head_tilt_look_down)
        signal.signal(signal.SIGINT, self.shutdown)

        self.process_size = rospy.get_param('/image_processing/process_size', [160, 120])
        self.image_width = int(self.process_size[0]) if len(self.process_size) > 0 else 160
        self.image_height = int(self.process_size[1]) if len(self.process_size) > 1 else 120
        self.image_process_size = [self.image_width, self.image_height]
        self.head_settle_sec = float(rospy.get_param('/tasks/line_following/head_settle_sec', 0.12))

        self.load_orange_guide_config()
        self.setup_ros_interface()
        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)

        self.motion_manager.run_action('walk_ready')
        common.loginfo('Orange guide initialized (enter 在 /orange_guide/start 时执行)')

    def load_orange_guide_config(self):
        cfg = rospy.get_param('/tasks/orange_guide', {})
        self.orange_color = cfg.get('color', 'orange')
        self.left_roi = cfg.get('left_roi', [0.2, 1.0, 0.0, 0.5])
        self.right_roi = cfg.get('right_roi', [0.2, 1.0, 0.5, 1.0])
        self.min_area = int(cfg.get('min_area', 10))
        self.both_stable_frames = int(cfg.get('both_stable_frames', 6))
        self.guide_linear_x = float(cfg.get('guide_linear_x', 0.01))
        self.guide_yaw = int(cfg.get('guide_yaw', 4))
        self.timeout_sec = float(cfg.get('timeout_sec', 8.0))
        # A 结束门控：即便双线稳定，也需右侧橙线 y_max 达到阈值才允许完成（<=1 解释为比例）
        self.right_stop_y_threshold = float(cfg.get('right_stop_y_threshold', 0.0))
        self.right_y_log_throttle_sec = float(cfg.get('right_y_log_throttle_sec', 0.2))
        # 双线稳定达标后、结束 A 前再前进一段时间（秒）；0 表示立即结束
        self.after_align_forward_sec = float(cfg.get('after_align_forward_sec', 0.0))
        self.after_align_linear_x = float(cfg.get('after_align_linear_x', self.guide_linear_x))
        self.after_align_yaw = int(cfg.get('after_align_yaw', 0))
        self._both_stable_counter = 0
        self._phase_start_time = 0.0
        self._after_align_start_time = 0.0
        self.walking_param = self.gait_manager.get_gait_param()
        self.dsp_param = [320, 0.25, 0.022]
        self.arm_swap = 30
        # 在任务切换边界上，前一个节点 stop 时会清空 /color_detection/update_detect。
        # orange_guide 周期性重发自己的橙线 ROI，避免单次配置被后到的空配置冲掉。
        self.detect_refresh_sec = float(cfg.get('detect_refresh_sec', 0.5))

    def setup_ros_interface(self):
        self.status_pub = rospy.Publisher('/orange_guide/status', String, queue_size=1)
        self.active_pub = rospy.Publisher('/orange_guide/active', Bool, queue_size=1)
        self.start_service = rospy.Service('/orange_guide/start', Trigger, self.start_service_callback)
        self.stop_service = rospy.Service('/orange_guide/stop', Trigger, self.stop_service_callback)
        self.reset_service = rospy.Service('/orange_guide/reset', Trigger, self.reset_service_callback)

    def publish_status(self, status):
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
            rospy.logwarn('orange_guide enter unavailable: %s', e)
            return False

    def try_start_color_detection(self):
        try:
            self.start_srv_callback(None)
            return True
        except Exception as e:
            rospy.logwarn('orange_guide start unavailable: %s', e)
            return False

    def try_stop_color_detection(self):
        try:
            self.stop_srv_callback(None)
            return True
        except Exception as e:
            rospy.logwarn('orange_guide stop unavailable: %s', e)
            return False

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

    def configure_orange_detection(self):
        left_param = self.build_orange_side_param(self.left_roi)
        right_param = self.build_orange_side_param(self.right_roi)
        self.detect_pub.publish([left_param, right_param])
        self._last_detect_config_time = rospy.get_time()
        common.loginfo('orange_guide set orange side detect roi/color')

    def refresh_orange_detection_if_needed(self, force=False):
        now = rospy.get_time()
        if force or self.detect_refresh_sec <= 0.0:
            self.configure_orange_detection()
            return
        if (now - self._last_detect_config_time) >= self.detect_refresh_sec:
            self.configure_orange_detection()

    def get_color_callback(self, msg):
        self.objects_info = msg.data

    def get_orange_side_objects(self):
        orange = []
        for obj in self.objects_info:
            if getattr(obj, 'type', '') == 'side' and getattr(obj, 'label', '') == self.orange_color:
                orange.append(obj)
        return orange

    def _pixel_frame_width(self):
        """与主控/跨栏/台阶一致：像素坐标以 ObjectInfo.width 为原图宽度（非 process_size）。"""
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

    def side_y_max(self, obj, fallback_y=0):
        """优先用 side 两端点最大 y，避免中心点 y 偏小导致门控滞后。"""
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

    def right_stop_y_threshold_px(self, right_obj):
        ty = float(self.right_stop_y_threshold)
        if ty <= 0:
            return 0
        frame_h = int(getattr(right_obj, 'height', self.image_height)) or self.image_height
        if ty <= 1.0:
            ty = float(ty) * float(frame_h)
            rospy.logwarn_once('orange_guide.right_stop_y_threshold <= 1, interpreted as ratio using frame height')
        return int(ty)

    def log_right_y_for_tune(self, right_seen, right_y_max, right_stop_y):
        if not right_seen:
            return
        now = rospy.get_time()
        if self.right_y_log_throttle_sec > 0.0 and (now - self._last_right_y_log_time) < self.right_y_log_throttle_sec:
            return
        self._last_right_y_log_time = now
        if right_stop_y > 0:
            rospy.loginfo('og R y_max=%d/%d', right_y_max, right_stop_y)
        else:
            rospy.loginfo('og R y_max=%d', right_y_max)

    def start_service_callback(self, req):
        self.load_orange_guide_config()
        self.task_completed = False
        self.active = False
        self.current_state = 'starting'
        self.objects_info = []
        self._both_stable_counter = 0
        self._phase_start_time = 0.0
        self._after_align_start_time = 0.0
        self._last_right_y_log_time = 0.0
        self._last_detect_config_time = 0.0
        if not self.try_enter_color_detection():
            self.current_state = 'idle'
            self.publish_status('waiting_color_detection')
            return TriggerResponse(success=False, message='color_detection/enter unavailable')
        self.configure_orange_detection()
        if not self.try_start_color_detection():
            self.current_state = 'idle'
            self.publish_status('waiting_color_detection')
            return TriggerResponse(success=False, message='color_detection/start unavailable')
        # 再发一次，覆盖上一个任务 stop 过程中可能晚到的空检测配置。
        self.refresh_orange_detection_if_needed(force=True)
        self.motion_manager.run_action('walk_ready')
        if self.head_settle_sec > 0:
            rospy.sleep(self.head_settle_sec)
        self._phase_start_time = rospy.get_time()
        self.current_state = 'guiding'
        self.active = True
        self.publish_status('started')
        return TriggerResponse(success=True, message='orange guide started')

    def stop_service_callback(self, req):
        self.active = False
        self.current_state = 'idle'
        self.gait_manager.stop()
        self.try_stop_color_detection()
        self.publish_status('stopped')
        return TriggerResponse(success=True, message='orange guide stopped')

    def reset_service_callback(self, req):
        self.active = False
        self.task_completed = False
        self.current_state = 'idle'
        self._both_stable_counter = 0
        self.gait_manager.stop()
        self.try_stop_color_detection()
        self.publish_status('reset')
        return TriggerResponse(success=True, message='orange guide reset')

    def run_guide_motion_always(self):
        """无论当前是否看到橙线，均保持前向 + 左偏航，直到左右双线同时稳定若干帧。"""
        self.gait_manager.set_step(
            self.dsp_param,
            self.guide_linear_x,
            0.0,
            int(self.guide_yaw),
            self.walking_param,
            arm_swap=self.arm_swap,
            step_num=0,
        )

    def run_after_align_motion(self):
        """双线已稳定达标后的附加前进：可纯直行（yaw=0）或微调偏航。"""
        self.gait_manager.set_step(
            self.dsp_param,
            self.after_align_linear_x,
            0.0,
            int(self.after_align_yaw),
            self.walking_param,
            arm_swap=self.arm_swap,
            step_num=0,
        )

    def run(self):
        self.publish_status('ready')
        while self.running and not rospy.is_shutdown():
            if not self.active:
                time.sleep(0.02)
                continue

            now = rospy.get_time()
            if self.current_state == 'guiding':
                self.refresh_orange_detection_if_needed()
                self.run_guide_motion_always()
                left_obj, right_obj = self.split_left_right()
                left_seen = left_obj is not None
                right_seen = right_obj is not None
                if left_seen and right_seen:
                    self._both_stable_counter += 1
                else:
                    self._both_stable_counter = 0
                right_y_max = self.side_y_max(right_obj, fallback_y=0) if right_seen else 0
                right_stop_y = self.right_stop_y_threshold_px(right_obj) if right_seen else 0
                self.log_right_y_for_tune(right_seen, right_y_max, right_stop_y)
                right_y_ready = (right_stop_y <= 0) or (right_seen and right_y_max >= right_stop_y)
                if left_seen and right_seen:
                    self.publish_status('guiding_both_lines')
                elif right_seen:
                    self.publish_status('guiding_right_only')
                elif left_seen:
                    self.publish_status('guiding_left_only')
                else:
                    self.publish_status('guiding_blind')
                if self._both_stable_counter >= self.both_stable_frames and right_y_ready:
                    if self.after_align_forward_sec > 0.0:
                        self.current_state = 'after_align_forward'
                        self._after_align_start_time = now
                        self.publish_status('after_align_forward')
                        common.loginfo(
                            'orange_guide: dual-line stable, extra forward %.2fs before complete'
                            % self.after_align_forward_sec
                        )
                    else:
                        self.gait_manager.stop()
                        self.active = False
                        self.task_completed = True
                        self.current_state = 'done'
                        self.publish_status('completed_orange_guide')
                    continue
                if self._both_stable_counter >= self.both_stable_frames and right_seen and not right_y_ready:
                    self.publish_status('guiding_wait_right_y')

            if self.current_state == 'after_align_forward':
                self.refresh_orange_detection_if_needed()
                self.run_after_align_motion()
                if (now - self._after_align_start_time) >= self.after_align_forward_sec:
                    self.gait_manager.stop()
                    self.active = False
                    self.task_completed = True
                    self.current_state = 'done'
                    self.publish_status('completed_orange_guide')
                else:
                    self.publish_status('after_align_forward')
                continue

            if self.current_state == 'guiding' and (now - self._phase_start_time) > self.timeout_sec:
                self.gait_manager.stop()
                self.active = False
                self.current_state = 'timeout'
                self.publish_status('timeout')

            time.sleep(0.02)

        self.gait_manager.stop()
        self.try_stop_color_detection()
        self.motion_manager.run_action('stand')
        self.publish_status('shutdown')
        rospy.signal_shutdown('shutdown')

    def shutdown(self, signum, frame):
        with self.lock:
            self.running = False
            self.active = False
            self.gait_manager.stop()
            self.try_stop_color_detection()
            self.motion_manager.run_action('stand')
            self.publish_status('shutdown')
            common.loginfo('%s shutdown' % self.name)


if __name__ == '__main__':
    try:
        OrangeGuide('orange_guide').run()
    except Exception as e:
        rospy.logerr('Orange guide initialization error: %s', e)
