#!/usr/bin/env python3
# encoding: utf-8

import math
import signal
import time

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from rosman_sdk import common
from rosman_example.color_common import Common
from rosman_interfaces.srv import SetString
from rosman_interfaces.msg import ObjectsInfo, ColorDetect


class CompetitionNode(Common):
    """
    楂橀€熺洿琛?+ 绔栫櫧绾夸笉鏂閬胯锛堟棤 PID锛?    - 榛樿鎸佺画蹇€熷墠杩?    - 鍙湪妫€娴嬪埌“绔栫洿鐧界嚎”鏃惰Е鍙戦伩璁?    - 鐧界嚎寰€宸?鍦ㄥ乏 -> 鏈哄櫒浜哄線鍙宠皟
    - 鐧界嚎寰€鍙?鍦ㄥ彸 -> 鏈哄櫒浜哄線宸﹁皟
    - 妯嚎蹇界暐
    - 鍓嶈繘淇閲囩敤锛氬彸璋冧竴浼氬効 -> 涓嶈皟涓€浼氬効
    - 澧炲姞鐩告満妗嗘樉绀轰笌璋冭瘯淇℃伅
    """

    line_roi = [
        (5 / 12, 6 / 12, 1 / 4, 3 / 4),
        (6 / 12, 7 / 12, 0, 1),
        (7 / 12, 8 / 12, 0, 1),
    ]
    image_process_size = [160, 120]

    forward_step_action = 'forward_step_1'
    turn_left_action = 'go_turn_left_20'
    turn_right_action = 'go_turn_right_20'

    def __init__(self, name):
        rospy.init_node(name)
        self.name = name
        
        self.running = True
        self.objects_info = []
        self.start_forward_signal = True

        self.head_pan_init = 500
        self.head_tilt_init = 520
        super().__init__(name, self.head_pan_init, self.head_tilt_init)

        # 绔栫嚎鍒ゅ畾鍙傛暟
        self.vertical_angle_threshold = 22.0
        self.horizontal_angle_threshold = 20.01
        self.vertical_ratio_threshold = 1.45

        # 榛樿鍓嶈繘鍙傛暟
        self.use_gait = hasattr(self, 'gait_manager') and self.gait_manager is not None
        self.walking_param = 1.1
        self.last_forward_cmd_time = 0.0
        self.forward_interval = 0.16
        self.fast_step_distance = 0.033
        self.arm_swap = 37
        self.dsp_param = [230, 0.25, 0.02]

        # 妯櫧绾垮姞閫燂細鐪嬪埌鏄庣‘妯嚎鏃讹紝涓嶉伩璁╋紝鐩存帴鏇村揩鐩村啿
        self.horizontal_boost_enabled = True
        self.horizontal_boost_interval = 0.1
        self.horizontal_boost_step_distance = 0.034

        # 鍙宠皟涓€浼氬効 -> 涓嶈皟涓€浼氬効
        self.forward_yaw_bias = 0
        self.pulse_right_yaw_bias = -3#鍙冲亸骞呭害瑕佸噺灏?
        self.pulse_on_steps = 1
        self.pulse_off_steps = 1
        self.forward_pulse_index = 0

        # 鍔ㄤ綔缁勫洖閫€妯″紡淇
        self.forward_cmd_count = 0
        self.action_pulse_turn_every = 2

        # 閬跨嚎鐘舵€?
        self.last_avoid_time = 1.0
        self.avoid_cooldown = 0.45
        self.avoid_turn_steps = 2
        self.last_line_side = 1  # 1=鍙充晶锛?1=宸︿晶

        # 涓嶆柇姝ラ伩绾匡細杩炵画鍓嶈繘鍑犳媿锛屼絾涓存椂鍔犲ぇ yaw
        self.escape_mode = False
        self.escape_direction = None      # 'left' / 'right'
        self.escape_forward_remaining = 0
        self.escape_forward_steps = 4
        self.escape_left_yaw = 7
        self.escape_right_yaw = -7

        # 鐧界嚎杩愬姩鏂瑰悜浼拌
        self.last_line_x = None
        self.last_line_time = 0.0
        self.line_motion_deadband = 4.0
        self.line_track_timeout = 0.35
        self.last_motion_direction = None  # 'left' / 'right'

        # 鏄剧ず涓庣浉鏈?
        self.bridge = CvBridge()
        self.current_image = None
        self.show_debug_window = rospy.get_param('~show_debug_window', True)
        self.debug_window_name = 'white_line_debug'
        self.camera_topic = rospy.get_param('~camera_topic', '/camera/image_raw')

        if self.use_gait:
            try:
                self.walking_param = self.gait_manager.get_gait_param()
                self.walking_param['step_height'] = 0.025
                self.walking_param['pelvis_offset'] = 5
                self.walking_param['hip_pitch_offset'] = 10
                self.walking_param['body_height'] = 0.055
                self.walking_param['init_y_offset'] = 0.0
                common.loginfo('use gait_manager fast forward white motion mode')
            except Exception as e:
                common.loginfo('gait_manager init failed, fallback to action mode: %s' % e)
                self.use_gait = False
                self.walking_param = None

        signal.signal(signal.SIGINT, self.shutdown)

        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)
        rospy.Subscriber(self.camera_topic, Image, self.image_callback, queue_size=1)
        rospy.Service('~set_color', SetString, self.set_color_srv_callback)

        self.motion_manager.run_action('walk_ready')

        if rospy.get_param('~start', True):
            self.enter_func(None)
            self.set_color_srv_callback(None)
            self.start_srv_callback(None)
            common.loginfo('start fast forward white-line motion + box + pulse-right + no-stop-escape (no pid)')

    def shutdown(self, signum=None, frame=None):
        self.running = False
        try:
            if self.use_gait:
                self.gait_manager.stop()
        except Exception:
            pass
        try:
            if self.show_debug_window:
                cv2.destroyAllWindows()
        except Exception:
            pass
        common.loginfo('%s shutdown' % self.name)

    def image_callback(self, msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            rospy.logwarn_throttle(2.0, 'image convert failed: %s' % e)

    def set_visual_patrol_color(self, color, roi, image_process_size):
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

    def set_color_srv_callback(self, msg):
        line_param = self.set_visual_patrol_color(
            color='white',
            roi=self.line_roi,
            image_process_size=self.image_process_size,
        )
        
        red_line_param = self.set_visual_patrol_color(
            color='red',
            roi=self.line_roi,
            image_process_size=self.image_process_size,
        )
        
        self.detect_pub.publish([line_param, red_line_param])
        common.loginfo('%s set_color' % self.name)
        return [True, 'set_color']

    def get_color_callback(self, msg):
        self.objects_info = msg.data

    def _safe_get(self, obj, *names):
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return None

    def _calc_angle_from_points(self, x1, y1, x2, y2):
        dx = float(x2) - float(x1)
        dy = float(y2) - float(y1)
        angle = math.degrees(math.atan2(dy, dx))
        angle = abs(angle) % 180.0
        if angle > 90.0:
            angle = 180.0 - angle
        return angle

    def _get_msg_dims(self, line_data):
        msg_w = self._safe_get(line_data, 'width', 'image_width', 'img_width')
        msg_h = self._safe_get(line_data, 'height', 'image_height', 'img_height')
        if hasattr(line_data, 'rect'):
            rect = line_data.rect
            if msg_w is None:
                msg_w = self._safe_get(rect, 'img_width', 'image_width', 'width_max')
            if msg_h is None:
                msg_h = self._safe_get(rect, 'img_height', 'image_height', 'height_max')
        try:
            msg_w = float(msg_w) if msg_w is not None else None
            msg_h = float(msg_h) if msg_h is not None else None
        except Exception:
            msg_w, msg_h = None, None
        return msg_w, msg_h

    def _scale_xy(self, x, y, line_data, img_shape):
        img_h, img_w = img_shape[:2]
        msg_w, msg_h = self._get_msg_dims(line_data)
        sx = float(img_w) / msg_w if msg_w not in (None, 0) else 1.0
        sy = float(img_h) / msg_h if msg_h not in (None, 0) else 1.0
        return int(round(float(x) * sx)), int(round(float(y) * sy))

    def _scale_wh(self, w, h, line_data, img_shape):
        img_h, img_w = img_shape[:2]
        msg_w, msg_h = self._get_msg_dims(line_data)
        sx = float(img_w) / msg_w if msg_w not in (None, 0) else 1.0
        sy = float(img_h) / msg_h if msg_h not in (None, 0) else 1.0
        return int(round(float(w) * sx)), int(round(float(h) * sy))

    def get_line_orientation(self, line_data):
        angle = self._safe_get(line_data, 'angle', 'theta')
        if angle is not None:
            try:
                angle = abs(float(angle)) % 180.0
                if angle > 90.0:
                    angle = 180.0 - angle
                if abs(90.0 - angle) <= self.vertical_angle_threshold:
                    return 'vertical'
                if angle <= self.horizontal_angle_threshold:
                    return 'horizontal'
            except Exception:
                pass

        w = self._safe_get(line_data, 'w', 'rect_w', 'box_w')
        h = self._safe_get(line_data, 'h', 'rect_h', 'box_h')
        if w is None and hasattr(line_data, 'rect'):
            rect = line_data.rect
            w = self._safe_get(rect, 'w', 'width')
            h = self._safe_get(rect, 'h', 'height')

        if w is not None and h is not None:
            try:
                w = float(w)
                h = float(h)
                if h > w * self.vertical_ratio_threshold:
                    return 'vertical'
                if w > h * self.vertical_ratio_threshold:
                    return 'horizontal'
            except Exception:
                pass

        x1 = self._safe_get(line_data, 'x1')
        y1 = self._safe_get(line_data, 'y1')
        x2 = self._safe_get(line_data, 'x2')
        y2 = self._safe_get(line_data, 'y2')
        if None not in (x1, y1, x2, y2):
            try:
                angle = self._calc_angle_from_points(x1, y1, x2, y2)
                if abs(90.0 - angle) <= self.vertical_angle_threshold:
                    return 'vertical'
                if angle <= self.horizontal_angle_threshold:
                    return 'horizontal'
            except Exception:
                pass

        return None

    def get_line_center_x(self, line_data):
        x = self._safe_get(line_data, 'x', 'cx', 'center_x')
        if x is not None:
            try:
                return float(x)
            except Exception:
                pass
        return None

    def get_line_side(self, line_data):
        x = self.get_line_center_x(line_data)
        width = self._safe_get(line_data, 'width', 'image_width', 'img_width')
        if x is not None and width is not None:
            try:
                dx = float(x) - float(width) / 2.0
                if dx < 0:
                    return -1
                if dx > 0:
                    return 1
            except Exception:
                pass
        return 0

    def update_line_motion_direction(self, line_data):
        x = self.get_line_center_x(line_data)
        now = time.time()
        direction = None

        if x is None:
            self.last_line_x = None
            self.last_line_time = 0.0
            self.last_motion_direction = None
            return None

        if self.last_line_x is not None and (now - self.last_line_time) <= self.line_track_timeout:
            dx = float(x) - float(self.last_line_x)
            if dx >= self.line_motion_deadband:
                direction = 'right'
            elif dx <= -self.line_motion_deadband:
                direction = 'left'

        self.last_line_x = float(x)
        self.last_line_time = now
        self.last_motion_direction = direction
        return direction

    def stop_forward_gait(self):
        if self.use_gait:
            try:
                self.gait_manager.stop()
                time.sleep(0.05)
            except Exception:
                pass

    def get_pulsed_yaw_bias(self):
        total = self.pulse_on_steps + self.pulse_off_steps
        if total <= 0:
            return self.forward_yaw_bias
        phase = self.forward_pulse_index % total
        if phase < self.pulse_on_steps:
            return self.forward_yaw_bias + self.pulse_right_yaw_bias
        return self.forward_yaw_bias

    def set_forward_step(self, yaw_bias=None, x_amplitude=None, interval_override=None):
        now = time.time()
        min_interval = self.forward_interval if interval_override is None else interval_override
        if now - self.last_forward_cmd_time < min_interval:
            return False

        use_pulse_schedule = yaw_bias is None
        if yaw_bias is None:
            yaw_bias = self.get_pulsed_yaw_bias()

        if x_amplitude is None:
            x_amplitude = self.fast_step_distance
        x_amplitude = max(-0.04, min(0.04, float(x_amplitude)))


        if self.use_gait and self.walking_param is not None:
            try:
                self.gait_manager.set_step(
                    self.dsp_param,
                    x_amplitude,
                    0,
                    yaw_bias,
                    self.walking_param,
                    self.arm_swap,
                    0,
                )
                self.last_forward_cmd_time = now
                if use_pulse_schedule:
                    self.forward_pulse_index += 1
                return True
            except Exception as e:
                common.loginfo('gait forward failed, fallback to action: %s' % e)
                self.use_gait = False

        self.motion_manager.run_action(self.forward_step_action)
        if use_pulse_schedule:
            total = self.pulse_on_steps + self.pulse_off_steps
            if total > 0:
                phase = self.forward_pulse_index % total
                if phase < self.pulse_on_steps and (self.forward_cmd_count % self.action_pulse_turn_every == 0):
                    self.motion_manager.run_action(self.turn_right_action)
            self.forward_pulse_index += 1
        self.forward_cmd_count += 1
        self.last_forward_cmd_time = now
        return True

    def do_turn_left(self, repeat=1):
        self.stop_forward_gait()
        for _ in range(repeat):
            self.motion_manager.run_action(self.turn_left_action)
        self.last_forward_cmd_time = 0.0

    def do_turn_right(self, repeat=1):
        self.stop_forward_gait()
        for _ in range(repeat):
            self.motion_manager.run_action(self.turn_right_action)
        self.last_forward_cmd_time = 0.0

    def start_escape(self, direction):
        """
        寮€濮嬩笉鏂閬跨嚎锛?        涓嶅仠姝ワ紝涓嶈浆鍔ㄤ綔缁勶紝鑰屾槸杩炵画鍓嶈繘鍑犳媿骞朵复鏃跺姞澶?yaw銆?        """
        self.escape_mode = True
        self.escape_direction = direction
        self.escape_forward_remaining = self.escape_forward_steps
        self.last_avoid_time = time.time()

    def run_escape_step(self):
        """
        姣忔涓诲惊鐜窇涓€鎷嶉伩绾垮墠鍐层€?        鍙湁鐪熸鍙戝嚭涓€姝ュ墠杩涘懡浠ゆ椂锛宺emaining 鎵嶅噺 1銆?        """
        if not self.escape_mode:
            return False

        yaw_bias = self.escape_right_yaw if self.escape_direction == 'right' else self.escape_left_yaw
        sent = self.set_forward_step(yaw_bias=yaw_bias)
        if sent:
            self.escape_forward_remaining -= 1

        if self.escape_forward_remaining <= 0:
            self.escape_mode = False
            self.escape_direction = None

        return True

    def choose_escape_direction(self, line_data):
        motion_direction = self.update_line_motion_direction(line_data)
        if motion_direction == 'left':
            return 'right'
        if motion_direction == 'right':
            return 'left'

        side = self.get_line_side(line_data)
        if side < 0:
            self.last_line_side = -1
            return 'right'
        if side > 0:
            self.last_line_side = 1
            return 'left'

        return 'right' if self.last_line_side < 0 else 'left'

    def reverse_line_patrol_no_pid(self, line_data):
        # 姝ｅ正在鎵ц涓嶆柇姝ラ伩绾挎椂锛屽厛鎶婅繖涓€娈佃窇瀹?        
        if self.escape_mode:
            self.run_escape_step()
            return

        if line_data is None:
            self.last_line_x = None
            self.last_line_time = 0.0
            self.last_motion_direction = None
            self.set_forward_step()
            return

        orientation = self.get_line_orientation(line_data)
        if orientation == 'horizontal':
            self.update_line_motion_direction(line_data)
            if self.horizontal_boost_enabled:
                self.set_forward_step(
                    yaw_bias=self.forward_yaw_bias,
                    x_amplitude=self.horizontal_boost_step_distance,
                    interval_override=self.horizontal_boost_interval,
                )
            else:
                self.set_forward_step()
            return

        if orientation != 'vertical':
            self.update_line_motion_direction(line_data)
            self.set_forward_step()
            return

        now = time.time()
        if now - self.last_avoid_time < self.avoid_cooldown:
            self.update_line_motion_direction(line_data)
            self.set_forward_step()
            return

        direction = self.choose_escape_direction(line_data)

        # gait 妯″紡涓嬶細涓嶆柇姝ラ伩绾匡紝杈硅蛋杈规嫄
        if self.use_gait:
            self.start_escape(direction)
            if direction == 'right':
                self.last_line_side = -1
            else:
                self.last_line_side = 1
            return

        # action 妯″紡涓嬶細淇濈暀鍘熸潵鐨勫姩浣滅粍閬跨嚎
        if direction == 'right':
            self.do_turn_right(repeat=self.avoid_turn_steps)
            self.last_line_side = -1
        else:
            self.do_turn_left(repeat=self.avoid_turn_steps)
            self.last_line_side = 1
        self.last_avoid_time = time.time()

    def draw_debug(self, line_data):
        if self.current_image is None or not self.show_debug_window:
            return

        img = self.current_image.copy()
        img_h, img_w = img.shape[:2]

        # 鐢?ROI
        roi_colors = [(255, 200, 0), (0, 255, 255), (0, 180, 0)]
        for idx, roi in enumerate(self.line_roi):
            y_min = int(roi[0] * img_h)
            y_max = int(roi[1] * img_h)
            x_min = int(roi[2] * img_w)
            x_max = int(roi[3] * img_w)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), roi_colors[idx], 2)
            cv2.putText(img, 'ROI-%d' % (idx + 1), (x_min + 4, y_min + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_colors[idx], 1)

        orientation = None
        side = 0
        motion_dir = None
        if line_data is not None:
            orientation = self.get_line_orientation(line_data)
            side = self.get_line_side(line_data)
            motion_dir = self.last_motion_direction

            # 鐢荤嚎娈?            
            x1 = self._safe_get(line_data, 'x1')
            y1 = self._safe_get(line_data, 'y1')
            x2 = self._safe_get(line_data, 'x2')
            y2 = self._safe_get(line_data, 'y2')
            if None not in (x1, y1, x2, y2):
                p1 = self._scale_xy(x1, y1, line_data, img.shape)
                p2 = self._scale_xy(x2, y2, line_data, img.shape)
                cv2.line(img, p1, p2, (0, 0, 255), 3)

            # 鐢绘锛堝敖閲忓吋瀹?center+w+h / rect / top-left+w+h锛?            
            cx = self._safe_get(line_data, 'cx', 'center_x', 'x')
            cy = self._safe_get(line_data, 'cy', 'center_y', 'y')
            w = self._safe_get(line_data, 'w', 'rect_w', 'box_w')
            h = self._safe_get(line_data, 'h', 'rect_h', 'box_h')
            if (w is None or h is None) and hasattr(line_data, 'rect'):
                rect = line_data.rect
                w = self._safe_get(rect, 'w', 'width')
                h = self._safe_get(rect, 'h', 'height')
                if cx is None:
                    cx = self._safe_get(rect, 'cx', 'center_x', 'x')
                if cy is None:
                    cy = self._safe_get(rect, 'cy', 'center_y', 'y')

            if None not in (cx, cy, w, h):
                try:
                    sx, sy = self._scale_xy(cx, cy, line_data, img.shape)
                    sw, sh = self._scale_wh(w, h, line_data, img.shape)
                    left = int(sx - sw / 2)
                    top = int(sy - sh / 2)
                    right = int(sx + sw / 2)
                    bottom = int(sy + sh / 2)
                    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.circle(img, (sx, sy), 4, (255, 0, 255), -1)
                except Exception:
                    pass
            elif cx is not None and cy is not None:
                try:
                    sx, sy = self._scale_xy(cx, cy, line_data, img.shape)
                    cv2.circle(img, (sx, sy), 6, (255, 0, 255), -1)
                except Exception:
                    pass

        total = self.pulse_on_steps + self.pulse_off_steps
        phase = (self.forward_pulse_index % total) if total > 0 else 0
        pulse_state = 'RIGHT_BIAS' if total > 0 and phase < self.pulse_on_steps else 'FREE'
        side_text = 'LEFT' if side < 0 else ('RIGHT' if side > 0 else 'CENTER')
        boost_state = 'ON' if orientation == 'horizontal' and self.horizontal_boost_enabled else 'OFF'
        shown_x_amp = self.horizontal_boost_step_distance if boost_state == 'ON' else self.fast_step_distance

        info_lines = [
            'orientation: %s' % str(orientation),
            'line_side: %s' % side_text,
            'motion_dir: %s' % str(motion_dir),
            'pulse: %s (%d/%d)' % (pulse_state, phase, total),
            'h_boost: %s' % boost_state,
            'gait: %s' % ('ON' if self.use_gait else 'ACTION'),
            'step_h: %.3f' % (self.walking_param['step_height'] if self.walking_param else 0.0),
            'x_amp: %.3f' % max(-0.02, min(0.02, float(shown_x_amp))),
        ]

        y = 26
        for text in info_lines:
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 24

        cv2.imshow(self.debug_window_name, img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            self.shutdown()

    def run(self):
        while self.running and not rospy.is_shutdown():
            if self.start:
                line_data = None
                red_line_data = None
                
                for object_info in self.objects_info:
                    if object_info.type == 'line':
                        color = getattr(object_info, 'color_name', getattr(object_info, 'name', ''))
                        if color == 'red':
                            red_line_data = object_info
                        elif color == 'white' or not color:
                            line_data = object_info

                if self.start_forward_signal:
                    for _ in range(5):
                        self.set_forward_step()
                        self.draw_debug(line_data)
                        rospy.sleep(self.forward_interval)
                    self.start_forward_signal = False

                if red_line_data is not None:
                    self.set_forward_step(
                        yaw_bias=0.0,
                        x_amplitude=self.horizontal_boost_step_distance if self.horizontal_boost_enabled else self.fast_step_distance,
                        interval_override=self.horizontal_boost_interval if self.horizontal_boost_enabled else self.forward_interval
                    )
                    self.draw_debug(red_line_data)
                else:
                    self.reverse_line_patrol_no_pid(line_data)
                    self.draw_debug(line_data)
                
                rospy.sleep(0.01)
            else:
                self.draw_debug(None)
                rospy.sleep(0.01)

        try:
            self.stop_forward_gait()
        except Exception:
            pass
        self.init_action(self.head_pan_init, self.head_tilt_init)
        self.stop_srv_callback(None)
        rospy.signal_shutdown('shutdown')


if __name__ == '__main__':
    CompetitionNode('competition_line_fast_white_motion_box_pulse_nostop_hboost').run()


