#!/usr/bin/env python3
# encoding: utf-8
# v7.21_5 — _4的动作组版：前进/左右横移用motion_manager.run_action()，转向保留gait

import signal
import time
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from rosman_sdk import common
from rosman_example.color_common import Common


class AdvancedPenaltyNode(Common):

    # ============================================================
    #  [ 可调参区域 ]
    # ============================================================

    # ---------- 图像尺寸 ----------
    IMG_W = 320
    IMG_H = 240
    IMG_CX = 160
    IMG_CY = 120

    # ---------- 舵机角度 ----------
    HEAD_PAN = 500
    HEAD_TILT_INIT = 380
    HEAD_TILT_MIN  = 250
    HEAD_TILT_MAX  = 750

    # ---------- ★ 开局前冲 (START_WALK) ----------
    START_WALK_STEPS = 4       # 开局前进几步
    START_WALK_DIST  = 0.022   # 每步步幅
    START_WALK_SLEEP = 0.7     # 每步时间

    # ---------- ★ 左侧横移绕行 (MOVE_LEFT) ----------
    MOVE_LEFT_STEPS = 1        # 向左侧挪动几步
    MOVE_LEFT_SLEEP = 1.2      # 每步动作后的等待时间

    # ---------- 逼近球 (APPROACH_BALL) ----------
    APPROACH_DX_STOP         = 40
    APPROACH_DX_TURN         = 30
    APPROACH_TILT_STOP       = 390
    APPROACH_TURN_SLEEP      = 0.4
    APPROACH_FWD_SLEEP       = 0.9    # 前进动作后的等待时间
    APPROACH_TILT_RATE       = 0.2
    APPROACH_TILT_DEADBAND   = 15

    # ---------- [抬头] 看球门 (LOOK_UP) ----------
    LOOK_UP_TILT  = 520
    LOOK_UP_SLEEP = 2.0

    # ---------- 对准球门 (ALIGN_GOAL) ----------
    ALIGN_DX_STOP  = 30
    ALIGN_SLEEP    = 1

    # ---------- [低头] 看球 (LOOK_DOWN) ----------
    LOOK_DOWN_TILT  = 350
    LOOK_DOWN_SLEEP = 1.5

    # ---------- 粗调步态 — START_WALK 前冲 ----------
    COARSE_STEP_HEIGHT      = 0.025
    COARSE_BODY_HEIGHT      = 0.045
    COARSE_PELVIS_OFFSET    = 5
    COARSE_HIP_PITCH_OFFSET = 13
    COARSE_INIT_Y_OFFSET    = 0.0

    # ---------- 微调步态 — DECIDE_FOOT 后退 ----------
    FINE_STEP_HEIGHT      = 0.02
    FINE_BODY_HEIGHT      = 0.045
    FINE_PELVIS_OFFSET    = 5
    FINE_HIP_PITCH_OFFSET = 13
    FINE_INIT_Y_OFFSET    = 0.0

    # ---------- 调距 / 选脚 (DECIDE_FOOT) ----------
    DECIDE_CY_TOO_FAR    = 200
    DECIDE_CY_TOO_CLOSE  = 400
    DECIDE_MOVE_SLEEP    = 0.9    # 进退动作后的等待时间
    DECIDE_FWD_PER_TURN  = 2      # 逼近时每前进N步转一次
    DECIDE_TURN_STEPS    = 2      # 每次转M步
    DECIDE_TURN_WHEN_FWD = 'turn_right'  # 逼近时转向方向

    # ---------- 横向微调 — 强制球停在指定侧 ----------
    FORCE_BALL_SIDE       = 'right'
    FINE_LATERAL_SLEEP    = 0.7    # 横移动作后的等待时间
    FINE_LATERAL_MAX      = 60
    FINE_LATERAL_MIN      = 10

    # ---------- 射门 (KICKING) ----------
    KICK_HEAD_TILT   = 520
    KICK_HEAD_SLEEP  = 0.5
    KICK_IDLE_WAIT   = 8.0
    FORCE_KICK_FOOT  = 'right_shot'

    # ---------- 找球 / 搜门 ----------
    SEARCH_SLEEP = 0.7

    # ---------- 视觉：找紫球 ----------
    BALL_RADIUS_MIN = 8

    # ---------- 视觉：找球门 ----------
    GOAL_CONTOUR_AREA_MIN = 10
    GOAL_RECT_W_MIN = 40
    GOAL_RECT_H_MIN = 10

    # ---------- 视觉：红色障碍物 ----------
    OBS_AREA_MIN      = 50
    OBS_DIST_TO_GOAL  = 60
    OBS_MARGIN        = 40

    # ============================================================
    #  [ 参数区结束 ]
    # ============================================================

    def __init__(self, name):
        self.head_pan = self.HEAD_PAN
        self.head_tilt = self.HEAD_TILT_INIT
        rospy.init_node(name)
        super().__init__(name, self.head_pan, self.head_tilt)

        self.name = name
        self.running = True
        self.bridge = CvBridge()

        self.target_cx = -1
        self.target_cy = -1
        self.target_radius = 0

        self.obs_cx = -1
        self.obs_cy = -1
        self.obs_w = 0

        self.current_image = None
        self.show_debug_window = rospy.get_param('~show_debug_window', True)

        self.state = 'START_WALK'
        self.vision_mode = 'ball'

        self.act_kick = 'right_shot'
        self.act_stand = 'walk_ready'
        self.use_gait = True

        self._step_counter = 0
        self._fwd_step_counter = 0    # DECIDE_FOOT前进步数计数

        # 步态参数（两组：粗调 / 微调）
        self._gait_params_coarse = {
            'step_height': self.COARSE_STEP_HEIGHT,
            'body_height': self.COARSE_BODY_HEIGHT,
            'pelvis_offset': self.COARSE_PELVIS_OFFSET,
            'hip_pitch_offset': self.COARSE_HIP_PITCH_OFFSET,
            'init_y_offset': self.COARSE_INIT_Y_OFFSET,
        }
        self._gait_params_fine = {
            'step_height': self.FINE_STEP_HEIGHT,
            'body_height': self.FINE_BODY_HEIGHT,
            'pelvis_offset': self.FINE_PELVIS_OFFSET,
            'hip_pitch_offset': self.FINE_HIP_PITCH_OFFSET,
            'init_y_offset': self.FINE_INIT_Y_OFFSET,
        }
        try:
            self.walking_param = self.gait_manager.get_gait_param()
            self._apply_gait('coarse')
            common.loginfo('gait params configured (coarse + fine)')
        except Exception as e:
            common.loginfo('gait param init failed: %s' % e)
            self.use_gait = False
            self.walking_param = None

        rospy.Subscriber('/camera/image_raw', Image, self.image_callback, queue_size=1)
        signal.signal(signal.SIGINT, self.shutdown)

        self.motion_manager.run_action(self.act_stand)
        rospy.sleep(1.0)

        self.start = True
        common.loginfo('>>> v7.21_5 Action-Group Version Started! <<<')
        self._print_params()

    def _apply_gait(self, mode):
        """切换步态参数: 'coarse' = 粗调, 'fine' = 微调"""
        if not self.use_gait or self.walking_param is None:
            return
        params = self._gait_params_coarse if mode == 'coarse' else self._gait_params_fine
        for key, val in params.items():
            self.walking_param[key] = val

    def _print_params(self):
        common.loginfo('========== CURRENT PARAMS ==========')
        common.loginfo(f'HEAD: init={self.HEAD_TILT_INIT} look_up={self.LOOK_UP_TILT} look_down={self.LOOK_DOWN_TILT} kick_head={self.KICK_HEAD_TILT}')
        common.loginfo(f'START_WALK: steps={self.START_WALK_STEPS} dist={self.START_WALK_DIST}')
        common.loginfo(f'MOVE_LEFT: steps={self.MOVE_LEFT_STEPS}')
        common.loginfo(f'APPROACH: dx_stop={self.APPROACH_DX_STOP} tilt_stop={self.APPROACH_TILT_STOP}')
        common.loginfo(f'DECIDE_FOOT: cy_too_far={self.DECIDE_CY_TOO_FAR} cy_too_close={self.DECIDE_CY_TOO_CLOSE}')
        common.loginfo(f'FORCE: kick_foot={self.FORCE_KICK_FOOT} ball_side={self.FORCE_BALL_SIDE}')
        common.loginfo(f'LATERAL: min={self.FINE_LATERAL_MIN} max={self.FINE_LATERAL_MAX}')
        common.loginfo(f'KICK_IDLE_WAIT: {self.KICK_IDLE_WAIT}s')
        common.loginfo(f'GAIT_COARSE: step_h={self.COARSE_STEP_HEIGHT} body_h={self.COARSE_BODY_HEIGHT}')
        common.loginfo(f'GAIT_FINE:   step_h={self.FINE_STEP_HEIGHT} body_h={self.FINE_BODY_HEIGHT}')
        common.loginfo('====================================')

    def shutdown(self, signum=None, frame=None):
        self.running = False
        if self.show_debug_window:
            cv2.destroyAllWindows()
        common.loginfo('%s shutdown properly.' % self.name)

    # ============================================================
    #  视觉
    # ============================================================
    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError:
            return

        img_resize = cv2.resize(cv_img, (self.IMG_W, self.IMG_H))
        self.current_image = img_resize.copy()

        if self.vision_mode == 'ball':
            hsv = cv2.cvtColor(img_resize, cv2.COLOR_BGR2HSV)
            hsv_blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

            lower_purple = np.array([115, 60, 60])
            upper_purple = np.array([160, 255, 255])
            mask = cv2.inRange(hsv_blurred, lower_purple, upper_purple)

            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

            if contours:
                c = max(contours, key=cv2.contourArea)
                ((cx, cy), radius) = cv2.minEnclosingCircle(c)
                if radius > self.BALL_RADIUS_MIN:
                    self.target_cx, self.target_cy = int(cx), int(cy)
                    self.target_radius = int(radius)
                else:
                    self.target_cx = -1
            else:
                self.target_cx = -1

        elif self.vision_mode == 'goal':
            top_half = img_resize[0:self.IMG_CY, :]
            gray = cv2.cvtColor(top_half, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
            edges = cv2.dilate(edges, None, iterations=2)
            goal_contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

            goal_cx = -1
            goal_cy = -1
            if goal_contours:
                valid_contours = [c for c in goal_contours if cv2.contourArea(c) > self.GOAL_CONTOUR_AREA_MIN]
                if valid_contours:
                    all_points = np.vstack(valid_contours)
                    x, y, w, h = cv2.boundingRect(all_points)
                    if w > self.GOAL_RECT_W_MIN and h > self.GOAL_RECT_H_MIN:
                        goal_cx = x + w // 2
                        goal_cy = y + h // 2

            hsv = cv2.cvtColor(img_resize, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([160, 120, 70]), np.array([180, 255, 255]))
            red_mask = cv2.bitwise_or(mask1, mask2)
            red_mask = cv2.erode(red_mask, None, iterations=2)
            red_mask = cv2.dilate(red_mask, None, iterations=2)

            obs_contours = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

            self.obs_cx = -1
            if obs_contours:
                c_obs = max(obs_contours, key=cv2.contourArea)
                if cv2.contourArea(c_obs) > self.OBS_AREA_MIN:
                    ox, oy, ow, oh = cv2.boundingRect(c_obs)
                    self.obs_cx = ox + ow // 2
                    self.obs_cy = oy + oh // 2
                    self.obs_w = ow

            if goal_cx != -1:
                self.target_cx = goal_cx
                self.target_cy = goal_cy
                if self.obs_cx != -1 and abs(self.obs_cx - goal_cx) < self.OBS_DIST_TO_GOAL:
                    if self.obs_cx < goal_cx:
                        self.target_cx = self.obs_cx + self.obs_w // 2 + self.OBS_MARGIN
                    else:
                        self.target_cx = self.obs_cx - self.obs_w // 2 - self.OBS_MARGIN
            elif self.obs_cx != -1:
                self.target_cx = self.obs_cx + self.obs_w // 2 + self.OBS_MARGIN
                self.target_cy = self.obs_cy
            else:
                self.target_cx = -1

        self.draw_debug_window()

    def draw_debug_window(self):
        if self.current_image is None or not self.show_debug_window:
            return

        img = self.current_image
        cv2.line(img, (self.IMG_CX, 0), (self.IMG_CX, self.IMG_H), (255, 255, 255), 1)
        cv2.line(img, (0, self.IMG_CY), (self.IMG_W, self.IMG_CY), (255, 255, 255), 1)

        if self.vision_mode == 'ball' and self.target_cx != -1:
            cv2.circle(img, (self.target_cx, self.target_cy), self.target_radius, (255, 0, 255), 2)

        elif self.vision_mode == 'goal':
            if self.obs_cx != -1:
                top_left = (int(self.obs_cx - self.obs_w/2), int(self.obs_cy - 10))
                bottom_right = (int(self.obs_cx + self.obs_w/2), int(self.obs_cy + 10))
                cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)
                cv2.putText(img, "Obstacle", (top_left[0], top_left[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            if self.target_cx != -1:
                cv2.circle(img, (self.target_cx, self.target_cy), 5, (0, 255, 0), -1)
                cv2.putText(img, "Aim Here", (self.target_cx+10, self.target_cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.putText(img, f"STATE: {self.state}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(img, f"TILT: {self.head_tilt}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow('Pure_ROS_Vision_Debug', img)
        cv2.waitKey(1)

    # ============================================================
    #  状态机
    # ============================================================
    def process_state_machine(self):

        if self.target_cx == -1:
            if self.state in ['SEARCH_BALL', 'ALIGN_GOAL']:
                self.motion_manager.run_action('turn_left')
                rospy.sleep(self.SEARCH_SLEEP)
                return
            elif self.state in ['START_WALK', 'MOVE_LEFT', 'LOOK_UP', 'LOOK_DOWN', 'KICKING']:
                pass

        # ==================== START_WALK ====================
        if self.state == 'START_WALK':
            if self._step_counter < self.START_WALK_STEPS:
                common.loginfo(f">>> 开局前冲 ({self._step_counter + 1}/{self.START_WALK_STEPS}) <<<")
                self.gait_manager.move(1, self.START_WALK_DIST, 0, 0)
                rospy.sleep(self.START_WALK_SLEEP)
                self._step_counter += 1
                return
            self.gait_manager.stop()
            rospy.sleep(0.5)
            self._step_counter = 0
            self.state = 'MOVE_LEFT'
            return

        # ==================== MOVE_LEFT ====================
        if self.state == 'MOVE_LEFT':
            if self._step_counter < self.MOVE_LEFT_STEPS:
                common.loginfo(f">>> 左侧横移 ({self._step_counter + 1}/{self.MOVE_LEFT_STEPS}) <<<")
                self.motion_manager.run_action('move_right')
                rospy.sleep(self.MOVE_LEFT_SLEEP)
                self._step_counter += 1
                return
            self._step_counter = 0
            self.state = 'SEARCH_BALL'
            return

        # ==================== SEARCH_BALL ====================
        if self.state == 'SEARCH_BALL':
            self.state = 'APPROACH_BALL'

        # ==================== APPROACH_BALL ====================
        if self.state == 'APPROACH_BALL':
            err_y = self.target_cy - self.IMG_CY
            if abs(err_y) > self.APPROACH_TILT_DEADBAND:
                tilt_adjustment = int(err_y * self.APPROACH_TILT_RATE)
                self.head_tilt -= tilt_adjustment
                self.head_tilt = max(self.HEAD_TILT_MIN,
                                     min(self.HEAD_TILT_MAX, self.head_tilt))
                self.init_action(self.head_pan, self.head_tilt)

            dx = self.target_cx - self.IMG_CX

            if self.head_tilt < self.APPROACH_TILT_STOP and abs(dx) < self.APPROACH_DX_STOP:
                common.loginfo(">>> 走到球前，抬头看门对准！ <<<")
                self.state = 'LOOK_UP'
                return

            if dx > self.APPROACH_DX_TURN:
                self.motion_manager.run_action('turn_right')
                rospy.sleep(self.APPROACH_TURN_SLEEP)
            elif dx < -self.APPROACH_DX_TURN:
                self.motion_manager.run_action('turn_left')
                rospy.sleep(self.APPROACH_TURN_SLEEP)
            else:
                self.motion_manager.run_action('forward_one_step_1')
                rospy.sleep(self.APPROACH_FWD_SLEEP)

        # ==================== LOOK_UP ====================
        elif self.state == 'LOOK_UP':
            common.loginfo(f">>> [抬头] 抬头看球门！ tilt={self.LOOK_UP_TILT} <<<")
            self.head_tilt = self.LOOK_UP_TILT
            self.init_action(self.head_pan, self.head_tilt)
            rospy.sleep(self.LOOK_UP_SLEEP)
            self.vision_mode = 'goal'
            self.state = 'ALIGN_GOAL'

        # ==================== ALIGN_GOAL ====================
        elif self.state == 'ALIGN_GOAL':
            dx = self.target_cx - self.IMG_CX
            if abs(dx) < self.ALIGN_DX_STOP:
                common.loginfo(">>> 目标已对正，低头看球！ <<<")
                rospy.sleep(self.ALIGN_SLEEP)
                self.state = 'LOOK_DOWN'
            else:
                if dx > 0:
                    self.motion_manager.run_action('turn_right')
                else:
                    self.motion_manager.run_action('turn_left')
                rospy.sleep(self.ALIGN_SLEEP)

        # ==================== LOOK_DOWN ====================
        elif self.state == 'LOOK_DOWN':
            common.loginfo(f">>> [低头] 低头找球！ tilt={self.LOOK_DOWN_TILT} <<<")
            self.head_tilt = self.LOOK_DOWN_TILT
            self.init_action(self.head_pan, self.head_tilt)
            rospy.sleep(self.LOOK_DOWN_SLEEP)
            self.vision_mode = 'ball'
            self._fwd_step_counter = 0
            self.state = 'DECIDE_FOOT'

        # ==================== DECIDE_FOOT ====================
        elif self.state == 'DECIDE_FOOT':
            if self.target_cy < self.DECIDE_CY_TOO_FAR:
                if self._fwd_step_counter >= self.DECIDE_FWD_PER_TURN:
                    common.loginfo(f">>> 球太远，前进了{self.DECIDE_FWD_PER_TURN}步，{self.DECIDE_TURN_WHEN_FWD} {self.DECIDE_TURN_STEPS}步 <<<")
                    for _ in range(self.DECIDE_TURN_STEPS):
                        self.motion_manager.run_action(self.DECIDE_TURN_WHEN_FWD)
                        rospy.sleep(self.DECIDE_MOVE_SLEEP)
                    self._fwd_step_counter = 0
                else:
                    common.loginfo(f">>> 球太远 (cy={self.target_cy})，前进微调 ({self._fwd_step_counter+1}/{self.DECIDE_FWD_PER_TURN}) <<<")
                    self.motion_manager.run_action('forward_step_1')
                    rospy.sleep(self.DECIDE_MOVE_SLEEP)
                    self._fwd_step_counter += 1
                return

            elif self.target_cy > self.DECIDE_CY_TOO_CLOSE:
                common.loginfo(f">>> 球太近 (cy={self.target_cy})，后退微调 <<<")
                self._apply_gait('fine')
                # TODO: 需要 backward 动作组，目前暂用 gait_manager
                self.gait_manager.move(1, -0.015, 0, 0)
                rospy.sleep(self.DECIDE_MOVE_SLEEP)
                return

            # 横向微调：不对称死区
            if self.FORCE_BALL_SIDE is not None:
                dx = self.target_cx - self.IMG_CX
                if self.FORCE_BALL_SIDE == 'left' and dx > -self.FINE_LATERAL_MIN:
                    common.loginfo(f">>> 球不够左 (dx={dx})，右侧步把球推到左边 <<<")
                    self.motion_manager.run_action('move_right')
                    rospy.sleep(self.FINE_LATERAL_SLEEP)
                    return
                elif self.FORCE_BALL_SIDE == 'left' and dx < -self.FINE_LATERAL_MAX:
                    common.loginfo(f">>> 球太左 (dx={dx})，左侧步把球拉回来 <<<")
                    self.motion_manager.run_action('move_left')
                    rospy.sleep(self.FINE_LATERAL_SLEEP)
                    return
                elif self.FORCE_BALL_SIDE == 'right' and dx < self.FINE_LATERAL_MIN:
                    common.loginfo(f">>> 球不够右 (dx={dx})，左侧步把球推到右边 <<<")
                    self.motion_manager.run_action('move_left')
                    rospy.sleep(self.FINE_LATERAL_SLEEP)
                    return
                elif self.FORCE_BALL_SIDE == 'right' and dx > self.FINE_LATERAL_MAX:
                    common.loginfo(f">>> 球太右 (dx={dx})，右侧步把球拉回来 <<<")
                    self.motion_manager.run_action('move_right')
                    rospy.sleep(self.FINE_LATERAL_SLEEP)
                    return

            if self.target_cx == -1:
                common.loginfo(">>> 丢球！重回搜索 <<<")
                self.head_pan = self.HEAD_PAN
                self.head_tilt = self.HEAD_TILT_INIT
                self.init_action(self.head_pan, self.head_tilt)
                self.state = 'SEARCH_BALL'
                return

            dx = self.target_cx - self.IMG_CX
            if self.FORCE_KICK_FOOT is not None:
                self.act_kick = self.FORCE_KICK_FOOT
            elif dx < 0:
                self.act_kick = 'left_shot'
            else:
                self.act_kick = 'right_shot'

            common.loginfo(f">>> 距离刚好 (cy={self.target_cy})，选脚: {self.act_kick} <<<")
            self.state = 'KICKING'

        # ==================== KICKING ====================
        elif self.state == 'KICKING':
            common.loginfo(f">>> 执行射门: {self.act_kick} <<<")
            self.motion_manager.run_action(self.act_stand)
            rospy.sleep(0.5)

            self.head_tilt = self.KICK_HEAD_TILT
            self.init_action(self.head_pan, self.head_tilt)
            rospy.sleep(self.KICK_HEAD_SLEEP)

            self.motion_manager.run_action(self.act_kick)
            rospy.sleep(1.5)
            self.motion_manager.run_action(self.act_stand)
            rospy.sleep(1.5)

            common.loginfo(f">>> 踢完，原地等待 {self.KICK_IDLE_WAIT}s ... <<<")
            rospy.sleep(self.KICK_IDLE_WAIT)

            self.head_pan = self.HEAD_PAN
            self.head_tilt = self.HEAD_TILT_INIT
            self.init_action(self.head_pan, self.head_tilt)
            rospy.sleep(1.0)

            self.vision_mode = 'ball'
            self.target_cx = -1
            self.target_cy = -1

            self.state = 'START_WALK'
            self._step_counter = 0

    def run(self):
        while self.running and not rospy.is_shutdown():
            if self.state == 'DONE':
                break
            if self.start:
                self.process_state_machine()
                rospy.sleep(0.01)
            else:
                rospy.sleep(0.1)
        rospy.signal_shutdown('Task Completed.')


if __name__ == '__main__':
    AdvancedPenaltyNode('advanced_penalty_node').run()


