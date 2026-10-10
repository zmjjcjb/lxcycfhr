#!/usr/bin/env python3
# encoding: utf-8
# @data:2023-07-06
# @author:aiden
# Line following logic processing(巡线逻辑处理)
import math
import time
from rosman_sdk import misc, common
from rosman_example import gait_posture

class VisualPatrol:
    # 丢线保护阈值。set_step 用的是 step_num=0 的连续步态，丢线后若不再发指令，
    # 步态引擎会继续执行最后那条已经饱和的转向指令往前冲 —— 越偏越远直到彻底出线，
    # 这就是"直线越来越偏"的正反馈。
    # 注意单位：on_no_line() 是在调用方的主循环里被反复调的（节点循环约 100Hz，
    # 远快于识别的 20Hz），所以这里必须用"持续时间"而不是"次数"来判，否则阈值会缩短数倍。
    LOST_HOLD_SEC = 0.5    # 前 0.5 秒原地站住，容忍瞬时遮挡和十字处短暂丢线
    LOST_SEARCH_SEC = 3.0  # 0.5~3 秒朝最后偏离的反方向原地转找线；再找不到就 stop
    SEARCH_YAW = 10        # 找线时原地转的幅度(deg)，用满厂家上限
    SAT_YAW = 10           # 修正量达到该值说明转向已饱和
    SAT_X = 0.010          # 饱和时的步幅
    CORNER_YAW = 7         # 修正量超过该值认为正在过弯
    CORNER_X = 0.014       # 过弯步幅
    START_STAND_SEC = 2.5  # 启动后先原地站住这么久再迈步。实测开局 0~7 秒识别目标会乱跳
                           # （x 在 14~288 之间跳、命中排在 218/252/306 之间换），因为头部舵机
                           # 还在到位、color_detection 还在用旧参数出的头几帧、相机自动曝光未稳。
                           # 这段宽限期同时避开这三件事。设 0 关闭。
    # ---- 下面三个是"直线越纠越偏"的旋钮，现在都是关闭/原值，别和近排优先一起改 ----
    DEADBAND = 10          # 死区(px)：|偏差|小于它完全不纠。想更早介入就调小，建议先试 5
    KI = 0.0               # 积分增益：专治长期单边偏置（实测有符号偏差均值 -40px、68% 帧偏左）。
                           # 0=关闭。要开就从 0.0004 起，别直接上大值，会 windup
    I_MAX = 3.0            # 积分项最多贡献几度，防 windup
    TRACE_SEC = 0.25       # 逐帧追踪日志的限频(秒)，设 0 关闭。用于查"跟错目标/哪排ROI命中"

    def __init__(self, gait_manager):
        self.calib_config = common.get_yaml_data('/home/ubuntu/ros_ws/src/rosman_example/config/calib.yaml')
        self.x_range = [0, gait_posture.X_MAX]  # The step length for walking forward in units of m(前进脚步大小单位m)
        # 转弯范围。厂家 gait_manager 的上限是 ±10（rotation_angle_range=[0,10]），
        # 原来只用 ±8：25~53 秒那段误差长期钉在 -120px、转向压在 -7 饱和不收敛，
        # 就是修正权限不够 + 降速太狠一起把自己锁死了。这里用满 ±10。
        self.yaw_range = [-10, 10]
        self.gait_manager = gait_manager
        # 姿态参数统一由 gait_posture 提供。原来这里 go/turn 各写一遍、且 turn 的两行
        # 误写到 go_gait_param 上；y_swap_amplitude 写进 gait_param 也是无效项，已一并去掉。
        self.go_gait_param = gait_posture.build(gait_manager)
        self.turn_gait_param = gait_posture.build(gait_manager, turn=True)
        self.go_dsp = list(gait_posture.DSP)
        self.go_arm_swap = gait_posture.ARM_SWAP
        self.turn_dsp = list(gait_posture.DSP)
        self.turn_arm_swap = gait_posture.ARM_SWAP

        self.x_max = self.x_range[1]
        self.lost_since = None
        self.last_rot = 0
        self.lost_stopped = False
        self.trace_t = 0.0
        self.start_t = time.monotonic()
        self.integ = 0.0

    def update_turn_gait(self, dsp=None, x_max=None, arm_swap=None, walking_param=None):
        if walking_param is not None:
            self.turn_gait_param = walking_param
        if dsp is not None:
            self.turn_dsp = dsp
        if x_max is not None:
            self.x_max = x_max
        if arm_swap is not None:
            self.turn_arm_swap = arm_swap

    def update_go_gait(self, dsp=None, x_max=None, arm_swap=None, walking_param=None):
        if walking_param is not None:
            self.go_gait_param = walking_param
        if dsp is not None:
            self.go_dsp = dsp
        if x_max is not None:
            self.x_max = x_max
        if arm_swap is not None:
            self.go_arm_swap = arm_swap

    def process(self, x, width, y=0):
        # 启动宽限期：先原地站住，等头部舵机到位、新颜色/ROI 参数生效、相机自动曝光稳定。
        # 这期间识别目标乱跳，跟着走只会把机器人带偏（实测开局偏差能一路涨到 -309px）。
        if self.START_STAND_SEC and time.monotonic() - self.start_t < self.START_STAND_SEC:
            self.gait_manager.set_step(self.go_dsp, 0, 0, 0, self.go_gait_param,
                                       arm_swap=self.go_arm_swap, step_num=0)
            return
        # Adjust left and right based on x-coordinate value(左右根据x坐标值进行调整)
        #width=640
        # y 只用于追踪日志：识别层是按 ROI 顺序第一个命中就返回，三个 ROI 的 y 区间不同，
        # 所以 y 能反推出"这一帧到底是哪一排框命中的"——排查弯道跟错目标必需。
        center_x = width/2 + self.calib_config['center_x_offset'] - 20
        err = x - center_x
        if abs(err) < self.DEADBAND:
            yaw_output = 0
        elif abs(err) < width/6:
            yaw_output = math.copysign(1, err) + misc.val_map(err, -width/6, width/6, self.yaw_range[0] + 3, self.yaw_range[1] - 3)
        elif abs(err) < width/4:
            yaw_output = math.copysign(1, err) + misc.val_map(err, -width/4, width/4, self.yaw_range[0], self.yaw_range[1])
        else:
            yaw_output = math.copysign(self.yaw_range[1], err)

        # 积分项：纯 P 控制消不掉长期单边偏置（实测有符号偏差均值 -40px、68% 帧偏左，
        # 机器人会长期骑在线的一侧）。KI=0 时这段完全不参与，行为与加它之前逐位一致。
        if self.KI:
            self.integ = max(-self.I_MAX, min(self.I_MAX, self.integ + err * self.KI))
            yaw_output = max(self.yaw_range[0], min(self.yaw_range[1], yaw_output + self.integ))

        # 看到线就清掉丢线计时与 stop 标志
        self.lost_since = None
        self.lost_stopped = False

        # When turning, decreases forward step length(转弯时前进步幅减小)
        if abs(yaw_output) >= self.SAT_YAW:
            x_output = self.SAT_X          # 转向已饱和还往前冲，只会越偏越多
        elif abs(yaw_output) > self.CORNER_YAW:
            x_output = self.CORNER_X
        else:
            x_output = self.x_max

        rot = int(-yaw_output)
        self.last_rot = rot
        if abs(yaw_output) < 3:
            self.gait_manager.set_step(self.go_dsp, x_output, 0, rot, self.go_gait_param, arm_swap=self.go_arm_swap, step_num=0)
        else:
            self.gait_manager.set_step(self.turn_dsp, x_output, 0, rot, self.turn_gait_param, arm_swap=self.turn_arm_swap, step_num=0)

        now = time.monotonic()
        if self.TRACE_SEC and now - self.trace_t >= self.TRACE_SEC:
            self.trace_t = now
            common.loginfo('[巡线] x=%d y=%d 宽=%d 目标=%d 偏差=%+d 转向=%+.1f 步幅=%.3f' % (
                x, y, width, int(center_x), int(x - center_x), yaw_output, x_output))

    def on_no_line(self):
        '''没检测到线时必须调用：主动覆盖上一条步态指令，不再沿旧转向继续前冲
        (must be called when no line is detected, otherwise the last steering keeps running)'''
        now = time.monotonic()
        if self.lost_since is None:
            self.lost_since = now
        dt = now - self.lost_since
        self.integ = 0.0    # 看不到线时积分必须清零，否则会把找回动作带偏

        if self.TRACE_SEC and now - self.trace_t >= self.TRACE_SEC:
            self.trace_t = now
            common.loginfo('[巡线] 丢线 %.2fs 上次转向=%+d' % (dt, self.last_rot))

        if dt <= self.LOST_SEARCH_SEC:
            # 0.5 秒内原地站住；之后朝最后偏离的反方向原地转找回线。
            # 注意：即使没有明确方向（丢线前是直行），也必须发一条 x=0 的指令，
            # 否则步态引擎会按上一条全速直行继续走。
            rot = 0
            if dt > self.LOST_HOLD_SEC and self.last_rot:
                rot = -int(math.copysign(self.SEARCH_YAW, self.last_rot))
            if rot:
                self.gait_manager.set_step(self.turn_dsp, 0, 0, rot, self.turn_gait_param,
                                           arm_swap=self.turn_arm_swap, step_num=0)
            else:
                self.gait_manager.set_step(self.go_dsp, 0, 0, 0, self.go_gait_param,
                                           arm_swap=self.go_arm_swap, step_num=0)
            return

        if not self.lost_stopped:
            self.lost_stopped = True
            self.gait_manager.stop()
            common.loginfo('[巡线] 超过 %.0f 秒找不到线，已停步等待处理' % self.LOST_SEARCH_SEC)
