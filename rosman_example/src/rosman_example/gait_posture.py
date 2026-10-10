#!/usr/bin/env python3
# encoding: utf-8
# 步态姿态的单一来源：和前倾、横向稳定有关的旋钮全部集中在这里。
# 改姿态只改这一份，不要在 competition.py 或各节点里再写一遍数值。
#
# 符号约定：init_x_offset 为负使身体更前倾（实测结论，来自原 approach_object 的注释）。
# 机器人没有 IMU 平衡闭环——WalkingParam 里的 balance_enable 和四个 balance_* 增益
# 在整个 rosman_driver 中无人读取，也没有任何 IMU 订阅，步态是纯开环几何的。
# 所以前倾只能靠姿态偏移和步幅解决，重心一旦越过脚尖就一定会倒，不要指望它自己稳住。

# 直行姿态。以机器人当前参数为底（保留动作上位机调过的 roll/yaw），只覆盖下面这些项。
#
# 【当前这组是 2026-09-23 实机验证过的配置，不要随意改回"上次比赛值"】
# 验证方式：init_x_offset=0.0 + 步幅0.02，5 步，用 /imu 记录前倾角曲线：
#   基线 19.94° → 行走均值 19.85°（前后对称摆动，无单向累积）→ 停后稳定 19.31°、角速度 1.3°/s
#   结论：能前进、不栽、走完自己站回去。
# 前提是机械零位正常：摔一次会让静止前倾角从 18.3° 变到 24.8°（重新拧紧舵盘后回到 19.96°）。
# 所以每次摔过之后，先量静止前倾角再谈调参。
PATROL_OVERRIDE = {
    # 前后偏移(m)。负值使身体更前倾。完整实测矩阵：
    #   -0.01 + 步幅 0.02 → 前进，但前倾栽倒（这就是"上次比赛"的组合）
    #    0.0 + 步幅 0.01 / 0.014 → 不栽，但原地倒腿不前进
    #    0.0 + 步幅 0.02 → 前进且不栽 ← 当前采用
    'init_x_offset': 0.0,
    'init_y_offset': -0.002,        # 左右偏移(m)
    'body_height': 0.025,           # 比完全站直低多少(m)，范围 0.015~0.06，越低越蹲越前倾
    'hip_pitch_offset': 15,         # 髋关节前后倾斜角(deg)，越大越弯腰。厂家 APP 硬编码 15
    'init_pitch_offset': 0,         # 躯干俯仰角(deg)，前倾最直接的旋钮。未动过
    'step_height': 0.02,            # 抬脚高度(m)，范围 0.01~0.04
    'z_swap_amplitude': 0.006,      # 躯体上下浮动(m)
    'pelvis_offset': 3,             # 髋左右摆角(deg)
    'step_fb_ratio': 0.028,         # 左右脚前后错开(m)，决定支撑多边形的前后位置
}

# 转身姿态：抬脚略低、上下浮动略大（09-22 实机原值）
TURN_OVERRIDE = {
    'step_height': 0.015,
    'z_swap_amplitude': 0.010,
    'init_y_offset': 0,
}

# dsp = [period_time(ms), dsp_ratio(一个周期内双脚着地占比), y_swap_amplitude(左右摆幅 m)]
# 注意：y_swap_amplitude 只从 dsp 第三位取，写进 walking_param 是无效的
# （gait_manager.update_param 里 walking_param 字典根本没有这一项）
# 【单变量回退】09-23 曾把三项一起提速（period 500→400、yaw ±8→±10、弯道档 0.006/0.003→0.014/0.010），
# 结果约 24 秒就前倾摔倒（前倾角冲到 98°，电池 11.94V 健康，不是电量问题）。
# 打包改导致无法归因，是我的错。现在只回退 period 这一项（400→500），
# yaw ±10 与弯道档位保持不变 —— 用来判断"摔"到底是步频造成的还是那两项造成的。
# 若这次仍摔：下一步回退 yaw_range(±10→±8)；若不摔且不再卡死：说明步频就是元凶，提速只能另找路子。
DSP = [500, 0.4, 0.02]
DSP_FAST = [420, 0.4, 0.02]

# 巡线步幅(m)。厂家上限 0.02，也是上次实际在用的值。
# 0.014 和 0.01 实测都变成"原地倒腿"，所以基线回到 0.02。
X_MAX = 0.02

# 摆臂幅度，单位度。radians(30)=0.52，competition.py 里 start_forward 曾误填的 0.5 就是这个弧度值
ARM_SWAP = 30

# 接近物体（approach_object）单独一组。init_x_offset 原 -0.015 是 09-23 前的旧值，
# 2026-09-24 翻栅栏实测第一步前倾栽倒，与巡线组当时踩的坑同源，已归零；
# 其余项（init_y -0.005、pelvis 5、dsp_ratio 0.2）仍是历史值，一次一个变量，暂不动。
APPROACH_OVERRIDE = {
    'init_x_offset': 0.0,
    'init_y_offset': -0.005,
    'pelvis_offset': 5,
    'step_height': 0.02,
    'z_swap_amplitude': 0.006,
}
APPROACH_DSP = [500, 0.2, 0.02]


def build(gait_manager, turn=False):
    '''生成统一姿态的 walking_param(build walking_param with the unified posture)'''
    param = gait_manager.get_gait_param()
    param.update(PATROL_OVERRIDE)
    if turn:
        param.update(TURN_OVERRIDE)
    return param


def build_approach(gait_manager):
    '''接近物体姿态，在巡线姿态之上再覆盖 APPROACH_OVERRIDE'''
    param = gait_manager.get_gait_param()
    param.update(PATROL_OVERRIDE)
    param.update(APPROACH_OVERRIDE)
    return param
