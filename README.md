# ROSMan 仿人机器人比赛代码（睿抗 RAICOM / 信安智能安全赛）

ROSMan（树莓派 5 + 人形双足，ROS1 Noetic 跑在机器人 Docker 容器 `rosman` 内）的比赛开发工作区。
**本仓库为 private，仅限队内协作**——文档中含机器人登录凭据与内网信息，请勿转公开。

## 仓库结构

| 目录/文件 | 用途 |
|---|---|
| `rosman_iscc/` | **信安赛主力包**：`scripts/iscc_controller.py` 是全任务链状态机入口（巡线→跨栏→台阶→双橙线走廊→沙包投掷→绿块抓取→回程放置） |
| `rosman_example/` | 睿抗主程序 `competition.py` + 公共库（color_common/common/approach_object/gait_posture，iscc 也依赖它） |
| `iscc部署/` | 实机部署脚本 `deploy_iscc.sh` + iscc 动作组（.d6a） |
| `动作组版本-跨栏/` | 跨栏动作组 v2~v8 调参轨迹（v9/v10 仅在机器人上，待回同步） |
| `进度记录-信安抓取-20261010.md` | 当前进度、抓取链代码分析、问题清单 |
| `国庆改动清单-机器人-20261010.md` | 机器人上国庆期间改动的完整核查清单（含相机锁定运维要点） |
| `测试记录-跨栏-20260924.md` | 跨栏调参记录 |

## 开发纪律（详见工作区 AGENTS.md）

1. **改代码只改本仓库**（实时版）；部署到机器人容器 `~/ros_ws/src/` 后实机验证；
   机器人上如有现场微调，先回同步到本仓库再继续开发。
2. 步态姿态参数唯一来源：`rosman_example/src/rosman_example/gait_posture.py`，别在节点里散写数值。
3. 机器人可能领先本仓库（多台电脑开发）——**改前先与实机 md5 对比**，以实机为准回同步。
4. 大文件不入库：onnx 模型、视频、备份归档（见 .gitignore）。

## 机器人运维要点（易踩坑）

- 机器人 SSH：`pi@<机器人IP>`（IP 由热点 DHCP 分配，mDNS 名 `raspberrypi.local`），
  ROS 在容器内：`docker exec -it -u ubuntu -w /home/ubuntu rosman /bin/bash`。
- **`/home/pi/set_cam_lock.sh` 无自启动：机器人每次重启或 USB 相机重插后必须手动重跑**，
  否则整套 LAB 颜色阈值漂移、视觉全废（10-02~10-06 标定全部绑定该参数组）。
- 机器人上跑节点一律 `python3 -u`。
