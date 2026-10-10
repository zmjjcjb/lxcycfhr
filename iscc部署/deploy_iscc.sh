#!/usr/bin/env bash
# =============================================================================
# rosman_iscc 部署脚本（并行模式：与常驻 bringup 共存）
#
# 用法（本机执行）:
#   ./deploy_iscc.sh stage      上传包与动作组到机器人共享目录（bind 挂载，容器重建不丢）
#   ./deploy_iscc.sh install    解到 ros_ws/src + 动作组"只增不删"拷入
#   ./deploy_iscc.sh build      catkin build rosman_iscc（纯 Python 包，估 <1 分钟）
#   ./deploy_iscc.sh verify     rospack / roslaunch --files / 脚本语法 静态校验
#   ./deploy_iscc.sh rollback   回滚：删 ros_ws/src/rosman_iscc（动作组默认保留）
#   ./deploy_iscc.sh all        stage + install + build + verify
#   ./deploy_iscc.sh status     查看机器人上包/动作组/共享目录现状（只读）
#
# 环境变量:
#   ROBOT=pi@192.168.x.x   指定机器人（默认用 mDNS 解析 raspberrypi.local）
#   PASS=raspberrypi       主机 pi 的密码
#
# 干跑验证（不下场地、不发步态；在机器人 SSH 里执行）:
#   docker exec -u ubuntu rosman bash -lc 'source /opt/ros/noetic/setup.bash; \
#     source /home/ubuntu/ros_ws/devel/setup.bash; \
#     roslaunch rosman_iscc iscc_rosman_launch.launch gait_switch:=false'
#   注意1: iscc_config.yaml 里 auto_open_rqt_result: true 会在 VNC 桌面弹 rqt 窗口
#   注意2: apriltag_tags.yaml 写 size: 0.1，规则是 50mm，需拿实卡核对后再改
#
# 已知约束（2026-09-23 实机核实）:
#   - 容器名 rosman，不是教程写的 puppypi；ROS 在容器内，~/ros_ws 与 ~/software
#     只在 overlay2 可写层，不在 bind 挂载上 → 本脚本先把文件放 /home/pi/docker/src
#     （容器内 /home/ubuntu/share/src），再往里装，容器重建后共享目录仍在
#   - 老师给的 12 个 py 没有可执行位，install 时统一 chmod +x（launch 用 type="xx.py"）
#   - 机器人上 bringup 常驻，iscc launch 已改为并行模式（不加 enable_controller 等开关）
#   - 远程命令一律用 stdin 传脚本（bash -ls <<'EOS'），避免嵌套引号被吃掉
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PKG_PARENT="$SCRIPT_DIR/.."          # 新赛季开发/（rosman_iscc 在此）
LOCAL_AG_DIR="$SCRIPT_DIR/ActionGroups_iscc"
CT=rosman
SHARE_HOST=/home/pi/docker/src             # 主机侧共享目录
PASS="${PASS:-raspberrypi}"

if [[ -z "${ROBOT:-}" ]]; then
  IP="$(avahi-resolve -4 -n raspberrypi.local | awk '{print $2}')"
  ROBOT="pi@$IP"
fi

rsh()   { sshpass -p "$PASS" ssh -o StrictHostKeyChecking=accept-new "$ROBOT" "$@"; }
rcopy() { rsh "cat > $1"; }                                      # stdin -> 机器人主机文件
# 远程容器脚本：整段脚本经 stdin 进容器 bash，路径已在脚本里写死
rc()    { rsh "docker exec -i -u ubuntu $CT bash -ls"; }
# 动作组目录 /home/ubuntu/software/rosman_controller/ActionGroups 属主是 root
# （前人调的，ls -ld 实测 root:root drwxr-xr-x），ubuntu 写不进去，这里必须用 root
rcroot() { rsh "docker exec -i -u root $CT bash -ls"; }

stage() {
  echo "[stage] 上传 rosman_iscc 到共享目录（持久层）"
  tar czf - -C "$LOCAL_PKG_PARENT" rosman_iscc | rcopy "$SHARE_HOST/rosman_iscc.tgz"
  echo "[stage] 上传 10 个 ISCC 动作组"
  tar czf - -C "$LOCAL_AG_DIR" . | rsh "mkdir -p $SHARE_HOST/ActionGroups_iscc && tar xzf - -C $SHARE_HOST/ActionGroups_iscc"
  echo "[stage] OK"
}

install() {
  echo "[install] 解包到容器 ros_ws/src（带包名校验，非本包拒删）"
  rc <<'EOS'
set -e
PKG=/home/ubuntu/ros_ws/src/rosman_iscc
if [ -e "$PKG" ]; then
  grep -q '<name>rosman_iscc</name>' "$PKG/package.xml" || { echo "目标不是 rosman_iscc 包，拒绝覆盖"; exit 9; }
  rm -rf "$PKG"
fi
tar xzf /home/ubuntu/share/src/rosman_iscc.tgz -C /home/ubuntu/ros_ws/src
chmod +x "$PKG"/scripts/*.py
echo "包解压 + chmod +x OK（$(ls "$PKG"/scripts/*.py | wc -l) 个脚本）"
EOS
  echo "[install] 动作组只增不删（cp -n，目录属主 root → 用 root 写入）"
  rcroot <<'EOS'
set -e
AG=/home/ubuntu/software/rosman_controller/ActionGroups
cp -n /home/ubuntu/share/src/ActionGroups_iscc/*.d6a "$AG"/
echo "动作组总数: $(ls "$AG"/*.d6a | wc -l)"
EOS
  echo "[install] OK"
}

build() {
  echo "[build] catkin build rosman_iscc（纯 Python 包，est <1min）"
  rc <<'EOS'
set -e
source /opt/ros/noetic/setup.bash
cd /home/ubuntu/ros_ws
catkin build rosman_iscc
EOS
  echo "[build] OK"
}

verify() {
  echo "[verify] rospack find + roslaunch --files（解析 launch 引用链）"
  rc <<'EOS'
set -e
source /opt/ros/noetic/setup.bash
source /home/ubuntu/ros_ws/devel/setup.bash
rospack find rosman_iscc
roslaunch --files rosman_iscc iscc_rosman_launch.launch
EOS
  echo "[verify] scripts 语法（ast 解析，不写 pyc）"
  rc <<'EOS'
set -e
cd /home/ubuntu/ros_ws/src/rosman_iscc/scripts
python3 -c "import ast,glob; [ast.parse(open(f,encoding='utf-8').read()) for f in glob.glob('*.py')]; print('scripts 语法 OK')"
EOS
  echo "[verify] 10 个动作组是否齐"
  rc <<'EOS'
set -e
cd /home/ubuntu/software/rosman_controller/ActionGroups
miss=0
for f in descend_stairs_iscc hurdles_iscc move_left_iscc move_right_iscc pick_green_iscc pick_sandbag pick_sandbag_iscc place_green_iscc throw_sandbag_iscc throw_sandbag_iscc2; do
  [ -f "$f.d6a" ] || { echo "缺 $f"; miss=1; }
done
[ "$miss" = 0 ] && echo "10 个动作组齐全"
EOS
  echo "[verify] OK"
}

rollback() {
  echo "[rollback] 删除容器 ros_ws/src/rosman_iscc（同样的包名校验后才删）"
  rc <<'EOS'
set -e
PKG=/home/ubuntu/ros_ws/src/rosman_iscc
if [ -f "$PKG/package.xml" ] && grep -q '<name>rosman_iscc</name>' "$PKG/package.xml"; then
  rm -rf "$PKG"; echo "已删除包"
else
  echo "包不存在或校验失败，未删除"
fi
source /opt/ros/noetic/setup.bash
cd /home/ubuntu/ros_ws
catkin clean rosman_iscc -y 2>/dev/null || true
echo clean 完成
EOS
  echo "[rollback] 动作组默认保留（纯新增文件，删了会丢 ISCC 专用动作）"
  echo "          确认要删时手动进容器: rm /home/ubuntu/software/rosman_controller/ActionGroups/{descend_stairs_iscc,hurdles_iscc,move_left_iscc,move_right_iscc,pick_green_iscc,pick_sandbag,pick_sandbag_iscc,place_green_iscc,throw_sandbag_iscc,throw_sandbag_iscc2}.d6a"
}

status() {
  echo "[status] 机器人: $ROBOT"
  rc <<'EOS'
PKG=/home/ubuntu/ros_ws/src/rosman_iscc
[ -f "$PKG/package.xml" ] && echo "包已装: $(grep -o '<version>.*</version>' "$PKG/package.xml")" || echo "包未装"
echo "动作组数: $(ls /home/ubuntu/software/rosman_controller/ActionGroups/*.d6a | wc -l)"
echo "共享目录: $(ls /home/ubuntu/share/src)"
EOS
}

case "${1:-}" in
  stage)    stage ;;
  install)  install ;;
  build)    build ;;
  verify)   verify ;;
  rollback) rollback ;;
  status)   status ;;
  all)      stage; install; build; verify ;;
  *) sed -n '2,32p' "$0"; exit 1 ;;
esac
