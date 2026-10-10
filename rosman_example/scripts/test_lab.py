import cv2
import numpy as np

# 全局变量：存储阈值（滑块调节时更新）
L_min, L_max = 0, 255
A_min, A_max = 160, 253
B_min, B_max = 117, 197

current_frame = None  # 存储当前摄像头帧（用于暂停时显示）

# ---------------------- 滑块回调函数（更新全局阈值）----------------------
def update_L_min(val):
    global L_min
    L_min = val

def update_L_max(val):
    global L_max
    L_max = val

def update_A_min(val):
    global A_min
    A_min = val

def update_A_max(val):
    global A_max
    A_max = val

def update_B_min(val):
    global B_min
    B_min = val

def update_B_max(val):
    global B_max
    B_max = val

# ---------------------- 帧处理函数（转换LAB + 计算掩码）----------------------
def process_frame(frame):
    # 转换为LAB颜色空间
    lab_img = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab_img)
    
    # 计算各通道阈值掩码
    mask_L = cv2.inRange(L, L_min, L_max)
    mask_A = cv2.inRange(A, A_min, A_max)
    mask_B = cv2.inRange(B, B_min, B_max)
    
    # 三通道掩码交集（只有全部满足才保留）
    mask = cv2.bitwise_and(mask_L, cv2.bitwise_and(mask_A, mask_B))
    
    return mask

# ---------------------- 主函数（初始化+运行循环）----------------------
def main(camera_index=0):
    global current_frame
    # 初始化摄像头
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"错误：无法打开摄像头（索引{camera_index}）")
        return
    
    # 设置摄像头参数（可选）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 读取第一帧作为初始画面
    ret, frame = cap.read()
    if not ret:
        print("错误：无法读取摄像头画面")
        cap.release()
        return
    current_frame = frame
    
    # 创建窗口和滑块
    cv2.namedWindow("LAB", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("L_min", "LAB", L_min, 255, update_L_min)
    cv2.createTrackbar("L_max", "LAB", L_max, 255, update_L_max)
    cv2.createTrackbar("A_min", "LAB", A_min, 255, update_A_min)
    cv2.createTrackbar("A_max", "LAB", A_max, 255, update_A_max)
    cv2.createTrackbar("B_min", "LAB", B_min, 255, update_B_min)
    cv2.createTrackbar("B_max", "LAB", B_max, 255, update_B_max)
    
    
    paused = False
    while True:
        # 读取摄像头帧（暂停时不读取）
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("警告：摄像头帧读取失败")
                break
            current_frame = frame
        
        # 处理帧和拼接显示画面
        mask = process_frame(current_frame)
        
        # 显示画面
        cv2.imshow("cs", mask)
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # 退出
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("程序退出")

if __name__ == "__main__":
    # 运行主函数（camera_index=0 为默认摄像头，外接摄像头可改为1）
    main(camera_index=0)