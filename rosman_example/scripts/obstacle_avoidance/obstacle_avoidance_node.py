#!/usr/bin/env python3
# encoding: utf-8
# @data: 2024/05/10
# @author: aiden
import time
import math
import rospy
import signal
import threading
import numpy as np
import pandas as pd
from rosman_sdk import VL53L0X
from rosman_kinematics.gait_manager import GaitManager
from rosman_kinematics.motion_manager import MotionManager

# Create a VL53L0X object
tof = VL53L0X.VL53L0X()

# Start ranging
tof.start_ranging(VL53L0X.VL53L0X_BETTER_ACCURACY_MODE)

timing = tof.get_timing()
if (timing < 20000):
    timing = 20000
print ("Timing %d ms" % (timing/1000))

running = True
thread_running = True
def shutdown(signum, frame):
    global thread_running
    print("Shutdown obstacle_avoidance_node")
    thread_running = False

signal.signal(signal.SIGINT, shutdown)

distance = -1
last_distance = 999
#Thread to perform action groups (执行动作组线程)
def run():
    global distance, last_distance, running
    while thread_running:
        if distance >= 0:
            if distance > 250 or distance < 100: # When detected distance is greater than 250mm (检测距离大于250mm时)
                if 100 <= last_distance <= 250: # If last distance was between 100 and 250mm, meaning just turned but not fully aligned (如果上次距离大于350mm, 说明是刚转到检测不到障碍物，但是没有完全转正)
                    last_distance = distance
                    gait_manager.move(2, 0, 0, 8, step_num=4) # Turn right (右转)
                else:
                    last_distance = distance
                    gait_manager.move(2, 0.01, 0, 0) # Move forward (直走)
            elif 150 < distance <= 250: # When detected distance is between 150 and 250mm (检测距离在150-350mm时)
                last_distance = distance
                gait_manager.move(2, 0, 0, 8, step_num=4) # Turn right (右转)
            else:
                last_distance = distance
                gait_manager.move(2, -0.01, 0, 0) # Move backward (后退)
            time.sleep(timing/1000000.00)
        else:
            time.sleep(0.01)
    
    running = False

rospy.init_node('obstacle_avoidance_node', anonymous=True)
gait_manager = GaitManager()
motion_manager = MotionManager()
motion_manager.run_action('walk_ready') # Initial posture (初始姿态)

#Start the action thread (启动动作的线程)
threading.Thread(target=run, daemon=True).start()

distance_data = []
while running:
    dist = tof.get_distance() # Get distance measurement (获取测距)

    if dist < 30 or dist == 8190:
        dist = 0
    distance_data.append(dist)
    data = pd.DataFrame(distance_data)
    data_ = data.copy()
    u = data_.mean()
    std = data_.std()
    data_c = data[np.abs(data - u) <= std]
    d = data_c.mean()[0]

    if len(distance_data) == 5:
        distance_data.remove(distance_data[0])
        if math.isnan(d):
            distance = 0
        else:
            distance = int(d)
        print ("%d mm" % distance)
    time.sleep(timing/1000000.00)
    if not thread_running:
        break
gait_manager.stop()
tof.stop_ranging()  # Stop ranging (关闭测距)
