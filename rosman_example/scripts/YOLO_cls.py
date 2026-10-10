from ultralytics import YOLO
import time
import cv2

model=YOLO('/home/ubuntu/ros_ws/src/rosman_example/scripts/raicom_new.onnx',verbose=False)

cls_ten=[]
cls=['equipment','doctor','boom']

while 1:
    with open('/home/ubuntu/ros_ws/src/rosman_example/scripts/txt/YOLO_bridge_ros.txt','r') as f:
        is_yolo=f.readline()
    if is_yolo !='true':
        time.sleep(1)
        continue
    break

print('开始推理')

for i in range(5):
    img=cv2.imread(f'/home/ubuntu/ros_ws/src/rosman_example/scripts/img__/{i}.jpg')
    results=model(img)
    if results is not None:    
        cls_ten.append(results[0].names[int(results[0].boxes.cls[0].item())])
    if len(cls_ten)>=2:
        break

a=[1 if s=='equipment' else 0 for s in cls_ten]
b=[1 if s=='doctor' else 0 for s in cls_ten]
c=[1 if s=='boom' else 0 for s in cls_ten]
score=[sum(a),sum(b),sum(c)]

max_index = max(enumerate(score), key=lambda x: x[1])[0]

with open('/home/ubuntu/ros_ws/src/rosman_example/scripts/txt/YOLO_class.txt','w') as f:
    f.write(cls[max_index])
