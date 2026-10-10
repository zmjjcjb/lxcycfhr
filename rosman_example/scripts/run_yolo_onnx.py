import onnxruntime as ort
import numpy as np
import time
import PIL
import cv2

def letterbox(img, target_size=(640, 640), color=(114, 114, 114)):
    img_h, img_w = img.shape[:2]
    # 目标宽高
    target_w, target_h = target_size
    print(img_h)
    
    # 1. 按比例缩放：计算缩放因子（取宽高方向最小因子，避免图像超出目标尺寸）
    scale = min(target_w / img_w, target_h / img_h)
    # 计算缩放后的实际宽高（确保按比例）
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    
    # 缩放图像
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 2. 填充黑边：计算水平/垂直方向需填充的像素（左右/上下对称填充）
    pad_w = (target_w - new_w) // 2  # 水平填充（左）
    pad_h = (target_h - new_h) // 2  # 垂直填充（上）
    # 处理整除残留（确保总填充量满足目标尺寸）
    pad_w2 = target_w - new_w - pad_w  # 水平填充（右）
    pad_h2 = target_h - new_h - pad_h  # 垂直填充（下）
    
    # 填充操作（OpenCV的copyMakeBorder函数）
    padded_img = cv2.copyMakeBorder(
        resized_img, pad_h, pad_h2, pad_w, pad_w2,
        cv2.BORDER_CONSTANT, value=color
    )
    
    return padded_img, scale, (pad_w, pad_h)

# 1. 配置推理会话（指定CPU后端，树莓派无GPU）
# 若模型较大，可添加 providers=["CPUExecutionProvider"] 明确指定
session = ort.InferenceSession(
    "/home/ubuntu/ros_ws/src/rosman_example/scripts/raicom.onnx",  # 模型路径
    providers=["CPUExecutionProvider"]
)

# 2. 获取模型输入/输出信息（关键，需匹配模型定义）
input_name = session.get_inputs()[0].name  # 输入张量名称
input_shape = session.get_inputs()[0].shape  # 输入维度（如(1,3,224,224)）
input_dtype = session.get_inputs()[0].type  # 输入数据类型（如tensor(float)）
output_name = session.get_outputs()[0].name  # 输出张量名称

print(f"输入信息：名称={input_name}，形状={input_shape}，类型={input_dtype}")

img=cv2.imread('./20.jpg')
img=np.array(img).astype(np.float32)
print(img.shape)
img,_,_=letterbox(img=img)
print(img.shape)
img=np.transpose(img,(2,0,1))
print(img.shape)
img=img[np.newaxis,...]
print(img.shape)
# 3. 准备输入数据（需严格匹配输入维度和类型）
# 示例：生成随机输入（实际场景需替换为真实数据，如图像预处理后的数据）
# input_data = np.random.randn(*input_shape).astype(np.float32)  # 转换为float32（ONNX默认类型）

# 4. 执行推理（统计耗时）
start_time = time.time()
outputs = session.run([output_name], {input_name: img})  # 推理核心调用
infer_time = time.time() - start_time

# 5. 解析输出结果
output_result = outputs[0]  # outputs为列表，首个元素为核心输出
print(f"推理耗时：{infer_time:.4f}秒")
print(f"输出形状：{output_result.shape}")
print(f"输出前5个值：{output_result.flatten()[:5]}")  # 打印部分结果验证
temp=np.array(outputs)[0][0]
print(temp.shape)
print(temp[:,8000])
