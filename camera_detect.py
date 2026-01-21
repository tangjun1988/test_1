from ultralytics import YOLO
import cv2

# 加载训练好的模型
model = YOLO(r'runs\detect\train11\weights\best.pt')

# 打开摄像头（0表示默认摄像头，如果是外接摄像头可以尝试1、2等）
cap = cv2.VideoCapture(0)

# 设置摄像头分辨率（可选，根据你的摄像头调整）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("摄像头已启动，按 'q' 键退出程序...")

while True:
    # 读取摄像头画面
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面，请检查摄像头是否连接正常")
        break
    
    # 使用模型进行预测
    # conf=0.5 表示置信度阈值，只有置信度大于0.5的检测结果才会显示
    # 你可以根据需要调整这个值（0.0-1.0之间）
    results = model(frame, conf=0.25, imgsz=1280)
    
    # 在画面上绘制检测结果（包括边界框、类别名称、置信度）
    annotated_frame = results[0].plot()
    
    # 显示结果
    cv2.imshow('摄像头实时检测 - kunkun/fan/xiang', annotated_frame)
    
    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("正在退出...")
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
print("程序已退出")

