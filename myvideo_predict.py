from ultralytics import YOLO

# 加载训练好的模型
model = YOLO(r'F:\Deeplearning\yolo_source8.3.163\ultralytics\runs\detect\train11\weights\best.pt')

# 检测视频文件
model.predict(
    source=r'F:\Deeplearning\yolo_source8.3.163\ultralytics\datasets\make_dataset\videos\001.mp4',
    save=True,          # 保存标注后的视频
    show=False,         # 不实时显示（设为True会在处理时显示）
    conf=0.25,          # 置信度阈值（与训练时一致）
    imgsz=1280,         # 输入尺寸（与训练时一致）
    save_txt=False,     # 视频不需要保存txt（如果需要可以设为True，会保存每帧的检测结果）
)

