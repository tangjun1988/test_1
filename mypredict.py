from ultralytics import YOLO

model = YOLO(r"F:\Deeplearning\yolo_source8.3.163\ultralytics\runs\detect\train7\weights\best.pt")

# 先测试在验证集上是否能检测到
print("=" * 50)
print("在验证集上测试...")
val_results = model.predict(
    source=r"F:\Deeplearning\yolo_source8.3.163\ultralytics\datasets\xz_dataset\images\val",
    save=True,
    show=False,
    save_txt=True,
    conf=0.01,
    project="runs/detect",
    name="predict_val"
)

# 统计验证集检测结果
total_detections = 0
for result in val_results:
    detections = len(result.boxes)
    total_detections += detections
    img_name = str(result.path).split('\\')[-1] if isinstance(result.path, str) else str(result.path)
    print(f"图像 {img_name}: 检测到 {detections} 个目标")
    if detections > 0:
        for i, box in enumerate(result.boxes[:5]):  # 只显示前5个
            conf_value = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
            print(f"  - 目标{i+1}: 类别={result.names[int(box.cls)]}, 置信度={conf_value:.3f}")

print(f"\n验证集总共检测到 {total_detections} 个目标")

# 然后在测试图像上预测
print("\n" + "=" * 50)
print("在测试图像上预测...")
test_results = model.predict(
    source=r"F:\Deeplearning\yolo_source8.3.163\make_dataset\images", 
    save=True,
    show=False,
    save_txt=True,
    conf=0.01,  # 进一步降低置信度阈值，查看是否有任何检测结果
)

# 统计测试集检测结果
total_test_detections = 0
for result in test_results:
    detections = len(result.boxes)
    total_test_detections += detections
    img_name = str(result.path).split('\\')[-1] if isinstance(result.path, str) else str(result.path)
    print(f"图像 {img_name}: 检测到 {detections} 个目标")
    if detections > 0:
        for i, box in enumerate(result.boxes[:5]):  # 只显示前5个
            conf_value = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
            print(f"  - 目标{i+1}: 类别={result.names[int(box.cls)]}, 置信度={conf_value:.3f}")

print(f"\n测试集总共检测到 {total_test_detections} 个目标")

