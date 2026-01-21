# YOLO共享内存通信系统 - 部署说明

## 📋 项目结构

```
项目目录/
├── config.yaml              # 配置文件（所有参数都在这里）
├── config_loader.py         # 配置加载模块
├── logger_setup.py          # 日志系统模块
├── data_source_shm.py       # 程序A：数据源（读取图像并写入共享内存）
├── inference_shm.py         # 程序B：推理引擎（从共享内存读取并推理）
├── start.sh                 # 启动脚本（Linux/Mac）
├── stop.sh                  # 停止脚本（Linux/Mac）
├── requirements.txt         # Python依赖包列表
└── logs/                    # 日志目录（自动创建）
    ├── app.log              # 应用日志
    ├── data_source.log      # 数据源程序日志
    └── inference.log        # 推理程序日志
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

或者单独安装：
```bash
pip install opencv-python numpy ultralytics pyyaml
```

### 2. 配置系统

编辑 `config.yaml` 文件，修改以下参数：

```yaml
data_source:
  type: "video"  # 可选: video, camera, images
  video_path: "你的视频路径.mp4"
  fps: 24

inference:
  model_path: "yolov8n.pt"
  fps: 24
  conf_threshold: 0.3
  classes: [0]  # 0=人, 1=自行车, 2=汽车 等
```

### 3. 运行系统

#### 方式一：使用启动脚本（推荐，Linux/Mac）

```bash
# 启动系统
./start.sh

# 停止系统
./stop.sh
```

#### 方式二：手动启动（Windows/Linux/Mac）

**终端1（数据源）：**
```bash
python data_source_shm.py
```

**终端2（推理）：**
```bash
python inference_shm.py
```

按 `q` 键退出推理程序，按 `Ctrl+C` 退出数据源程序。

## 📝 配置文件说明

### config.yaml 参数详解

#### 数据源配置 (data_source)

- `type`: 数据源类型
  - `"video"`: 从视频文件读取
  - `"camera"`: 从摄像头读取
  - `"images"`: 从图片文件夹读取

- `video_path`: 视频文件路径（type="video"时使用）
- `camera_index`: 摄像头索引（type="camera"时使用，0=第一个摄像头）
- `images_dir`: 图片文件夹路径（type="images"时使用）
- `fps`: 发送频率（Hz），每秒发送多少帧

#### 共享内存配置 (shared_memory)

- `name`: 共享内存名称（程序A和B必须一致）
- `max_width`: 最大图像宽度（像素）
- `max_height`: 最大图像高度（像素）
- `max_channels`: 通道数（RGB图像是3）

#### 推理配置 (inference)

- `model_path`: YOLO模型文件路径
- `fps`: 推理频率（Hz），建议和data_source.fps一致
- `conf_threshold`: 置信度阈值（0.0-1.0）
- `iou_threshold`: IoU阈值（0.0-1.0）
- `classes`: 检测类别列表（空列表[]表示检测所有类别）

#### 日志配置 (logging)

- `level`: 日志级别（DEBUG, INFO, WARNING, ERROR）
- `file`: 日志文件路径（None表示不保存到文件）
- `console`: 是否输出到控制台

## 📊 查看日志

### 实时查看日志

```bash
# 查看数据源日志
tail -f logs/data_source.log

# 查看推理日志
tail -f logs/inference.log

# 查看应用日志（如果配置了）
tail -f logs/app.log
```

### Windows PowerShell 查看日志

```powershell
# 查看最后50行
Get-Content logs/app.log -Tail 50

# 实时查看（类似tail -f）
Get-Content logs/app.log -Wait -Tail 20
```

## 🔧 常见问题

### 1. 配置文件找不到

**问题：** `FileNotFoundError: 配置文件不存在: config.yaml`

**解决：** 确保 `config.yaml` 文件在项目根目录，或者程序会使用默认配置。

### 2. 共享内存连接失败

**问题：** `无法连接到共享内存`

**解决：** 
- 确保程序A（data_source_shm.py）先启动
- 检查两个程序的 `SHM_NAME` 配置是否一致
- 在Linux上，检查 `/dev/shm` 目录权限

### 3. 摄像头无法打开

**问题：** `无法打开摄像头`

**解决：**
- 检查摄像头是否连接
- 尝试修改 `camera_index`（0, 1, 2...）
- Windows上可能需要安装摄像头驱动

### 4. 模块导入错误

**问题：** `ModuleNotFoundError: No module named 'xxx'`

**解决：**
```bash
pip install -r requirements.txt
```

## 📈 性能优化建议

1. **调整FPS**：根据硬件性能调整 `fps` 参数
2. **调整图像尺寸**：如果图像太大，可以降低 `max_width` 和 `max_height`
3. **选择合适模型**：`yolov8n.pt`（最快）< `yolov8s.pt` < `yolov8m.pt` < `yolov8l.pt` < `yolov8x.pt`（最准）

## 🎯 下一步：部署到生产环境

1. **确认相机型号**：询问导师具体相机型号和接口类型
2. **集成工业相机**：根据相机SDK实现相机接口
3. **系统服务化**：创建systemd服务（Linux）或Windows服务
4. **监控和告警**：添加健康检查和异常告警

## 📞 需要帮助？

如果遇到问题，请检查：
1. 日志文件（`logs/` 目录）
2. 配置文件是否正确
3. Python依赖是否安装完整

---

**祝部署顺利！** 🎉
