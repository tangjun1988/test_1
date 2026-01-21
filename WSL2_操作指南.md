# WSL2 + Unix Domain Socket 操作指南

## 📋 准备工作

### 1. 确认 WSL2 和 Ubuntu 已安装

在 Windows PowerShell 中运行：
```powershell
wsl --list --verbose
```

应该能看到 Ubuntu，并且 VERSION 是 2。

如果没安装，运行：
```powershell
wsl --install -d Ubuntu
```

---

## 🔧 第一步：在 WSL2 中安装 Python 和依赖

### 1.1 打开 WSL2（Ubuntu）

在 Windows 开始菜单搜索 "Ubuntu" 或 "WSL"，打开 Ubuntu 终端。

### 1.2 更新系统（可选但推荐）

```bash
sudo apt update
sudo apt upgrade -y
```

### 1.3 安装 Python 和 pip

```bash
sudo apt install python3 python3-pip -y
```

### 1.4 安装必要的库

```bash
pip3 install ultralytics opencv-python
```

**注意**：如果提示权限问题，可以加 `--user`：
```bash
pip3 install --user ultralytics opencv-python
```

---

## 📁 第二步：准备代码文件

### 2.1 在 WSL2 中访问 Windows 文件

你的 Windows 文件在 WSL2 中的路径是：
- `F:\` 盘 → `/mnt/f/`
- 你的项目路径：
  ```
  /mnt/f/Deeplearning/yolo_source8.3.163/ultralytics/
  ```

### 2.2 进入项目目录

```bash
cd /mnt/f/Deeplearning/yolo_source8.3.163/ultralytics
```

### 2.3 确认文件存在

```bash
ls -la
```

应该能看到：
- `data_source.py`
- `inference.py`

### 2.4 确认视频文件路径

检查视频文件是否存在：
```bash
ls -la /mnt/f/Deeplearning/yolo_source8.3.163/ultralytics/datasets/make_dataset/videos/
```

如果视频文件名不是 `result.mp4`，需要修改 `data_source.py` 中的 `VIDEO_PATH`。

---

## 🚀 第三步：运行程序

### 3.1 打开两个 WSL2 终端窗口

需要两个终端窗口：
1. **终端1**：运行 `inference.py`（程序B - 推理）
2. **终端2**：运行 `data_source.py`（程序A - 数据源）

### 3.2 在终端1中运行程序B（推理程序）

```bash
# 进入项目目录
cd /mnt/f/Deeplearning/yolo_source8.3.163/ultralytics

# 运行推理程序
python3 inference.py
```

**预期输出**：
```
============================================================
程序 B：推理进程（负责从 Socket 读取图像并做实时检测）
模型: yolo12n.pt  (预训练，识别 person 等 COCO 类别)
Unix Domain Socket 路径: /tmp/yolo_image_socket
============================================================
正在加载模型，请稍候...
模型加载完成！
正在连接程序 A ...
```

程序会卡在"正在连接程序 A ..."，这是正常的，等待程序A连接。

**⚠️ 重要**：不要关闭这个终端，让它一直运行！

---

### 3.3 在终端2中运行程序A（数据源程序）

打开**新的** WSL2 终端窗口（或新标签页），运行：

```bash
# 进入项目目录
cd /mnt/f/Deeplearning/yolo_source8.3.163/ultralytics

# 运行数据源程序
python3 data_source.py
```

**预期输出**：
```
============================================================
程序 A：数据源进程（负责读取视频/图片/摄像头，并发送给程序 B）
数据源类型: video
发送频率: 10 Hz
Unix Domain Socket 路径: /tmp/yolo_image_socket
============================================================
等待程序 B 连接...
程序 B 已连接！
开始发送图像帧（按 Ctrl+C 终止）...
已发送帧数: 30
已发送帧数: 60
...
```

---

### 3.4 查看结果

当程序A连接成功后，程序B的窗口会：
1. 显示"已连接程序 A，开始接收图像并推理。"
2. 弹出一个 OpenCV 窗口，显示检测结果（画框标注人体）
3. 在终端中显示"已推理帧数: X"

---

## 🛑 停止程序

### 方法1：关闭显示窗口
在 OpenCV 显示窗口中按键盘 `q` 键，程序B会退出。

### 方法2：使用 Ctrl+C
在任意一个终端中按 `Ctrl + C`，对应的程序会停止。

**建议**：先停止程序A（终端2），再停止程序B（终端1）。

---

## 🔍 常见问题排查

### 问题1：找不到模块

**错误**：`ModuleNotFoundError: No module named 'ultralytics'`

**解决**：
```bash
pip3 install --user ultralytics opencv-python
```

如果还不行，检查 Python 版本：
```bash
python3 --version
# 应该是 Python 3.8 或更高版本
```

---

### 问题2：找不到视频文件

**错误**：`无法打开视频文件: /mnt/f/...`

**解决**：
1. 检查视频文件是否存在：
   ```bash
   ls -la /mnt/f/Deeplearning/yolo_source8.3.163/ultralytics/datasets/make_dataset/videos/
   ```

2. 如果文件名不对，修改 `data_source.py` 中的 `VIDEO_PATH`

3. 确保路径使用 Linux 格式（用 `/` 而不是 `\`）

---

### 问题3：Socket 连接失败

**错误**：`Connection refused` 或 `No such file or directory`

**解决**：
1. 确保先运行程序B（inference.py），再运行程序A（data_source.py）
2. 检查 socket 文件：
   ```bash
   ls -la /tmp/yolo_image_socket
   ```
3. 如果 socket 文件存在但连接失败，删除它：
   ```bash
   rm /tmp/yolo_image_socket
   ```
   然后重新运行程序

---

### 问题4：无法显示窗口（X11 问题）

**错误**：`cannot connect to X server` 或 `qt.qpa.xcb: could not connect to display`

**解决**：WSL2 默认不支持图形界面，需要安装 X11 转发：

**方法1：使用 VcXsrv（推荐，适用于 Windows 10/11）**

#### 步骤1：安装并启动 VcXsrv

1. **下载安装 VcXsrv**：
   - 下载地址：https://sourceforge.net/projects/vcxsrv/
   - 安装后启动 **XLaunch**

2. **配置 XLaunch**（重要：必须按以下步骤配置）：
   - **Display settings**: 选择 "Multiple windows" → Next
   - **Client startup**: 选择 "Start no client" → Next
   - **Extra settings**: **必须勾选 "Disable access control"** → Finish
   - 启动后，Windows 系统托盘会出现 VcXsrv 图标

#### 步骤2：在 WSL2 中配置 DISPLAY

```bash
# 设置 DISPLAY 环境变量
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0

# 验证设置
echo $DISPLAY
# 应该显示类似：172.x.x.x:0.0

# 安装必要的 X11 库（如果还没有）
sudo apt update
sudo apt install -y libxcb-xinerama0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 libxcb-xkb1 libxkbcommon-x11-0
```

#### 步骤3：永久配置（可选）

将 DISPLAY 设置添加到 `~/.bashrc`，这样每次打开终端都会自动设置：

```bash
# 编辑 ~/.bashrc
nano ~/.bashrc

# 在文件末尾添加：
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0

# 保存后重新加载
source ~/.bashrc
```

#### 步骤4：测试 X11 连接

```bash
# 测试 X11 是否可用（如果安装了 x11-apps）
sudo apt install -y x11-apps
xeyes  # 应该会弹出眼睛窗口

# 或者测试 OpenCV
python3 -c "import cv2; cv2.namedWindow('test'); cv2.destroyAllWindows(); print('X11 可用')"
```

#### 步骤5：配置 Windows 防火墙（如果连接失败）

如果仍然无法连接，需要在 Windows 防火墙中允许 VcXsrv：

在 **Windows PowerShell（管理员）** 中运行：
```powershell
New-NetFirewallRule -DisplayName "VcXsrv X11 Server" -Direction Inbound -Program "C:\Program Files\VcXsrv\vcxsrv.exe" -Action Allow
```

**方法2：使用 WSLg（仅 Windows 11）**

如果你用的是 Windows 11，WSLg 应该已经自动支持图形界面：

```bash
# 检查 WSLg 是否可用
echo $DISPLAY
# 如果显示 :0 或类似值，说明 WSLg 已启用

# 如果 DISPLAY 未设置，手动设置
export DISPLAY=:0

# 安装必要的库


# 运行程序
python3 inference.py
```

**方法3：使用自动模式（推荐）**

代码已支持自动检测 X11 可用性。只需设置：

```python
DISPLAY_MODE = 'auto'  # 在 inference.py 中
```

程序会自动检测：
- 如果 X11 可用 → 显示窗口
- 如果 X11 不可用 → 自动保存到文件

这样即使 X11 未配置，程序也能正常运行。

---

### 问题5：模型文件不存在

**错误**：`FileNotFoundError: yolo12n.pt`

**解决**：
1. 检查模型文件是否存在：
   ```bash
   ls -la /mnt/f/Deeplearning/yolo_source8.3.163/ultralytics/yolo12n.pt
   ```

2. 如果不存在，YOLO 会自动下载，但需要网络连接

3. 或者修改 `inference.py` 中的 `MODEL_PATH` 为存在的模型文件

---

## 📝 修改配置

### 更改数据源类型

编辑 `data_source.py`，修改：
```python
SOURCE_TYPE = "video"  # 改为 "camera" 或 "images"
```

### 更改发送/推理频率

在两个文件中修改：
```python
FPS = 10  # 改为其他数字，比如 5（5Hz）或 20（20Hz）
```

### 更改视频路径

编辑 `data_source.py`，修改：
```python
VIDEO_PATH = "/mnt/f/你的/视频/路径.mp4"
```

---

## ✅ 验证系统是否正常工作

### 快速测试步骤：

1. **终端1**：运行 `python3 inference.py`，等待连接
2. **终端2**：运行 `python3 data_source.py`，开始发送
3. 应该看到：
   - 程序A显示"已发送帧数: X"
   - 程序B显示"已推理帧数: X"
   - 弹出窗口显示检测结果

如果这些都正常，说明系统运行成功！🎉

---

## 🔄 下一步

现在你的系统已经可以：
- ✅ 使用 Unix Domain Socket 进行进程间通信
- ✅ 程序A持续发送图像数据
- ✅ 程序B持续接收并推理
- ✅ 实时显示检测结果

接下来可以：
1. 尝试使用摄像头（修改 `SOURCE_TYPE = "camera"`）
2. 尝试使用图片文件夹（修改 `SOURCE_TYPE = "images"`）
3. 接入真实相机（需要配置相机驱动和路径）

