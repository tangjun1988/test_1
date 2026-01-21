#!/bin/bash
# YOLO共享内存系统启动脚本
# 作用：一键启动数据源程序和推理程序

echo "=========================================="
echo "YOLO共享内存系统启动脚本"
echo "=========================================="

# 创建必要目录
echo "创建必要目录..."
mkdir -p logs
mkdir -p config

# 检查配置文件是否存在
if [ ! -f "config.yaml" ]; then
    echo "⚠️  警告：配置文件 config.yaml 不存在！"
    echo "将使用代码中的默认配置"
else
    echo "✅ 配置文件存在"
fi

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误：未找到 python3，请先安装Python"
    exit 1
fi

# 检查必要的Python模块
echo "检查Python依赖..."
python3 -c "import cv2, numpy, yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  警告：缺少必要的Python模块，请运行: pip install -r requirements.txt"
fi

# 启动数据源程序（后台运行）
echo ""
echo "启动数据源程序..."
python3 data_source_shm.py > logs/data_source.log 2>&1 &
DATA_SOURCE_PID=$!
echo "数据源程序PID: $DATA_SOURCE_PID"

# 等待2秒，确保共享内存已创建
echo "等待共享内存创建..."
sleep 2

# 启动推理程序
echo "启动推理程序..."
python3 inference_shm.py > logs/inference.log 2>&1 &
INFERENCE_PID=$!
echo "推理程序PID: $INFERENCE_PID"

# 保存PID到文件（用于停止脚本）
echo $DATA_SOURCE_PID > logs/data_source.pid
echo $INFERENCE_PID > logs/inference.pid

echo ""
echo "=========================================="
echo "✅ 系统已启动！"
echo "=========================================="
echo "数据源PID: $DATA_SOURCE_PID"
echo "推理PID: $INFERENCE_PID"
echo ""
echo "查看日志："
echo "  tail -f logs/data_source.log   # 数据源日志"
echo "  tail -f logs/inference.log     # 推理日志"
echo "  tail -f logs/app.log           # 应用日志（如果配置了）"
echo ""
echo "停止系统："
echo "  ./stop.sh"
echo "=========================================="
