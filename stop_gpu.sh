#!/bin/bash
# YOLO GPU显存版本停止脚本
# 作用：停止GPU版本的数据源程序和推理程序

echo "=========================================="
echo "YOLO GPU显存版本停止脚本"
echo "=========================================="

# 停止数据源程序
if [ -f "logs/data_source_gpu.pid" ]; then
    PID=$(cat logs/data_source_gpu.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "停止GPU数据源程序 (PID: $PID)..."
        kill $PID 2>/dev/null
        sleep 1
        # 如果还在运行，强制杀死
        if ps -p $PID > /dev/null 2>&1; then
            kill -9 $PID 2>/dev/null
        fi
        echo "✅ GPU数据源程序已停止"
    else
        echo "⚠️  GPU数据源程序未运行"
    fi
    rm -f logs/data_source_gpu.pid
else
    echo "⚠️  未找到GPU数据源程序PID文件"
fi

# 停止推理程序
if [ -f "logs/inference_gpu.pid" ]; then
    PID=$(cat logs/inference_gpu.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "停止GPU推理程序 (PID: $PID)..."
        kill $PID 2>/dev/null
        sleep 1
        # 如果还在运行，强制杀死
        if ps -p $PID > /dev/null 2>&1; then
            kill -9 $PID 2>/dev/null
        fi
        echo "✅ GPU推理程序已停止"
    else
        echo "⚠️  GPU推理程序未运行"
    fi
    rm -f logs/inference_gpu.pid
else
    echo "⚠️  未找到GPU推理程序PID文件"
fi

echo ""
echo "=========================================="
echo "✅ GPU版本系统已停止"
echo "=========================================="
