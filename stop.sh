#!/bin/bash
# YOLO共享内存系统停止脚本
# 作用：停止数据源程序和推理程序

echo "=========================================="
echo "YOLO共享内存系统停止脚本"
echo "=========================================="

# 停止数据源程序
if [ -f "logs/data_source.pid" ]; then
  PID=$(cat logs/data_source.pid)
  if ps -p $PID > /dev/null 2>&1; then
    echo "停止数据源程序 (PID: $PID)..."
    kill $PID 2> /dev/null
    sleep 1
    # 如果还在运行，强制杀死
    if ps -p $PID > /dev/null 2>&1; then
      kill -9 $PID 2> /dev/null
    fi
    echo "✅ 数据源程序已停止"
  else
    echo "⚠️  数据源程序未运行"
  fi
  rm -f logs/data_source.pid
else
  echo "⚠️  未找到数据源程序PID文件"
fi

# 停止推理程序
if [ -f "logs/inference.pid" ]; then
  PID=$(cat logs/inference.pid)
  if ps -p $PID > /dev/null 2>&1; then
    echo "停止推理程序 (PID: $PID)..."
    kill $PID 2> /dev/null
    sleep 1
    # 如果还在运行，强制杀死
    if ps -p $PID > /dev/null 2>&1; then
      kill -9 $PID 2> /dev/null
    fi
    echo "✅ 推理程序已停止"
  else
    echo "⚠️  推理程序未运行"
  fi
  rm -f logs/inference.pid
else
  echo "⚠️  未找到推理程序PID文件"
fi

echo ""
echo "=========================================="
echo "✅ 系统已停止"
echo "=========================================="
