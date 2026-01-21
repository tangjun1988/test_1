import os
import time
import numpy as np
import struct
import socket
import cv2
from ultralytics import YOLO

# ================== 配置区域 ==================

# ========== Socket 配置 ==========
# 这些参数必须和程序A（data_source.py）中的参数完全一致
HOST = "127.0.0.1"  # 服务器地址（必须和程序A一致）
PORT = 8888          # 服务器端口（必须和程序A一致）

# ========== YOLO模型配置 ==========
MODEL_PATH = "yolo12n.pt"  # 模型文件路径（YOLO会自动下载如果不存在）

# ========== 推理配置 ==========
FPS = 10  # 推理频率：每秒处理10帧（必须和程序A的发送频率一致）

# ========== 检测类别配置 ==========
PERSON_CLASS_ID = 0  # 只检测"人"这个类别（COCO数据集中的class 0）

# ========== 保存配置 ==========
OUTPUT_DIR = "result"  # 结果保存目录
SAVE_EVERY_N_FRAMES = 10  # 每处理多少帧保存一次（例如：10表示每10帧保存1帧）

# ==================================================


def receive_frame(sock, debug=False):
    """
    从socket接收一帧图像
    
    参数：
        sock: socket对象
        debug: 是否打印调试信息
    
    返回：
        numpy数组：图像数据，形状为 (高度, 宽度, 通道数)
        None：如果接收失败或数据无效
    
    接收格式（和程序A发送的格式一致）：
        先接收12字节元数据：宽度(4字节) + 高度(4字节) + 通道数(4字节)
        然后接收图像数据（BGR格式）
    """
    try:
        # 接收元数据（图像尺寸信息）
        meta_data = b""
        while len(meta_data) < 12:
            chunk = sock.recv(12 - len(meta_data))
            if not chunk:
                if debug:
                    print("调试：连接已关闭")
                return None
            meta_data += chunk
        
        # 解析元数据
        w, h, c = struct.unpack("III", meta_data)
        
        if debug:
            print(f"调试：接收到的尺寸 w={w}, h={h}, c={c}")
        
        # 检查尺寸是否有效
        if w == 0 or h == 0 or c == 0 or w > 10000 or h > 10000:
            if debug:
                print(f"调试：尺寸无效，返回 None")
            return None
        
        # 接收图像数据
        data_size = w * h * c
        frame_data = b""
        while len(frame_data) < data_size:
            chunk = sock.recv(min(4096, data_size - len(frame_data)))
            if not chunk:
                if debug:
                    print("调试：接收图像数据时连接关闭")
                return None
            frame_data += chunk
        
        # 转换为numpy数组
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((h, w, c))
        return frame
        
    except socket.timeout:
        if debug:
            print("调试：接收超时")
        return None
    except Exception as e:
        if debug:
            print(f"调试：接收帧时出错: {e}")
        return None


def main():
    sock = None
    
    try:
        print("=" * 60)
        print("程序 B：推理进程（Socket版本）")
        print(f"模型: {MODEL_PATH}")
        print(f"Socket地址: {HOST}:{PORT}")
        print(f"结果保存目录: {OUTPUT_DIR}")
        print(f"保存频率: 每 {SAVE_EVERY_N_FRAMES} 帧保存一次")
        print("=" * 60)

        # 加载YOLO模型
        print("正在加载模型，请稍候...")
        model = YOLO(MODEL_PATH)
        print("✅ 模型加载完成！")

        # 连接到Socket服务器
        print(f"正在连接到服务器 {HOST}:{PORT}...")
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)  # 设置超时时间
                sock.connect((HOST, PORT))
                print("✅ 已连接到服务器！")
                break
            except (ConnectionRefusedError, socket.timeout, OSError) as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"等待服务器启动... ({retry_count}/{max_retries})")
                    time.sleep(1)
                else:
                    print(f"❌ 无法连接到服务器 {HOST}:{PORT}")
                    print("请确保程序A（data_source.py）已启动")
                    return
            except Exception as e:
                print(f"❌ 连接服务器时出错: {e}")
                return
        
        # 设置接收超时
        sock.settimeout(10)

        # 创建输出目录
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"结果将保存到: {OUTPUT_DIR}/")

        # 初始化变量
        last_frame = None
        frame_count = 0
        saved_count = 0
        interval = 1.0 / FPS
        no_frame_count = 0

        print("开始推理并保存结果（按 Ctrl+C 终止）...")

        while True:
            # 从socket接收图像
            frame = receive_frame(sock, debug=(no_frame_count < 3))
            
            if frame is None:
                no_frame_count += 1
                if no_frame_count == 1:
                    print("⚠️  警告：未接收到有效图像数据，请检查：")
                    print("   1. 程序A（data_source.py）是否正在运行？")
                    print("   2. 摄像头是否正常工作？")
                elif no_frame_count % 30 == 0:
                    print(f"⚠️  仍在等待图像数据... (已等待 {no_frame_count} 次)")
                
                if last_frame is None:
                    time.sleep(interval)
                    continue
                frame = last_frame
            else:
                if no_frame_count > 0:
                    print(f"✅ 已接收到图像数据！图像尺寸: {frame.shape}")
                    no_frame_count = 0
                last_frame = frame

            # YOLO推理
            results = model.predict(
                frame,
                conf=0.3,
                iou=0.5,
                classes=[PERSON_CLASS_ID],
                verbose=False,
            )

            annotated = results[0].plot()
            frame_count += 1

            # 按配置的频率保存结果
            if frame_count % SAVE_EVERY_N_FRAMES == 0:
                output_path = os.path.join(OUTPUT_DIR, f"frame_{saved_count:06d}.jpg")
                cv2.imwrite(output_path, annotated)
                saved_count += 1
                print(f"已保存第 {saved_count} 张图片: {output_path}")

            # 每3秒打印一次进度
            if frame_count % (FPS * 3) == 0:
                print(f"已推理帧数: {frame_count}, 已保存图片数: {saved_count}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，准备退出...")
    except Exception as e:
        print(f"❌ 推理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if sock is not None:
            sock.close()
            print("✅ Socket连接已关闭")
        print(f"程序 B 已退出。共处理 {frame_count} 帧，保存了 {saved_count} 张图片。")


if __name__ == "__main__":
    try:
        print("程序开始运行...")
        main()
    except Exception as e:
        print(f"程序启动时发生错误: {e}")
        import traceback
        traceback.print_exc()
        input("按回车键退出...")  # 防止窗口立即关闭
