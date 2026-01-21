import cv2
import time
import numpy as np
import struct
import socket

# ================== 配置区域 ==================

# 视频文件路径
VIDEO_PATH = r"F:\Deeplearning\yolo_source8.3.163\ultralytics\datasets\make_dataset\videos\result.mp4"

# 向程序 B 发送图像的频率（Hz），10 表示每秒 10 帧
FPS = 10

# 是否循环播放视频（播放到结尾后重新开始）
LOOP_VIDEO = True

# ========== Socket 配置 ==========
HOST = "127.0.0.1"  # 服务器地址
PORT = 8888          # 服务器端口

# ==================================================


def iter_frames():
    """
    从视频文件不断产生图像帧（numpy 数组）。
    这是一个生成器函数，程序 B 会持续消费这些帧。
    """
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {VIDEO_PATH}")

    while True:
        ret, frame = cap.read()
        if not ret:
            # 播放到结尾后，从头重新开始
            if LOOP_VIDEO:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        yield frame


def send_frame(sock, frame):
    """
    通过socket发送图像帧
    
    参数：
        sock: socket对象
        frame: numpy数组，形状为 (高度, 宽度, 通道数)
    
    返回：
        True: 发送成功
        False: 发送失败
    
    发送格式：
        先发送12字节元数据：宽度(4字节) + 高度(4字节) + 通道数(4字节)
        然后发送图像数据（BGR格式）
    """
    try:
        h, w, c = frame.shape
        
        # 发送元数据（图像尺寸信息）
        meta_data = struct.pack("III", w, h, c)
        sock.sendall(meta_data)
        
        # 发送图像数据
        frame_bytes = frame.tobytes()
        sock.sendall(frame_bytes)
        
        return True
    except Exception as e:
        print(f"⚠️  发送帧时出错: {e}")
        return False


def main():
    server_socket = None
    client_socket = None
    cap = None
    
    try:
        print("=" * 60)
        print("程序 A：数据源进程（Socket版本）")
        print(f"视频文件: {VIDEO_PATH}")
        print(f"发送频率: {FPS} Hz")
        print(f"循环播放: {LOOP_VIDEO}")
        print(f"Socket地址: {HOST}:{PORT}")
        print("=" * 60)

        # 测试视频文件是否能打开
        print(f"正在测试视频文件 {VIDEO_PATH}...")
        test_cap = cv2.VideoCapture(VIDEO_PATH)
        if not test_cap.isOpened():
            raise RuntimeError(f"无法打开视频文件 {VIDEO_PATH}，请检查：\n"
                             f"1. 视频文件路径是否正确\n"
                             f"2. 视频文件是否存在\n"
                             f"3. 视频格式是否支持")
        # 尝试读取一帧来确认视频真的可用
        ret, test_frame = test_cap.read()
        total_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = test_cap.get(cv2.CAP_PROP_FPS)
        test_cap.release()
        if not ret or test_frame is None:
            raise RuntimeError(f"视频文件 {VIDEO_PATH} 无法读取图像数据")
        print(f"✅ 视频文件测试成功！")
        print(f"   总帧数: {total_frames}")
        print(f"   视频FPS: {fps_video:.2f}")
        print(f"   视频尺寸: {test_frame.shape[1]}x{test_frame.shape[0]}")
        
        # 创建socket服务器
        print(f"正在创建Socket服务器 {HOST}:{PORT}...")
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"✅ Socket服务器已启动，等待客户端连接...")
        
        # 等待客户端连接
        client_socket, client_address = server_socket.accept()
        print(f"✅ 客户端已连接: {client_address}")
        
        # 初始化视频文件
        print("正在打开视频文件...")
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {VIDEO_PATH}")
        
        interval = 1.0 / FPS
        frame_count = 0

        print("开始发送图像帧（按 Ctrl+C 终止）...")
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️  警告：读取到的帧为 None，跳过")
                time.sleep(interval)
                continue
            
            if frame_count == 0:
                print(f"✅ 第一帧图像尺寸: {frame.shape}")
            
            if send_frame(client_socket, frame):
                frame_count += 1
                if frame_count == 1:
                    print(f"✅ 成功发送第一帧！")
                if frame_count % (FPS * 3) == 0:  # 每 3 秒提示一次
                    print(f"已发送帧数: {frame_count}, 当前图像尺寸: {frame.shape}")
            else:
                print(f"⚠️  警告：发送帧失败，图像尺寸: {frame.shape}")
                break  # 发送失败，退出循环
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，准备退出...")
    except Exception as e:
        print(f"\n❌ 程序发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        try:
            if cap is not None:
                cap.release()
                print("✅ 视频文件已释放")
        except Exception as e:
            print(f"⚠️  释放视频文件时出错: {e}")
        
        try:
            if client_socket is not None:
                client_socket.close()
                print("✅ 客户端连接已关闭")
        except Exception as e:
            print(f"⚠️  关闭客户端连接时出错: {e}")
        
        try:
            if server_socket is not None:
                server_socket.close()
                print("✅ Socket服务器已关闭")
        except Exception as e:
            print(f"⚠️  关闭服务器时出错: {e}")
        
        print("程序 A 已退出。")


if __name__ == "__main__":
    try:
        print("程序开始运行...")
        main()
    except Exception as e:
        print(f"程序启动时发生错误: {e}")
        import traceback
        traceback.print_exc()
        input("按回车键退出...")  # 防止窗口立即关闭
