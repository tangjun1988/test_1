"""
程序A：数据源（使用共享内存版本）

这个程序的作用：
1. 从视频文件/摄像头/图片文件夹读取图像帧
2. 将图像数据写入共享内存，供程序B（推理程序）读取
3. 以固定频率（10Hz）持续写入新的图像帧

共享内存是什么？
- 共享内存是一块可以被多个程序同时访问的内存区域
- 程序A写入数据，程序B读取数据，不需要通过网络传输
- 比Socket方式更快，因为不需要复制数据

工作流程：
视频/摄像头 → 读取帧 → 写入共享内存 → 程序B读取并推理
"""
# ========== 导入必要的库 ==========
import cv2          # OpenCV，用于读取视频/图片/摄像头
import time         # 时间库，用于控制帧率
import numpy as np  # NumPy，用于处理图像数组
from pathlib import Path  # 路径处理库，用于处理文件路径
from multiprocessing import shared_memory  # 共享内存库，用于进程间通信
import struct       # 结构体库，用于将整数转换为字节（二进制数据）

# ========== 导入配置和日志模块 ==========
try:
    from config_loader import Config
    from logger_setup import setup_logger
    USE_CONFIG = True
except ImportError:
    # 如果导入失败，使用默认配置（向后兼容）
    USE_CONFIG = False
    print("警告：无法导入配置模块，使用默认配置")

# ================== 配置区域 ==================
# 优先从配置文件读取，如果没有配置文件则使用默认值

if USE_CONFIG:
    try:
        config = Config("config.yaml")
        logger = setup_logger(
            name="data_source",
            log_level=config.logging.get('level', 'INFO'),
            log_file=config.logging.get('file'),
            console=config.logging.get('console', True)
        )
        logger.info("成功加载配置文件")
        
        # 从配置文件读取参数
        SOURCE_TYPE = config.data_source.get('type', 'video')
        VIDEO_PATH = config.data_source.get('video_path', '')
        CAMERA_INDEX = config.data_source.get('camera_index', 0)
        IMAGES_DIR = config.data_source.get('images_dir', '')
        FPS = config.data_source.get('fps', 24)
        
        SHM_NAME = config.shared_memory.get('name', 'yolo_image_shm')
        MAX_WIDTH = config.shared_memory.get('max_width', 1920)
        MAX_HEIGHT = config.shared_memory.get('max_height', 1080)
        MAX_CHANNELS = config.shared_memory.get('max_channels', 3)
    except Exception as e:
        print(f"警告：加载配置失败，使用默认配置: {e}")
        USE_CONFIG = False

# 如果没有配置文件或加载失败，使用默认配置
if not USE_CONFIG:
    # 创建简单的日志器（只输出到控制台）
    try:
        logger = setup_logger("data_source", console=True)
    except:
        # 如果日志模块也失败，使用print
        class SimpleLogger:
            def info(self, msg): print(f"[INFO] {msg}")
            def warning(self, msg): print(f"[WARNING] {msg}")
            def error(self, msg): print(f"[ERROR] {msg}")
        logger = SimpleLogger()
    
    # 默认配置（保持原有代码）
    SOURCE_TYPE = "video"
    VIDEO_PATH = "/mnt/f/Deeplearning/yolo_source8.3.163/ultralytics/datasets/make_dataset/videos/result.mp4"
    CAMERA_INDEX = 0
    IMAGES_DIR = "/mnt/f/Deeplearning/yolo_source8.3.163/ultralytics/datasets/make_dataset/images"
    FPS = 24
    SHM_NAME = "yolo_image_shm"
    MAX_WIDTH = 1920
    MAX_HEIGHT = 1080
    MAX_CHANNELS = 3

# 共享内存大小计算：
# 共享内存 = 元数据（图像尺寸信息） + 图像数据（像素值）
SHM_META_SIZE = 12  # 元数据大小：3个整数（width, height, channels），每个4字节，共12字节
# 图像数据大小 = 宽度 × 高度 × 通道数 × 每个像素的字节数（uint8是1字节）
SHM_DATA_SIZE = MAX_WIDTH * MAX_HEIGHT * MAX_CHANNELS * np.dtype(np.uint8).itemsize
# 总大小 = 元数据 + 图像数据
SHM_TOTAL_SIZE = SHM_META_SIZE + SHM_DATA_SIZE

# ==================================================


def iter_frames():
    """
    根据 SOURCE_TYPE 不断产生图像帧（numpy 数组）
    
    这是一个生成器函数（generator），使用 yield 关键字
    每次调用 next() 时，会返回下一帧图像
    
    返回：numpy 数组，形状为 (高度, 宽度, 通道数)，例如 (1080, 1920, 3)
    """
    if SOURCE_TYPE == "video":
        # ========== 从视频文件读取 ==========
        # 打开视频文件
        cap = cv2.VideoCapture(VIDEO_PATH)
        # 检查是否成功打开
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {VIDEO_PATH}")
        
        # 无限循环读取视频帧
        while True:
            ret, frame = cap.read()  # ret表示是否成功，frame是图像数据
            if not ret:
                # 如果读取失败（比如视频结束了），从头开始播放
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            yield frame  # 返回这一帧图像

    elif SOURCE_TYPE == "camera":
        # ========== 从摄像头读取 ==========
        # 打开摄像头
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头: {CAMERA_INDEX}")
        
        # 无限循环读取摄像头画面
        while True:
            ret, frame = cap.read()
            if not ret:
                # 如果读取失败，跳过这一帧，继续读取下一帧
                continue
            yield frame

    elif SOURCE_TYPE == "images":
        # ========== 从图片文件夹读取 ==========
        img_dir = Path(IMAGES_DIR)  # 将路径字符串转换为Path对象
        if not img_dir.is_dir():
            raise RuntimeError(f"图片文件夹不存在: {IMAGES_DIR}")
        
        # 找到文件夹中所有的图片文件（jpg, jpeg, png, bmp）
        # sorted() 按文件名排序，确保顺序一致
        image_paths = sorted(
            [p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        )
        if not image_paths:
            raise RuntimeError(f"图片文件夹中没有找到图片: {IMAGES_DIR}")
        
        # 循环播放图片
        idx = 0  # 当前图片的索引
        while True:
            img_path = image_paths[idx]  # 获取当前图片路径
            frame = cv2.imread(str(img_path))  # 读取图片
            if frame is not None:
                yield frame  # 返回图片数据
            # 移动到下一张图片，如果到了最后一张，回到第一张（循环播放）
            idx = (idx + 1) % len(image_paths)
    else:
        raise ValueError(f"未知的 SOURCE_TYPE: {SOURCE_TYPE}")


def write_frame_to_shm(shm, frame):
    """
    将图像帧写入共享内存
    
    参数：
        shm: 共享内存对象
        frame: numpy数组，形状为 (高度, 宽度, 通道数)
    
    返回：
        True: 写入成功
        False: 写入失败（图像尺寸超出限制）
    
    共享内存布局（内存中的排列方式）：
        [0-3字节]   宽度 (width)，4字节整数
        [4-7字节]   高度 (height)，4字节整数
        [8-11字节]  通道数 (channels)，4字节整数
        [12字节开始] 图像数据 (BGR格式，每个像素3个值：蓝、绿、红)
    
    为什么要这样布局？
    - 程序B需要知道图像的尺寸，才能正确读取数据
    - 先写尺寸信息（元数据），再写图像数据
    """
    # 获取图像的尺寸：h=高度, w=宽度, c=通道数
    h, w, c = frame.shape
    
    # 检查图像尺寸是否超出限制
    # 如果超出，无法写入（因为共享内存大小是固定的）
    if w > MAX_WIDTH or h > MAX_HEIGHT or c > MAX_CHANNELS:
        logger.warning(f"图像尺寸 {w}x{h}x{c} 超出限制 {MAX_WIDTH}x{MAX_HEIGHT}x{MAX_CHANNELS}，将跳过")
        return False
    
    # ========== 将共享内存转换为numpy数组 ==========
    # 这样可以直接用numpy的方式操作内存
    # buffer=shm.buf 表示使用共享内存的缓冲区
    shm_array = np.ndarray((SHM_TOTAL_SIZE,), dtype=np.uint8, buffer=shm.buf)
    
    # ========== 写入元数据（图像尺寸信息）==========
    # struct.pack("I", w) 将整数w转换为4字节的二进制数据
    # "I" 表示无符号整数（uint32），占4字节
    # np.frombuffer() 将二进制数据转换为numpy数组
    shm_array[0:4] = np.frombuffer(struct.pack("I", w), dtype=np.uint8)   # 写入宽度
    shm_array[4:8] = np.frombuffer(struct.pack("I", h), dtype=np.uint8)   # 写入高度
    shm_array[8:12] = np.frombuffer(struct.pack("I", c), dtype=np.uint8)  # 写入通道数
    
    # ========== 写入图像数据（从第12字节开始）==========
    data_start = SHM_META_SIZE  # 数据开始位置：12字节（元数据之后）
    data_end = data_start + w * h * c  # 数据结束位置：开始位置 + 图像大小
    # frame.flatten() 将二维/三维图像数组展平成一维数组
    # 例如：(1080, 1920, 3) -> (6220800,) 一维数组
    shm_array[data_start:data_end] = frame.flatten()
    
    return True


def main():
    """
    主函数：程序的入口点
    
    工作流程：
    1. 打印配置信息
    2. 创建共享内存
    3. 循环读取图像并写入共享内存
    4. 退出时清理资源
    """
    # ========== 打印程序信息 ==========
    logger.info("=" * 60)
    logger.info("程序 A：数据源进程（共享内存版本）")
    logger.info(f"数据源类型: {SOURCE_TYPE}")
    logger.info(f"发送频率: {FPS} Hz")  # Hz表示每秒多少次
    logger.info(f"共享内存名称: {SHM_NAME}")
    logger.info(f"共享内存大小: {SHM_TOTAL_SIZE / 1024 / 1024:.2f} MB")  # 转换为MB显示
    logger.info(f"最大图像尺寸: {MAX_WIDTH}x{MAX_HEIGHT}x{MAX_CHANNELS}")
    logger.info("=" * 60)

    # ========== 创建或获取共享内存 ==========
    try:
        # 尝试创建新的共享内存
        # create=True 表示如果不存在就创建
        # size 指定内存大小（字节）
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_TOTAL_SIZE)
        logger.info(f"✅ 创建共享内存成功: {SHM_NAME}")
    except FileExistsError:
        # 如果共享内存已存在（比如上次程序异常退出），先删除再创建
        logger.warning(f"共享内存已存在，正在清理...")
        try:
            # 连接到旧的共享内存
            old_shm = shared_memory.SharedMemory(name=SHM_NAME)
            old_shm.close()   # 关闭连接
            old_shm.unlink()  # 删除共享内存
        except:
            pass  # 如果删除失败，忽略错误
        # 重新创建
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_TOTAL_SIZE)
        logger.info(f"✅ 重新创建共享内存成功: {SHM_NAME}")

    # ========== 初始化变量 ==========
    interval = 1.0 / FPS  # 每次循环的间隔时间（秒）
    # 例如：FPS=10，则 interval=0.1秒，也就是每0.1秒发送一帧
    frame_gen = iter_frames()  # 创建图像生成器
    frame_count = 0  # 计数器：已写入的帧数

    logger.info("开始写入图像帧到共享内存（按 Ctrl+C 终止）...")
    try:
        # ========== 主循环：持续读取和写入图像 ==========
        while True:
            # 从生成器获取下一帧图像
            frame = next(frame_gen)
            
            # 检查是否读取成功
            if frame is None:
                logger.warning("读取到的帧为 None，跳过")
                time.sleep(interval)  # 等待一段时间后继续
                continue
            
            # 打印第一帧的信息（用于调试）
            if frame_count == 0:
                logger.info(f"✅ 第一帧图像尺寸: {frame.shape}")
            
            # 将图像写入共享内存
            if write_frame_to_shm(shm, frame):
                frame_count += 1  # 成功写入，计数器+1
                if frame_count == 1:
                    logger.info(f"✅ 成功写入第一帧到共享内存！")
                # 每3秒打印一次进度（FPS * 3 = 30帧）
                if frame_count % (FPS * 3) == 0:
                    logger.info(f"已写入帧数: {frame_count}, 当前图像尺寸: {frame.shape}")
            else:
                # 写入失败（通常是图像尺寸超出限制）
                logger.warning(f"写入帧失败，图像尺寸: {frame.shape}")
            
            # 等待一段时间，控制发送频率
            time.sleep(interval)
            
    except KeyboardInterrupt:
        # 用户按 Ctrl+C 中断程序
        logger.info("\n检测到 Ctrl+C，准备退出...")
    except Exception as e:
        # 发生其他错误
        logger.error(f"写入过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())  # 记录详细的错误信息
    finally:
        # ========== 清理资源 ==========
        # finally 块中的代码无论是否出错都会执行
        logger.info("\n正在清理资源...")
        
        try:
            shm.close()  # 关闭共享内存连接
            logger.info("✅ 共享内存连接已关闭")
        except Exception as e:
            logger.warning(f"关闭共享内存连接时出错: {e}")
        
        try:
            shm.unlink()  # 删除共享内存（释放系统资源）
            logger.info("✅ 共享内存已删除")
        except FileNotFoundError:
            # 共享内存已经被删除，忽略
            pass
        except Exception as e:
            # 如果删除失败（比如程序B还在使用），忽略错误
            logger.warning(f"删除共享内存时出错（可能程序B还在使用）: {e}")
        
        logger.info("程序 A 已退出。")


# ========== 程序入口 ==========
# 当直接运行这个文件时（而不是被其他文件导入），执行main()函数
if __name__ == "__main__":
    main()

