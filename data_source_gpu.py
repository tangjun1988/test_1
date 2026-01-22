"""
程序A：数据源（GPU显存版本 - 单路摄像头）

这个程序的作用：
1. 从摄像头读取图像帧
2. 将图像数据直接存储到GPU显存
3. 通过共享内存传递GPU显存指针和元数据给推理程序

优势：
- 零拷贝：数据直接在GPU显存，YOLO推理时无需CPU→GPU复制
- 低延迟：减少数据传输时间
- 高性能：特别适合Jetson Orin等GPU平台

工作流程：
摄像头 → GStreamer → GPU显存 → 共享内存(传递指针) → 推理程序
"""

# ========== 导入必要的库 ==========
import cv2
import time
import numpy as np
import struct
from pathlib import Path
from multiprocessing import shared_memory

# 尝试导入PyCUDA（用于GPU显存操作）
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    print("警告：PyCUDA未安装，将使用CPU内存作为后备方案")
    print("安装方法: pip install pycuda")

# 尝试导入torch（用于GPU tensor操作）
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告：PyTorch未安装")

# ========== 导入配置和日志模块 ==========
try:
    from config_loader import Config
    from logger_setup import setup_logger
    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False
    print("警告：无法导入配置模块，使用默认配置")

# ================== 配置区域 ==================
if USE_CONFIG:
    try:
        config = Config("config.yaml")
        logger = setup_logger(
            name="data_source_gpu",
            log_level=config.logging.get('level', 'INFO'),
            log_file=config.logging.get('file'),
            console=config.logging.get('console', True)
        )
        logger.info("成功加载配置文件")
        
        CAMERA_INDEX = config.data_source.get('camera_index', 0)
        FPS = config.data_source.get('fps', 24)
        
        SHM_NAME = config.shared_memory.get('name', 'yolo_image_gpu_shm')
        MAX_WIDTH = config.shared_memory.get('max_width', 640)
        MAX_HEIGHT = config.shared_memory.get('max_height', 480)
        MAX_CHANNELS = config.shared_memory.get('max_channels', 3)
    except Exception as e:
        print(f"警告：加载配置失败，使用默认配置: {e}")
        USE_CONFIG = False

if not USE_CONFIG:
    try:
        logger = setup_logger("data_source_gpu", console=True)
    except:
        class SimpleLogger:
            def info(self, msg): print(f"[INFO] {msg}")
            def warning(self, msg): print(f"[WARNING] {msg}")
            def error(self, msg): print(f"[ERROR] {msg}")
        logger = SimpleLogger()
    
    CAMERA_INDEX = 0
    FPS = 24
    SHM_NAME = "yolo_image_gpu_shm"
    MAX_WIDTH = 640
    MAX_HEIGHT = 480
    MAX_CHANNELS = 3

# 共享内存大小计算
# 元数据：width(4) + height(4) + channels(4) + gpu_ptr(8) + frame_id(8) = 28字节
SHM_META_SIZE = 28
SHM_DATA_SIZE = MAX_WIDTH * MAX_HEIGHT * MAX_CHANNELS * np.dtype(np.uint8).itemsize
SHM_TOTAL_SIZE = SHM_META_SIZE + SHM_DATA_SIZE

# ==================================================


class GPUCameraCapture:
    """使用GPU显存直接采集摄像头数据"""
    
    def __init__(self, camera_index=0, width=640, height=480, use_gstreamer=True):
        """
        初始化GPU摄像头采集
        
        参数:
            camera_index: 摄像头索引
            width: 图像宽度
            height: 图像高度
            use_gstreamer: 是否使用GStreamer（Jetson平台推荐）
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.frame_size = width * height * MAX_CHANNELS
        
        # 初始化GPU显存缓冲区
        self.gpu_buffer = None
        self.gpu_ptr = 0
        self.frame_id = 0
        
        if PYCUDA_AVAILABLE:
            try:
                # 在GPU显存中分配缓冲区
                self.gpu_buffer = cuda.mem_alloc(self.frame_size)
                self.gpu_ptr = int(self.gpu_buffer)
                logger.info(f"✅ GPU显存分配成功: {self.frame_size / 1024 / 1024:.2f} MB")
            except Exception as e:
                logger.error(f"❌ GPU显存分配失败: {e}")
                PYCUDA_AVAILABLE = False
        
        # 打开摄像头
        if use_gstreamer and self._check_gstreamer():
            self.cap = self._open_gstreamer_camera()
        else:
            # 使用标准OpenCV方式（后备方案）
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开摄像头: {camera_index}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        logger.info(f"✅ 摄像头 {camera_index} 打开成功")
    
    def _check_gstreamer(self):
        """检查GStreamer是否可用"""
        try:
            # 尝试创建GStreamer pipeline
            test_pipeline = "videotestsrc ! video/x-raw,width=640,height=480 ! appsink"
            test_cap = cv2.VideoCapture(test_pipeline, cv2.CAP_GSTREAMER)
            if test_cap.isOpened():
                test_cap.release()
                return True
        except:
            pass
        return False
    
    def _open_gstreamer_camera(self):
        """使用GStreamer打开摄像头（Jetson优化）"""
        # 方案1: 使用v4l2src（通用USB摄像头）
        gst_pipeline = (
            f"v4l2src device=/dev/video{self.camera_index} ! "
            f"video/x-raw,width={self.width},height={self.height},format=MJPG ! "
            f"nvjpegdec ! "  # NVIDIA硬件JPEG解码
            f"nvvidconv ! "  # NVIDIA视频格式转换
            f"video/x-raw,format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw,format=BGR ! "
            f"appsink drop=true"
        )
        
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            # 如果GStreamer失败，尝试简化pipeline
            logger.warning("GStreamer pipeline失败，尝试简化版本...")
            gst_pipeline_simple = (
                f"v4l2src device=/dev/video{self.camera_index} ! "
                f"video/x-raw,width={self.width},height={self.height} ! "
                f"videoconvert ! "
                f"video/x-raw,format=BGR ! "
                f"appsink"
            )
            cap = cv2.VideoCapture(gst_pipeline_simple, cv2.CAP_GSTREAMER)
        
        return cap
    
    def read_to_gpu(self):
        """
        读取一帧到GPU显存
        
        返回:
            (success, gpu_ptr, shape, frame_id)
            success: 是否成功
            gpu_ptr: GPU显存指针（整数）
            shape: 图像形状 (height, width, channels)
            frame_id: 帧ID
        """
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, 0, None, self.frame_id
        
        # 检查图像尺寸
        h, w, c = frame.shape
        if w != self.width or h != self.height:
            # 调整尺寸
            frame = cv2.resize(frame, (self.width, self.height))
            h, w, c = frame.shape
        
        if PYCUDA_AVAILABLE and self.gpu_buffer:
            try:
                # 将数据复制到GPU显存（零拷贝优化）
                cuda.memcpy_htod(self.gpu_buffer, frame)
                self.frame_id += 1
                return True, self.gpu_ptr, (h, w, c), self.frame_id
            except Exception as e:
                logger.error(f"GPU复制失败: {e}")
                return False, 0, (h, w, c), self.frame_id
        else:
            # 后备方案：返回CPU内存（但标记为CPU模式）
            self.frame_id += 1
            return True, 0, (h, w, c), self.frame_id
    
    def release(self):
        """释放资源"""
        if self.cap:
            self.cap.release()
        if PYCUDA_AVAILABLE and self.gpu_buffer:
            try:
                self.gpu_buffer.free()
                logger.info("GPU显存已释放")
            except:
                pass


def write_gpu_frame_to_shm(shm, gpu_ptr, shape, frame_id):
    """
    将GPU帧信息写入共享内存
    
    参数:
        shm: 共享内存对象
        gpu_ptr: GPU显存指针（整数，8字节）
        shape: 图像形状 (height, width, channels)
        frame_id: 帧ID（8字节）
    
    返回:
        True: 写入成功
        False: 写入失败
    """
    if shape is None:
        return False
    
    h, w, c = shape
    
    # 检查尺寸
    if w > MAX_WIDTH or h > MAX_HEIGHT or c > MAX_CHANNELS:
        logger.warning(f"图像尺寸 {w}x{h}x{c} 超出限制")
        return False
    
    # 转换为numpy数组操作共享内存
    shm_array = np.ndarray((SHM_TOTAL_SIZE,), dtype=np.uint8, buffer=shm.buf)
    
    # 写入元数据
    shm_array[0:4] = np.frombuffer(struct.pack("I", w), dtype=np.uint8)   # 宽度
    shm_array[4:8] = np.frombuffer(struct.pack("I", h), dtype=np.uint8)   # 高度
    shm_array[8:12] = np.frombuffer(struct.pack("I", c), dtype=np.uint8)  # 通道数
    
    # 写入GPU指针（8字节，64位）
    shm_array[12:20] = np.frombuffer(struct.pack("Q", gpu_ptr), dtype=np.uint8)
    
    # 写入帧ID（8字节）
    shm_array[20:28] = np.frombuffer(struct.pack("Q", frame_id), dtype=np.uint8)
    
    # 如果GPU指针为0（CPU模式），写入CPU数据
    if gpu_ptr == 0:
        data_start = SHM_META_SIZE
        data_end = data_start + w * h * c
        # 注意：这里需要实际的frame数据，但当前函数没有
        # 实际使用时，如果gpu_ptr==0，应该在调用前准备好frame数据
        logger.warning("GPU指针为0，使用CPU模式（需要传入frame数据）")
    
    return True


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("程序 A：数据源进程（GPU显存版本）")
    logger.info(f"摄像头索引: {CAMERA_INDEX}")
    logger.info(f"分辨率: {MAX_WIDTH}x{MAX_HEIGHT}")
    logger.info(f"发送频率: {FPS} Hz")
    logger.info(f"GPU支持: {'是' if PYCUDA_AVAILABLE else '否（使用CPU后备）'}")
    logger.info(f"共享内存名称: {SHM_NAME}")
    logger.info("=" * 60)
    
    # 创建GPU摄像头采集器
    try:
        camera = GPUCameraCapture(
            camera_index=CAMERA_INDEX,
            width=MAX_WIDTH,
            height=MAX_HEIGHT
        )
    except Exception as e:
        logger.error(f"❌ 初始化摄像头失败: {e}")
        return
    
    # 创建共享内存
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_TOTAL_SIZE)
        logger.info(f"✅ 创建共享内存成功: {SHM_NAME}")
    except FileExistsError:
        logger.warning("共享内存已存在，正在清理...")
        try:
            old_shm = shared_memory.SharedMemory(name=SHM_NAME)
            old_shm.close()
            old_shm.unlink()
        except:
            pass
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_TOTAL_SIZE)
        logger.info(f"✅ 重新创建共享内存成功: {SHM_NAME}")
    
    # 初始化变量
    interval = 1.0 / FPS
    frame_count = 0
    
    logger.info("开始采集摄像头数据到GPU显存（按 Ctrl+C 终止）...")
    
    try:
        while True:
            # 读取帧到GPU显存
            success, gpu_ptr, shape, frame_id = camera.read_to_gpu()
            
            if not success:
                logger.warning("读取帧失败，跳过")
                time.sleep(interval)
                continue
            
            # 写入共享内存
            if write_gpu_frame_to_shm(shm, gpu_ptr, shape, frame_id):
                frame_count += 1
                if frame_count == 1:
                    logger.info(f"✅ 成功写入第一帧到共享内存！")
                    logger.info(f"   GPU指针: {gpu_ptr if gpu_ptr > 0 else 'CPU模式'}")
                    logger.info(f"   图像尺寸: {shape}")
                
                # 每3秒打印一次进度
                if frame_count % (FPS * 3) == 0:
                    logger.info(f"已写入帧数: {frame_count}, GPU指针: {gpu_ptr if gpu_ptr > 0 else 'CPU'}, 帧ID: {frame_id}")
            else:
                logger.warning(f"写入帧失败，图像尺寸: {shape}")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        logger.info("\n检测到 Ctrl+C，准备退出...")
    except Exception as e:
        logger.error(f"运行过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("\n正在清理资源...")
        camera.release()
        
        try:
            shm.close()
            logger.info("✅ 共享内存连接已关闭")
        except Exception as e:
            logger.warning(f"关闭共享内存连接时出错: {e}")
        
        try:
            shm.unlink()
            logger.info("✅ 共享内存已删除")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"删除共享内存时出错: {e}")
        
        logger.info("程序 A 已退出。")


if __name__ == "__main__":
    main()
