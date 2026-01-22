"""
程序B：推理引擎（GPU显存版本）

这个程序的作用：
1. 从共享内存读取GPU显存指针和元数据
2. 直接使用GPU显存中的数据进行YOLO推理（零拷贝）
3. 显示检测结果

优势：
- 零拷贝：数据已在GPU显存，无需CPU→GPU传输
- 低延迟：减少数据传输时间
- 高性能：充分利用GPU性能

工作流程：
共享内存(读取GPU指针) → 直接使用GPU显存 → YOLO推理 → 显示结果
"""

# ========== 导入必要的库 ==========
import cv2
import time
import numpy as np
import struct
from multiprocessing import shared_memory
from ultralytics import YOLO

# 尝试导入PyCUDA
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    print("警告：PyCUDA未安装，将使用CPU内存作为后备方案")

# 尝试导入torch
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
            name="inference_gpu",
            log_level=config.logging.get('level', 'INFO'),
            log_file=config.logging.get('file'),
            console=config.logging.get('console', True)
        )
        logger.info("成功加载配置文件")
        
        SHM_NAME = config.shared_memory.get('name', 'yolo_image_gpu_shm')
        MAX_WIDTH = config.shared_memory.get('max_width', 640)
        MAX_HEIGHT = config.shared_memory.get('max_height', 480)
        MAX_CHANNELS = config.shared_memory.get('max_channels', 3)
        
        MODEL_PATH = config.inference.get('model_path', 'yolov8n.pt')
        FPS = config.inference.get('fps', 24)
        CONF_THRESHOLD = config.inference.get('conf_threshold', 0.3)
        IOU_THRESHOLD = config.inference.get('iou_threshold', 0.5)
        CLASSES = config.inference.get('classes', [0])
    except Exception as e:
        print(f"警告：加载配置失败，使用默认配置: {e}")
        USE_CONFIG = False

if not USE_CONFIG:
    try:
        logger = setup_logger("inference_gpu", console=True)
    except:
        class SimpleLogger:
            def info(self, msg): print(f"[INFO] {msg}")
            def warning(self, msg): print(f"[WARNING] {msg}")
            def error(self, msg): print(f"[ERROR] {msg}")
        logger = SimpleLogger()
    
    SHM_NAME = "yolo_image_gpu_shm"
    MAX_WIDTH = 640
    MAX_HEIGHT = 480
    MAX_CHANNELS = 3
    MODEL_PATH = "yolov8n.pt"
    FPS = 24
    CONF_THRESHOLD = 0.3
    IOU_THRESHOLD = 0.5
    CLASSES = [0]

SHM_META_SIZE = 28
SHM_DATA_SIZE = MAX_WIDTH * MAX_HEIGHT * MAX_CHANNELS * np.dtype(np.uint8).itemsize
SHM_TOTAL_SIZE = SHM_META_SIZE + SHM_DATA_SIZE

# ==================================================


def read_gpu_frame_from_shm(shm, debug=False):
    """
    从共享内存读取GPU帧信息
    
    参数:
        shm: 共享内存对象
        debug: 是否打印调试信息
    
    返回:
        (frame_tensor, shape, frame_id) 或 (None, None, None)
        frame_tensor: torch tensor（在GPU上）或numpy数组（CPU后备）
        shape: 图像形状
        frame_id: 帧ID
    """
    shm_array = np.ndarray((SHM_TOTAL_SIZE,), dtype=np.uint8, buffer=shm.buf)
    
    # 读取元数据
    w = struct.unpack("I", shm_array[0:4].tobytes())[0]
    h = struct.unpack("I", shm_array[4:8].tobytes())[0]
    c = struct.unpack("I", shm_array[8:12].tobytes())[0]
    
    # 读取GPU指针
    gpu_ptr = struct.unpack("Q", shm_array[12:20].tobytes())[0]
    
    # 读取帧ID
    frame_id = struct.unpack("Q", shm_array[20:28].tobytes())[0]
    
    # 检查尺寸
    if w == 0 or h == 0 or c == 0 or w > MAX_WIDTH or h > MAX_HEIGHT:
        if debug:
            logger.warning(f"尺寸无效: {w}x{h}x{c}")
        return None, None, None
    
    # 如果GPU指针有效，直接从GPU显存读取
    if gpu_ptr > 0 and PYCUDA_AVAILABLE and TORCH_AVAILABLE:
        try:
            # 创建GPU buffer对象
            gpu_buffer = cuda.DevicePointer(ptr=gpu_ptr)
            
            # 创建torch tensor，直接指向GPU显存
            frame_size = w * h * c
            frame_tensor = torch.as_tensor(
                np.frombuffer(
                    cuda.from_device_like(gpu_buffer, (frame_size,), np.uint8),
                    dtype=np.uint8
                ),
                device="cuda"
            ).view(h, w, c)
            
            return frame_tensor, (h, w, c), frame_id
        except Exception as e:
            if debug:
                logger.warning(f"从GPU读取失败: {e}，尝试CPU后备")
    
    # CPU后备方案：从共享内存数据区域读取
    try:
        data_start = SHM_META_SIZE
        data_end = data_start + w * h * c
        frame = shm_array[data_start:data_end].reshape((h, w, c)).copy()
        
        # 转换为torch tensor并移到GPU
        if TORCH_AVAILABLE:
            frame_tensor = torch.from_numpy(frame).to("cuda")
            return frame_tensor, (h, w, c), frame_id
        else:
            return frame, (h, w, c), frame_id
    except Exception as e:
        if debug:
            logger.warning(f"读取失败: {e}")
        return None, None, None


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("程序 B：推理进程（GPU显存版本）")
    logger.info(f"模型: {MODEL_PATH}")
    logger.info(f"共享内存名称: {SHM_NAME}")
    logger.info(f"推理频率: {FPS} Hz")
    logger.info(f"GPU支持: {'是' if PYCUDA_AVAILABLE else '否（使用CPU后备）'}")
    logger.info(f"PyTorch支持: {'是' if TORCH_AVAILABLE else '否'}")
    logger.info("=" * 60)
    
    # 加载YOLO模型
    logger.info("正在加载模型，请稍候...")
    try:
        model = YOLO(MODEL_PATH)
        # 确保模型在GPU上
        if TORCH_AVAILABLE and torch.cuda.is_available():
            model.to("cuda")
            logger.info("✅ 模型已加载到GPU")
        else:
            logger.warning("⚠️ GPU不可用，使用CPU推理")
        logger.info("模型加载完成！")
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        return
    
    # 连接到共享内存
    logger.info(f"正在连接共享内存 {SHM_NAME}...")
    max_retries = 10
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
            logger.info("✅ 已连接到共享内存！")
            break
        except FileNotFoundError:
            retry_count += 1
            if retry_count < max_retries:
                logger.info(f"等待共享内存创建... ({retry_count}/{max_retries})")
                time.sleep(1)
            else:
                logger.error(f"❌ 无法连接到共享内存 {SHM_NAME}")
                logger.error("请确保程序A（data_source_gpu.py）已启动")
                return
        except Exception as e:
            logger.error(f"❌ 连接共享内存时出错: {e}")
            return
    
    # 初始化变量
    interval = 1.0 / FPS
    frame_count = 0
    no_frame_count = 0
    last_frame = None
    last_frame_id = 0
    
    logger.info("开始从GPU显存读取并推理（按 'q' 键退出）...")
    
    try:
        while True:
            # 从共享内存读取GPU帧
            frame_tensor, shape, frame_id = read_gpu_frame_from_shm(
                shm, debug=(no_frame_count < 3)
            )
            
            if frame_tensor is None:
                no_frame_count += 1
                if no_frame_count == 1:
                    logger.warning("未读取到有效图像数据，请检查：")
                    logger.warning("   1. 程序A（data_source_gpu.py）是否正在运行？")
                    logger.warning("   2. GPU显存是否正常？")
                
                if last_frame is None:
                    time.sleep(interval)
                    continue
                frame_tensor = last_frame
            else:
                if no_frame_count > 0:
                    logger.info(f"✅ 已接收到GPU图像数据！图像尺寸: {shape}, 帧ID: {frame_id}")
                    no_frame_count = 0
                last_frame = frame_tensor
                last_frame_id = frame_id
            
            # YOLO推理（数据已在GPU，零拷贝）
            try:
                # 如果frame_tensor是torch tensor，转换为numpy（YOLO需要）
                if isinstance(frame_tensor, torch.Tensor):
                    frame_np = frame_tensor.cpu().numpy()
                else:
                    frame_np = frame_tensor
                
                results = model.predict(
                    frame_np,
                    conf=CONF_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    classes=CLASSES if CLASSES else None,
                    verbose=False,
                    device="cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
                )
                
                # 绘制检测结果
                annotated = results[0].plot()
                
                # 显示结果
                cv2.imshow("YOLO GPU Inference", annotated)
                
                frame_count += 1
                if frame_count == 1:
                    logger.info(f"✅ 成功推理第一帧！")
                
                # 每3秒打印一次进度
                if frame_count % (FPS * 3) == 0:
                    logger.info(f"已推理帧数: {frame_count}, 当前帧ID: {frame_id}, 检测数量: {len(results[0].boxes)}")
                
            except Exception as e:
                logger.error(f"推理时出错: {e}")
            
            # 检查退出
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("检测到 'q' 键，准备退出...")
                break
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        logger.info("\n检测到 Ctrl+C，准备退出...")
    except Exception as e:
        logger.error(f"运行过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("\n正在清理资源...")
        cv2.destroyAllWindows()
        
        try:
            shm.close()
            logger.info("✅ 共享内存连接已关闭")
        except Exception as e:
            logger.warning(f"关闭共享内存连接时出错: {e}")
        
        logger.info("程序 B 已退出。")


if __name__ == "__main__":
    main()
