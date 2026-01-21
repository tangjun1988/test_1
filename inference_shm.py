"""
程序B：推理引擎（使用共享内存版本）

这个程序的作用：
1. 从共享内存读取图像数据（由程序A写入）
2. 使用YOLO模型对图像进行目标检测（识别人体）
3. 在图像上画出检测框
4. 显示检测结果（或保存到文件）

工作流程：
共享内存 → 读取图像 → YOLO推理 → 画检测框 → 显示/保存结果

与程序A的关系：
- 程序A负责写入图像数据到共享内存
- 程序B负责从共享内存读取数据并进行推理
- 两个程序通过共享内存名称（SHM_NAME）连接
"""
# ========== 导入必要的库 ==========
import cv2          # OpenCV，用于显示图像和处理图像
import time         # 时间库，用于控制推理频率
import numpy as np  # NumPy，用于处理图像数组
import struct       # 结构体库，用于将字节转换为整数
from multiprocessing import shared_memory  # 共享内存库，用于读取共享内存
from ultralytics import YOLO  # YOLO模型库，用于目标检测

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
# 注意：共享内存配置必须和 data_source_shm.py 完全一致！
# 优先从配置文件读取，如果没有配置文件则使用默认值

if USE_CONFIG:
    try:
        config = Config("config.yaml")
        logger = setup_logger(
            name="inference",
            log_level=config.logging.get('level', 'INFO'),
            log_file=config.logging.get('file'),
            console=config.logging.get('console', True)
        )
        logger.info("成功加载配置文件")
        
        # 从配置文件读取参数
        SHM_NAME = config.shared_memory.get('name', 'yolo_image_shm')
        MAX_WIDTH = config.shared_memory.get('max_width', 1920)
        MAX_HEIGHT = config.shared_memory.get('max_height', 1080)
        MAX_CHANNELS = config.shared_memory.get('max_channels', 3)
        
        MODEL_PATH = config.inference.get('model_path', 'yolov8n.pt')
        FPS = config.inference.get('fps', 10)
        CONF_THRESHOLD = config.inference.get('conf_threshold', 0.3)
        IOU_THRESHOLD = config.inference.get('iou_threshold', 0.5)
        CLASSES = config.inference.get('classes', [0])  # 默认只检测人
    except Exception as e:
        print(f"警告：加载配置失败，使用默认配置: {e}")
        USE_CONFIG = False

# 如果没有配置文件或加载失败，使用默认配置
if not USE_CONFIG:
    # 创建简单的日志器（只输出到控制台）
    try:
        logger = setup_logger("inference", console=True)
    except:
        # 如果日志模块也失败，使用print
        class SimpleLogger:
            def info(self, msg): print(f"[INFO] {msg}")
            def warning(self, msg): print(f"[WARNING] {msg}")
            def error(self, msg): print(f"[ERROR] {msg}")
        logger = SimpleLogger()
    
    # 默认配置（保持原有代码）
    SHM_NAME = "yolo_image_shm"
    MAX_WIDTH = 1920
    MAX_HEIGHT = 1080
    MAX_CHANNELS = 3
    MODEL_PATH = "yolov8n.pt"
    FPS = 10
    CONF_THRESHOLD = 0.3
    IOU_THRESHOLD = 0.5
    CLASSES = [0]  # 只检测人

# 共享内存大小计算（必须和程序A一致）
SHM_META_SIZE = 12  # 元数据大小：12字节
SHM_DATA_SIZE = MAX_WIDTH * MAX_HEIGHT * MAX_CHANNELS * np.dtype(np.uint8).itemsize
SHM_TOTAL_SIZE = SHM_META_SIZE + SHM_DATA_SIZE

# ==================================================


def read_frame_from_shm(shm, debug=False):
    """
    从共享内存读取一帧图像
    
    参数：
        shm: 共享内存对象
        debug: 是否打印调试信息
    
    返回：
        numpy数组：图像数据，形状为 (高度, 宽度, 通道数)
        None：如果读取失败或数据无效
    
    共享内存布局（和程序A写入的格式一致）：
        [0-3字节]   宽度 (width)，4字节整数
        [4-7字节]   高度 (height)，4字节整数
        [8-11字节]  通道数 (channels)，4字节整数
        [12字节开始] 图像数据 (BGR格式，每个像素3个值)
    """
    # ========== 将共享内存转换为numpy数组 ==========
    # 这样可以直接用numpy的方式读取数据
    shm_array = np.ndarray((SHM_TOTAL_SIZE,), dtype=np.uint8, buffer=shm.buf)
    
    # ========== 读取元数据（图像尺寸信息）==========
    # struct.unpack("I", ...) 将4字节的二进制数据转换为整数
    # "I" 表示无符号整数（uint32）
    # tobytes() 将numpy数组转换为字节
    w = struct.unpack("I", shm_array[0:4].tobytes())[0]   # 读取宽度
    h = struct.unpack("I", shm_array[4:8].tobytes())[0]   # 读取高度
    c = struct.unpack("I", shm_array[8:12].tobytes())[0]  # 读取通道数
    
    # 打印调试信息（如果需要）
    #if debug:
    #    print(f"调试：读取到的尺寸 w={w}, h={h}, c={c}")
    
    # ========== 检查尺寸是否有效 ==========
    # 如果尺寸为0或超出限制，说明数据无效
    if w == 0 or h == 0 or c == 0 or w > MAX_WIDTH or h > MAX_HEIGHT:
        if debug:
            print(f"调试：尺寸无效，返回 None")
        return None
    
    # ========== 读取图像数据 ==========
    data_start = SHM_META_SIZE  # 数据开始位置：12字节（元数据之后）
    data_end = data_start + w * h * c  # 数据结束位置：开始位置 + 图像大小
    
    # ========== 将一维数组重塑为图像形状 ==========
    try:
        # reshape((h, w, c)) 将一维数组重新排列成三维数组
        # 例如：(6220800,) -> (1080, 1920, 3)
        frame = shm_array[data_start:data_end].reshape((h, w, c))
        # 返回副本，避免直接修改共享内存中的数据
        return frame.copy()
    except Exception as e:
        # 如果重塑失败（比如数据损坏），返回None
        if debug:
            print(f"调试：重塑图像时出错: {e}")
        return None


def main():
    """
    主函数：程序的入口点
    
    工作流程：
    1. 加载YOLO模型
    2. 连接到共享内存（等待程序A创建）
    3. 循环读取图像、推理、显示结果
    4. 退出时清理资源
    """
    # ========== 打印程序信息 ==========
    logger.info("=" * 60)
    logger.info("程序 B：推理进程（共享内存版本）")
    logger.info(f"模型: {MODEL_PATH}")
    logger.info(f"共享内存名称: {SHM_NAME}")
    logger.info(f"共享内存大小: {SHM_TOTAL_SIZE / 1024 / 1024:.2f} MB")
    logger.info(f"推理频率: {FPS} Hz")
    logger.info(f"置信度阈值: {CONF_THRESHOLD}")
    logger.info(f"IoU阈值: {IOU_THRESHOLD}")
    logger.info(f"检测类别: {CLASSES}")
    logger.info("=" * 60)

    # ========== 加载YOLO模型 ==========
    logger.info("正在加载模型，请稍候...")
    # YOLO() 会加载模型文件，如果文件不存在会自动下载
    model = YOLO(MODEL_PATH)
    logger.info("模型加载完成！")

    # ========== 连接到共享内存 ==========
    logger.info(f"正在连接共享内存 {SHM_NAME}...")
    max_retries = 10  # 最多重试10次
    retry_count = 0
    
    # 循环尝试连接共享内存（因为程序A可能还没启动）
    while retry_count < max_retries:
        try:
            # create=False 表示不创建，只连接已存在的共享内存
            shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
            logger.info("✅ 已连接到共享内存！")
            break  # 连接成功，退出循环
        except FileNotFoundError:
            # 共享内存不存在（程序A还没启动）
            retry_count += 1
            if retry_count < max_retries:
                logger.info(f"等待共享内存创建... ({retry_count}/{max_retries})")
                time.sleep(1)  # 等待1秒后重试
            else:
                # 重试次数用完，退出程序
                logger.error(f"❌ 无法连接到共享内存 {SHM_NAME}")
                logger.error("请确保程序A（data_source_shm.py）已启动")
                return
        except Exception as e:
            # 其他错误
            logger.error(f"❌ 连接共享内存时出错: {e}")
            return

    # ========== 初始化变量 ==========
    last_frame = None  # 保存上一帧图像（如果读取失败，显示上一帧）
    frame_count = 0    # 计数器：已推理的帧数
    interval = 1.0 / FPS  # 每次循环的间隔时间（秒）
    no_frame_count = 0  # 统计连续未收到帧的次数

    logger.info("按 'q' 关闭窗口并退出程序。")

    try:
        # ========== 主循环：持续读取、推理、显示 ==========
        while True:
            # 从共享内存读取图像（前3次失败时打印调试信息）
            frame = read_frame_from_shm(shm, debug=(no_frame_count < 3))
            
            if frame is None:
                # ========== 读取失败的处理 ==========
                no_frame_count += 1
                if no_frame_count == 1:
                    # 第一次失败，打印提示
                    logger.warning("未读取到有效图像数据，请检查：")
                    logger.warning("   1. 程序A（data_source_shm.py）是否正在运行？")
                    logger.warning("   2. 视频文件路径是否正确？")
                elif no_frame_count % 30 == 0:  # 每30次（约3秒）提示一次
                    logger.warning(f"仍在等待图像数据... (已等待 {no_frame_count} 次)")
                
                # 如果没有收到新帧，使用上一帧（维持画面，避免黑屏）
                if last_frame is None:
                    # 如果连上一帧都没有，等待后继续
                    time.sleep(interval)
                    continue
                frame = last_frame  # 使用上一帧
            else:
                # ========== 读取成功的处理 ==========
                if no_frame_count > 0:
                    # 之前失败过，现在成功了，打印提示
                    logger.info(f"✅ 已接收到图像数据！图像尺寸: {frame.shape}")
                    no_frame_count = 0  # 重置失败计数
                last_frame = frame  # 保存这一帧，以备下次使用

            # ========== YOLO推理 ==========
            # model.predict() 对图像进行目标检测
            results = model.predict(
                frame,                    # 输入图像
                conf=CONF_THRESHOLD,      # 置信度阈值
                iou=IOU_THRESHOLD,        # IoU阈值：用于非极大值抑制（NMS）
                classes=CLASSES if CLASSES else None,  # 检测类别（空列表表示所有类别）
                verbose=False,            # 不打印详细信息
            )

            # ========== 绘制检测结果 ==========
            # results[0].plot() 在图像上画出检测框和标签
            # 返回绘制好的图像（numpy数组）
            annotated = results[0].plot()

            # ========== 显示结果 ==========
            # cv2.imshow() 显示图像窗口
            # 窗口标题使用英文避免乱码
            cv2.imshow("YOLO Real-time Inference (Shared Memory)", annotated)
            frame_count += 1  # 推理成功，计数器+1
            
            # 每3秒打印一次进度（FPS * 3 = 30帧）
            if frame_count % (FPS * 3) == 0:
                logger.info(f"已推理帧数: {frame_count}, 图像尺寸: {annotated.shape}")

            # ========== 检查用户输入 ==========
            # cv2.waitKey(1) 等待1毫秒，检查是否有键盘输入
            # & 0xFF 取低8位
            # ord("q") 获取字符'q'的ASCII码
            # 如果用户按了'q'键，退出循环
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # 等待一段时间，控制推理频率
            time.sleep(interval)

    except KeyboardInterrupt:
        # 用户按 Ctrl+C 中断程序
        logger.info("\n检测到 Ctrl+C，准备退出...")
    except Exception as e:
        # 发生其他错误
        logger.error(f"❌ 推理过程中发生错误: {e}")
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
        
        # 尝试删除共享内存（如果程序A已经退出，或者这是最后一个连接）
        try:
            shm.unlink()  # 删除共享内存
            logger.info("✅ 共享内存已删除")
        except FileNotFoundError:
            # 共享内存已经被程序A删除，这是正常的
            pass
        except Exception as e:
            # 其他错误（比如程序A还在使用），忽略
            pass
        
        try:
            cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
        except:
            pass
        
        logger.info("程序 B 已退出。")


# ========== 程序入口 ==========
# 当直接运行这个文件时（而不是被其他文件导入），执行main()函数
if __name__ == "__main__":
    main()

