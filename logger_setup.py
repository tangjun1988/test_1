"""
日志系统设置模块.

作用：设置日志系统，记录程序运行信息并保存到文件
类比：像记录员，记录重要信息并保存
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(
    name: str = "yolo_shm", log_level: str = "INFO", log_file: str | None = None, console: bool = True
) -> logging.Logger:
    """设置日志系统.

    参数:
        name: 日志器名称（用于区分不同模块的日志）
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        log_file: 日志文件路径（None表示不保存到文件）
        console: 是否输出到控制台

    返回:
        配置好的日志器

    使用示例:
        logger = setup_logger("data_source", log_file="logs/app.log")
        logger.info("程序启动")
        logger.warning("警告信息")
        logger.error("错误信息")
    """
    # 创建日志器
    logger = logging.getLogger(name)

    # 设置日志级别
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # 避免重复添加处理器（如果已经配置过，直接返回）
    if logger.handlers:
        return logger

    # 日志格式：时间 - 名称 - 级别 - 消息
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 控制台输出处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件输出处理器
    if log_file:
        # 创建日志目录（如果不存在）
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # 使用RotatingFileHandler，自动轮转日志文件
        # maxBytes: 单个日志文件最大10MB
        # backupCount: 保留5个备份文件
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,  # 保留5个备份文件
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
