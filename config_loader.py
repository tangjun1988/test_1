"""
配置文件加载模块.

作用：读取 config.yaml 配置文件，并提供给程序使用
类比：像服务员，把菜单（配置文件）的内容告诉厨房（程序）
"""

from pathlib import Path
from typing import Any

import yaml


class Config:
    """配置类，用于加载和管理配置."""

    def __init__(self, config_path: str = "config.yaml"):
        """加载配置文件.

        参数:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """加载YAML配置文件."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config is None:
            config = {}

        return config

    @property
    def data_source(self) -> dict[str, Any]:
        """获取数据源配置."""
        return self._config.get("data_source", {})

    @property
    def shared_memory(self) -> dict[str, Any]:
        """获取共享内存配置."""
        return self._config.get("shared_memory", {})

    @property
    def inference(self) -> dict[str, Any]:
        """获取推理配置."""
        return self._config.get("inference", {})

    @property
    def logging(self) -> dict[str, Any]:
        """获取日志配置."""
        return self._config.get("logging", {})

    def get(self, key: str, default=None):
        """获取配置值（支持点号分隔的嵌套键，如 'data_source.type'）.

        参数:
            key: 配置键，支持点号分隔（如 'data_source.type'）
            default: 默认值（如果键不存在）

        返回:
            配置值
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value
