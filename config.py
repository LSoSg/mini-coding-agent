"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"


class ConfigurationError(ValueError):
    """Raised when required application configuration is invalid."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from the environment and an optional local .env file."""
        load_dotenv()

        api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise ConfigurationError(
                "未配置 DASHSCOPE_API_KEY，请先在环境变量或 .env 文件中设置它。"
            )

        base_url = os.getenv("LLM_BASE_URL", "").strip() or DEFAULT_BASE_URL
        model = os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODEL

        return cls(api_key=api_key, base_url=base_url, model=model)
