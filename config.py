"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_BUILDER_MODEL = "qwen-plus"
DEFAULT_VERIFIER_MODEL = "deepseek-v4-flash"
# Kept as a compatibility alias for code importing the v0.6 constant.
DEFAULT_MODEL = DEFAULT_BUILDER_MODEL


class ConfigurationError(ValueError):
    """Raised when required application configuration is invalid."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    verifier_model: str = DEFAULT_VERIFIER_MODEL

    @property
    def builder_model(self) -> str:
        return self.model

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
        model = (
            os.getenv("BUILDER_MODEL", "").strip()
            or os.getenv("LLM_MODEL", "").strip()
            or DEFAULT_BUILDER_MODEL
        )
        verifier_model = (
            os.getenv("VERIFIER_MODEL", "").strip()
            or DEFAULT_VERIFIER_MODEL
        )

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            verifier_model=verifier_model,
        )
