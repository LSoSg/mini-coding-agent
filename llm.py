"""Minimal client for an OpenAI-compatible chat completion API."""

from collections.abc import Mapping, Sequence
from typing import Any

from openai import OpenAI, OpenAIError

from config import Settings


class LLMError(RuntimeError):
    """Raised when an LLM request fails or returns unusable content."""


class LLMClient:
    """Small wrapper around the OpenAI-compatible Chat Completions API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        try:
            self._client = OpenAI(
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
            )
        except (OpenAIError, OSError) as exc:
            raise LLMError(
                "模型客户端初始化失败，请检查网络和 SSL 证书环境变量："
                f"{exc}"
            ) from exc

    def chat(self, messages: Sequence[Mapping[str, Any]]) -> str:
        """Send standard Chat Completions messages and return response text."""
        try:
            completion = self._client.chat.completions.create(
                model=self.settings.model,
                messages=list(messages),
            )
        except OpenAIError as exc:
            raise LLMError(f"模型请求失败：{exc}") from exc

        if not completion.choices:
            raise LLMError("模型返回异常：响应中没有可用选项。")

        content = completion.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise LLMError("模型返回异常：响应内容为空。")

        return content
