"""Minimal client for an OpenAI-compatible chat completion API."""

from collections.abc import Mapping, Sequence
from typing import Any, overload

from openai import OpenAI, OpenAIError

from config import Settings


class LLMError(RuntimeError):
    """Raised when an LLM request fails or returns unusable content."""


class LLMClient:
    """Small wrapper around the OpenAI-compatible Chat Completions API."""

    def __init__(
        self, settings: Settings | None = None, *, model: str | None = None
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.model = model or self.settings.model
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

    @overload
    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: None = None,
    ) -> str: ...

    @overload
    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> Any: ...

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> str | Any:
        """Send messages and return text or an assistant tool-calling message."""
        request: dict[str, Any] = {
            "model": getattr(self, "model", self.settings.model),
            "messages": list(messages),
        }
        if tools is not None:
            request["tools"] = list(tools)

        try:
            completion = self._client.chat.completions.create(**request)
        except OpenAIError as exc:
            raise LLMError(f"模型请求失败：{exc}") from exc

        if not completion.choices:
            raise LLMError("模型返回异常：响应中没有可用选项。")

        message = completion.choices[0].message
        if tools is not None:
            return message

        content = message.content
        if not isinstance(content, str) or not content.strip():
            raise LLMError("模型返回异常：响应内容为空。")

        return content
