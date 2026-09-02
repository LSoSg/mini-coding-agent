"""Unit tests for LLMClient response modes without network access."""

from types import SimpleNamespace
from typing import Any

from llm import LLMClient


class FakeCompletions:
    def __init__(self, message: Any) -> None:
        self.message = message
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=self.message)])


def make_client(message: Any) -> tuple[LLMClient, FakeCompletions]:
    completions = FakeCompletions(message)
    client = LLMClient.__new__(LLMClient)
    client.settings = SimpleNamespace(model="test-model")
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    return client, completions


def test_chat_without_tools_preserves_plain_text_mode() -> None:
    message = SimpleNamespace(content="plain response")
    client, completions = make_client(message)

    result = client.chat([{"role": "user", "content": "hello"}])

    assert result == "plain response"
    assert "tools" not in completions.requests[0]


def test_chat_with_tools_returns_assistant_message() -> None:
    message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="call")])
    client, completions = make_client(message)
    schemas = [{"type": "function", "function": {"name": "read_file"}}]

    result = client.chat(
        [{"role": "user", "content": "read"}],
        tools=schemas,
    )

    assert result is message
    assert completions.requests[0]["tools"] == schemas


def test_explicit_model_selects_independent_pipeline_model() -> None:
    message = SimpleNamespace(content="review")
    client, completions = make_client(message)
    client.model = "deepseek-v4-flash"

    client.chat([{"role": "user", "content": "review code"}])

    assert completions.requests[0]["model"] == "deepseek-v4-flash"
