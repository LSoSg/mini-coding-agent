"""Dual-model environment configuration tests."""

import pytest

from config import Settings


def test_dual_models_share_key_and_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "legacy-builder")
    monkeypatch.setenv("BUILDER_MODEL", "qwen-plus")
    monkeypatch.setenv("VERIFIER_MODEL", "deepseek-v4-flash")

    settings = Settings.from_env()

    assert settings.api_key == "test-key"
    assert settings.base_url == "https://example.invalid/v1"
    assert settings.builder_model == "qwen-plus"
    assert settings.verifier_model == "deepseek-v4-flash"


def test_legacy_llm_model_remains_builder_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "legacy-builder")
    # Explicit empty values prevent a developer's local .env from influencing the test.
    monkeypatch.setenv("BUILDER_MODEL", "")
    monkeypatch.setenv("VERIFIER_MODEL", "")

    settings = Settings.from_env()

    assert settings.builder_model == "legacy-builder"
    assert settings.verifier_model == "deepseek-v4-flash"
