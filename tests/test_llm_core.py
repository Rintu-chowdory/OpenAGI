import os

import pytest

from pyopenagi.core import llm as llm_core


def test_llm_config_groq_default(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAGI_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAGI_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAGI_LLM_MODEL", raising=False)
    cfg = llm_core.llm_config()
    assert cfg["base_url"] == "https://api.groq.com/openai/v1"
    assert cfg["api_key"] == "gsk_test"
    assert cfg["model"] == "llama-3.3-70b-versatile"


def test_llm_config_env_override(monkeypatch):
    monkeypatch.setenv("OPENAGI_LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("OPENAGI_LLM_MODEL", "my-model")
    monkeypatch.setenv("OPENAGI_LLM_API_KEY", "k")
    cfg = llm_core.llm_config()
    assert cfg["base_url"] == "http://localhost:8000/v1"
    assert cfg["model"] == "my-model"
    assert cfg["api_key"] == "k"


def test_execution_mode_standalone(monkeypatch):
    monkeypatch.setenv("OPENAGI_EXECUTION_MODE", "standalone")
    assert llm_core.execution_mode() == "standalone"
    assert llm_core.use_aios() is False


def test_execution_mode_invalid_falls_back_to_auto(monkeypatch):
    monkeypatch.setenv("OPENAGI_EXECUTION_MODE", "bogus")
    assert llm_core.execution_mode() == "auto"


def test_aios_not_available_in_test_env():
    # this environment does not have the aios kernel installed
    assert llm_core.aios_available() is False


def test_mock_llm_plan_then_answer():
    mock = llm_core.MockLLM()
    content, tool_calls = mock.chat([{"role": "user", "content": "hi"}], json_mode=True)
    assert content.startswith("[")
    content2, tool_calls2 = mock.chat([{"role": "user", "content": "step"}])
    assert "Final answer" in content2
    assert tool_calls is None and tool_calls2 is None


def test_mock_llm_tool_call_script():
    script = [("plan", [{"name": "wikipedia", "parameters": {"query": "crm"}}])]
    mock = llm_core.MockLLM(script=script)
    content, tool_calls = mock.chat([])
    assert content == "plan"
    assert tool_calls[0]["name"] == "wikipedia"


def test_use_aios_mode_forces_error(monkeypatch):
    monkeypatch.setenv("OPENAGI_EXECUTION_MODE", "aios")
    with pytest.raises(RuntimeError, match="aios kernel is not installed"):
        llm_core.use_aios()
