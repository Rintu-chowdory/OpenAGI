import os

from pyopenagi.manager.manager import AgentManager


def test_cache_dir_not_hardcoded(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAGI_CACHE_DIR", str(tmp_path / "cache"))
    mgr = AgentManager()
    assert str(mgr.cache_dir).startswith(str(tmp_path))
    assert "/Users/rama2r" not in str(mgr.cache_dir)


def test_base_url_env_override(monkeypatch):
    monkeypatch.setenv("OPENAGI_AGENT_HUB_URL", "http://localhost:9999")
    mgr = AgentManager()
    assert mgr.base_url == "http://localhost:9999"


def test_base_url_default(monkeypatch):
    monkeypatch.delenv("OPENAGI_AGENT_HUB_URL", raising=False)
    mgr = AgentManager()
    assert mgr.base_url == "https://openagi-beta.vercel.app"
