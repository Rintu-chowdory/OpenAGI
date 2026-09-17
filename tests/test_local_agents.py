import json
import os

import pytest

from click.testing import CliRunner

from pyopenagi.cli import main as cli_main


AGENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "pyopenagi", "agents")
TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "pyopenagi", "tools")


def all_local_agents():
    from pyopenagi.cli import _local_agents
    return _local_agents()


def test_rintu_agents_exist():
    agents = all_local_agents()
    assert ("rintu", "dev_ops_agent") in agents
    assert ("rintu", "security_audit_agent") in agents
    assert ("rintu", "saas_idea_agent") in agents
    assert ("rintu", "base44_builder_agent") in agents


def test_agent_configs_valid():
    for author, name in all_local_agents():
        cfg_path = os.path.join(AGENTS_DIR, author, name, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        assert isinstance(cfg["description"], list) and cfg["description"]
        assert isinstance(cfg.get("tools", []), list)
        if author == "rintu":
            assert cfg["meta"]["author"] == author
        assert os.path.isfile(os.path.join(AGENTS_DIR, author, name, "agent.py"))
        assert os.path.isfile(os.path.join(AGENTS_DIR, author, name, "meta_requirements.txt"))
        # strict name check for rintu's own agents
        if author == "rintu":
            assert cfg["name"] == name, f"{author}/{name}: config name mismatch"


def test_agent_tool_references_resolve():
    """Every tool referenced by a local agent config exists in pyopenagi/tools.

    Tools are files: pyopenagi/tools/<org>/<name>.py
    """
    for author, name in all_local_agents():
        with open(os.path.join(AGENTS_DIR, author, name, "config.json")) as f:
            cfg = json.load(f)
        for tool in cfg.get("tools", []):
            org, tool_name = tool.split("/")
            assert tool_name, f"{author}/{name} has malformed tool ref {tool!r}"
            assert os.path.isfile(os.path.join(TOOLS_DIR, org, tool_name + ".py")), \
                f"{author}/{name} references missing tool {tool}"


def test_cli_lists_rintu_agents():
    runner = CliRunner()
    res = runner.invoke(cli_main, ["list"])
    assert res.exit_code == 0
    assert "rintu/dev_ops_agent" in res.output
    assert "rintu/security_audit_agent" in res.output
    assert "rintu/saas_idea_agent" in res.output
    assert "rintu/base44_builder_agent" in res.output


def test_cli_tools_command():
    runner = CliRunner()
    res = runner.invoke(cli_main, ["tools"])
    assert res.exit_code == 0
    assert "wikipedia/wikipedia" in res.output


def test_cli_doctor_reports_missing_key(monkeypatch):
    from pyopenagi.core import llm as llm_core
    monkeypatch.setenv("OPENAGI_EXECUTION_MODE", "standalone")
    monkeypatch.delenv("OPENAGI_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm_core.set_llm(llm_core.StandaloneLLM(api_key=None))
    runner = CliRunner()
    res = runner.invoke(cli_main, ["doctor"])
    assert res.exit_code == 1
    assert "MISSING" in res.output
