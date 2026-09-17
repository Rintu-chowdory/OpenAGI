"""End-to-end agent run in standalone mode using the MockLLM.

No network access and no API keys required.
"""

import pytest

from pyopenagi.agents.agent_process import AgentProcessFactory
from pyopenagi.agents.react_agent import ReactAgent
from pyopenagi.core import llm as llm_core
from pyopenagi.utils.chat_template import Query


@pytest.fixture()
def standalone_mode(monkeypatch):
    monkeypatch.setenv("OPENAGI_EXECUTION_MODE", "standalone")
    llm_core.set_llm(llm_core.MockLLM())
    yield
    llm_core.set_llm(None)


def _make_agent(agent_name="rintu/dev_ops_agent", task="review my ci pipeline", tmp_path=None):
    class _MinimalAgent(ReactAgent):
        def __init__(self, *args, **kwargs):
            ReactAgent.__init__(self, *args, **kwargs)
            self.workflow_mode = "manual"
            self.manual_steps = [
                {"message": "review the pipeline", "tool_use": []},
            ]

        def manual_workflow(self):
            return self.manual_steps

    agent = _MinimalAgent(agent_name, task, AgentProcessFactory(), "console")
    return agent


def test_standalone_run_end_to_end(standalone_mode):
    # manual workflow already supplies the plan; every LLM call should give an answer
    llm_core.set_llm(llm_core.MockLLM(script=[None]))
    agent = _make_agent(task="review my ci pipeline")
    result = agent.run()
    assert result["agent_name"] == "rintu/dev_ops_agent"
    assert "rounds" in result and result["rounds"] >= 1
    assert result["request_turnaround_times"], "no LLM calls were recorded"
    # mock llm leaves a final answer in the message stream
    assert any("[MockLLM] Final answer" in m.get("content", "") for m in agent.messages if isinstance(m, dict))


def test_standalone_automatic_workflow_generates_plan(standalone_mode):
    agent = _make_agent(task="plan my deployment")
    agent.workflow_mode = "automatic"
    workflow = agent.automatic_workflow()
    assert workflow is not None
    assert workflow[0]["message"] == "answer the task directly"


def test_standalone_get_response_shape(standalone_mode):
    agent = _make_agent()
    response, starts, ends, waits, turns = agent.get_response(
        query=Query(messages=[{"role": "user", "content": "hello"}])
    )
    assert response.response_message
    assert response.tool_calls is None
    assert len(starts) == 1 and len(ends) == 1 and len(waits) == 1 and len(turns) == 1


def test_standalone_without_key_raises(monkeypatch):
    monkeypatch.setenv("OPENAGI_EXECUTION_MODE", "standalone")
    for var in ("OPENAGI_LLM_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    llm_core.set_llm(llm_core.StandaloneLLM(api_key=None))
    agent = _make_agent()
    with pytest.raises(RuntimeError, match="No LLM API key"):
        agent.get_response(query=Query(messages=[{"role": "user", "content": "x"}]))
    llm_core.set_llm(None)


def test_full_rintu_agent_runs(standalone_mode, monkeypatch):
    """The actual rintu/dev_ops_agent (automatic workflow) runs with the mock."""
    from pyopenagi.agents.agent_factory import AgentFactory

    factory = AgentFactory(agent_process_factory=AgentProcessFactory(), agent_log_mode="console")
    agent_class = factory.load_agent_instance("rintu/dev_ops_agent")
    agent = agent_class("rintu/dev_ops_agent", "how do I cache docker layers?", AgentProcessFactory(), "console")
    result = agent.run()
    assert result and result["agent_name"] == "rintu/dev_ops_agent"


def test_context_trim_fits_token_cap():
    from pyopenagi.agents.base_agent import BaseAgent
    big = [
        {"role": "system", "content": "x" * 200},
        {"role": "user", "content": "the original task"},
    ] + [{"role": "assistant", "content": "y" * 4000} for _ in range(10)]
    trimmed = BaseAgent._trim_context(big, max_tokens=6000)
    assert trimmed[0]["role"] == "system"
    assert trimmed[1]["content"] == "the original task"
    est = sum(len(str(m.get("content", ""))) for m in trimmed) // 4
    assert est <= 6000
    assert len(trimmed) < len(big)


def test_context_trim_leaves_short_conversations_alone():
    from pyopenagi.agents.base_agent import BaseAgent
    small = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    assert BaseAgent._trim_context(small, max_tokens=6000) is small
