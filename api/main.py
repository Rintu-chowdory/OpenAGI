"""OpenAGI Agent API — FastAPI wrapper around the OpenAGI Personal Edition agents.

Endpoints:
    GET  /health        liveness check
    GET  /agents        list locally available agents
    POST /run           run an agent on a task and return its result

Auth: set AGENT_API_KEY to require an X-API-Key header on /run and /agents.
"""

import os
import time

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pyopenagi.agents.agent_factory import AgentFactory
from pyopenagi.agents.agent_process import AgentProcessFactory
from pyopenagi.core import llm as llm_core

app = FastAPI(title="OpenAGI Agent API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _local_agents():
    """List "author/name" for all agents under pyopenagi/agents."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pyopenagi", "agents")
    base = os.path.normpath(base)
    agents = []
    if not os.path.isdir(base):
        return agents
    for author in sorted(os.listdir(base)):
        author_dir = os.path.join(base, author)
        if not os.path.isdir(author_dir) or author.startswith(("_", ".")):
            continue
        for name in sorted(os.listdir(author_dir)):
            if os.path.isfile(os.path.join(author_dir, name, "agent.py")):
                agents.append(f"{author}/{name}")
    return agents


def _check_auth(x_api_key: str):
    expected = os.environ.get("AGENT_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key header")


class RunRequest(BaseModel):
    agent: str  # e.g. "rintu/dev_ops_agent"
    task: str
    log_mode: str = "console"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "execution_mode": "aios" if llm_core.use_aios() else "standalone",
        "llm_configured": llm_core.get_llm().configured if not llm_core.use_aios() else True,
    }


@app.get("/agents")
def list_agents(x_api_key: str = Header(default="")):
    _check_auth(x_api_key)
    return {"agents": _local_agents()}


@app.post("/run")
def run_agent(req: RunRequest, x_api_key: str = Header(default="")):
    _check_auth(x_api_key)

    if "/" not in req.agent:
        raise HTTPException(status_code=400, detail='agent must be "author/agent_name", e.g. rintu/dev_ops_agent')
    if req.agent not in _local_agents():
        raise HTTPException(status_code=404, detail={"error": f'agent "{req.agent}" not found',
                                                     "available": _local_agents()})

    if not llm_core.use_aios():
        client = llm_core.get_llm()
        if not client.configured:
            raise HTTPException(
                status_code=503,
                detail="No LLM configured. Set OPENAGI_LLM_API_KEY (+ OPENAGI_LLM_BASE_URL / OPENAGI_LLM_MODEL).",
            )

    start = time.time()
    try:
        factory = AgentFactory(agent_process_factory=AgentProcessFactory(), agent_log_mode=req.log_mode)
        agent_class = factory.load_agent_instance(req.agent)
        agent = agent_class(req.agent, req.task, AgentProcessFactory(), req.log_mode)
        result = agent.run()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"agent run failed: {exc}") from exc

    if not result:
        raise HTTPException(status_code=500, detail="agent run failed (empty result)")

    content = result.get("result")
    if isinstance(content, dict) and "content" in content:
        content = content["content"]

    return {
        "agent": req.agent,
        "task": req.task,
        "result": content,
        "rounds": result.get("rounds"),
        "elapsed_seconds": round(time.time() - start, 2),
        "raw": result,
    }
