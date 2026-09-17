# OpenAGI — Personal Edition

My personal fork of [OpenAGI](https://github.com/agiresearch/OpenAGI), rebuilt to actually run standalone: agents execute directly against any **OpenAI-compatible LLM API** (Groq, OpenAI, Together, OpenRouter, vLLM…) — no AIOS kernel required.

```
[rintu/dev_ops_agent] Generated workflow is: [{'message': 'answer the question using reasoning', 'tool_use': []}]

RESULT
**Final answer:** CRM systems manage customer relationships...
rounds: 2 · turnaround: 0.01s
```

## What's different from upstream

- **Standalone execution engine** (`pyopenagi/core/llm.py`) — when the `aios` kernel isn't installed, agents call the LLM API directly with full function-calling support. Same `ReactAgent` flow, zero infrastructure. Retries with backoff on rate limits and trims long conversations to fit token caps — long runs work even on the free Groq tier.
- **CLI** — `openagi run`, `openagi list`, `openagi tools`, `openagi doctor` (registered as a console script).
- **My agents** (`pyopenagi/agents/rintu/`):
  - `dev_ops_agent` — CI/CD, Docker, K8s and deployment-strategy consultant
  - `security_audit_agent` — internal audit, ISO 27001/NIST/BSI controls, AI governance
  - `saas_idea_agent` — SaaS ideation, MVP scoping, go-to-market (uses the Wikipedia tool)
  - `base44_builder_agent` — drafts complete Base44 micro-SaaS build plans (entities, pages, workflows, build order)
- **Bug fixes**:
  - removed hardcoded `/Users/rama2r/...` cache path → `~/.openagi/cache` (or `OPENAGI_CACHE_DIR`)
  - agent-hub URL now configurable (`OPENAGI_AGENT_HUB_URL`)
  - fixed malformed tool reference in `example/travel_agent` config
- **Tests** — 23 tests incl. end-to-end agent runs against a mock LLM; CI runs on every push.

## Quickstart

```bash
git clone https://github.com/Rintu-chowdory/OpenAGI.git
cd OpenAGI
pip install -e .

# Groq (recommended — fast + cheap gpt-oss-120b)
export OPENAGI_LLM_BASE_URL=https://api.groq.com/openai/v1
export OPENAGI_LLM_MODEL=openai/gpt-oss-120b
export OPENAGI_LLM_API_KEY=gsk_...

openagi doctor                       # check your setup
openagi list                         # see available agents
openagi run rintu/dev_ops_agent "review my GitHub Actions pipeline for a Vite + Tailwind app"
openagi run rintu/security_audit_agent "create a GDPR-focused checklist for an AI chatbot"
openagi run rintu/saas_idea_agent "validate this idea: AI-powered invoice reminders for freelancers"
```

No key set? `openagi doctor` tells you exactly what's missing.

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `OPENAGI_LLM_API_KEY` | LLM API key (fallbacks: `GROQ_API_KEY`, `OPENAI_API_KEY`) | — |
| `OPENAGI_LLM_BASE_URL` | OpenAI-compatible endpoint | auto from key type |
| `OPENAGI_LLM_MODEL` | Model name | `openai/gpt-oss-120b` / `gpt-4o-mini` |
| `OPENAGI_EXECUTION_MODE` | `auto` \| `standalone` \| `aios` | `auto` |
| `OPENAGI_AGENT_HUB_URL` | Agent hub for upload/download | upstream hub |
| `OPENAGI_CACHE_DIR` | Agent cache location | `~/.openagi/cache` |
| `OPENAGI_MAX_CONTEXT_TOKENS` | Trim long conversations to this size (rate-limit safety) | `6000` |

See `.env.example` — a `.env` in the repo root is loaded automatically.

## Building your own agents

Create `pyopenagi/agents/<your-name>/<agent_name>/`:

```
- agent.py              # class <CamelCaseName>(ReactAgent)
- config.json           # name, description (system prompt), tools, meta
- meta_requirements.txt # pip packages the agent needs
```

Then run it: `openagi run <your-name>/<agent_name> "task"`. Tools live in `pyopenagi/tools/<org>/<tool>.py` and follow the OpenAI function-calling format — the agents in `pyopenagi/agents/rintu/` are reference implementations.

## Execution modes

- **auto** (default): use the AIOS kernel if installed, otherwise standalone direct-API calls.
- **standalone**: always direct API calls — what you want on a laptop or CI.
- **aios**: require the kernel (for AIOS scheduler research setups).

## Tests

```bash
pip install -e . pytest
pytest tests/          # no API keys or network needed
```

## Upstream

Based on [OpenAGI](https://github.com/agiresearch/OpenAGI) (NeurIPS 2023, *"OpenAGI: When LLM Meets Domain Experts"*, MIT license). Upstream has moved agent development to [Cerebrum](https://github.com/agiresearch/Cerebrum) — this fork keeps the classic ReactAgent flow and makes it self-contained.
