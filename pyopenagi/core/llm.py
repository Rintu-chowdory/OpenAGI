# OpenAGI Personal Edition — standalone LLM execution engine.
#
# This module decouples pyopenagi from the AIOS kernel: if `aios` is not
# installed (or OPENAGI_EXECUTION_MODE=standalone), agents execute directly
# against any OpenAI-compatible chat completions endpoint — Groq, OpenAI,
# Together, OpenRouter, or a local llama.cpp / vLLM server.
#
# Configuration (env vars, or a loaded .env):
#   OPENAGI_LLM_BASE_URL     e.g. https://api.groq.com/openai/v1
#   OPENAGI_LLM_MODEL        e.g. openai/gpt-oss-120b (Groq)
#   OPENAGI_LLM_API_KEY      the API key
#   OPENAGI_EXECUTION_MODE   "auto" (default) | "standalone" | "aios"
#
# Key fallback chain: OPENAGI_LLM_API_KEY -> GROQ_API_KEY -> OPENAI_API_KEY,
# with the base URL defaulting to the matching provider.

import json
import os
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"


def _env(name, default=None):
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else default


def llm_config():
    """Resolve the LLM configuration from the environment."""
    api_key = _env("OPENAGI_LLM_API_KEY") or _env("GROQ_API_KEY") or _env("OPENAI_API_KEY")
    base_url = _env("OPENAGI_LLM_BASE_URL")
    model = _env("OPENAGI_LLM_MODEL")

    if not base_url:
        if _env("GROQ_API_KEY") and not _env("OPENAI_API_KEY"):
            base_url = GROQ_BASE_URL
        else:
            base_url = OPENAI_BASE_URL
    if not model:
        model = "openai/gpt-oss-120b" if "groq" in base_url else "gpt-4o-mini"

    return {"base_url": base_url, "model": model, "api_key": api_key}


def execution_mode():
    """auto | standalone | aios"""
    mode = (_env("OPENAGI_EXECUTION_MODE") or "auto").lower()
    if mode not in ("auto", "standalone", "aios"):
        mode = "auto"
    return mode


def aios_available():
    """True if the real AIOS kernel hooks can be imported."""
    try:
        from aios.hooks.stores._global import global_llm_req_queue_add_message  # noqa: F401
        return True
    except Exception:
        return False


def use_aios():
    """Decide whether to route LLM requests through the AIOS kernel."""
    mode = execution_mode()
    if mode == "aios":
        if not aios_available():
            raise RuntimeError(
                "OPENAGI_EXECUTION_MODE=aios but the aios kernel is not installed. "
                "Install it from github.com/agiresearch/AIOS or switch to standalone mode."
            )
        return True
    if mode == "standalone":
        return False
    return aios_available()


class ChatResult:
    """Mirrors the AgentProcess response shape used across pyopenagi."""

    def __init__(self, response_message, tool_calls=None):
        self.response_message = response_message
        self.tool_calls = tool_calls


class StandaloneLLM:
    """Minimal OpenAI-compatible chat client built on `requests` (no new deps).

    Supports plain chat and function calling in the OpenAI wire format, so it
    works unchanged against Groq, OpenAI, Together, OpenRouter, vLLM, etc.
    """

    def __init__(self, base_url=None, model=None, api_key=None, timeout=120):
        cfg = llm_config()
        self.base_url = (base_url or cfg["base_url"]).rstrip("/")
        self.model = model or cfg["model"]
        self.api_key = api_key or cfg["api_key"]
        self.timeout = timeout

    @property
    def configured(self):
        return bool(self.api_key)

    def _providers(self):
        """Primary provider first, then an optional fallback provider.

        Fallback is enabled by setting OPENAGI_LLM_FALLBACK_API_KEY or
        OPENROUTER_API_KEY. Defaults target OpenRouter's free tier.
        """
        providers = [{"base_url": self.base_url, "model": self.model, "api_key": self.api_key}]
        fb_key = _env("OPENAGI_LLM_FALLBACK_API_KEY") or _env("OPENROUTER_API_KEY")
        if fb_key:
            providers.append({
                "base_url": (_env("OPENAGI_LLM_FALLBACK_BASE_URL")
                             or "https://openrouter.ai/api/v1").rstrip("/"),
                "model": _env("OPENAGI_LLM_FALLBACK_MODEL")
                         or "nvidia/nemotron-3-super-120b-a12b:free",
                "api_key": fb_key,
            })
        return providers

    def chat(self, messages, tools=None, temperature=0.0, json_mode=False, max_retries=4):
        """Return (content, tool_calls) where tool_calls is
        [{"name": ..., "parameters": {...}}, ...] or None.

        Retries with backoff on rate limits (429) and transient server
        errors (500/502/503/529) — the free Groq tier rate-limits easily.
        If every retry fails and a fallback provider is configured
        (OPENROUTER_API_KEY / OPENAGI_LLM_FALLBACK_*), the request is
        re-sent against the fallback instead of failing the agent run.
        """
        import time

        import requests

        last_err = None
        for pidx, pcfg in enumerate(self._providers()):
            payload = {
                "model": pcfg["model"],
                "messages": messages,
                "temperature": temperature,
            }
            if tools:
                payload["tools"] = tools
            if json_mode:
                payload["response_format"] = {"type": "json_object"}

            backoff = 5
            resp = None
            for attempt in range(max_retries + 1):
                try:
                    resp = requests.post(
                        f"{pcfg['base_url']}/chat/completions",
                        headers={"Authorization": f"Bearer {pcfg['api_key']}"},
                        json=payload,
                        timeout=self.timeout,
                    )
                except requests.RequestException as e:
                    resp = None
                    last_err = e
                    break
                if resp.status_code == 200:
                    break
                if resp.status_code in (429, 500, 502, 503, 529) and attempt < max_retries:
                    wait = backoff
                    try:  # honour the server's Retry-After if present
                        wait = max(wait, float(resp.headers.get("Retry-After", 0)))
                    except ValueError:
                        pass
                    print(f"[llm] {resp.status_code} from API, retrying in {wait:.0f}s "
                          f"(attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait)
                    backoff = min(backoff * 2, 60)
                    continue
                last_err = RuntimeError(
                    f"LLM request failed ({resp.status_code}): {resp.text[:500]}")
                break
            if resp is None or resp.status_code != 200:
                label = "primary" if pidx == 0 else "fallback"
                print(f"[llm] provider {label} ({pcfg['base_url']}) failed: {last_err}")
                continue
            if pidx > 0:
                print(f"[llm] fell back to {pcfg['base_url']} ({pcfg['model']})")
            message = resp.json()["choices"][0]["message"]
            tool_calls = None
            if message.get("tool_calls"):
                tool_calls = []
                for tc in message["tool_calls"]:
                    fn = tc.get("function", {})
                    try:
                        arguments = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    tool_calls.append({"name": fn.get("name"), "parameters": arguments})
            return message.get("content") or "", tool_calls
        raise last_err if last_err else RuntimeError("LLM request failed for unknown reasons")


class MockLLM:
    """Deterministic LLM stand-in for tests and offline demos.

    respond() receives (messages, tools) and returns (content, tool_calls).
    Override or pass a script; by default it produces a minimal valid plan
    and then a plain answer.
    """

    def __init__(self, script=None):
        self.script = script or [
            '[{"message": "answer the task directly", "tool_use": []}]',
            None,  # None -> produce a plain final answer
        ]
        self.calls = 0

    @property
    def configured(self):
        return True

    def chat(self, messages, tools=None, temperature=0.0, json_mode=False):
        idx = min(self.calls, len(self.script) - 1)
        self.calls += 1
        entry = self.script[idx]
        if entry is None:
            last = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "task")
            return f"[MockLLM] Final answer for: {last}", None
        if isinstance(entry, tuple):
            return entry  # (content, tool_calls)
        return entry, None


_llm = None


def get_llm():
    """Process-wide LLM client for standalone execution mode."""
    global _llm
    if _llm is None:
        _llm = StandaloneLLM()
    return _llm


def set_llm(client):
    """Inject an LLM client (used by tests and CLIs)."""
    global _llm
    _llm = client
