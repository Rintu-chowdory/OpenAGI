"""OpenAGI Personal Edition — command line interface.

Run your local agents from the terminal:

    openagi run rintu/dev_ops_agent "review my GitHub Actions pipeline"
    openagi list                 # list agents available locally
    openagi tools                # list available tool modules
    openagi doctor               # check LLM configuration
"""

import json
import os
import sys

import click

from .agents.agent_process import AgentProcessFactory
from .core import llm as llm_core


def _local_agents():
    """List (author, name) for all agents found locally under pyopenagi/agents."""
    base = os.path.dirname(os.path.abspath(__file__))
    agents_dir = os.path.join(base, "agents")
    agents = []
    if not os.path.isdir(agents_dir):
        return agents
    for author in sorted(os.listdir(agents_dir)):
        author_dir = os.path.join(agents_dir, author)
        if not os.path.isdir(author_dir) or author.startswith(("_", ".")):
            continue
        for name in sorted(os.listdir(author_dir)):
            name_dir = os.path.join(author_dir, name)
            if os.path.isfile(os.path.join(name_dir, "agent.py")):
                agents.append((author, name))
    return agents


def _local_tools():
    """List org/name for all tool modules found locally under pyopenagi/tools."""
    base = os.path.dirname(os.path.abspath(__file__))
    tools_dir = os.path.join(base, "tools")
    tools = []
    if not os.path.isdir(tools_dir):
        return tools
    for org in sorted(os.listdir(tools_dir)):
        org_dir = os.path.join(tools_dir, org)
        if not os.path.isdir(org_dir) or org.startswith(("_", ".")):
            continue
        for name in sorted(os.listdir(org_dir)):
            if name.endswith(".py") and not name.startswith("__"):
                tools.append(f"{org}/{name[:-3]}")
    return tools


@click.group()
@click.version_option(package_name="pyopenagi", prog_name="openagi")
def main():
    """OpenAGI Personal Edition — build and run agents locally."""
    pass


@main.command()
@click.argument("agent_name")  # e.g. rintu/dev_ops_agent
@click.argument("task")
@click.option("--log-mode", default="console", type=click.Choice(["console", "file"]), help="Where agent logs go.")
@click.option("--json", "as_json", is_flag=True, help="Print the raw result dict instead of plain text.")
def run(agent_name, task, log_mode, as_json):
    """Run a local agent on a task. AGENT_NAME is author/agent_name."""
    from .agents.agent_factory import AgentFactory

    if "/" not in agent_name:
        raise click.UsageError('AGENT_NAME must be in "author/agent_name" format, e.g. rintu/dev_ops_agent')

    author, name = agent_name.split("/", 1)
    if (author, name) not in _local_agents():
        available = "\n".join(f"  - {a}/{n}" for a, n in _local_agents())
        raise click.UsageError(
            f'Agent "{agent_name}" was not found locally.\n'
            f"Available agents:\n{available or '  (none)'}"
        )

    if not llm_core.use_aios():
        client = llm_core.get_llm()
        if not client.configured:
            raise click.ClickException(
                "No LLM API key found. Set OPENAGI_LLM_API_KEY (or GROQ_API_KEY / OPENAI_API_KEY).\n"
                "For Groq: export OPENAGI_LLM_BASE_URL=https://api.groq.com/openai/v1 "
                "OPENAGI_LLM_MODEL=llama-3.3-70b-versatile"
            )

    factory = AgentFactory(agent_process_factory=AgentProcessFactory(), agent_log_mode=log_mode)
    agent_class = factory.load_agent_instance(agent_name)
    agent = agent_class(agent_name, task, AgentProcessFactory(), log_mode)
    result = agent.run()

    if not result:
        raise click.ClickException("Agent run failed (see logs above).")

    if as_json:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        click.echo("")
        click.secho("─" * 60, dim=True)
        click.secho("RESULT", bold=True)
        content = result.get("result")
        if isinstance(content, dict) and "content" in content:
            content = content["content"]
        click.echo(content)
        click.secho("─" * 60, dim=True)
        click.secho(
            f"rounds: {result.get('rounds')} · turnaround: "
            f"{(result.get('agent_turnaround_time') or 0):.2f}s",
            dim=True,
        )


@main.command(name="list")
def list_agents():
    """List locally available agents."""
    for author, name in _local_agents():
        click.echo(f"{author}/{name}")


@main.command()
def tools():
    """List locally available tool modules."""
    for tool in _local_tools():
        click.echo(tool)


@main.command()
def doctor():
    """Diagnose the execution configuration."""
    mode = llm_core.execution_mode()
    use = "aios kernel" if llm_core.use_aios() else "standalone (direct LLM)"
    click.echo(f"execution mode:  {mode} -> {use}")
    click.echo(f"aios installed:  {llm_core.aios_available()}")
    if not llm_core.use_aios():
        cfg = llm_core.llm_config()
        click.echo(f"base url:        {cfg['base_url']}")
        click.echo(f"model:          {cfg['model']}")
        click.echo(f"api key:        {'set ✓' if cfg['api_key'] else 'MISSING ✗'}")
        if not cfg["api_key"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
