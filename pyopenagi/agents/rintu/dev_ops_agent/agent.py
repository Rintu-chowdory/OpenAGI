from ...react_agent import ReactAgent


class DevOpsAgent(ReactAgent):
    """Personal DevOps engineer: pipelines, containers, deployments."""

    def __init__(self, agent_name, task_input, agent_process_factory, log_mode: str):
        ReactAgent.__init__(self, agent_name, task_input, agent_process_factory, log_mode)
        self.workflow_mode = "automatic"

    def run(self):
        return super().run()
