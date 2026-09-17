from ...react_agent import ReactAgent


class SecurityAuditAgent(ReactAgent):
    """Internal-audit and security-advisor agent: controls, governance, checklists."""

    def __init__(self, agent_name, task_input, agent_process_factory, log_mode: str):
        ReactAgent.__init__(self, agent_name, task_input, agent_process_factory, log_mode)
        self.workflow_mode = "automatic"

    def run(self):
        return super().run()
