from ...react_agent import ReactAgent


class SaasIdeaAgent(ReactAgent):
    """SaaS product ideation partner: ideas, validation, MVP scope, GTM."""

    def __init__(self, agent_name, task_input, agent_process_factory, log_mode: str):
        ReactAgent.__init__(self, agent_name, task_input, agent_process_factory, log_mode)
        self.workflow_mode = "automatic"

    def run(self):
        return super().run()
