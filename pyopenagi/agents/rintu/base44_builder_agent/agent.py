from ...react_agent import ReactAgent


class Base44BuilderAgent(ReactAgent):
    """Drafts new Base44 SaaS apps in Rintu's portfolio style.

    Given an idea (or just a problem), it produces a concrete build plan:
    concept, entity schema, pages, workflows, backend functions and a
    build order — ready to hand to the Base44 builder.
    """

    def __init__(self, agent_name, task_input, agent_process_factory, log_mode: str):
        ReactAgent.__init__(self, agent_name, task_input, agent_process_factory, log_mode)
        self.workflow_mode = "automatic"

    def run(self):
        return super().run()
