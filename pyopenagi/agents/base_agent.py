import os

import json

from .agent_process import (
    AgentProcess
)

import time

from threading import Thread

from ..utils.logger import AgentLogger

from ..utils.chat_template import Query

import importlib

try:
    from aios.hooks.stores._global import global_llm_req_queue_add_message
except Exception:  # AIOS kernel not installed -> standalone execution mode
    global_llm_req_queue_add_message = None

class CustomizedThread(Thread):
    def __init__(self, target, args=()):
        super().__init__()
        self.target = target
        self.args = args
        self.result = None

    def run(self):
        self.result = self.target(*self.args)

    def join(self):
        super().join()
        return self.result

class BaseAgent:
    def __init__(self,
                 agent_name,
                 task_input,
                 agent_process_factory,
                 log_mode: str
        ):

        self.agent_name = agent_name
        self.config = self.load_config()
        self.tool_names = self.config["tools"]

        self.agent_process_factory = agent_process_factory

        self.tool_list = dict()
        self.tools = []
        self.tool_info = [] # simplified information of the tool: {"name": "xxx", "description": "xxx"}

        self.load_tools(self.tool_names)

        self.start_time = None
        self.end_time = None
        self.request_waiting_times: list = []
        self.request_turnaround_times: list = []
        self.task_input = task_input
        self.messages = []
        self.workflow_mode = "manual" # (mannual, automatic)
        self.rounds = 0

        self.log_mode = log_mode
        self.logger = self.setup_logger()
        self.logger.log("Initialized. \n", level="info")

        self.set_status("active")
        self.set_created_time(time.time())


    def run(self):
        '''Execute each step to finish the task.'''
        pass

    # can be customization
    def build_system_instruction(self):
        pass

    def check_workflow(self, message):
        try:
            # print(f"Workflow message: {message}")
            workflow = json.loads(message)
            if not isinstance(workflow, list):
                return None

            for step in workflow:
                if "message" not in step or "tool_use" not in step:
                    return None

            return workflow

        except json.JSONDecodeError:
            return None

    def automatic_workflow(self):
        for i in range(self.plan_max_fail_times):
            response, start_times, end_times, waiting_times, turnaround_times = self.get_response(
                query = Query(
                    messages = self.messages,
                    tools = None,
                    message_return_type="json"
                )
            )

            if self.rounds == 0:
                self.set_start_time(start_times[0])

            self.request_waiting_times.extend(waiting_times)
            self.request_turnaround_times.extend(turnaround_times)

            workflow = self.check_workflow(response.response_message)

            self.rounds += 1

            if workflow:
                return workflow

            else:
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": f"Fail {i+1} times to generate a valid plan. I need to regenerate a plan"
                    }
                )
        return None

    def manual_workflow(self):
        pass

    def check_path(self, tool_calls):
        script_path = os.path.abspath(__file__)
        save_dir = os.path.join(os.path.dirname(script_path), "output") # modify the customized output path for saving outputs
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for tool_call in tool_calls:
            try:
                for k in tool_call["parameters"]:
                    if "path" in k:
                        path = tool_call["parameters"][k]
                        if not path.startswith(save_dir):
                            tool_call["parameters"][k] = os.path.join(save_dir, os.path.basename(path))
            except Exception:
                continue
        return tool_calls

    def snake_to_camel(self, snake_str):
        components = snake_str.split('_')
        return ''.join(x.title() for x in components)

    def load_tools(self, tool_names):

        for tool_name in tool_names:
            org, name = tool_name.split("/")
            module_name = ".".join(["pyopenagi", "tools", org, name])
            class_name = self.snake_to_camel(name)
            tool_module = importlib.import_module(module_name)
            tool_class = getattr(tool_module, class_name)
            self.tool_list[name] = tool_class()
            tool_format = tool_class().get_tool_call_format()
            self.tools.append(tool_format)
            self.tool_info.append(
                {"name": tool_format["function"]["name"], "description": tool_format["function"]["description"]}
            )

    def pre_select_tools(self, tool_names):
        pre_selected_tools = []
        for tool_name in tool_names:
            for tool in self.tools:
                if tool["function"]["name"] == tool_name:
                    pre_selected_tools.append(tool)
                    break

        return pre_selected_tools

    def setup_logger(self):
        logger = AgentLogger(self.agent_name, self.log_mode)
        return logger

    def load_config(self):
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        config_file = os.path.join(script_dir, self.agent_name, "config.json")
        with open(config_file, "r") as f:
            config = json.load(f)
            return config

    # the default method used for getting response from AIOS
    def get_response(self,
            query,
            temperature=0.0
        ):

        from ..core.llm import use_aios

        if use_aios():
            thread = CustomizedThread(target=self.query_loop, args=(query, ))
            thread.start()
            return thread.join()
        return self.query_loop_standalone(query)

    @staticmethod
    def _trim_context(messages, max_tokens=None):
        """Trim the middle of a long conversation to fit token limits.

        Keeps the head (system prompt + original task) and the most recent
        messages. Free Groq tiers cap requests at 8000 TPM, so long agent
        runs need this. Override the cap with OPENAGI_MAX_CONTEXT_TOKENS.
        """
        import os as _os

        if max_tokens is None:
            try:
                max_tokens = int(_os.environ.get("OPENAGI_MAX_CONTEXT_TOKENS", 6000))
            except ValueError:
                max_tokens = 6000

        def est(msgs):
            return sum(len(str(m.get("content", ""))) for m in msgs) // 4

        if est(messages) <= max_tokens or len(messages) <= 4:
            return messages

        head = messages[:2]
        tail = messages[-6:]
        notice = [{"role": "user", "content": "[earlier steps trimmed to fit the model context]"}]
        trimmed = head + notice + tail
        while est(trimmed) > max_tokens and len(tail) > 1:
            tail = tail[:-1]
            trimmed = head + notice + tail
        return trimmed

    def query_loop_standalone(self, query):
        """Standalone execution: call an OpenAI-compatible LLM directly.

        Used when the AIOS kernel is not available. Returns the same tuple
        shape as query_loop so ReactAgent and CallCore work unchanged.
        """
        from ..core.llm import get_llm, ChatResult

        agent_process = self.create_agent_request(query)

        llm = get_llm()
        if not llm.configured:
            raise RuntimeError(
                "No LLM API key configured for standalone execution. "
                "Set OPENAGI_LLM_API_KEY (or GROQ_API_KEY / OPENAI_API_KEY), "
                "and optionally OPENAGI_LLM_BASE_URL / OPENAGI_LLM_MODEL."
            )

        start_time = time.time()
        content, tool_calls = llm.chat(
            messages=self._trim_context(query.messages),
            tools=query.tools,
            temperature=0.0,
            json_mode=(query.message_return_type == "json"),
        )
        end_time = time.time()

        response = ChatResult(content, tool_calls)
        agent_process.set_response(response)
        agent_process.set_status("done")
        agent_process.set_start_time(start_time)
        agent_process.set_end_time(end_time)

        waiting_time = 0.0
        turnaround_time = end_time - start_time
        return (
            agent_process.get_response(),
            [start_time],
            [end_time],
            [waiting_time],
            [turnaround_time],
        )

    def query_loop(self, query):
        agent_process = self.create_agent_request(query)

        completed_response, start_times, end_times, waiting_times, turnaround_times = "", [], [], [], []

        while agent_process.get_status() != "done":
            thread = Thread(target=self.listen, args=(agent_process, ))
            current_time = time.time()
            # reinitialize agent status
            agent_process.set_created_time(current_time)
            agent_process.set_response(None)

            global_llm_req_queue_add_message(agent_process)

            # LLMRequestQueue.add_message(agent_process)

            thread.start()
            thread.join()

            completed_response = agent_process.get_response()
            if agent_process.get_status() != "done":
                self.logger.log(
                    f"Suspended due to the reach of time limit ({agent_process.get_time_limit()}s). Current result is: {completed_response.response_message}\n",
                    level="suspending"
                )
            start_time = agent_process.get_start_time()
            end_time = agent_process.get_end_time()
            waiting_time = start_time - agent_process.get_created_time()
            turnaround_time = end_time - agent_process.get_created_time()

            start_times.append(start_time)
            end_times.append(end_time)
            waiting_times.append(waiting_time)
            turnaround_times.append(turnaround_time)
            # Re-start the thread if not done

        # self.agent_process_factory.deactivate_agent_process(agent_process.get_pid())

        return completed_response, start_times, end_times, waiting_times, turnaround_times

    def create_agent_request(self, query):
        agent_process = self.agent_process_factory.activate_agent_process(
            agent_name = self.agent_name,
            query = query
        )
        agent_process.set_created_time(time.time())
        # print("Already put into the queue")
        return agent_process

    def listen(self, agent_process: AgentProcess):
        """Response Listener for agent

        Args:
            agent_process (AgentProcess): Listened AgentProcess

        Returns:
            str: LLM response of Agent Process
        """
        while agent_process.get_response() is None:
            time.sleep(0.2)

        return agent_process.get_response()

    def set_aid(self, aid):
        self.aid = aid

    def get_aid(self):
        return self.aid

    def get_agent_name(self):
        return self.agent_name

    def set_status(self, status):

        """
        Status type: Waiting, Running, Done, Inactive
        """
        self.status = status

    def get_status(self):
        return self.status

    def set_created_time(self, time):
        self.created_time = time

    def get_created_time(self):
        return self.created_time

    def set_start_time(self, time):
        self.start_time = time

    def get_start_time(self):
        return self.start_time

    def set_end_time(self, time):
        self.end_time = time

    def get_end_time(self):
        return self.end_time
