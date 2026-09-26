from collections.abc import Callable
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from ..config import Settings
from ..model import create_chat_model
from ..state import AgentState
from ..tools.tavily import TavilyClient
from ..tools.tmdb import TMDBClient
from ..tools.workspace import WorkspaceTool


def _model(settings: Settings):
    return create_chat_model(settings)


def chat_node(state: AgentState, settings: Settings) -> dict:
    response = _model(settings).invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are the general conversation assistant. Have a helpful, natural "
                    "conversation. Do not use external tools, browse the web, access files, "
                    "or invent current facts. If the user needs one of those capabilities, "
                    "explain that they can ask for it directly."
                ),
            },
            *state["messages"],
        ]
    )
    return {"messages": [AIMessage(content=response.content)]}


def coding_node(
    state: AgentState,
    settings: Settings,
    approve_command: Callable[[str, Path], bool] | None = None,
) -> dict:
    workspace = WorkspaceTool(
        settings.coding_workspace_root,
        settings.command_timeout_seconds,
        approve_command=approve_command,
    )

    @tool
    def read_file(relative_path: str) -> str:
        """Read a UTF-8 text file relative to the coding workspace."""
        return workspace.read_file(relative_path)

    @tool
    def write_file(relative_path: str, content: str) -> str:
        """Write a UTF-8 text file relative to the coding workspace."""
        return workspace.write_file(relative_path, content)

    @tool
    def run_command(command: str) -> str:
        """Run one command in the coding workspace with the configured timeout."""
        return workspace.run_command(command)

    prompt = (
        "You are the coding specialist. Work only under the configured workspace. "
        "Use the available workspace operations when the request requires file changes or commands. "
        "Report exactly what you changed or ran.\n\n"
        f"User request:\n{state['messages'][-1].content}"
    )
    model = _model(settings).bind_tools([read_file, write_file, run_command])
    messages = [*state["messages"], AIMessage(content=prompt)]
    for _ in range(8):
        response = model.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return {"messages": [AIMessage(content=response.content)]}
        for call in response.tool_calls:
            tool_map = {
                "read_file": read_file,
                "write_file": write_file,
                "run_command": run_command,
            }
            result = tool_map[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    raise RuntimeError("Coding agent exceeded its tool-call limit")


def movies_node(state: AgentState, settings: Settings) -> dict:
    results = TMDBClient(settings.tmdb_api_key).search_movies(state["messages"][-1].content)
    prompt = (
        "You are the movie specialist. Recommend movies using these TMDB results. "
        "Be explicit when the results are limited and include release years.\n\n"
        f"Results:\n{results}\n\nUser request:\n{state['messages'][-1].content}"
    )
    response = _model(settings).invoke(prompt)
    return {"messages": [AIMessage(content=response.content)]}


def web_node(state: AgentState, settings: Settings) -> dict:
    results = TavilyClient(settings.tavily_api_key).search(state["messages"][-1].content)
    prompt = (
        "You are the web research specialist. Answer from the search results and cite each "
        "important claim with its source URL.\n\n"
        f"Results:\n{results}\n\nUser request:\n{state['messages'][-1].content}"
    )
    response = _model(settings).invoke(prompt)
    return {"messages": [AIMessage(content=response.content)]}
