from collections.abc import Callable
from pathlib import Path

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph

from .agents.specialists import chat_node, coding_node, movies_node, web_node
from .config import Settings, get_settings
from .model import create_chat_model
from .state import AgentState, Route


def _route(state: AgentState, settings: Settings) -> dict:
    classifier = create_chat_model(settings)
    prompt = SystemMessage(
        content=(
            "Classify the user's request as exactly one route: chat, coding, movies, or web. "
            "Use chat for casual conversation, brainstorming, explanations, or questions "
            "that do not need tools or current information; chat must not call external tools. "
            "Use coding for files, programming, and command execution; movies for film "
            "recommendations or TMDB facts; web for general current internet research. "
            "Return only the route word."
        )
    )
    result = classifier.invoke([prompt, *state["messages"]])
    route = result.content.strip().lower()
    if route not in {"chat", "coding", "movies", "web"}:
        raise ValueError(f"Supervisor returned invalid route: {route!r}")
    return {"route": route}


def _next(state: AgentState) -> Route:
    route = state.get("route")
    if route is None:
        raise ValueError("Supervisor did not set a route")
    return route


def create_graph(
    settings: Settings | None = None,
    approve_command: Callable[[str, Path], bool] | None = None,
):
    settings = settings or get_settings()
    builder = StateGraph(AgentState)
    builder.add_node("supervisor", lambda state: _route(state, settings))
    builder.add_node("chat", lambda state: chat_node(state, settings))
    builder.add_node("coding", lambda state: coding_node(state, settings, approve_command))
    builder.add_node("movies", lambda state: movies_node(state, settings))
    builder.add_node("web", lambda state: web_node(state, settings))
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        _next,
        {"chat": "chat", "coding": "coding", "movies": "movies", "web": "web"},
    )
    builder.add_edge("chat", END)
    builder.add_edge("coding", END)
    builder.add_edge("movies", END)
    builder.add_edge("web", END)
    return builder.compile()
