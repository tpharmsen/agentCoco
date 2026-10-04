from typing import Literal

from langgraph.graph import MessagesState

Route = Literal["chat", "coding", "movies", "web"]


class AgentState(MessagesState):
    route: Route | None
