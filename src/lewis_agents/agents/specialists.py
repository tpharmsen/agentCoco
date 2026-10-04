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
    client = TMDBClient(settings.tmdb_api_key)

    @tool
    def search_movies(query: str) -> list[dict]:
        """Search TMDB movies by title or keyword."""
        return client.search_movies(query)

    @tool
    def discover_movies(
        genre_id: int | None = None,
        year: int | None = None,
        sort_by: str = "popularity.desc",
        min_vote_average: float | None = None,
    ) -> list[dict]:
        """Discover movies using optional genre, year, rating, and sort filters."""
        filters: dict[str, str | int | float] = {"sort_by": sort_by}
        if genre_id is not None:
            filters["with_genres"] = genre_id
        if year is not None:
            filters["primary_release_year"] = year
        if min_vote_average is not None:
            filters["vote_average.gte"] = min_vote_average
        return client.discover_movies(**filters)

    @tool
    def movie_details(movie_id: int) -> dict:
        """Get full details for a TMDB movie ID."""
        return client.movie_details(movie_id)

    @tool
    def recommendations(movie_id: int) -> list[dict]:
        """Get TMDB recommendations based on a movie ID."""
        return client.recommendations(movie_id)

    @tool
    def similar_movies(movie_id: int) -> list[dict]:
        """Get movies similar to a TMDB movie ID."""
        return client.similar_movies(movie_id)

    @tool
    def credits(movie_id: int) -> dict:
        """Get cast and crew for a TMDB movie ID."""
        return client.credits(movie_id)

    @tool
    def watch_providers(movie_id: int) -> dict:
        """Get streaming providers by region for a TMDB movie ID."""
        return client.watch_providers(movie_id)

    @tool
    def videos(movie_id: int) -> list[dict]:
        """Get trailers and other videos for a TMDB movie ID."""
        return client.videos(movie_id)

    @tool
    def reviews(movie_id: int) -> list[dict]:
        """Get TMDB reviews for a movie ID."""
        return client.reviews(movie_id)

    @tool
    def genres() -> list[dict]:
        """List TMDB movie genres and IDs."""
        return client.genres()

    tools = [
        search_movies,
        discover_movies,
        movie_details,
        recommendations,
        similar_movies,
        credits,
        watch_providers,
        videos,
        reviews,
        genres,
    ]
    model = _model(settings).bind_tools(tools)
    messages = [
        {
            "role": "system",
            "content": (
                "You are the movie specialist. Use TMDB tools to ground every factual movie "
                "answer. Search for titles before using an ID-based endpoint when necessary. "
                "Use discover for filters, recommendations or similar_movies for related films, "
                "credits for cast/crew, watch_providers for streaming availability, videos for "
                "trailers, reviews for reviews, and genres for genre IDs. Do not invent TMDB "
                "results. State when TMDB returns no matches."
            ),
        },
        *state["messages"],
    ]
    for _ in range(8):
        response = model.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return {"messages": [AIMessage(content=response.content)]}
        tool_map = {item.name: item for item in tools}
        for call in response.tool_calls:
            result = tool_map[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    raise RuntimeError("Movie agent exceeded its tool-call limit")


def web_node(state: AgentState, settings: Settings) -> dict:
    results = TavilyClient(settings.tavily_api_key).search(state["messages"][-1].content)
    prompt = (
        "You are the web research specialist. Answer from the search results and cite each "
        "important claim with its source URL.\n\n"
        f"Results:\n{results}\n\nUser request:\n{state['messages'][-1].content}"
    )
    response = _model(settings).invoke(prompt)
    return {"messages": [AIMessage(content=response.content)]}
