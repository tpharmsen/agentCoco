from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from coco_agents.config import Settings
from coco_agents.graph import _route
from coco_agents.tools.tmdb import TMDBClient
from coco_agents.tools.workspace import WorkspaceTool


def test_workspace_rejects_paths_outside_root(tmp_path: Path) -> None:
    tool = WorkspaceTool(tmp_path, 5)
    with pytest.raises(ValueError, match="outside"):
        tool.read_file("../secret.txt")


def test_workspace_writes_and_reads_relative_files(tmp_path: Path) -> None:
    tool = WorkspaceTool(tmp_path, 5)
    tool.write_file("nested/note.txt", "hello")
    assert tool.read_file("nested/note.txt") == "hello"


def test_workspace_denies_commands_without_approval(tmp_path: Path) -> None:
    tool = WorkspaceTool(tmp_path, 5)
    with pytest.raises(PermissionError, match="human approval"):
        tool.run_command("echo should-not-run")


def test_workspace_runs_command_after_approval(tmp_path: Path) -> None:
    tool = WorkspaceTool(tmp_path, 5, approve_command=lambda command, root: True)
    assert tool.run_command("echo approved") == "approved"


def test_workspace_skips_rejected_command(tmp_path: Path) -> None:
    tool = WorkspaceTool(tmp_path, 5, approve_command=lambda command, root: False)
    assert "rejected" in tool.run_command("echo should-not-run")


def test_supervisor_accepts_chat_route(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def invoke(self, messages):
            return type("Response", (), {"content": "chat"})()

    monkeypatch.setattr("coco_agents.graph.create_chat_model", lambda settings: FakeModel())
    settings = Settings(
        _env_file=None,
        model_provider="ollama",
        model_name="test",
        tmdb_api_key="test",
        tavily_api_key="test",
    )

    result = _route({"messages": [HumanMessage(content="Tell me a joke")], "route": None}, settings)

    assert result == {"route": "chat"}


def test_tmdb_search_sends_authenticated_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"results": [{"title": "The Matrix"}]}

    def fake_get(url, *, params, headers, timeout):
        captured.update(url=url, params=params, headers=headers, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr("coco_agents.tools.tmdb.httpx.get", fake_get)

    results = TMDBClient("tmdb-test-key").search_movies("science fiction")

    assert results == [{"title": "The Matrix"}]
    assert captured == {
        "url": "https://api.themoviedb.org/3/search/movie",
        "params": {
            "query": "science fiction",
            "language": "en-US",
        },
        "headers": {
            "Authorization": "Bearer tmdb-test-key",
            "accept": "application/json",
        },
        "timeout": 15,
    }


def test_tmdb_supports_discover_and_movie_detail_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"results": [{"id": 550}]}

    def fake_get(url, *, params, headers, timeout):
        requests.append((url, params, headers, timeout))
        return FakeResponse()

    monkeypatch.setattr("coco_agents.tools.tmdb.httpx.get", fake_get)
    client = TMDBClient("tmdb-test-key")

    assert client.discover_movies(primary_release_year=2024, sort_by="vote_average.desc") == [
        {"id": 550}
    ]
    assert client.movie_details(550) == [{"id": 550}]
    assert requests[0][0].endswith("/discover/movie")
    assert requests[0][1]["primary_release_year"] == 2024
    assert requests[1][0].endswith("/movie/550")
    assert (
        requests[0][2]
        == requests[1][2]
        == {
            "Authorization": "Bearer tmdb-test-key",
            "accept": "application/json",
        }
    )
