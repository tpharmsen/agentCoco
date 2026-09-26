# Coco agents

Coco is a small LangGraph supervisor that routes requests to four specialists:

- **Chat**: handles normal conversation without calling external tools.
- **Coding**: works in a configured documents workspace and can run commands there.
- **Movies**: searches TMDB for movie data and recommendations.
- **Web**: searches the internet using Tavily and preserves source URLs.

## Setup

```bash
python -m pip install -e ".[dev]"
python -m coco_agents
```

In an interactive terminal, your input prompt is white `>>>` and agent responses are bold blue. Color is
automatically disabled when output is redirected; set `NO_COLOR=1` to disable it explicitly.

Use the existing local `.env` file for the selected provider's credentials, `TMDB_API_KEY`,
and `TAVILY_API_KEY`. It is ignored by Git and must not be committed.
Supported model providers are:

- `openai` (default)
- `anthropic`
- `google` (Gemini)
- `mistral`
- `ollama` (local)
- `openai-compatible` (Groq, Together, DeepSeek, or another OpenAI-compatible endpoint; set `MODEL_BASE_URL`)

Install only the provider adapter you need, or all adapters:

```bash
.venv/bin/pip install -e ".[anthropic]"
.venv/bin/pip install -e ".[all-providers]"
```

The coding specialist is limited to `CODING_WORKSPACE_ROOT`; choose that directory deliberately.
Every command proposed by the coding agent requires interactive approval in the CLI. Command
execution is denied by default for non-interactive callers unless they pass an explicit
`approve_command` callback to `create_graph`.

Movie requests use TMDB's authenticated `/3/search/movie` endpoint. Set `TMDB_API_KEY` to a
valid TMDB v3 API key; the movie agent sends the user's request as the search query and uses
the returned movie records to compose its response.

## Development

```bash
pytest
pytest tests/test_tools.py::test_workspace_rejects_paths_outside_root
ruff format .
ruff check .
```

The graph is assembled by `coco_agents.graph.create_graph`. Keep integrations in `coco_agents.tools` and specialist orchestration in `coco_agents.agents`.
