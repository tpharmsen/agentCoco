# Copilot instructions

## Repository purpose

This repository contains a Python LangGraph application with a supervisor that routes each user request to one specialist. The model provider is selected through environment configuration:

- `MODEL_PROVIDER`: `openai`, `anthropic`, `google`, `mistral`, `ollama`, or `openai-compatible`
- `MODEL_NAME`: provider-specific model identifier
- `MODEL_BASE_URL`: optional endpoint for OpenAI-compatible providers

- `coding`: reads and modifies files below `CODING_WORKSPACE_ROOT` and can run commands in that workspace.
- `movies`: queries TMDB through the `TMDB_API_KEY`-authenticated client.
- `web`: searches the internet through Tavily using `TAVILY_API_KEY`.
- `chat`: handles normal conversation without external tools.

The graph is intentionally a single-process orchestration layer. Specialists return their answer to the supervisor; they do not call one another directly. Keep external integrations behind tool modules so the graph and agents remain testable.

## Project layout

- `src/coco_agents/graph.py` builds and compiles the LangGraph.
- `src/coco_agents/model.py` is the provider-neutral chat-model factory.
- `src/coco_agents/terminal.py` owns CLI colors and disables ANSI output for non-interactive streams or `NO_COLOR`.
- `src/coco_agents/state.py` defines the shared message state and routing state.
- `src/coco_agents/agents/` contains specialist prompts and node functions.
- `src/coco_agents/tools/` contains filesystem/command, TMDB, and Tavily integrations.
- `src/coco_agents/config.py` loads environment configuration and validates required settings.
- `tests/` contains focused unit tests for routing and tool safety.

## Commands

Install the package and development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Use the existing local `.env` file, select a model provider, install its optional adapter, and provide the required credentials before running the application. `.env` is ignored by Git and must not be committed:

```bash
.venv/bin/pip install -e ".[all-providers]"
.venv/bin/python -m coco_agents
```

Run the full test suite:

```bash
pytest
```

Run one test or one test module:

```bash
pytest tests/test_tools.py
pytest tests/test_tools.py::test_workspace_rejects_paths_outside_root
```

Run formatting and linting:

```bash
ruff format .
ruff check .
```

## Architecture and conventions

- Use `create_graph()` as the public graph factory; do not construct a compiled graph at import time.
- Use `create_chat_model()` for all LLM construction. Do not import provider SDK wrappers directly in graph or agent nodes.
- Provider-specific features must be optional dependencies and selected lazily in `model.py`; the default installation must remain usable with the configured provider.
- The `chat` route must remain tool-free; do not attach filesystem, TMDB, or web tools to it.
- Command execution is human-in-the-loop: `WorkspaceTool` must require an approval callback and deny execution when one is absent. The CLI supplies an interactive callback; never add an unattended default.
- Nodes communicate through `AgentState`. Preserve the existing `messages` history and append an `AIMessage` for specialist results.
- Keep routing labels limited to the `Route` literal in `state.py`. If a new specialist is added, update the literal, supervisor prompt, graph edges, and tests together.
- Agent nodes should be thin orchestration functions. Put network calls, filesystem access, and subprocess execution in `tools/`.
- External tools must have explicit timeouts and raise useful errors. Do not swallow failures or return success-shaped fallback data.
- The coding workspace is a security boundary: resolve paths against `CODING_WORKSPACE_ROOT`, reject paths that escape it, and preserve the command timeout. Never broaden this boundary implicitly.
- Keep API keys in environment variables. Do not log prompts, credentials, raw API responses, or file contents by default.
- TMDB requests use `/search/movie` and `/movie/{id}` through the shared client; keep attribution and source URLs in movie responses.
- Web answers should retain Tavily result URLs so the user can inspect sources.
- Add or update focused tests when changing routing, workspace safety, or an external tool contract.
