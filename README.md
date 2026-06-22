# Multi-Agent Coding Prototype

A real multi-agent project builder for coding tasks. It has:

- A ChatGPT-style web UI with one request box.
- A dynamic workflow planner that creates named agents per request.
- A DAG workflow view that groups agents by dependency level.
- An Agents dashboard showing each agent's task, input, output, messages, and status.
- Explicit agent contracts: input format, expected output format, and logic are shown per agent.
- Inter-agent messages and a "nudge" action so agents can talk to each other.
- LlamaIndex orchestration with a root orchestrator that estimates complexity, optionally creates domain sub-orchestrators, then assigns specialist agents before project building, verification, and review.
- Specialist agents run as LlamaIndex FunctionAgents with scoped tools that can write real project files.
- Real project generation under `generated_projects/`.
- Static verification for generated Python and JavaScript files, plus stdlib Python unittest execution when tests exist.
- A framework profile endpoint showing the active LlamaIndex runtime.

Creating a run requires `OPENAI_API_KEY`; the generated source tree is written to disk.

## Run

```powershell
uv run python -m app.main
```

Open http://127.0.0.1:8000.

If you do not use `uv`, install the dependencies from `pyproject.toml` into your active environment first, then run:

```powershell
python -m app.main
```

## OpenAI Configuration

Copy `.env.example` to `.env`, keep it private, then set:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5.2
OPENAI_REASONING_EFFORT=low
```

The app refuses to start a workflow without `OPENAI_API_KEY`; there is no deterministic fallback.

## Framework Choice

The active runtime is `llamaindex`: the app uses LlamaIndex `FunctionAgent` agents for orchestration and specialist coding work. The root orchestrator first estimates complexity, estimated file count, and whether the request needs domain sub-orchestrators. Small and medium tasks use direct specialists. Large multi-domain tasks can use sub-orchestrators such as frontend, backend, or QA planners; each sub-orchestrator returns its own specialist plan.

Specialist plans can include `depends_on` relationships. The runtime turns those relationships into a DAG, runs each dependency level in order, and runs agents in the same level concurrently. Specialists and the Project Builder use scoped file tools to read, write, append, and list files under `generated_projects/`.

The hierarchy is intentionally bounded: sub-orchestrators may create specialists, but specialists do not create more agents.

The file tools reject absolute paths and parent-directory traversal, so agents can create real projects without writing outside the generated project folder. See [docs/FRAMEWORK_DECISION.md](docs/FRAMEWORK_DECISION.md).

The LlamaIndex dependency is already in `pyproject.toml`; to refresh it manually:

```powershell
uv add llama-index llama-index-llms-openai
```

Then set this in `.env`:

```text
AGENT_FRAMEWORK=llamaindex
```

The app exposes the active/recommended runtime at `GET /api/framework`.

## MCP-Style Tool Server

Run the stdio tool server:

```powershell
uv run python tools/mcp_agent_tool.py
```

It supports:

- `design_workflow`: returns the agents the orchestrator would create for a request.
- `agent_message`: creates a structured inter-agent message payload.
- `framework_profile`: reports the active and recommended framework runtime.
- `create_project`: runs the LlamaIndex multi-agent workflow and writes a generated project folder.

This is intentionally minimal so it can be embedded in a local MCP client later.

## Tests

```powershell
uv run python -m unittest discover -s tests
```
