# Framework Decision

## Choice

Use LlamaIndex as the active framework for the real multi-agent project builder.

## Why

The requirement is to move beyond chat simulation and JSON-only file manifests. The app requires `OPENAI_API_KEY`; a LlamaIndex root orchestrator estimates project complexity, expected file count, and whether the request needs domain sub-orchestrators. For small and medium work, it creates direct specialists. For larger multi-domain work, it creates bounded sub-orchestrators such as frontend, backend, data, or QA planners. Those sub-orchestrators return specialist-agent plans, including each agent's input format, expected output format, and logic.

The final specialist set is scheduled as a DAG. Specialist plans can include `depends_on` relationships; the runtime runs dependency levels in order and runs agents in the same level concurrently. Each specialist is a real LlamaIndex `FunctionAgent` with scoped file tools that can create, read, append, and list files inside one generated project folder.

This keeps the UI-visible workflow explicit while letting agents actually write code. The Project Builder also runs as a LlamaIndex agent with the same scoped tools, then the verifier performs local static checks plus Python `unittest` discovery when tests exist.

## Current Implementation

- Active runtime: `llamaindex`
- Model default: `gpt-5.2`
- Required configuration: `OPENAI_API_KEY`
- Generated project root: `generated_projects/`
- Pipeline shape: LlamaIndex root complexity estimate -> optional domain sub-orchestrators -> specialist-agent dependency DAG -> parallel scoped file writes by DAG level -> Project Builder completion -> Verification Agent -> Review Agent
- Framework profile endpoint: `/api/framework`
- MCP-style tool: `framework_profile`

Installed framework:

```powershell
uv add llama-index llama-index-llms-openai
```

The current app records the active runtime on each run and displays it in the Agents dashboard.

## Sources

- LlamaIndex multi-agent patterns: https://developers.llamaindex.ai/python/framework/understanding/agent/multi_agent/
- LlamaIndex FunctionAgent and AgentWorkflow basics: https://developers.llamaindex.ai/python/examples/agent/agent_workflow_basic/
