from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.frameworks import current_framework_profile
from app.llm import LLMConfigError, LLMRequestError, load_dotenv
from app.models import Message, Run
from app.orchestrator import design_workflow, run_workflow


TOOLS = [
    {
        "name": "design_workflow",
        "description": "Create a named orchestrator/workers workflow for a coding request.",
        "inputSchema": {
            "type": "object",
            "properties": {"request": {"type": "string"}},
            "required": ["request"],
        },
    },
    {
        "name": "agent_message",
        "description": "Create a structured message from one agent to another.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sender": {"type": "string"},
                "recipient": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["sender", "recipient", "content"],
        },
    },
    {
        "name": "framework_profile",
        "description": "Describe the active and recommended multi-agent framework runtime.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "create_project",
        "description": "Run the real LlamaIndex multi-agent workflow and create a generated project folder.",
        "inputSchema": {
            "type": "object",
            "properties": {"request": {"type": "string"}},
            "required": ["request"],
        },
    },
]


def handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    request_id = request.get("id")

    if method == "notifications/initialized":
        return None
    if method == "initialize":
        return _response(
            request_id,
            {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "multi-agent-prototype", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            },
        )
    if method == "tools/list":
        return _response(request_id, {"tools": TOOLS})
    if method == "tools/call":
        params = request.get("params", {})
        return _response(request_id, _call_tool(params))

    return _error(request_id, -32601, f"Unknown method: {method}")


def _call_tool(params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments", {})
    if name == "design_workflow":
        request = str(arguments.get("request", "")).strip()
        try:
            agents = [agent.to_dict() for agent in design_workflow(request)]
        except (LLMConfigError, LLMRequestError, ValueError) as exc:
            return _text_result(str(exc), is_error=True)
        return _text_result(json.dumps({"agents": agents}, indent=2))

    if name == "agent_message":
        message = Message(
            sender=str(arguments.get("sender", "")),
            recipient=str(arguments.get("recipient", "")),
            content=str(arguments.get("content", "")),
            kind="mcp",
        )
        return _text_result(json.dumps(message.to_dict(), indent=2))

    if name == "framework_profile":
        return _text_result(json.dumps(current_framework_profile().to_dict(), indent=2))

    if name == "create_project":
        request = str(arguments.get("request", "")).strip()
        if not request:
            return _text_result("request is required", is_error=True)
        run = Run(prompt=request, framework=current_framework_profile().active)
        run_workflow(run, delay=0)
        return _text_result(json.dumps(run.to_dict(), indent=2), is_error=run.status == "failed")

    return _text_result(f"Unknown tool: {name}", is_error=True)


def _text_result(text: str, is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def main() -> None:
    load_dotenv()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = handle_request(request)
        except json.JSONDecodeError as exc:
            response = _error(None, -32700, str(exc))
        if response is not None:
            print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
