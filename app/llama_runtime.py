from __future__ import annotations

import asyncio
from contextlib import suppress
from inspect import isawaitable
import re
from typing import Any
from collections.abc import Callable, Sequence

import httpx
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

from app.llm import configured_model, configured_reasoning_effort, require_openai_config


ToolFunction = Callable[..., str]
AGENT_MAX_ITERATIONS = 80


def run_llama_agent(
    name: str,
    instructions: str,
    input_text: str,
    tools: Sequence[ToolFunction] | None = None,
    max_tokens: int = 1600,
    timeout: int = 120,
) -> str:
    require_openai_config()
    tool_list = list(tools or [])
    llm_kwargs = {
        "model": configured_model(),
        "api_key": None,
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "timeout": float(timeout),
    }
    if not tool_list:
        reasoning_effort = _llama_reasoning_effort()
        if reasoning_effort:
            llm_kwargs["reasoning_effort"] = reasoning_effort
    async def _run() -> str:
        http_client = httpx.Client(timeout=float(timeout))
        async_http_client = httpx.AsyncClient(timeout=float(timeout))
        llm: Any | None = None
        try:
            llm = OpenAI(
                **llm_kwargs,
                http_client=http_client,
                async_http_client=async_http_client,
            )
            agent = FunctionAgent(
                name=_safe_agent_name(name),
                description=f"{name}: {instructions[:180]}",
                system_prompt=instructions,
                tools=tool_list,
                llm=llm,
                streaming=False,
                timeout=float(timeout),
                allow_parallel_tool_calls=False,
            )
            try:
                response = await agent.run(user_msg=input_text, max_iterations=AGENT_MAX_ITERATIONS)
            except TypeError as exc:
                if "max_iterations" not in str(exc):
                    raise
                response = await agent.run(user_msg=input_text)
            return str(response or "").strip()
        finally:
            await _close_llm_async_resources(llm)
            await _close_async_http_client(async_http_client)
            with suppress(Exception):
                http_client.close()
            # Let Windows transports process connection-lost callbacks before asyncio.run closes the loop.
            await asyncio.sleep(0.05)

    output = asyncio.run(_run())
    if not output:
        raise RuntimeError(f"{name} did not produce output.")
    return output


def _safe_agent_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_")
    return cleaned[:64] or "Agent"


def _llama_reasoning_effort() -> str | None:
    effort = configured_reasoning_effort().lower()
    return effort if effort in {"none", "minimal", "low", "medium", "high", "xhigh"} else None


async def _close_llm_async_resources(llm: Any | None) -> None:
    if llm is None:
        return
    for attr in ("_aclient", "_client"):
        client = getattr(llm, attr, None)
        close = getattr(client, "close", None)
        if close is None:
            continue
        with suppress(Exception):
            result = close()
            if isawaitable(result):
                await result


async def _close_async_http_client(client: httpx.AsyncClient) -> None:
    with suppress(Exception):
        await client.aclose()
