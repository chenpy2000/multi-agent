from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

from app.llm import configured_model, openai_configured


@dataclass(frozen=True)
class FrameworkProfile:
    active: str
    requested: str
    recommended: str
    available: bool
    adapter_ready: bool
    rationale: str
    install_hint: str
    docs: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def current_framework_profile() -> FrameworkProfile:
    requested = os.getenv("AGENT_FRAMEWORK", "llamaindex").strip().lower() or "llamaindex"
    active = "llamaindex"
    configured = openai_configured()
    rationale = (
        f"Real project generation is active through LlamaIndex FunctionAgent workflows using {configured_model()}. "
        "The orchestrator designs request-specific specialist agents, delegates work through LlamaIndex tools, "
        "specialists write scoped files under generated_projects/, and the verifier runs local checks."
    )

    return FrameworkProfile(
        active=active,
        requested=requested,
        recommended="llamaindex",
        available=configured,
        adapter_ready=True,
        rationale=rationale,
        install_hint="uv add llama-index llama-index-llms-openai; copy .env.example to .env and set OPENAI_API_KEY.",
        docs=[
            {
                "name": "LlamaIndex multi-agent patterns",
                "url": "https://developers.llamaindex.ai/python/framework/understanding/agent/multi_agent/",
            },
            {
                "name": "LlamaIndex FunctionAgent",
                "url": "https://developers.llamaindex.ai/python/examples/agent/agent_workflow_basic/",
            },
        ],
    )
