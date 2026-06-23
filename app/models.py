from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


@dataclass
class Message:
    sender: str
    content: str
    recipient: str = "all"
    kind: str = "agent"
    id: str = field(default_factory=lambda: new_id("msg"))
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Agent:
    name: str
    purpose: str
    task: str
    input: str
    input_format: str = ""
    expected_output_format: str = ""
    logic: str = ""
    deliverable: str = ""
    depends_on: list[str] = field(default_factory=list)
    parallel_group: int = 0
    status: str = "queued"
    output: str = ""
    id: str = field(default_factory=lambda: new_id("agent"))
    messages: list[Message] = field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["messages"] = [message.to_dict() for message in self.messages]
        return data


@dataclass
class Run:
    prompt: str
    id: str = field(default_factory=lambda: new_id("run"))
    status: str = "planning"
    summary: str = ""
    framework: str = "llamaindex"
    framework_note: str = ""
    project_path: str = ""
    selected_project: str = ""
    project_mode: str = "new"
    repo_summary: str = ""
    code_index_summary: str = ""
    relationship_summary: str = ""
    modified_files: dict[str, list[str]] = field(default_factory=dict)
    agents: list[Agent] = field(default_factory=list)
    transcript: list[Message] = field(default_factory=list)
    artifacts: list[dict[str, str]] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def touch(self) -> None:
        self.updated_at = now_iso()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["agents"] = [agent.to_dict() for agent in self.agents]
        data["transcript"] = [message.to_dict() for message in self.transcript]
        return data
