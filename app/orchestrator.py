from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from threading import Lock
from typing import Any

from app.llm import LLMConfigError, LLMRequestError, ask_openai
from app.models import Agent, Message, Run, now_iso
from app.project_runtime import ProjectRuntimeError, project_agent_plan, run_project_workflow


RUN_DELAY_SECONDS = 0.25
ModelClient = Callable[[str, str], str]


class WorkflowError(RuntimeError):
    """Raised when the real workflow cannot be planned or executed."""


def design_workflow(prompt: str, model_client: ModelClient | None = None) -> list[Agent]:
    if model_client is None:
        return project_agent_plan(prompt)
    model = model_client or ask_openai
    plan_text = model(_planner_system_prompt(), _planner_user_prompt(prompt))
    return _parse_agent_plan(plan_text, prompt)


def run_workflow(
    run: Run,
    on_change: Callable[[Run], None] | None = None,
    delay: float = RUN_DELAY_SECONDS,
    model_client: ModelClient | None = None,
) -> None:
    model = model_client or ask_openai

    def emit(message: Message, agent: Agent | None = None) -> None:
        run.transcript.append(message)
        if agent is not None:
            agent.messages.append(message)
        run.touch()
        if on_change:
            on_change(run)

    if model_client is None:
        try:
            run_project_workflow(run, emit)
        except (LLMConfigError, LLMRequestError, ProjectRuntimeError, ValueError) as exc:
            _fail_run(run, str(exc), emit)
        except Exception as exc:
            _fail_run(run, f"Agents SDK workflow failed: {exc}", emit)
        return

    try:
        run.status = "planning"
        run.agents = design_workflow(run.prompt, model)
        emit(
            Message(
                sender="Orchestrator",
                recipient="all",
                content=f"Planned a real {len(run.agents)} agent workflow for: {run.prompt}",
                kind="system",
            )
        )
        time.sleep(delay)

        run.status = "running"
        previous_outputs: list[str] = []
        for index, agent in enumerate(run.agents):
            agent.status = "running"
            agent.started_at = now_iso()
            emit(
                Message(
                    sender=agent.name,
                    content=f"Starting: {agent.task}",
                    recipient="Orchestrator",
                    kind="status",
                ),
                agent,
            )
            time.sleep(delay)

            agent.output = _run_agent(agent, run.prompt, previous_outputs, model)
            previous_outputs.append(f"{agent.name}: {agent.output}")
            agent.status = "done"
            agent.finished_at = now_iso()
            emit(Message(sender=agent.name, recipient="all", content=agent.output), agent)

            if index + 1 < len(run.agents):
                next_agent = run.agents[index + 1]
                handoff_text = _handoff(agent, next_agent, model)
                handoff = Message(
                    sender=agent.name,
                    recipient=next_agent.name,
                    content=handoff_text,
                    kind="handoff",
                )
                next_agent.messages.append(handoff)
                emit(handoff, agent)

            time.sleep(delay)

        run.status = "complete"
        run.summary = _summary(run, model)
        run.artifacts = [
            {
                "name": "workflow.json",
                "description": "Structured run state available through the API.",
            },
            {
                "name": "agent-transcript.md",
                "description": "The live transcript shown in the chat timeline.",
            },
        ]
        emit(Message(sender="Orchestrator", recipient="all", content=run.summary, kind="summary"))
    except (LLMConfigError, LLMRequestError, WorkflowError, ValueError) as exc:
        _fail_run(run, str(exc), emit)


def manual_agent_message(
    run: Run,
    sender_id: str,
    recipient_id: str,
    content: str,
    model_client: ModelClient | None = None,
) -> list[Message]:
    agent_by_id = {agent.id: agent for agent in run.agents}
    sender = agent_by_id.get(sender_id)
    recipient = agent_by_id.get(recipient_id)
    if sender is None or recipient is None:
        raise ValueError("Unknown sender or recipient agent.")

    user_message = Message(sender=sender.name, recipient=recipient.name, content=content, kind="manual")
    sender.messages.append(user_message)
    recipient.messages.append(user_message)
    run.transcript.append(user_message)
    run.touch()

    messages = [user_message]
    if model_client is not None:
        reply = _agent_reply_to_message(run, recipient, sender, content, model_client)
        reply_message = Message(sender=recipient.name, recipient=sender.name, content=reply, kind="agent-chat")
        sender.messages.append(reply_message)
        recipient.messages.append(reply_message)
        run.transcript.append(reply_message)
        run.touch()
        messages.append(reply_message)

    return messages


def _planner_system_prompt() -> str:
    return (
        "You are a senior software orchestrator for a multi-agent coding workflow. "
        "Return only valid JSON. Do not wrap it in markdown. The JSON schema is: "
        '{"agents":[{"name":"Agent name","purpose":"Why this agent exists",'
        '"task":"The concrete task it owns","input":"The context it receives"}]}. '
        "Create 3 to 5 specialized agents. The first agent must be Orchestrator. "
        "Use focused coding roles such as planner, interface designer, backend engineer, "
        "implementation agent, reviewer, documentation agent, or test engineer when relevant."
    )


def _planner_user_prompt(prompt: str) -> str:
    return f"User request:\n{prompt}\n\nDesign the agent workflow for this coding task."


def _parse_agent_plan(plan_text: str, prompt: str) -> list[Agent]:
    payload = _extract_json_object(plan_text)
    raw_agents = payload.get("agents")
    if not isinstance(raw_agents, list) or not raw_agents:
        raise WorkflowError("Planner did not return an agents list.")

    agents: list[Agent] = []
    for index, raw_agent in enumerate(raw_agents):
        if not isinstance(raw_agent, dict):
            raise WorkflowError("Planner returned an invalid agent entry.")
        name = _clean_required(raw_agent.get("name"), "agent name")
        purpose = _clean_required(raw_agent.get("purpose"), f"{name} purpose")
        task = _clean_required(raw_agent.get("task"), f"{name} task")
        agent_input = _clean_required(raw_agent.get("input"), f"{name} input")
        if index == 0 and name.lower() != "orchestrator":
            name = "Orchestrator"
        agents.append(Agent(name=name, purpose=purpose, task=task, input=agent_input or prompt))

    return agents


def _run_agent(agent: Agent, user_prompt: str, previous_outputs: list[str], model: ModelClient) -> str:
    context = "\n\n".join(previous_outputs) if previous_outputs else "No previous agent output yet."
    system_prompt = (
        f"You are {agent.name}, a specialist in a real multi-agent coding workflow. "
        f"Purpose: {agent.purpose}. Own your task and produce concrete work product. "
        "Do not pretend files were edited unless your output is explicitly a proposed patch or plan. "
        "Return no more than 180 words. Be concise, specific, and hand off useful details to the next agent."
    )
    user_prompt_text = (
        f"Original user request:\n{user_prompt}\n\n"
        f"Your assigned task:\n{agent.task}\n\n"
        f"Your input:\n{agent.input}\n\n"
        f"Previous agent outputs:\n{context}"
    )
    return _clean_required(model(system_prompt, user_prompt_text), f"{agent.name} output")


def _handoff(current_agent: Agent, next_agent: Agent, model: ModelClient) -> str:
    system_prompt = (
        "You write concise handoff notes between specialist coding agents. "
        "Return one or two sentences with the key context and constraints the next agent needs."
    )
    user_prompt = (
        f"Current agent: {current_agent.name}\n"
        f"Current output:\n{current_agent.output}\n\n"
        f"Next agent: {next_agent.name}\n"
        f"Next task: {next_agent.task}"
    )
    return _clean_required(model(system_prompt, user_prompt), "handoff")


def _agent_reply_to_message(
    run: Run,
    recipient: Agent,
    sender: Agent,
    content: str,
    model: ModelClient,
) -> str:
    transcript = "\n".join(f"{message.sender} -> {message.recipient}: {message.content}" for message in run.transcript[-10:])
    system_prompt = (
        f"You are {recipient.name} in a multi-agent coding workflow. "
        f"Purpose: {recipient.purpose}. Reply to another agent with actionable context."
    )
    user_prompt = (
        f"Original request:\n{run.prompt}\n\n"
        f"Your task:\n{recipient.task}\n\n"
        f"Recent transcript:\n{transcript}\n\n"
        f"{sender.name} says:\n{content}"
    )
    return _clean_required(model(system_prompt, user_prompt), f"{recipient.name} reply")


def _summary(run: Run, model: ModelClient) -> str:
    outputs = "\n\n".join(f"{agent.name}: {agent.output}" for agent in run.agents)
    system_prompt = (
        "You are the final orchestrator. Summarize what the agent team completed, "
        "what remains uncertain, and what the user should inspect next. Keep it under 120 words."
    )
    user_prompt = f"Original request:\n{run.prompt}\n\nAgent outputs:\n{outputs}"
    return _clean_required(model(system_prompt, user_prompt), "workflow summary")


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise WorkflowError("Planner did not return valid JSON.") from None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise WorkflowError("Planner returned malformed JSON.") from exc

    if not isinstance(payload, dict):
        raise WorkflowError("Planner JSON must be an object.")
    return payload


def _clean_required(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise WorkflowError(f"Missing {label}.")
    return text


def _fail_run(run: Run, error: str, emit: Callable[[Message, Agent | None], None]) -> None:
    run.status = "failed"
    run.summary = error
    for agent in run.agents:
        if agent.status == "running":
            agent.status = "failed"
            agent.finished_at = now_iso()
    emit(Message(sender="System", recipient="all", content=error, kind="error"))


class RunStore:
    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}
        self._lock = Lock()

    def add(self, run: Run) -> None:
        with self._lock:
            self._runs[run.id] = run

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            return self._runs.get(run_id)

    def list(self) -> list[Run]:
        with self._lock:
            return sorted(self._runs.values(), key=lambda item: item.created_at, reverse=True)

    def update(self, run: Run) -> None:
        with self._lock:
            self._runs[run.id] = run
