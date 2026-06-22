from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_OPENAI_MODEL = "gpt-5.2"


class LLMConfigError(RuntimeError):
    """Raised when the real model-backed runtime is not configured."""


class LLMRequestError(RuntimeError):
    """Raised when the OpenAI API call fails."""


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def configured_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL


def configured_reasoning_effort() -> str:
    return os.getenv("OPENAI_REASONING_EFFORT", "low").strip() or "low"


def openai_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def require_openai_config() -> None:
    if not openai_configured():
        raise LLMConfigError("OPENAI_API_KEY is required. Copy .env.example to .env and set your key.")


def ask_openai(system_prompt: str, user_prompt: str, timeout: int = 60) -> str:
    require_openai_config()

    body = {
        "model": configured_model(),
        "input": [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_output_tokens": 1600,
        "reasoning": {"effort": configured_reasoning_effort()},
        "text": {"verbosity": "low"},
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = _read_error_detail(exc)
        raise LLMRequestError(f"OpenAI API error {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LLMRequestError(f"OpenAI API request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LLMRequestError("OpenAI API returned invalid JSON.") from exc

    if isinstance(payload.get("output_text"), str):
        text = payload["output_text"].strip()
        if text:
            return text

    chunks: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    text = "\n".join(chunks).strip()
    if not text:
        incomplete = payload.get("incomplete_details")
        if isinstance(incomplete, dict) and incomplete.get("reason"):
            raise LLMRequestError(f"OpenAI API response did not include text output: {incomplete['reason']}.")
        raise LLMRequestError("OpenAI API response did not include text output.")
    return text


def _read_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8")
        payload = json.loads(raw)
    except Exception:
        return exc.reason or "request failed"

    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    return raw[:500] if raw else "request failed"
