from __future__ import annotations

import unittest
from unittest.mock import patch

from app.llama_runtime import run_llama_agent


class FakeFunctionAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def run(self, user_msg: str) -> str:
        return f"handled {user_msg}"


class LlamaRuntimeTests(unittest.TestCase):
    def test_tool_enabled_agents_omit_reasoning_effort(self) -> None:
        captured: dict[str, object] = {}

        def fake_openai(**kwargs):
            captured["llm_kwargs"] = kwargs
            return object()

        def fake_agent(**kwargs):
            captured["agent_kwargs"] = kwargs
            return FakeFunctionAgent(**kwargs)

        def sample_tool() -> str:
            """Sample tool."""
            return "ok"

        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-5.5", "OPENAI_REASONING_EFFORT": "low"},
            clear=True,
        ):
            with patch("app.llama_runtime.OpenAI", fake_openai):
                with patch("app.llama_runtime.FunctionAgent", fake_agent):
                    output = run_llama_agent("Tool Agent", "Use tools.", "input", tools=[sample_tool])

        self.assertEqual(output, "handled input")
        self.assertNotIn("reasoning_effort", captured["llm_kwargs"])
        self.assertTrue(captured["llm_kwargs"]["http_client"].is_closed)
        self.assertTrue(captured["llm_kwargs"]["async_http_client"].is_closed)
        self.assertEqual(captured["agent_kwargs"]["tools"], [sample_tool])

    def test_toolless_agents_keep_reasoning_effort(self) -> None:
        captured: dict[str, object] = {}

        def fake_openai(**kwargs):
            captured["llm_kwargs"] = kwargs
            return object()

        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-5.5", "OPENAI_REASONING_EFFORT": "low"},
            clear=True,
        ):
            with patch("app.llama_runtime.OpenAI", fake_openai):
                with patch("app.llama_runtime.FunctionAgent", lambda **kwargs: FakeFunctionAgent(**kwargs)):
                    output = run_llama_agent("Planner", "Plan.", "input")

        self.assertEqual(output, "handled input")
        self.assertEqual(captured["llm_kwargs"]["reasoning_effort"], "low")
        self.assertTrue(captured["llm_kwargs"]["http_client"].is_closed)
        self.assertTrue(captured["llm_kwargs"]["async_http_client"].is_closed)


if __name__ == "__main__":
    unittest.main()
