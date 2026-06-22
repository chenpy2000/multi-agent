from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from threading import Event, Lock
from unittest.mock import patch

from app.frameworks import current_framework_profile
from app.models import Run
from app.orchestrator import design_workflow, manual_agent_message, run_workflow
from app.project_runtime import ProjectRuntimeError, ProjectWorkspace


def fake_model(system_prompt: str, user_prompt: str) -> str:
    if "Return only valid JSON" in system_prompt:
        return json.dumps(
            {
                "agents": [
                    {
                        "name": "Orchestrator",
                        "purpose": "Plan the coding workflow.",
                        "task": "Define the agent sequence.",
                        "input": "The user request.",
                    },
                    {
                        "name": "Interface Designer",
                        "purpose": "Design the UI.",
                        "task": "Create the chat and agents dashboard interaction plan.",
                        "input": "The workflow plan.",
                    },
                    {
                        "name": "Systems Engineer",
                        "purpose": "Design APIs and state.",
                        "task": "Specify endpoints and agent state transitions.",
                        "input": "The UI workflow.",
                    },
                    {
                        "name": "Implementation Agent",
                        "purpose": "Implement the plan.",
                        "task": "Build the smallest maintainable version.",
                        "input": "The full plan.",
                    },
                ]
            }
        )
    if "final orchestrator" in system_prompt:
        return "The real agent team completed the workflow and left concrete outputs to inspect."
    if "handoff notes" in system_prompt:
        return "Use the previous output as direct context and preserve the stated constraints."
    if "Reply to another agent" in system_prompt:
        return "I will use that constraint in my next step."
    return "Model-backed agent output for this assigned task."


def _call_tool(tools: list, name: str, *args: str) -> str:
    for tool in tools or []:
        if getattr(tool, "__name__", "") == name:
            return tool(*args)
    raise AssertionError(f"Missing tool: {name}")


def _run_all_specialist_tools(tools: list) -> None:
    for tool in tools or []:
        if getattr(tool, "__name__", "").startswith("run_"):
            tool(f"Complete {tool.__name__} and write real project files.")


def fake_tiny_llama_agent(
    name: str,
    instructions: str,
    input_text: str,
    max_tokens: int,
    tools: list | None = None,
    timeout: int = 120,
) -> str:
    if name == "Orchestrator":
        return json.dumps(
            {
                "project_name": "tiny-tool",
                "project_summary": "A tiny Python utility requested from chat.",
                "acceptance_criteria": ["Generate a Python module", "Include unittest coverage"],
                "agents": [
                    {
                        "name": "Requirements Analyst",
                        "purpose": "Translate the chat request into concrete behavior.",
                        "task": "Define the user-facing behavior and edge cases.",
                        "input": "Original request.",
                        "input_format": "Plain text user request.",
                        "expected_output_format": "README behavior notes.",
                        "logic": "Extract required behavior and edge cases from the request.",
                        "deliverable": "Behavior spec for the builder.",
                    },
                    {
                        "name": "Python Engineer",
                        "purpose": "Build the Python implementation.",
                        "task": "Create the module and callable function.",
                        "input": "Requirements Analyst output.",
                        "input_format": "Workflow plan plus requirements.",
                        "expected_output_format": "Python source file.",
                        "logic": "Map requirements into a small callable Python module.",
                        "deliverable": "Implementation files.",
                    },
                    {
                        "name": "Test Engineer",
                        "purpose": "Define verification coverage.",
                        "task": "Create unittest cases for the generated module.",
                        "input": "Requirements and implementation notes.",
                        "input_format": "Workflow plan plus Python Engineer output.",
                        "expected_output_format": "unittest file.",
                        "logic": "Cover the core function behavior with unittest.",
                        "deliverable": "Test files.",
                    },
                ],
            }
        )
    if name == "Orchestrator Coordinator":
        _run_all_specialist_tools(tools or [])
        return "Coordinated Requirements Analyst, Python Engineer, and Test Engineer."
    if name == "Requirements Analyst":
        _call_tool(tools or [], "write_file", "README.md", "# Tiny Tool\n\nReturns ok.\n")
        return "Defined behavior and wrote README.md."
    if name == "Python Engineer":
        _call_tool(tools or [], "write_file", "tiny_tool.py", "def main():\n    return 'ok'\n")
        return "Implemented tiny_tool.py."
    if name == "Test Engineer":
        _call_tool(
            tools or [],
            "write_file",
            "tests/test_tiny_tool.py",
            (
                "import unittest\n"
                "from tiny_tool import main\n\n"
                "class TinyToolTests(unittest.TestCase):\n"
                "    def test_main(self):\n"
                "        self.assertEqual(main(), 'ok')\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        )
        return "Created unittest coverage."
    if name == "Project Builder":
        _call_tool(tools or [], "read_file", "tiny_tool.py")
        return "Inspected and completed the tiny Python project."
    if name == "Review Agent":
        return "Generated files were reviewed and static verification passed."
    return f"{name} output"


def fake_tic_tac_toe_llama_agent(
    name: str,
    instructions: str,
    input_text: str,
    max_tokens: int,
    tools: list | None = None,
    timeout: int = 120,
) -> str:
    if name == "Orchestrator":
        return json.dumps(
            {
                "project_name": "tic-tac-toe-browser-game",
                "project_summary": "A playable browser Tic Tac Toe game.",
                "acceptance_criteria": [
                    "Render a 3x3 board",
                    "Alternate X and O turns",
                    "Detect wins and draws",
                    "Provide a reset control",
                ],
                "agents": [
                    {
                        "name": "Gameplay Engineer",
                        "purpose": "Own game state and rules.",
                        "task": "Implement turn handling, legal moves, win detection, draw detection, and reset.",
                        "input": "Browser game request.",
                        "input_format": "Workflow plan and user request.",
                        "expected_output_format": "Executable JavaScript gameplay source.",
                        "logic": "Model the board, winning lines, current player, status, and reset behavior.",
                        "deliverable": "app.js with gameplay logic.",
                    },
                    {
                        "name": "Interface Designer",
                        "purpose": "Own browser UI and styling.",
                        "task": "Create accessible HTML structure and CSS for the game.",
                        "input": "Gameplay requirements.",
                        "input_format": "Workflow plan plus gameplay notes.",
                        "expected_output_format": "index.html and style.css.",
                        "logic": "Render a board, status text, and reset button with responsive styling.",
                        "deliverable": "Static browser UI files.",
                    },
                    {
                        "name": "Quality Engineer",
                        "purpose": "Review executable behavior.",
                        "task": "Document manual verification scenarios.",
                        "input": "Generated game files.",
                        "input_format": "Current file tree and acceptance criteria.",
                        "expected_output_format": "README and testing notes.",
                        "logic": "Check win, draw, invalid move, and reset scenarios.",
                        "deliverable": "README.md with test scenarios.",
                    },
                ],
            }
        )
    if name == "Orchestrator Coordinator":
        _run_all_specialist_tools(tools or [])
        return "Distributed gameplay, interface, and quality work across specialist agents."
    if name == "Gameplay Engineer":
        _call_tool(tools or [], "write_file", "generated_projects/tic-tac-toe-browser-game/app.js", TIC_TAC_TOE_JS)
        return "Implemented executable Tic Tac Toe state, turn, win, draw, and reset logic."
    if name == "Interface Designer":
        _call_tool(tools or [], "write_file", "generated_projects/tic-tac-toe-browser-game/index.html", TIC_TAC_TOE_HTML)
        _call_tool(tools or [], "write_file", "generated_projects/tic-tac-toe-browser-game/style.css", TIC_TAC_TOE_CSS)
        return "Created the browser UI, board, status area, reset button, and styling."
    if name == "Quality Engineer":
        _call_tool(tools or [], "write_file", "TESTING.md", TIC_TAC_TOE_TESTING)
        return "Documented gameplay verification scenarios."
    if name == "Project Builder":
        files = json.loads(_call_tool(tools or [], "list_files"))
        if "README.md" not in files:
            _call_tool(tools or [], "write_file", "README.md", TIC_TAC_TOE_README)
        return "Inspected generated files and completed README.md."
    if name == "Review Agent":
        return "The Tic Tac Toe browser game is executable and verification passed."
    return f"{name} output"


def fake_hierarchical_llama_agent(
    name: str,
    instructions: str,
    input_text: str,
    max_tokens: int,
    tools: list | None = None,
    timeout: int = 120,
) -> str:
    if name == "Orchestrator":
        return json.dumps(
            {
                "project_name": "large-browser-game-suite",
                "project_summary": "A larger browser game project that needs domain planning.",
                "complexity": "large",
                "estimated_files": 14,
                "requires_sub_orchestrators": True,
                "acceptance_criteria": [
                    "Create a playable browser Tic Tac Toe game",
                    "Include UI, game logic, documentation, and verification notes",
                ],
                "sub_orchestrators": [
                    {
                        "name": "Frontend Sub-Orchestrator",
                        "domain": "frontend",
                        "purpose": "Plan browser UI and interaction specialists.",
                        "task": "Split frontend work into gameplay and interface specialists.",
                        "input": "Browser game requirements.",
                        "input_format": "Root plan plus frontend domain assignment.",
                        "expected_output_format": "JSON agents list for frontend specialists.",
                        "logic": "Separate gameplay state from interface and styling work.",
                        "deliverable": "Frontend specialist plan.",
                    },
                    {
                        "name": "Quality Sub-Orchestrator",
                        "domain": "quality",
                        "purpose": "Plan quality and verification specialists.",
                        "task": "Split quality work into docs and gameplay checks.",
                        "input": "Acceptance criteria and generated file expectations.",
                        "input_format": "Root plan plus quality domain assignment.",
                        "expected_output_format": "JSON agents list for quality specialists.",
                        "logic": "Convert acceptance criteria into test and review artifacts.",
                        "deliverable": "Quality specialist plan.",
                    },
                ],
                "agents": [],
            }
        )
    if name == "Frontend Sub-Orchestrator":
        return json.dumps(
            {
                "agents": [
                    {
                        "name": "Gameplay Engineer",
                        "purpose": "Own game state and rules.",
                        "task": "Implement turn handling, legal moves, win detection, draw detection, and reset.",
                        "input": "Frontend domain plan.",
                        "input_format": "Root and frontend plans.",
                        "expected_output_format": "Executable JavaScript gameplay source.",
                        "logic": "Model board state, winning lines, current player, status, and reset behavior.",
                        "deliverable": "app.js with gameplay logic.",
                    },
                    {
                        "name": "Interface Designer",
                        "purpose": "Own browser UI and styling.",
                        "task": "Create accessible HTML structure and CSS for the game.",
                        "input": "Frontend domain plan.",
                        "input_format": "Root and frontend plans.",
                        "expected_output_format": "index.html and style.css.",
                        "logic": "Render a board, status text, and reset button with responsive styling.",
                        "deliverable": "Static browser UI files.",
                    },
                ]
            }
        )
    if name == "Quality Sub-Orchestrator":
        return json.dumps(
            {
                "agents": [
                    {
                        "name": "Quality Engineer",
                        "purpose": "Review executable behavior.",
                        "task": "Document manual verification scenarios.",
                        "input": "Quality domain plan.",
                        "input_format": "Acceptance criteria and current file expectations.",
                        "expected_output_format": "README and testing notes.",
                        "logic": "Check win, draw, invalid move, and reset scenarios.",
                        "deliverable": "README.md and TESTING.md.",
                    }
                ]
            }
        )
    if name == "Orchestrator Coordinator":
        _run_all_specialist_tools(tools or [])
        return "Coordinated specialists produced by frontend and quality sub-orchestrators."
    if name == "Gameplay Engineer":
        _call_tool(tools or [], "write_file", "app.js", TIC_TAC_TOE_JS)
        return "Implemented executable Tic Tac Toe state, turn, win, draw, and reset logic."
    if name == "Interface Designer":
        _call_tool(tools or [], "write_file", "index.html", TIC_TAC_TOE_HTML)
        _call_tool(tools or [], "write_file", "style.css", TIC_TAC_TOE_CSS)
        return "Created the browser UI, board, status area, reset button, and styling."
    if name == "Quality Engineer":
        _call_tool(tools or [], "write_file", "README.md", TIC_TAC_TOE_README)
        _call_tool(tools or [], "write_file", "TESTING.md", TIC_TAC_TOE_TESTING)
        return "Documented gameplay verification scenarios."
    if name == "Project Builder":
        _call_tool(tools or [], "read_file", "app.js")
        return "Inspected generated files and completed the game project."
    if name == "Review Agent":
        return "The hierarchical Tic Tac Toe project is executable and verification passed."
    return f"{name} output"


TIC_TAC_TOE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tic Tac Toe</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <main class="game" aria-labelledby="title">
    <h1 id="title">Tic Tac Toe</h1>
    <p id="status" aria-live="polite">Player X's turn</p>
    <section id="board" class="board" aria-label="Tic Tac Toe board"></section>
    <button id="reset" type="button">Reset game</button>
  </main>
  <script src="app.js"></script>
</body>
</html>
"""


TIC_TAC_TOE_CSS = """body {
  margin: 0;
  min-height: 100vh;
  display: grid;
  place-items: center;
  font-family: Arial, sans-serif;
  background: #f4f7fb;
  color: #172033;
}
.game {
  text-align: center;
}
.board {
  display: grid;
  grid-template-columns: repeat(3, 96px);
  gap: 8px;
  margin: 20px 0;
}
.cell {
  width: 96px;
  height: 96px;
  border: 2px solid #30415d;
  background: #ffffff;
  font-size: 42px;
  font-weight: 700;
}
button {
  cursor: pointer;
}
"""


TIC_TAC_TOE_JS = """const boardElement = document.querySelector('#board');
const statusElement = document.querySelector('#status');
const resetButton = document.querySelector('#reset');
const winningLines = [
  [0, 1, 2],
  [3, 4, 5],
  [6, 7, 8],
  [0, 3, 6],
  [1, 4, 7],
  [2, 5, 8],
  [0, 4, 8],
  [2, 4, 6],
];

let board = Array(9).fill('');
let currentPlayer = 'X';
let winner = null;
let draw = false;

function checkWinner() {
  for (const line of winningLines) {
    const [a, b, c] = line;
    if (board[a] && board[a] === board[b] && board[a] === board[c]) {
      return board[a];
    }
  }
  return null;
}

function updateStatus() {
  if (winner) {
    statusElement.textContent = `Player ${winner} wins!`;
  } else if (draw) {
    statusElement.textContent = 'Draw game.';
  } else {
    statusElement.textContent = `Player ${currentPlayer}'s turn`;
  }
}

function renderBoard() {
  boardElement.innerHTML = '';
  board.forEach((value, index) => {
    const cell = document.createElement('button');
    cell.type = 'button';
    cell.className = 'cell';
    cell.textContent = value;
    cell.disabled = Boolean(value || winner || draw);
    cell.setAttribute('aria-label', `Cell ${index + 1}${value ? ` occupied by ${value}` : ''}`);
    cell.addEventListener('click', () => handleCellClick(index));
    boardElement.appendChild(cell);
  });
  updateStatus();
}

function handleCellClick(index) {
  if (board[index] || winner || draw) {
    return;
  }
  board[index] = currentPlayer;
  winner = checkWinner();
  draw = !winner && board.every(Boolean);
  if (!winner && !draw) {
    currentPlayer = currentPlayer === 'X' ? 'O' : 'X';
  }
  renderBoard();
}

function resetGame() {
  board = Array(9).fill('');
  currentPlayer = 'X';
  winner = null;
  draw = false;
  renderBoard();
}

resetButton.addEventListener('click', resetGame);
renderBoard();
"""


TIC_TAC_TOE_README = """# Tic Tac Toe Browser Game

Open `index.html` in a browser to play a two-player Tic Tac Toe game.

The game alternates X and O turns, prevents illegal moves, detects wins and draws, and includes a reset button.
"""


TIC_TAC_TOE_TESTING = """# Testing

- X wins across a row, column, or diagonal.
- O wins across a row, column, or diagonal.
- A filled board with no winner shows a draw.
- Clicking an occupied cell does not change the board.
- Reset clears the board and returns to player X.
"""


class OrchestratorTests(unittest.TestCase):
    def test_default_workflow_creates_project_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("app.project_runtime.GENERATED_ROOT", Path(temp_dir)):
                with patch("app.project_runtime._run_llama_agent", fake_tiny_llama_agent):
                    run = Run(prompt="Create a tiny Python tool")
                    run_workflow(run, delay=0)

                self.assertEqual(run.status, "complete")
                self.assertTrue(run.project_path)
                project_dir = Path(run.project_path)
                self.assertTrue((project_dir / "README.md").exists())
                self.assertTrue((project_dir / "tiny_tool.py").exists())
                self.assertIn("Requirements Analyst", [agent.name for agent in run.agents])
                self.assertIn("Python Engineer", [agent.name for agent in run.agents])
                self.assertIn("Test Engineer", [agent.name for agent in run.agents])
                python_engineer = next(agent for agent in run.agents if agent.name == "Python Engineer")
                self.assertIn("Workflow plan", python_engineer.input_format)
                self.assertIn("Python source", python_engineer.expected_output_format)
                self.assertIn("Map requirements", python_engineer.logic)
                self.assertIn("tiny_tool.py", python_engineer.output)
                self.assertTrue(any(agent.name == "Verification Agent" and agent.status == "done" for agent in run.agents))
                verification_agent = next(agent for agent in run.agents if agent.name == "Verification Agent")
                self.assertIn("unittest", verification_agent.output)
                self.assertIn("Verification: passed", run.summary)

    def test_browser_tic_tac_toe_workflow_creates_executable_game(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("app.project_runtime.GENERATED_ROOT", Path(temp_dir)):
                with patch("app.project_runtime._run_llama_agent", fake_tic_tac_toe_llama_agent):
                    run = Run(prompt="Create a browser Tic Tac Toe game")
                    run_workflow(run, delay=0)

                self.assertEqual(run.status, "complete")
                project_dir = Path(run.project_path)
                self.assertTrue((project_dir / "index.html").exists())
                self.assertTrue((project_dir / "app.js").exists())
                self.assertTrue((project_dir / "style.css").exists())
                self.assertTrue((project_dir / "README.md").exists())

                agent_names = [agent.name for agent in run.agents]
                self.assertIn("Gameplay Engineer", agent_names)
                self.assertIn("Interface Designer", agent_names)
                self.assertIn("Quality Engineer", agent_names)

                html = (project_dir / "index.html").read_text(encoding="utf-8")
                js = (project_dir / "app.js").read_text(encoding="utf-8")
                self.assertIn('src="app.js"', html)
                for expected in ["winningLines", "currentPlayer", "checkWinner", "draw", "resetGame", "handleCellClick"]:
                    self.assertIn(expected, js)
                self.assertIn("Verification: passed", run.summary)
                verification_agent = next(agent for agent in run.agents if agent.name == "Verification Agent")
                self.assertIn("node --check", verification_agent.output)

    def test_large_workflow_uses_sub_orchestrators_before_specialists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("app.project_runtime.GENERATED_ROOT", Path(temp_dir)):
                with patch("app.project_runtime._run_llama_agent", fake_hierarchical_llama_agent):
                    run = Run(prompt="Create a larger browser game suite with UI, game logic, docs, and QA")
                    run_workflow(run, delay=0)

                self.assertEqual(run.status, "complete")
                project_dir = Path(run.project_path)
                self.assertTrue((project_dir / "index.html").exists())
                self.assertTrue((project_dir / "app.js").exists())
                self.assertTrue((project_dir / "style.css").exists())
                self.assertTrue((project_dir / "README.md").exists())

                agent_names = [agent.name for agent in run.agents]
                self.assertIn("Frontend Sub-Orchestrator", agent_names)
                self.assertIn("Quality Sub-Orchestrator", agent_names)
                self.assertIn("Gameplay Engineer", agent_names)
                self.assertIn("Interface Designer", agent_names)
                self.assertIn("Quality Engineer", agent_names)

                frontend = next(agent for agent in run.agents if agent.name == "Frontend Sub-Orchestrator")
                quality = next(agent for agent in run.agents if agent.name == "Quality Sub-Orchestrator")
                self.assertEqual(frontend.status, "done")
                self.assertEqual(quality.status, "done")
                self.assertIn("Specialists planned", frontend.output)
                self.assertIn("Specialists planned", quality.output)
                self.assertLess(agent_names.index("Frontend Sub-Orchestrator"), agent_names.index("Gameplay Engineer"))
                self.assertIn("Verification: passed", run.summary)

    def test_independent_specialists_run_in_parallel_before_dependents(self) -> None:
        spec_started = Event()
        ui_started = Event()
        timings: dict[str, float] = {}
        timing_lock = Lock()

        def fake_parallel_llama_agent(
            name: str,
            instructions: str,
            input_text: str,
            max_tokens: int,
            tools: list | None = None,
            timeout: int = 120,
        ) -> str:
            if name == "Orchestrator":
                return json.dumps(
                    {
                        "project_name": "parallel-dag-project",
                        "project_summary": "A project with parallel specialist work.",
                        "complexity": "medium",
                        "estimated_files": 4,
                        "requires_sub_orchestrators": False,
                        "acceptance_criteria": ["Run independent specialists before integration"],
                        "agents": [
                            {
                                "name": "Spec Writer",
                                "purpose": "Define behavior.",
                                "task": "Write the behavior specification.",
                                "input": "User request.",
                                "input_format": "Prompt and plan.",
                                "expected_output_format": "spec.md",
                                "logic": "Document behavior independently.",
                                "deliverable": "Specification file.",
                                "depends_on": [],
                            },
                            {
                                "name": "UI Engineer",
                                "purpose": "Create UI shell.",
                                "task": "Write the HTML shell.",
                                "input": "User request.",
                                "input_format": "Prompt and plan.",
                                "expected_output_format": "index.html",
                                "logic": "Create UI independently.",
                                "deliverable": "HTML file.",
                                "depends_on": [],
                            },
                            {
                                "name": "Integrator",
                                "purpose": "Join spec and UI into runnable behavior.",
                                "task": "Write JavaScript after spec and UI exist.",
                                "input": "Spec and UI outputs.",
                                "input_format": "Completed dependency files.",
                                "expected_output_format": "app.js",
                                "logic": "Read dependency files and create JavaScript.",
                                "deliverable": "JavaScript file.",
                                "depends_on": ["Spec Writer", "UI Engineer"],
                            },
                        ],
                    }
                )
            if name == "Spec Writer":
                with timing_lock:
                    timings["spec_start"] = time.perf_counter()
                spec_started.set()
                if not ui_started.wait(1):
                    raise AssertionError("Spec Writer did not run in parallel with UI Engineer.")
                _call_tool(tools or [], "write_file", "spec.md", "# Spec\n")
                with timing_lock:
                    timings["spec_end"] = time.perf_counter()
                return "Wrote spec.md."
            if name == "UI Engineer":
                with timing_lock:
                    timings["ui_start"] = time.perf_counter()
                ui_started.set()
                if not spec_started.wait(1):
                    raise AssertionError("UI Engineer did not run in parallel with Spec Writer.")
                _call_tool(tools or [], "write_file", "index.html", "<!doctype html><script src=\"app.js\"></script>\n")
                with timing_lock:
                    timings["ui_end"] = time.perf_counter()
                return "Wrote index.html."
            if name == "Integrator":
                with timing_lock:
                    timings["integrator_start"] = time.perf_counter()
                _call_tool(tools or [], "read_file", "spec.md")
                _call_tool(tools or [], "read_file", "index.html")
                _call_tool(tools or [], "write_file", "app.js", "console.log('ok');\n")
                return "Wrote app.js."
            if name == "Project Builder":
                _call_tool(tools or [], "read_file", "app.js")
                return "Inspected integrated project."
            if name == "Review Agent":
                return "Reviewed parallel DAG project."
            return f"{name} output"

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("app.project_runtime.GENERATED_ROOT", Path(temp_dir)):
                with patch("app.project_runtime._run_llama_agent", fake_parallel_llama_agent):
                    run = Run(prompt="Create a project that has independent UI and spec work before integration")
                    run_workflow(run, delay=0)

                self.assertEqual(run.status, "complete")
                project_dir = Path(run.project_path)
                self.assertTrue((project_dir / "spec.md").exists())
                self.assertTrue((project_dir / "index.html").exists())
                self.assertTrue((project_dir / "app.js").exists())
                spec_agent = next(agent for agent in run.agents if agent.name == "Spec Writer")
                ui_agent = next(agent for agent in run.agents if agent.name == "UI Engineer")
                integrator = next(agent for agent in run.agents if agent.name == "Integrator")
                self.assertEqual(spec_agent.parallel_group, ui_agent.parallel_group)
                self.assertGreater(integrator.parallel_group, spec_agent.parallel_group)
                self.assertEqual(integrator.depends_on, ["Orchestrator", "Spec Writer", "UI Engineer"])
                self.assertLess(timings["spec_start"], timings["ui_end"])
                self.assertLess(timings["ui_start"], timings["spec_end"])
                self.assertGreaterEqual(timings["integrator_start"], timings["spec_end"])
                self.assertGreaterEqual(timings["integrator_start"], timings["ui_end"])

    def test_project_workspace_rejects_unsafe_paths_and_preserves_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("app.project_runtime.GENERATED_ROOT", Path(temp_dir)):
                (Path(temp_dir) / "safe-project").mkdir()
                workspace = ProjectWorkspace("safe-project")
                workspace.ensure_created()
                written = workspace.write_file("docs/notes.txt", "hello \u2713\n")
                self.assertEqual(written, "docs/notes.txt")
                self.assertEqual(workspace.read_file("docs/notes.txt"), "hello \u2713\n")
                self.assertEqual(workspace.project_dir.name, "safe-project-2")

                self.assertEqual(
                    workspace.write_file(f"generated_projects/{workspace.project_dir.name}/index.html", "<!doctype html>\n"),
                    "index.html",
                )
                self.assertEqual(
                    workspace.write_file(f"generated_projects/{workspace.name}/app.js", "console.log('ok');\n"),
                    "app.js",
                )
                self.assertEqual(
                    workspace.write_file(f"{workspace.project_dir.name}/style.css", "body { margin: 0; }\n"),
                    "style.css",
                )
                self.assertEqual(
                    workspace.write_file("generated_projects/src/main.py", "print('ok')\n"),
                    "src/main.py",
                )
                self.assertEqual(
                    workspace.write_file("generated_projects/safe-projct-2/typo.md", "typo prefix\n"),
                    "typo.md",
                )
                self.assertEqual(
                    workspace.list_files(),
                    ["app.js", "docs/notes.txt", "index.html", "src/main.py", "style.css", "typo.md"],
                )
                self.assertFalse((workspace.project_dir / "generated_projects").exists())

                for path in ["../escape.txt", "/absolute.txt", "C:/absolute.txt", "docs/../escape.txt"]:
                    with self.assertRaises(ProjectRuntimeError):
                        workspace.write_file(path, "bad")

    def test_design_workflow_adds_relevant_agents(self) -> None:
        agents = design_workflow("Build a web UI with an MCP backend for coding agents")
        names = [agent.name for agent in agents]
        self.assertIn("Specialist Agents", names)
        self.assertIn("Project Builder", names)
        self.assertIn("Verification Agent", names)

    def test_run_workflow_completes_with_transcript(self) -> None:
        run = Run(prompt="Create a small README generator web UI")
        run_workflow(run, delay=0, model_client=fake_model)
        self.assertEqual(run.status, "complete")
        self.assertTrue(run.summary)
        self.assertTrue(all(agent.status == "done" for agent in run.agents))
        self.assertTrue(any(message.kind == "handoff" for message in run.transcript))

    def test_manual_agent_message_links_two_agents(self) -> None:
        run = Run(prompt="Build a CLI tool")
        run.agents = design_workflow(run.prompt, fake_model)
        messages = manual_agent_message(run, run.agents[0].id, run.agents[1].id, "Compare notes.", fake_model)
        message = messages[0]
        self.assertEqual(message.kind, "manual")
        self.assertIn(message, run.transcript)
        self.assertIn(message, run.agents[0].messages)
        self.assertIn(message, run.agents[1].messages)
        self.assertEqual(messages[1].kind, "agent-chat")

    def test_framework_profile_has_recommendation(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            profile = current_framework_profile()
        self.assertEqual(profile.active, "llamaindex")
        self.assertTrue(profile.adapter_ready)
        self.assertEqual(profile.recommended, "llamaindex")
        self.assertTrue(profile.available)
        self.assertTrue(profile.docs)


if __name__ == "__main__":
    unittest.main()
