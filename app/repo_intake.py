from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


IGNORE_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "dist",
    "node_modules",
}
MANIFEST_NAMES = {
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "uv.lock",
    "vite.config.js",
    "tsconfig.json",
}
DOC_NAMES = {"README.md", "TESTING.md", "CONTRIBUTING.md", "CHANGELOG.md"}
TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".cjs",
    ".js",
    ".jsx",
    ".json",
    ".md",
    ".mjs",
    ".py",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
MAX_INDEXED_FILES = 500
MAX_README_CHARS = 1200
MAX_SEARCH_FILE_CHARS = 80_000


@dataclass(frozen=True)
class RepoIntake:
    project_name: str
    project_dir: Path
    file_count: int
    total_bytes: int
    files: list[str]
    languages: dict[str, int]
    manifests: list[str]
    docs: list[str]
    likely_entrypoints: list[str]
    test_commands: list[str]
    readme_excerpt: str = ""
    notes: list[str] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        file_tree = "\n".join(f"- {path}" for path in self.files[:120]) or "- No files found."
        if self.file_count > 120:
            file_tree += f"\n- ... {self.file_count - 120} more files omitted"
        language_text = ", ".join(f"{key}:{value}" for key, value in self.languages.items()) or "unknown"
        manifest_text = ", ".join(self.manifests) or "none"
        doc_text = ", ".join(self.docs) or "none"
        entrypoint_text = ", ".join(self.likely_entrypoints) or "unknown"
        test_text = "\n".join(f"- {command}" for command in self.test_commands) or "- No test commands detected."
        readme = self.readme_excerpt or "No README excerpt available."
        notes = "\n".join(f"- {note}" for note in self.notes) or "- No additional notes."
        return (
            f"Existing project selected: {self.project_name}\n"
            f"Project folder: {self.project_dir}\n"
            f"File count: {self.file_count}\n"
            f"Approximate size: {self.total_bytes} bytes\n"
            f"Languages/extensions: {language_text}\n"
            f"Manifests: {manifest_text}\n"
            f"Docs: {doc_text}\n"
            f"Likely entrypoints: {entrypoint_text}\n"
            f"Detected test commands:\n{test_text}\n"
            f"README excerpt:\n{readme}\n"
            f"Current files:\n{file_tree}\n"
            f"Intake notes:\n{notes}"
        )


def build_repo_intake(project_dir: Path, project_name: str | None = None) -> RepoIntake:
    project_dir = project_dir.resolve()
    files = _project_files(project_dir)
    languages: dict[str, int] = {}
    manifests: list[str] = []
    docs: list[str] = []
    entrypoints: list[str] = []
    total_bytes = 0

    for relative_path in files:
        path = project_dir / relative_path
        try:
            total_bytes += path.stat().st_size
        except OSError:
            continue
        suffix = path.suffix.lower() or "[no extension]"
        languages[suffix] = languages.get(suffix, 0) + 1
        if path.name in MANIFEST_NAMES:
            manifests.append(relative_path)
        if path.name in DOC_NAMES or path.name.lower().startswith("readme"):
            docs.append(relative_path)
        if _looks_like_entrypoint(relative_path):
            entrypoints.append(relative_path)

    notes: list[str] = []
    if len(files) >= MAX_INDEXED_FILES:
        notes.append(f"File listing was capped at {MAX_INDEXED_FILES} files.")
    if not manifests:
        notes.append("No common package manifest was found.")
    if not docs:
        notes.append("No common README or testing document was found.")

    return RepoIntake(
        project_name=project_name or project_dir.name,
        project_dir=project_dir,
        file_count=len(files),
        total_bytes=total_bytes,
        files=files,
        languages=dict(sorted(languages.items())),
        manifests=sorted(manifests),
        docs=sorted(docs),
        likely_entrypoints=sorted(entrypoints)[:20],
        test_commands=_detect_test_commands(project_dir, files),
        readme_excerpt=_read_readme_excerpt(project_dir, files),
        notes=notes,
    )


def search_project_files(project_dir: Path, query: str, max_results: int = 8) -> list[dict[str, str]]:
    terms = [term.casefold() for term in query.split() if term.strip()]
    files = _project_files(project_dir)
    if not terms:
        return [{"path": path, "reason": "listed file", "snippet": ""} for path in files[:max_results]]

    scored: list[tuple[int, dict[str, str]]] = []
    for relative_path in files:
        path_score = sum(4 for term in terms if term in relative_path.casefold())
        snippet = ""
        content_score = 0
        path = project_dir / relative_path
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = _read_text(path, MAX_SEARCH_FILE_CHARS)
            if text:
                lowered = text.casefold()
                content_score = sum(lowered.count(term) for term in terms)
                snippet = _snippet_for_terms(text, terms)
        score = path_score + content_score
        if score:
            reason = "path match" if path_score else "content match"
            scored.append((score, {"path": relative_path, "reason": reason, "snippet": snippet}))

    return [item for _, item in sorted(scored, key=lambda row: (-row[0], row[1]["path"]))[:max_results]]


def _project_files(project_dir: Path) -> list[str]:
    if not project_dir.exists():
        return []
    paths: list[str] = []
    for path in project_dir.rglob("*"):
        if not path.is_file() or any(part in IGNORE_DIRS for part in path.parts):
            continue
        paths.append(path.relative_to(project_dir).as_posix())
        if len(paths) >= MAX_INDEXED_FILES:
            break
    return sorted(paths)


def _detect_test_commands(project_dir: Path, files: list[str]) -> list[str]:
    commands: list[str] = []
    package_json = project_dir / "package.json"
    if package_json.exists():
        payload = _read_json(package_json)
        scripts = payload.get("scripts") if isinstance(payload, dict) else None
        if isinstance(scripts, dict):
            if "test" in scripts:
                commands.append("npm test")
            if "lint" in scripts:
                commands.append("npm run lint")
            if "build" in scripts:
                commands.append("npm run build")
        elif any(path.endswith(".js") for path in files):
            commands.append("node --check <changed .js files>")
    if "pyproject.toml" in files or any(path.endswith(".py") for path in files):
        commands.append("python -m py_compile <changed .py files>")
        if any(path.startswith("tests/") and path.endswith(".py") for path in files):
            commands.append("python -m unittest discover -s tests")
    if any(path.endswith(".js") for path in files) and not any(command.startswith("node --check") for command in commands):
        commands.append("node --check <changed .js files>")
    return commands


def _looks_like_entrypoint(relative_path: str) -> bool:
    name = Path(relative_path).name
    return name in {"app.py", "main.py", "index.html", "index.js", "app.js", "server.js"} or relative_path.startswith("src/")


def _read_readme_excerpt(project_dir: Path, files: list[str]) -> str:
    for relative_path in files:
        name = Path(relative_path).name.lower()
        if name.startswith("readme"):
            return _read_text(project_dir / relative_path, MAX_README_CHARS)
    return ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_text(path: Path, limit: int) -> str:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    return content[:limit]


def _snippet_for_terms(text: str, terms: list[str]) -> str:
    lowered = text.casefold()
    first_match = min((lowered.find(term) for term in terms if term in lowered), default=-1)
    if first_match < 0:
        return ""
    start = max(0, first_match - 90)
    end = min(len(text), first_match + 220)
    return text[start:end].replace("\n", " ").strip()
