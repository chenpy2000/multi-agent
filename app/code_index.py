from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core.schema import Document

from app.repo_intake import IGNORE_DIRS, TEXT_SUFFIXES


MAX_CODE_FILES = 700
MAX_CODE_FILE_CHARS = 120_000
PY_SYMBOL_RE = re.compile(r"^\s*(class|def)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(\(.*)?\s*:", re.MULTILINE)
JS_FUNCTION_RE = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([a-zA-Z_$][\w$]*)\s*\(", re.MULTILINE)
JS_CLASS_RE = re.compile(r"^\s*(?:export\s+)?class\s+([a-zA-Z_$][\w$]*)", re.MULTILINE)
JS_ARROW_RE = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([a-zA-Z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^=]*\)|[a-zA-Z_$][\w$]*)\s*=>", re.MULTILINE)
IMPORT_RE = re.compile(r"^\s*(?:import\s+.+|from\s+\S+\s+import\s+.+|const\s+.+\s*=\s*require\(.+\))", re.MULTILINE)


@dataclass(frozen=True)
class CodeSymbol:
    name: str
    kind: str
    path: str
    line: int
    signature: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "path": self.path,
            "line": self.line,
            "signature": self.signature,
        }


@dataclass
class CodeIndexEntry:
    path: str
    suffix: str
    size: int
    line_count: int
    text: str
    symbols: list[CodeSymbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)


@dataclass
class ProjectCodeIndex:
    project_dir: Path
    entries: dict[str, CodeIndexEntry]
    documents: list[Document]
    file_documents: dict[str, Document] = field(default_factory=dict)

    def to_prompt_context(self) -> str:
        symbol_count = sum(len(entry.symbols) for entry in self.entries.values())
        top_files = sorted(
            self.entries.values(),
            key=lambda entry: (-(len(entry.symbols) * 3 + len(entry.imports)), entry.path),
        )[:12]
        file_lines = "\n".join(
            f"- {entry.path}: {entry.line_count} lines, {len(entry.symbols)} symbols"
            for entry in top_files
        ) or "- No indexed text files."
        symbols = sorted(
            [symbol for entry in self.entries.values() for symbol in entry.symbols],
            key=lambda item: (item.path, item.line),
        )[:18]
        symbol_lines = "\n".join(
            f"- {symbol.kind} {symbol.name} in {symbol.path}:{symbol.line}"
            for symbol in symbols
        ) or "- No symbols detected."
        return (
            f"Code index: {len(self.entries)} text files indexed as LlamaIndex Documents.\n"
            f"Detected symbols: {symbol_count}\n"
            f"Important indexed files:\n{file_lines}\n"
            f"Top symbols:\n{symbol_lines}"
        )


def build_project_code_index(project_dir: Path) -> ProjectCodeIndex:
    project_dir = project_dir.resolve()
    entries: dict[str, CodeIndexEntry] = {}
    file_documents: dict[str, Document] = {}
    for relative_path in _code_files(project_dir):
        entry = _build_entry(project_dir, relative_path)
        if entry is None:
            continue
        entries[relative_path] = entry
        file_documents[relative_path] = _entry_document(entry)
    return ProjectCodeIndex(
        project_dir=project_dir,
        entries=entries,
        documents=[file_documents[path] for path in sorted(file_documents)],
        file_documents=file_documents,
    )


def refresh_project_code_index_files(index: ProjectCodeIndex, relative_paths: list[str]) -> ProjectCodeIndex:
    project_dir = index.project_dir.resolve()
    for relative_path in sorted(set(relative_paths)):
        entry = _build_entry(project_dir, relative_path)
        if entry is None:
            index.entries.pop(relative_path, None)
            index.file_documents.pop(relative_path, None)
            continue
        index.entries[relative_path] = entry
        index.file_documents[relative_path] = _entry_document(entry)
    index.documents = [index.file_documents[path] for path in sorted(index.file_documents)]
    return index


def _code_files(project_dir: Path) -> list[str]:
    paths: list[str] = []
    for path in project_dir.rglob("*"):
        if not path.is_file() or any(part in IGNORE_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        paths.append(path.relative_to(project_dir).as_posix())
        if len(paths) >= MAX_CODE_FILES:
            break
    return sorted(paths)


def _build_entry(project_dir: Path, relative_path: str) -> CodeIndexEntry | None:
    path = (project_dir / relative_path).resolve()
    try:
        path.relative_to(project_dir)
    except ValueError:
        return None
    if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
        return None
    text = _read_text(path)
    if not text:
        return None
    return CodeIndexEntry(
        path=relative_path,
        suffix=path.suffix.lower(),
        size=path.stat().st_size,
        line_count=text.count("\n") + 1,
        text=text,
        symbols=_extract_symbols(relative_path, text),
        imports=_extract_imports(text),
    )


def _entry_document(entry: CodeIndexEntry) -> Document:
    return Document(
        text=entry.text,
        metadata={"kind": "source_file", "path": entry.path, "suffix": entry.suffix},
    )


def _extract_symbols(relative_path: str, text: str) -> list[CodeSymbol]:
    suffix = Path(relative_path).suffix.lower()
    if suffix == ".py":
        return [
            CodeSymbol(match.group(2), match.group(1), relative_path, _line_number(text, match.start()), match.group(0).strip())
            for match in PY_SYMBOL_RE.finditer(text)
        ]
    if suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        symbols: list[CodeSymbol] = []
        for match in JS_FUNCTION_RE.finditer(text):
            symbols.append(CodeSymbol(match.group(1), "function", relative_path, _line_number(text, match.start()), match.group(0).strip()))
        for match in JS_CLASS_RE.finditer(text):
            symbols.append(CodeSymbol(match.group(1), "class", relative_path, _line_number(text, match.start()), match.group(0).strip()))
        for match in JS_ARROW_RE.finditer(text):
            symbols.append(CodeSymbol(match.group(1), "function", relative_path, _line_number(text, match.start()), match.group(0).strip()))
        return sorted(symbols, key=lambda symbol: symbol.line)
    if suffix == ".html":
        return _html_symbols(relative_path, text)
    if suffix == ".css":
        return _css_symbols(relative_path, text)
    return []


def _html_symbols(relative_path: str, text: str) -> list[CodeSymbol]:
    symbols: list[CodeSymbol] = []
    for match in re.finditer(r"\bid=[\"']([^\"']+)[\"']", text):
        symbols.append(CodeSymbol(match.group(1), "html-id", relative_path, _line_number(text, match.start()), match.group(0)))
    return symbols[:80]


def _css_symbols(relative_path: str, text: str) -> list[CodeSymbol]:
    symbols: list[CodeSymbol] = []
    for match in re.finditer(r"^\s*([.#][a-zA-Z0-9_-]+)\s*[,{]", text, flags=re.MULTILINE):
        symbols.append(CodeSymbol(match.group(1), "css-selector", relative_path, _line_number(text, match.start()), match.group(0).strip()))
    return symbols[:80]


def _extract_imports(text: str) -> list[str]:
    return [match.group(0).strip()[:180] for match in IMPORT_RE.finditer(text)][:40]


def _read_text(path: Path) -> str:
    try:
        if path.stat().st_size > MAX_CODE_FILE_CHARS:
            return path.read_text(encoding="utf-8")[:MAX_CODE_FILE_CHARS]
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1
