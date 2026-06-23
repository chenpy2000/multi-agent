from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

from llama_index.core.schema import Document

from app.repo_intake import TEXT_SUFFIXES


HTML_REF_RE = re.compile(r"\b(?:src|href)=['\"]([^'\"]+)['\"]", re.IGNORECASE)
JS_IMPORT_RE = re.compile(
    r"(?:import\s+(?:[^'\"]+\s+from\s+)?|import\s*\(\s*|require\s*\(\s*)['\"]([^'\"]+)['\"]",
    re.MULTILINE,
)
CSS_IMPORT_RE = re.compile(r"@import\s+(?:url\()?\s*['\"]?([^'\"\)]+)['\"]?\s*\)?", re.IGNORECASE)
PY_IMPORT_RE = re.compile(r"^\s*import\s+([a-zA-Z_][\w.]*)(?:\s+as\s+\w+)?", re.MULTILINE)
PY_FROM_IMPORT_RE = re.compile(r"^\s*from\s+([.\w]+)\s+import\s+(.+)", re.MULTILINE)


@dataclass(frozen=True)
class RelationshipEntry:
    path: str
    summary: str
    targets: tuple[str, ...]
    document: Document


@dataclass(frozen=True)
class RelationshipMapUpdate:
    entries: dict[str, RelationshipEntry]
    removed: list[str]


@dataclass(frozen=True)
class RelationshipEvidence:
    path: str
    role_hint: str
    deterministic_hint: str
    direct_targets: tuple[str, ...]
    content: str


@dataclass(frozen=True)
class RelationshipEvidenceBatch:
    files: list[RelationshipEvidence]
    removed: list[str]


def build_relationship_map_update(
    project_dir: Path,
    selected_files: list[str],
    all_files: list[str],
) -> RelationshipMapUpdate:
    project_dir = project_dir.resolve()
    all_file_set = set(all_files)
    module_map = _python_module_map(all_files)
    entries: dict[str, RelationshipEntry] = {}
    removed: list[str] = []

    for relative_path in sorted(set(selected_files)):
        path = (project_dir / relative_path).resolve()
        try:
            path.relative_to(project_dir)
        except ValueError:
            continue
        if not path.is_file():
            removed.append(relative_path)
            continue

        text = _read_text(path) if path.suffix.lower() in TEXT_SUFFIXES else ""
        relationships = _direct_relationships(relative_path, text, all_file_set, module_map)
        summary = _relationship_summary(relative_path, relationships)
        targets = tuple(sorted({target for values in relationships.values() for target in values}))
        document = Document(
            text=summary,
            metadata={
                "kind": "relationship_map_entry",
                "path": relative_path,
                "targets": list(targets),
            },
        )
        entries[relative_path] = RelationshipEntry(
            path=relative_path,
            summary=summary,
            targets=targets,
            document=document,
        )

    return RelationshipMapUpdate(entries=entries, removed=removed)


def build_relationship_map_evidence(
    project_dir: Path,
    selected_files: list[str],
    all_files: list[str],
    max_file_chars: int = 12_000,
) -> RelationshipEvidenceBatch:
    project_dir = project_dir.resolve()
    all_file_set = set(all_files)
    module_map = _python_module_map(all_files)
    files: list[RelationshipEvidence] = []
    removed: list[str] = []

    for relative_path in sorted(set(selected_files)):
        path = (project_dir / relative_path).resolve()
        try:
            path.relative_to(project_dir)
        except ValueError:
            continue
        if not path.is_file():
            removed.append(relative_path)
            continue
        text = _read_text(path) if path.suffix.lower() in TEXT_SUFFIXES else ""
        relationships = _direct_relationships(relative_path, text, all_file_set, module_map)
        direct_targets = tuple(sorted({target for values in relationships.values() for target in values}))
        files.append(
            RelationshipEvidence(
                path=relative_path,
                role_hint=_file_role(relative_path),
                deterministic_hint=_relationship_summary(relative_path, relationships),
                direct_targets=direct_targets,
                content=text[:max_file_chars],
            )
        )
    return RelationshipEvidenceBatch(files=files, removed=removed)


def parse_relationship_map_response(
    text: str,
    selected_files: list[str],
    removed_files: list[str],
) -> RelationshipMapUpdate:
    payload = _extract_json_object(text)
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("Relationship Map Agent response must include an entries list.")

    selected = set(selected_files)
    entries: dict[str, RelationshipEntry] = {}
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "")).strip().replace("\\", "/")
        if path not in selected or path in removed_files:
            continue
        summary = str(item.get("summary", "")).strip()
        if not summary:
            continue
        if not summary.startswith("- "):
            summary = f"- {summary}"
        direct_files = item.get("direct_files", [])
        targets = tuple(
            sorted(
                str(target).strip().replace("\\", "/")
                for target in direct_files
                if isinstance(target, str) and str(target).strip()
            )
        )
        document = Document(
            text=summary,
            metadata={
                "kind": "relationship_map_entry",
                "path": path,
                "targets": list(targets),
                "source": "llm_relationship_map_agent",
            },
        )
        entries[path] = RelationshipEntry(path=path, summary=summary, targets=targets, document=document)

    expected = selected.difference(removed_files)
    missing = sorted(path for path in expected if path not in entries)
    if missing:
        raise ValueError(f"Relationship Map Agent omitted selected files: {', '.join(missing)}")
    return RelationshipMapUpdate(entries=entries, removed=sorted(set(removed_files)))


def relationship_map_context(summaries_by_path: dict[str, str], max_entries: int = 120) -> str:
    if not summaries_by_path:
        return "Project relationship map:\n- No project files have been mapped yet."
    lines = [summaries_by_path[path] for path in sorted(summaries_by_path)[:max_entries]]
    omitted = len(summaries_by_path) - len(lines)
    if omitted > 0:
        lines.append(f"- ... {omitted} more files omitted from this compact map.")
    return "Project relationship map:\n" + "\n".join(lines)


def relationship_map_document(summaries_by_path: dict[str, str]) -> Document:
    return Document(
        text=relationship_map_context(summaries_by_path, max_entries=len(summaries_by_path) or 1),
        metadata={"kind": "relationship_map", "path": "__relationship_map__"},
    )


def _direct_relationships(
    relative_path: str,
    text: str,
    all_files: set[str],
    module_map: dict[str, str],
) -> dict[str, set[str]]:
    suffix = Path(relative_path).suffix.lower()
    relationships: dict[str, set[str]] = {"loads": set(), "imports": set(), "references": set()}
    if not text:
        return relationships

    if suffix in {".html", ".htm"}:
        for specifier in HTML_REF_RE.findall(text):
            target = _resolve_local_reference(relative_path, specifier, all_files, _web_extensions(specifier))
            if target:
                relationships["loads"].add(target)
    elif suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        for specifier in JS_IMPORT_RE.findall(text):
            target = _resolve_local_reference(relative_path, specifier, all_files, _script_extensions(specifier))
            if target:
                relationships["imports"].add(target)
    elif suffix == ".css":
        for specifier in CSS_IMPORT_RE.findall(text):
            target = _resolve_local_reference(relative_path, specifier, all_files, [".css"])
            if target:
                relationships["imports"].add(target)
    elif suffix == ".py":
        for module in PY_IMPORT_RE.findall(text):
            target = module_map.get(module)
            if target and target != relative_path:
                relationships["imports"].add(target)
        for module, imported_names in PY_FROM_IMPORT_RE.findall(text):
            for target in _resolve_python_from_import(relative_path, module, imported_names, module_map):
                if target != relative_path:
                    relationships["imports"].add(target)

    return relationships


def _relationship_summary(relative_path: str, relationships: dict[str, set[str]]) -> str:
    role = _file_role(relative_path)
    phrases: list[str] = []
    for verb in ("loads", "imports", "references"):
        targets = sorted(relationships.get(verb, set()))
        if targets:
            phrases.append(f"directly {verb} {_format_targets(targets)}")
    if not phrases:
        return f"- `{relative_path}` {role} and does not directly load local project files."
    return f"- `{relative_path}` {role} and {' and '.join(phrases)}."


def _file_role(relative_path: str) -> str:
    path = Path(relative_path)
    name = path.name.lower()
    suffix = path.suffix.lower()
    parts = {part.lower() for part in path.parts}
    if name == "readme.md":
        return "documents the project"
    if name in {"package.json", "pyproject.toml", "requirements.txt"}:
        return "declares project metadata, dependencies, or scripts"
    if suffix in {".html", ".htm"}:
        return "is a browser page shell"
    if suffix == ".css":
        return "owns visual styling"
    if suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        if "test" in name or "tests" in parts:
            return "tests JavaScript behavior"
        return "owns JavaScript behavior"
    if suffix == ".py":
        if name.startswith("test_") or "tests" in parts:
            return "tests Python behavior"
        return "defines Python logic"
    if suffix in {".md", ".txt"}:
        return "contains project documentation"
    if suffix == ".json":
        return "contains structured project data"
    return "is a project file"


def _format_targets(targets: list[str]) -> str:
    quoted = [f"`{target}`" for target in targets]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + f" and {quoted[-1]}"


def _resolve_local_reference(
    source_path: str,
    specifier: str,
    all_files: set[str],
    extensions: list[str],
) -> str | None:
    specifier = _clean_specifier(specifier)
    if not specifier or _is_external_reference(specifier):
        return None
    base_dir = posixpath.dirname(source_path)
    if specifier.startswith("/"):
        candidate = posixpath.normpath(specifier.lstrip("/"))
    else:
        candidate = posixpath.normpath(posixpath.join(base_dir, specifier))
    if candidate == "." or candidate.startswith("../"):
        return None

    candidates = [candidate]
    if not posixpath.splitext(candidate)[1]:
        candidates.extend(candidate + extension for extension in extensions)
    candidates.extend(posixpath.join(candidate, "index" + extension) for extension in extensions)
    for item in candidates:
        if item in all_files:
            return item
    return None


def _resolve_python_from_import(
    source_path: str,
    module: str,
    imported_names: str,
    module_map: dict[str, str],
) -> set[str]:
    targets: set[str] = set()
    if module.startswith("."):
        base_module = _relative_python_module(source_path, module)
        if base_module and base_module in module_map:
            targets.add(module_map[base_module])
        if base_module:
            for imported_name in _imported_python_names(imported_names):
                candidate = f"{base_module}.{imported_name}" if base_module else imported_name
                if candidate in module_map:
                    targets.add(module_map[candidate])
        return targets

    if module in module_map:
        targets.add(module_map[module])
    for imported_name in _imported_python_names(imported_names):
        candidate = f"{module}.{imported_name}"
        if candidate in module_map:
            targets.add(module_map[candidate])
    return targets


def _relative_python_module(source_path: str, module: str) -> str:
    leading_dots = len(module) - len(module.lstrip("."))
    module_tail = module[leading_dots:]
    package_parts = source_path.removesuffix(".py").split("/")[:-1]
    if leading_dots > 1:
        package_parts = package_parts[: -(leading_dots - 1)] if leading_dots - 1 <= len(package_parts) else []
    if module_tail:
        package_parts.extend(part for part in module_tail.split(".") if part)
    return ".".join(package_parts)


def _imported_python_names(imported_names: str) -> list[str]:
    names: list[str] = []
    for raw_name in imported_names.split(","):
        name = raw_name.strip().split(" as ", 1)[0].strip()
        if name and name != "*":
            names.append(name)
    return names


def _python_module_map(all_files: list[str]) -> dict[str, str]:
    module_paths: dict[str, list[str]] = {}
    for path in all_files:
        if not path.endswith(".py"):
            continue
        stem = path[:-3]
        module = stem.removesuffix("/__init__").replace("/", ".")
        candidates = {module, module.split(".")[-1]}
        for candidate in candidates:
            if candidate:
                module_paths.setdefault(candidate, []).append(path)
    return {
        module: paths[0]
        for module, paths in module_paths.items()
        if len(paths) == 1
    }


def _clean_specifier(specifier: str) -> str:
    return specifier.strip().replace("\\", "/").split("#", 1)[0].split("?", 1)[0].strip()


def _is_external_reference(specifier: str) -> bool:
    lowered = specifier.lower()
    return (
        "://" in lowered
        or lowered.startswith("//")
        or lowered.startswith(("mailto:", "tel:", "data:", "javascript:", "#"))
    )


def _web_extensions(specifier: str) -> list[str]:
    lowered = specifier.lower()
    if lowered.endswith(".css"):
        return [".css"]
    if lowered.endswith((".js", ".mjs", ".cjs")):
        return [".js", ".mjs", ".cjs"]
    return [".html", ".js", ".mjs", ".cjs", ".css", ".json", ".png", ".jpg", ".jpeg", ".svg", ".webp"]


def _script_extensions(specifier: str) -> list[str]:
    lowered = specifier.lower()
    if lowered.endswith(".css"):
        return [".css"]
    return [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".json", ".css"]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _extract_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found in Relationship Map Agent response.")
    import json

    payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Relationship Map Agent response JSON must be an object.")
    return payload
