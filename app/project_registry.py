from __future__ import annotations

import shutil
from pathlib import Path

import app.project_runtime as runtime
from app.project_runtime import ProjectRuntimeError


def list_projects() -> list[dict[str, object]]:
    runtime.GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    projects: list[dict[str, object]] = []
    for path in sorted(runtime.GENERATED_ROOT.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_dir():
            continue
        files = [item for item in path.rglob("*") if item.is_file()]
        modified = max((item.stat().st_mtime for item in files), default=path.stat().st_mtime)
        projects.append(
            {
                "name": path.name,
                "path": str(path),
                "file_count": len(files),
                "updated_at": modified,
            }
        )
    return projects


def project_exists(name: str) -> bool:
    return _project_dir(name).is_dir()


def rename_project(current_name: str, new_name: str) -> dict[str, object]:
    source = _project_dir(current_name)
    if not source.is_dir():
        raise ProjectRuntimeError(f"Project not found: {current_name}")

    clean_name = runtime._slugify(new_name)
    if not clean_name:
        raise ProjectRuntimeError("Project name is required.")
    destination = _project_dir(clean_name)
    if destination.exists():
        raise ProjectRuntimeError(f"Project already exists: {clean_name}")

    source.rename(destination)
    return next(project for project in list_projects() if project["name"] == clean_name)


def delete_project(name: str) -> None:
    project_dir = _project_dir(name)
    if not project_dir.is_dir():
        raise ProjectRuntimeError(f"Project not found: {name}")
    shutil.rmtree(project_dir)


def _project_dir(name: str) -> Path:
    clean_name = runtime._slugify(name)
    if not clean_name:
        raise ProjectRuntimeError("Project name is required.")
    project_dir = (runtime.GENERATED_ROOT / clean_name).resolve()
    try:
        project_dir.relative_to(runtime.GENERATED_ROOT.resolve())
    except ValueError as exc:
        raise ProjectRuntimeError(f"Unsafe project name: {name}") from exc
    return project_dir
