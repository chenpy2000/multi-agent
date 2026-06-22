from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from app.frameworks import current_framework_profile
from app.llm import LLMConfigError, LLMRequestError, ask_openai, load_dotenv, require_openai_config
from app.models import Run
from app.orchestrator import RunStore, manual_agent_message, run_workflow


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
STORE = RunStore()


class AppHandler(BaseHTTPRequestHandler):
    server_version = "MultiAgentPrototype/0.1"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_file(STATIC_DIR / "index.html")
        elif path.startswith("/static/"):
            self._send_file(STATIC_DIR / unquote(path.removeprefix("/static/")))
        elif path == "/api/runs":
            self._send_json([run.to_dict() for run in STORE.list()])
        elif path == "/api/framework":
            self._send_json(current_framework_profile().to_dict())
        elif path.startswith("/api/runs/"):
            run_id = path.split("/", 3)[3]
            run = STORE.get(run_id)
            if run is None:
                self._send_error(HTTPStatus.NOT_FOUND, "Run not found.")
            else:
                self._send_json(run.to_dict())
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "Route not found.")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/runs":
            self._create_run()
        elif path.startswith("/api/runs/") and path.endswith("/messages"):
            run_id = path.split("/")[3]
            self._create_message(run_id)
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "Route not found.")

    def log_message(self, format: str, *args: object) -> None:
        return

    def _create_run(self) -> None:
        data = self._read_json()
        prompt = str(data.get("prompt", "")).strip()
        if not prompt:
            self._send_error(HTTPStatus.BAD_REQUEST, "Prompt is required.")
            return
        try:
            require_openai_config()
        except (LLMConfigError, LLMRequestError) as exc:
            self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            return

        framework = current_framework_profile()
        run = Run(prompt=prompt, framework=framework.active, framework_note=framework.rationale)
        STORE.add(run)
        thread = threading.Thread(target=run_workflow, args=(run, STORE.update), daemon=True)
        thread.start()
        self._send_json(run.to_dict(), HTTPStatus.CREATED)

    def _create_message(self, run_id: str) -> None:
        run = STORE.get(run_id)
        if run is None:
            self._send_error(HTTPStatus.NOT_FOUND, "Run not found.")
            return

        data = self._read_json()
        try:
            messages = manual_agent_message(
                run,
                str(data.get("sender_id", "")),
                str(data.get("recipient_id", "")),
                str(data.get("content", "")).strip() or "Please compare notes and unblock the next step.",
                ask_openai,
            )
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except (LLMConfigError, LLMRequestError) as exc:
            self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            return
        STORE.update(run)
        self._send_json([message.to_dict() for message in messages], HTTPStatus.CREATED)

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _send_file(self, path: Path) -> None:
        try:
            resolved = path.resolve()
            resolved.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self._send_error(HTTPStatus.FORBIDDEN, "Forbidden.")
            return

        if not resolved.exists() or not resolved.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "File not found.")
            return

        content_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        body = resolved.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status)


def build_server(host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), AppHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-agent prototype.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_dotenv()
    server = build_server(args.host, args.port)
    print(f"Multi-agent prototype running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
