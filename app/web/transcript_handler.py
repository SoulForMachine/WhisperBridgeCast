import json
import mimetypes
import queue
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse


class TranscriptRequestHandler(BaseHTTPRequestHandler):
    transcript_server = None

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            # Client hung up mid-request; ignore.
            pass

    def do_GET(self):
        server = self.transcript_server
        parsed = urlparse(self.path)
        if parsed.path in server._static_paths:
            self._handle_static(parsed.path)
        elif parsed.path == "/events":
            self._handle_events()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def log_message(self, fmt, *args):
        # Keep output clean; rely on caller logging.
        return

    # Static file handler ------------------------------------------
    def _handle_static(self, request_path: str):
        server = self.transcript_server
        target = server._static_paths.get(request_path)
        if not target or not target.exists() or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return

        content = target.read_bytes()
        content_type, _ = mimetypes.guess_type(str(target))
        if content_type is None:
            content_type = "application/octet-stream"

        self.send_response(HTTPStatus.OK)
        if content_type.startswith("text/") or content_type in (
            "application/javascript",
            "application/json",
        ):
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        else:
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    # SSE endpoint -------------------------------------------------
    def _handle_events(self):
        server = self.transcript_server

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            last_id = int(self.headers.get("Last-Event-ID", -1))
        except (TypeError, ValueError):
            last_id = -1

        client_q: queue.Queue = queue.Queue()
        server._register_client(client_q)

        # Send latest per-entry snapshots first.
        with server._lock:
            replay_events = list(server._entry_snapshots.values())
        for event in replay_events:
            if event["id"] <= last_id:
                continue
            if not self._write_event(event):
                break

        try:
            while not server._shutdown.is_set():
                try:
                    event = client_q.get(timeout=15)
                except queue.Empty:
                    # Keep-alive to prevent idle timeouts.
                    if not self._safe_write(b": ping\n\n"):
                        break
                    continue

                if event is None:
                    break

                if not self._write_event(event):
                    break
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            # Client went away; nothing else to do.
            pass
        finally:
            server._unregister_client(client_q)

    def _safe_write(self, data: bytes) -> bool:
        try:
            self.wfile.write(data)
            self.wfile.flush()
            return True
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            return False

    def _write_event(self, event: dict) -> bool:
        payload = json.dumps(event, ensure_ascii=False)
        if not self._safe_write(f"id: {event['id']}\n".encode("utf-8")):
            return False
        return self._safe_write(f"data: {payload}\n\n".encode("utf-8"))


def make_transcript_handler(server):
    class Handler(TranscriptRequestHandler):
        transcript_server = server

    return Handler