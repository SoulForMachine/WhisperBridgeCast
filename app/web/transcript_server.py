import json
import mimetypes
import threading
import time
import queue
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


class WebTranscriptServer:
    """
    Lightweight HTTP + Server-Sent Events server that shows live transcripts.

    Usage:
        server = WebTranscriptServer(host="0.0.0.0", port=8080)
        server.start()
        server.add_text({id="1", orig_text="Hello", orig_unconfirmed_text=" world", transl_text="Hallo", src_lang="en", target_lang="de", complete=False})
        server.add_text({id="1", orig_text="Hello world.", transl_text="Hallo Welt!", src_lang="en", target_lang="de", complete=True})
        server.stop()
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        certfile: str | None = None,
        keyfile: str | None = None,
    ):
        self.host = host
        self.port = port
        self.certfile = certfile
        self.keyfile = keyfile

        self._httpd: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._shutdown = threading.Event()

        self._clients: set[queue.Queue] = set()
        self._lock = threading.Lock()

        self._input_queue_thread = None
        self._input_queue: queue.Queue | None = None

        # SSE event id must be monotonic for Last-Event-ID resuming.
        self._event_counter = 0

        # Keep latest snapshot per entry for replay to reconnecting clients.
        self._entry_snapshots: OrderedDict[object, dict] = OrderedDict()
        self._history_limit = 500

        static_dir = Path(__file__).parent / "static"
        self._static_paths = {
          "/": static_dir / "index.html",
          "/static/style.css": static_dir / "style.css",
          "/static/app.js": static_dir / "app.js",
        }

    # Public API -----------------------------------------------------------
    def start(self, input_queue: queue.Queue | None = None):
        """Start the HTTP/SSE server in a background thread."""
        if self._httpd:
            return

        handler_cls = self._make_handler()
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler_cls)

        if self.certfile:
            import ssl

            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            context.load_cert_chain(certfile=self.certfile, keyfile=self.keyfile)
            self._httpd.socket = context.wrap_socket(self._httpd.socket, server_side=True)

        self._server_thread = threading.Thread(
            target=self._httpd.serve_forever, name="WebTranscriptServer", daemon=True
        )
        self._server_thread.start()

        if input_queue:
            self._input_queue = input_queue
            self._input_queue_thread = threading.Thread(
                target=self._input_queue_loop
            )
            self._input_queue_thread.start()

    def stop(self):
        """Stop the server and close all client streams."""
        if not self._httpd:
            return

        if self._input_queue_thread:
            self._input_queue.put(None)
            self._input_queue_thread.join()
            self._input_queue_thread = None
            self._input_queue = None

        self._shutdown.set()
        self._httpd.shutdown()
        self._httpd.server_close()

        with self._lock:
            for client in list(self._clients):
                client.put(None)
            self._clients.clear()

        if self._server_thread:
            self._server_thread.join()

        self._httpd = None
        self._server_thread = None
        self._shutdown.clear()
        self._entry_snapshots.clear()
        self._event_counter = 0

    def wait_until_ready(self):
        return True

    def add_text(self, message: dict):
        """
        Add or update a transcript block.

        - Updates are correlated by message id.
        - For original text, confirmed text is displayed normally and unconfirmed is grayed out.
        """
        self._broadcast_event(
            kind="entry",
            message=message,
        )

    # Internals ------------------------------------------------------------
    def _input_queue_loop(self):
        while True:
            item = self._input_queue.get()
            if item is None:
                break

            try:
                if isinstance(item, dict):
                    self.add_text(item)
            except Exception as e:
                print(f"Invalid input queue item: {item} ({e})")

    def _broadcast_event(self, kind: str, message: dict):
        with self._lock:
            self._event_counter += 1
            event_id = self._event_counter
            entry_id = message.get("id", event_id)

            event = {
                "type": kind,
                "entry_id": entry_id,
                "complete": message.get("complete", True),
                "id": event_id,
                "ts": time.time(),
            }

        if "orig_text" in message:
          event["original"] = message.get("orig_text")
        if "orig_unconfirmed_text" in message:
          event["unconfirmed"] = message.get("orig_unconfirmed_text")
        if "src_lang" in message:
          event["src_lang"] = message.get("src_lang")

        if "transl_text" in message:
          event["translation"] = message.get("transl_text")
        if "target_lang" in message:
          event["target_lang"] = message.get("target_lang")

        if "source_diff" in message:
            event["source_diff"] = message.get("source_diff")
        if "target_diff" in message:
            event["target_diff"] = message.get("target_diff")

        self._send_event(event)

    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def handle(self):
                try:
                    super().handle()
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
                    # Client hung up mid-request; ignore.
                    pass

            def do_GET(self):
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
                                target = server._static_paths.get(request_path)
                                if not target or not target.exists() or not target.is_file():
                                        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
                                        return

                                content = target.read_bytes()
                                content_type, _ = mimetypes.guess_type(str(target))
                                if content_type is None:
                                        content_type = "application/octet-stream"

                                self.send_response(HTTPStatus.OK)
                                if content_type.startswith("text/") or content_type in ("application/javascript", "application/json"):
                                        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
                                else:
                                        self.send_header("Content-Type", content_type)
                                self.send_header("Content-Length", str(len(content)))
                                self.end_headers()
                                self.wfile.write(content)

            # SSE endpoint -------------------------------------------------
            def _handle_events(self):
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

        return Handler

    def _register_client(self, q: queue.Queue):
        with self._lock:
            self._clients.add(q)

    def _unregister_client(self, q: queue.Queue):
        with self._lock:
            self._clients.discard(q)

    def _send_event(self, event: dict):
        with self._lock:
            entry_id = event.get("entry_id", event.get("id"))

            # Keep replay state separate so connected clients still receive
            # the original live event payload.
            previous = self._entry_snapshots.get(entry_id)
            snapshot = dict(previous) if previous is not None else {}
            snapshot.update(event)
            # Don't include diffs in snapshot since they are not needed for replay.
            if "source_diff" in snapshot:
                snapshot["source_diff"] = []
            if "target_diff" in snapshot:
                snapshot["target_diff"] = []

            self._entry_snapshots[entry_id] = snapshot
            self._entry_snapshots.move_to_end(entry_id)
            while len(self._entry_snapshots) > self._history_limit:
                self._entry_snapshots.popitem(last=False)

            clients = list(self._clients)

        for client in clients:
            try:
                client.put_nowait(event)
            except queue.Full:
                continue
