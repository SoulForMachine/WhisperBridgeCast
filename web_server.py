import json
import threading
import time
import queue
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


class WebTranscriptServer:
    """
    Lightweight HTTP + Server-Sent Events server that shows live transcripts.

    Usage:
        server = WebTranscriptServer(host="0.0.0.0", port=8080)
        server.start()
        server.add_text(original="Hello", translation="Hallo", complete=False)
        server.add_text(original="Hello world.", translation="Hallo Welt!", complete=True)
        server.stop()
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port

        self._httpd: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._shutdown = threading.Event()

        self._clients: set[queue.Queue] = set()
        self._lock = threading.Lock()

        self._input_queue_thread = None
        self._input_queue: queue.Queue | None = None

        # Track whether the last item was finalized so we know when to append vs. replace.
        self._stream_state = {"entry": {"last_complete": True, "counter": 0}}

        # Keep a small backlog so newcomers see recent lines.
        self._history: list[dict] = []
        self._history_limit = 500

    # Public API -----------------------------------------------------------
    def start(self, input_queue: queue.Queue | None = None):
        """Start the HTTP/SSE server in a background thread."""
        if self._httpd:
            return

        handler_cls = self._make_handler()
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler_cls)
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
        self._history.clear()
        self._stream_state = {"entry": {"last_complete": True, "counter": 0}}

    def add_text(self, translation: str, original: str | None = None, complete: bool = True):
        """
        Add or update a transcript block.

        - If complete=False, the last block is updated in place.
        - If complete=True, the block is finalized and the next call starts a new block.
        """
        self._broadcast_event(
            kind="entry",
            original=original,
            translation=translation,
            complete=complete,
        )

    # Internals ------------------------------------------------------------
    def _input_queue_loop(self):
        while True:
            item = self._input_queue.get()
            if item is None:
                break

            try:
                # Accept either a dict or a tuple (translation, complete, original?).
                if isinstance(item, dict):
                    self.add_text(
                        translation=item.get("translation", ""),
                        original=item.get("original"),
                        complete=bool(item.get("complete", True)),
                    )
                else:
                    # Fallback tuple format: (translation, complete[, original])
                    translation = item[0]
                    complete = bool(item[1]) if len(item) > 1 else True
                    original = item[2] if len(item) > 2 else None
                    self.add_text(translation=translation, original=original, complete=complete)
            except Exception as e:
                print(f"Invalid input queue item: {item} ({e})")

    def _broadcast_event(self, kind: str, original: str | None, translation: str, complete: bool):
        with self._lock:
            state = self._stream_state[kind]
            if state["last_complete"]:
                state["counter"] += 1
                action = "append"
            else:
                action = "replace"

            event_id = state["counter"]
            state["last_complete"] = complete

            event = {
                "type": kind,
                "original": original,
                "translation": translation,
                "complete": complete,
                "action": action,
                "id": event_id,
                "ts": time.time(),
            }

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
                if parsed.path == "/":
                    self._handle_index()
                elif parsed.path == "/events":
                    self._handle_events()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

            def log_message(self, fmt, *args):
                # Keep output clean; rely on caller logging.
                return

            # HTML page ----------------------------------------------------
            def _handle_index(self):
                content = self._index_html().encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
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

                # Send backlog first so newcomers see something immediately.
                for event in server._history:
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

            # HTML template ------------------------------------------------
            @staticmethod
            def _index_html() -> str:
                return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Live Transcript</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #111827;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --accent: #22c55e;
      --accent-2: #3b82f6;
      --pill: #1f2937;
      --border: #1f2937;
      --font: "Segoe UI", system-ui, -apple-system, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      line-height: 1.5;
    }
    header {
      position: sticky;
      top: 0;
      padding: 12px 16px;
      background: rgba(15, 23, 42, 0.9);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: baseline;
      gap: 12px;
      z-index: 1;
    }
    header h1 { margin: 0; font-size: 18px; letter-spacing: 0.02em; }
    header .hint { color: var(--muted); font-size: 13px; }
    main { max-width: 900px; margin: 0 auto; padding: 8px 16px 24px; }
    section.stream {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      min-height: 50vh;
      box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    section h2 {
      margin: 0 0 8px 0;
      font-size: 15px;
      color: var(--muted);
      letter-spacing: 0.02em;
    }
    #entries { display: flex; flex-direction: column; gap: 12px; }
    .entry {
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      background: rgba(255,255,255,0.02);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
    }
    .row {
      display: flex;
      gap: 10px;
      align-items: flex-start;
      margin: 4px 0;
    }
    .pill {
      padding: 2px 8px;
      border-radius: 999px;
      background: var(--pill);
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      flex-shrink: 0;
    }
    .pill.orig { background: rgba(34, 197, 94, 0.15); color: #bbf7d0; }
    .pill.transl { background: rgba(59, 130, 246, 0.15); color: #bfdbfe; }
    .content { white-space: pre-wrap; word-break: break-word; }
    .muted { color: var(--muted); }
  </style>
</head>
<body>
  <header>
    <h1>Live Transcript</h1>
    <div class="hint">Auto-scroll only when you are at the bottom. <a href="#bottom">Go to latest</a></div>
  </header>
  <main>
    <section class="stream">
      <h2>Transcript</h2>
      <div id="entries" aria-live="polite"></div>
    </section>
  </main>
  <script>
    const entriesBox = document.getElementById("entries");

    function atBottom() {
      const threshold = 16;
      return window.innerHeight + window.scrollY >= document.body.scrollHeight - threshold;
    }

    function renderEntry(action, original, translation) {
      const stick = atBottom();

      let entry = null;
      if (action === "replace" && entriesBox.lastElementChild) {
        entry = entriesBox.lastElementChild;
      }

      if (!entry) {
        entry = document.createElement("div");
        entry.className = "entry";
        entriesBox.appendChild(entry);
      }

      entry.replaceChildren();

      if (original) {
        const rowOrig = document.createElement("div");
        rowOrig.className = "row";
        const pillO = document.createElement("span");
        pillO.className = "pill orig";
        pillO.textContent = "Orig";
        rowOrig.appendChild(pillO);
        const bodyO = document.createElement("div");
        bodyO.className = "content";
        bodyO.textContent = original;
        rowOrig.appendChild(bodyO);
        entry.appendChild(rowOrig);
      }

      const rowTrans = document.createElement("div");
      rowTrans.className = "row";
      const pillT = document.createElement("span");
      pillT.className = "pill transl";
      pillT.textContent = "Transl";
      rowTrans.appendChild(pillT);
      const bodyT = document.createElement("div");
      bodyT.className = "content";
      bodyT.textContent = translation;
      rowTrans.appendChild(bodyT);
      entry.appendChild(rowTrans);

      if (stick) {
        window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
      }
    }

    const es = new EventSource("/events");
    es.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        const { type, original, translation, action } = payload;
        if (type === "entry") {
          renderEntry(action, original, translation);
        }
      } catch (err) {
        console.error("Bad event", err);
      }
    };

    es.onerror = () => {
      console.warn("Connection lost, retrying...");
    };
  </script>
</body>
</html>"""

        return Handler

    def _register_client(self, q: queue.Queue):
        with self._lock:
            self._clients.add(q)

    def _unregister_client(self, q: queue.Queue):
        with self._lock:
            self._clients.discard(q)

    def _send_event(self, event: dict):
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._history_limit:
                self._history = self._history[-self._history_limit :]
            clients = list(self._clients)

        for client in clients:
            try:
                client.put_nowait(event)
            except queue.Full:
                continue
