## Purpose

This document gives actionable, repository-specific guidance for AI coding agents working on WhisperBridgeCast. It focuses on the runtime architecture, data/protocol shapes, developer workflows, and the concrete files to touch for common changes.

## Big-picture architecture (short)

- Two main user-facing CLIs (declared in `pyproject.toml`): `whisper_server` (server) and `captioner_gui` (GUI client).
- Server receives a JSON params message over a simple TCP binary protocol, then streamed audio (float32 numpy bytes). See `app.common.net_common` for header/message format.
- Server pipeline: `app.server.tcp_server.WhisperServer` -> `app.server.pipeline.WhisperPipeline` which wires:
  - ASR (runs in a multiprocessing subprocess via `app.server.asr.ASRProcessor` / `whisper_streamer.asr_factory`)
  - Translator thread (`app.server.translation.Translator`)
  - Web transcript SSE server (`app.web.transcript_server.WebTranscriptServer`) for live replay/UI
  - Optional senders (Zoom / client caption sender)

## Important files to read first

- `pyproject.toml` — packaging & console scripts
- `app/server/main.py` — server CLI, sets multiprocessing start method (spawn) on startup
- `app/server/tcp_server.py` — connection handling, message loop, pipeline lifecycle
- `app/server/pipeline.py` — how ASR / Translator / web server are wired
- `app/common/net_common.py` — wire protocol (1-byte type + 4-byte length header); JSON vs audio frames
- `app/server/whisper_streamer.py` — ASR frontend & online processor logic (timestamps, VAD wrapper)
- `app/gui/network.py` — client-side protocol & state machine (connect, send params, stream audio, receive JSON)
- `app/web/transcript_server.py` — SSE server used for the web UI
- `app/server/asr_backends/` and `app/server/transl_backends/` — backend extension points
- `tests/test_translator.py` — concrete unit tests illustrating translator behavior and message shapes

## Runtime / developer workflows

- Install editable for development:
  - pip install -e .
  - Optional extras: pip install -e .[optional] or .[all]
- Run server (from repo root):
  - whisper_server --help
  - whisper_server --warmup-file data/samples_jfk.wav --log-level DEBUG
  - Note: on Windows the server sets mp start method to `spawn` in `app.server.main` — keep that when refactoring.
- Run GUI: `captioner_gui` (starts the desktop UI which uses `app.gui.network.WhisperClient`).
- Tests: `pytest` from repo root (see `tests/test_translator.py` for patterns). Prefer running single test modules during development.

## Protocol and data shapes (concrete)

- net_common header: 1 byte type + 4 bytes (big-endian) length. Types: JSON (1) or AUDIO (2).
- JSON messages: UTF-8 encoded JSON objects. The client MUST send params JSON first. Example server-ready handshake in `tcp_server.handle_client`:
  - server expects: msg_type == "json" and params dict
  - server replies: netc.send_json(conn, {"type": "status", "value": {"status": "ready"}})
- Audio messages: raw float32 little-endian bytes containing a numpy.ndarray.tobytes() representation. Use dtype=np.float32 on sender side (see `app.gui.network.WhisperClient`).
- Server-to-client messages: JSON with `type` values: `status`, `statistics`, `translation` (look at `tcp_server.sender_thread_func` and places that put messages on sender_queue).

## Concurrency & lifecycle patterns to respect

- Threads vs processes:
  - Networking + control runs on threads in `tcp_server`.
  - ASR heavy work runs in a multiprocessing.Process (see `ASRProcessor.start` -> `asr_subprocess_main`). Use mp-friendly primitives (mp.Queue, mp.Event, MPCountingQueue).
  - Translator and senders are threads and use `queue.Queue`.
- Queue types: prefer `multiprocessing.Queue` or `MPCountingQueue` when crossing process boundaries (MPCountingQueue is in `app.common.utils`). Use `queue.Queue` for same-process threads.
- Shutdown: many components follow pattern `.stop()` that signals an Event/puts None into queues and joins threads/processes. Mirror those patterns when adding subsystems.

## Extension points (where to add code)

- Add ASR backends: place new backend in `app/server/asr_backends/` and follow `base.py` interface used by `whisper_streamer.asr_factory`.
- Add translation backends: place new backend in `app/server/transl_backends/` and ensure it subclasses `TranslBase`; `discover_backend_classes()` is used to find backends.
- Client protocol features: `app.gui/network.py` and `app.common.net_common.py` together define how to add control messages (they use `{"type":"control", "command": ...}` messages).

## Project-specific conventions & gotchas

- Warmup audio: server uses `app/server/data/*.wav` by default. `ASRProcessor` resolves warmup path relative to server code, repo root, and CWD (see `WhisperOnline._resolve_warmup_file`). Tests and local runs rely on this behavior.
- Logging and events: components send `statistics` and `status` objects through `sender_queue` — tests and UI listen for these exact keys.
- Multiprocessing start method: `spawn` is forced in `app.server.main` to keep cross-platform behavior — don't remove that without verifying Windows compatibility.
- Audio dtype: audio chunks are float32 arrays (np.float32). Ensure third-party clients follow the same representation.

## Quick examples to reference in edits

- Add a small translation backend: inspect `app/server/transl_backends/base.py` and `tests/test_translator.py` for tests that instantiate engines directly (the repository uses a DummyBackend pattern there).
- To change the server protocol, modify `app/common/net_common.py` and update both `app/gui/network.py` and `app/server/tcp_server.py` accordingly.

## How to validate changes quickly

- Unit tests: run `pytest tests/test_translator.py::test_get_sentences_buffer_transitions_partial_then_complete` to exercise translator sentence logic quickly.
- Smoke run: `whisper_server --warmup-file data/samples_jfk.wav --log-level DEBUG` and then run a small client script that connects and sends the params JSON + a few float32 audio frames (see `app/gui/network.py` for a minimal example).

## Where to ask follow-ups / what I might be missing

- If you need runtime secrets or cloud keys (OpenAI / Google), they are supplied via optional extras and client params — there is no secrets store in repo.
- If you plan to change multiprocessing startup (fork vs spawn) or add native extensions, tell me which platform and I will flag needed adjustments.

---

If you'd like, I can iterate and (a) add a tiny example client script under `tools/` that sends a short audio chunk to the server for local smoke tests, or (b) include a short checklist for PR reviewers. Which would you prefer? 
