from __future__ import annotations

from app.server import tcp_server


def test_start_sets_running_before_thread_creation_and_is_idempotent(monkeypatch) -> None:
    created_threads = []
    server_ref = {}

    class FakeThread:
        def __init__(self, target=None, args=(), kwargs=None):
            assert server_ref["server"].is_running is True
            self.target = target
            self.args = args
            self.kwargs = kwargs or {}
            created_threads.append(self)

        def start(self):
            self.started = True

        def join(self):
            self.joined = True

    monkeypatch.setattr(tcp_server.threading, "Thread", FakeThread)

    server = tcp_server.WhisperServer()
    server_ref["server"] = server

    server.start()

    assert server.is_running is True
    assert len(created_threads) == 1

    server.start()

    assert len(created_threads) == 1

    server.stop()

    assert server.is_running is False
    assert server.is_stopping is False
    assert server.server_thread is None
