import logging
import queue
import socket
import threading

from app.common import net_common as netc
from app.common.utils import dataclass_from_dict

from app.server.pipeline import WhisperPipeline
from app.server.settings import PipelineSettings


logger = logging.getLogger(__name__)


class WhisperServer:
    def __init__(self, host="0.0.0.0", port=5000, warmup_file=None):
        self.server_thread = None
        self.pipeline_start_thread = None
        self.is_running = False
        self.is_stopping = False
        self.host = host
        self.port = port
        self.listen_socket = None
        self.client_socket = None
        self.warmup_file = warmup_file
        self.pipeline = None
        self.pipeline_state = "stopped"  # stopped | starting | ready | stopping
        self.pipeline_lock = threading.Lock()
        self.client_sender_queue = None
        self.client_sender_queue_lock = threading.Lock()
        self.lifecycle_lock = threading.Lock()

    def start(self):
        with self.lifecycle_lock:
            if self.is_running:
                return

            self.is_running = True
            self.is_stopping = False
            self.server_thread = threading.Thread(target=self._run)
            server_thread = self.server_thread

        server_thread.start()

    def stop(self):
        with self.lifecycle_lock:
            if not self.is_running:
                return

            self.is_stopping = True
            server_thread = self.server_thread
            listen_socket = self.listen_socket
            client_socket = self.client_socket

        if listen_socket is not None:
            self._close_socket(listen_socket)

        if client_socket is not None:
            self._close_socket(client_socket)

        if server_thread is not None:
            server_thread.join()

        with self.pipeline_lock:
            startup_thread = self.pipeline_start_thread
            self.pipeline_start_thread = None

        if startup_thread is not None:
            startup_thread.join()

        self._stop_pipeline()

        with self.lifecycle_lock:
            self.server_thread = None
            self.listen_socket = None
            self.client_socket = None
            self.is_stopping = False
            self.is_running = False

    def _run(self):
        listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with self.lifecycle_lock:
            self.listen_socket = listen_socket

        try:
            listen_socket.bind((self.host, self.port))
            listen_socket.listen(1)

            while True:
                with self.lifecycle_lock:
                    if self.is_stopping:
                        break

                logger.info(f"Server listening on {self.host}:{self.port}")
                client_socket, addr = listen_socket.accept()
                with self.lifecycle_lock:
                    self.client_socket = client_socket
                logger.info(f"Connected by {addr}")
                self._handle_client(client_socket)
                with self.lifecycle_lock:
                    self.client_socket = None
        except Exception as e:
            if not self.is_stopping:
                logger.error(f"Server error: {e}")
        finally:
            logger.info("Server shutting down.")
            self._close_socket(listen_socket)
            with self.lifecycle_lock:
                if self.listen_socket is listen_socket:
                    self.listen_socket = None
                self.is_running = False

    def _handle_client(self, conn: socket.socket):
        client_sender_queue = queue.Queue()
        sender_thread = None

        try:
            with self.client_sender_queue_lock:
                self.client_sender_queue = client_sender_queue

            sender_thread = threading.Thread(target=self._sender_thread_func, args=(conn, client_sender_queue))
            sender_thread.start()

            # Inform the client about the current server state.
            client_sender_queue.put({"type": "status", "value": {"status": "connected"}})

            with self.pipeline_lock:
                pipeline_state = self.pipeline_state

            match pipeline_state:
                case "ready":
                    client_sender_queue.put({"type": "status", "value": {"status": "ready"}})
                case "starting":
                    client_sender_queue.put({"type": "status", "value": {"status": "starting_pipeline"}})
                case "stopping":
                    client_sender_queue.put({"type": "status", "value": {"status": "stopping_pipeline"}})

            # Receive audio chunks and control messages.
            while True:
                msg_type, msg = netc.recv_message(conn)
                if msg is None:
                    logger.error("Invalid message received.")
                    break

                if msg_type == "audio":
                    if len(msg) == 0:
                        continue

                    with self.pipeline_lock:
                        pipeline = self.pipeline
                        pipeline_state = self.pipeline_state

                    if pipeline is not None and pipeline_state == "ready":
                        pipeline.process(msg)

                elif msg_type == "json":
                    if msg.get("type") == "control":
                        command = msg.get("command")
                        match command:
                            case "start_pipeline":
                                pipeline_payload = msg.get("settings")
                                if pipeline_payload is None:
                                    logger.error("Received start_pipeline without settings payload.")
                                    continue

                                with self.pipeline_lock:
                                    can_start = self.pipeline_state == "stopped" and self.pipeline is None

                                if can_start:
                                    import json

                                    logger.info(f"Received pipeline settings:\n{json.dumps(pipeline_payload, indent=2)}")
                                    pipeline_settings = dataclass_from_dict(PipelineSettings, pipeline_payload)
                                    self._start_pipeline_async(pipeline_settings)

                            case "stop_pipeline":
                                with self.pipeline_lock:
                                    pipeline_state = self.pipeline_state
                                if pipeline_state == "ready":
                                    self._stop_pipeline()
                                    client_sender_queue.put({"type": "status", "value": {"status": "connected"}})

                            case "start_sending_client_transcript":
                                with self.pipeline_lock:
                                    pipeline = self.pipeline
                                    pipeline_state = self.pipeline_state
                                if pipeline is not None and pipeline_state == "ready":
                                    pipeline.start_sending_client_transcript()

                            case "stop_sending_client_transcript":
                                with self.pipeline_lock:
                                    pipeline = self.pipeline
                                    pipeline_state = self.pipeline_state
                                if pipeline is not None and pipeline_state == "ready":
                                    pipeline.stop_sending_client_transcript()

                            case "stop":
                                logger.info("Client gracefully disconnecting.")
                                # Confirm shutdown request to the client.
                                client_sender_queue.put({"type": "status", "value": {"status": "conn_shutdown"}})
                                break

        except OSError as e:
            if not self.is_stopping:
                logger.error(f"Connection lost (receiver): {e}.")
        except Exception as e:
            logger.error(f"Receiver exception: {e}.")
        finally:
            with self.client_sender_queue_lock:
                if self.client_sender_queue is client_sender_queue:
                    self.client_sender_queue = None

            if sender_thread is not None:
                client_sender_queue.put(None)
                sender_thread.join()

            # Close the connection.
            self._close_socket(conn)
            logger.info("Connection closed.")

    def _start_pipeline_async(self, pipeline_settings: PipelineSettings):
        with self.pipeline_lock:
            if self.pipeline is not None or self.pipeline_state != "stopped":
                return

            self.pipeline_state = "starting"
            self.pipeline_start_thread = threading.Thread(
                target=self._start_pipeline,
                args=(pipeline_settings,),
                daemon=True,
            )
            self.pipeline_start_thread.start()

        self._send_to_client({"type": "status", "value": {"status": "starting_pipeline"}})

    def _start_pipeline(self, pipeline_settings: PipelineSettings):
        if self.warmup_file:
            pipeline_settings.asr.warmup_file = self.warmup_file

        pipeline = WhisperPipeline(pipeline_settings, self._send_to_client)
        pipeline.wait_until_ready()

        with self.pipeline_lock:
            self.pipeline = pipeline
            self.pipeline_state = "ready"
            self.pipeline_start_thread = None

        self._send_to_client({"type": "status", "value": {"status": "ready"}})

    def _stop_pipeline(self):
        with self.pipeline_lock:
            if self.pipeline_state == "stopped" or self.pipeline is None:
                return

            self.pipeline_state = "stopping"
            pipeline = self.pipeline
            self.pipeline = None

        self._send_to_client({"type": "status", "value": {"status": "stopping_pipeline"}})
        pipeline.stop()

        with self.pipeline_lock:
            self.pipeline_state = "stopped"

    def _send_to_client(self, message: dict):
        with self.client_sender_queue_lock:
            sender_queue = self.client_sender_queue

        if sender_queue is not None:
            sender_queue.put(message)

    def _sender_thread_func(self, conn: socket.socket, sender_queue: queue.Queue):
        try:
            while True:
                msg = sender_queue.get()
                if msg is None:
                    break
                netc.send_json(conn, msg)
        except OSError as e:
            if not self.is_stopping:
                logger.error(f"Connection lost (sender): {e}.")
        except Exception as e:
            logger.error(f"Sender exception: {e}.")

    def _close_socket(self, sock: socket.socket):
        try:
            sock.close()
        except Exception:
            pass

__all__ = ["WhisperServer"]
