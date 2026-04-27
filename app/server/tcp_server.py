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
        self.pipeline_ready = False
        self.pipeline_starting = False
        self.pipeline_lock = threading.Lock()
        self.client_sender_queue = None
        self.client_sender_queue_lock = threading.Lock()

    def start(self):
        if not self.is_running:
            self.server_thread = threading.Thread(target=self.run)
            self.server_thread.start()

    def stop(self):
        if self.is_running:
            self.is_stopping = True

            if self.listen_socket:
                self._close_socket(self.listen_socket)

            if self.client_socket:
                self._close_socket(self.client_socket)

            if self.server_thread:
                self.server_thread.join()
                self.server_thread = None

            self._stop_pipeline()

            self.listen_socket = None
            self.client_socket = None
            self.is_stopping = False

    def run(self):
        self.is_running = True
        self.listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listen_socket.bind((self.host, self.port))
        self.listen_socket.listen(1)

        try:
            while not self.is_stopping:
                logger.info(f"Server listening on {self.host}:{self.port}")
                self.client_socket, addr = self.listen_socket.accept()
                logger.info(f"Connected by {addr}")
                self.handle_client(self.client_socket)
                self.client_socket = None
        except Exception as e:
            if not self.is_stopping:
                logger.error(f"Server error: {e}")
        finally:
            logger.info("Server shutting down.")
            self._close_socket(self.listen_socket)
            self.is_running = False

    def handle_client(self, conn: socket.socket):
        client_sender_queue = queue.Queue()
        sender_thread = None

        try:
            with self.client_sender_queue_lock:
                self.client_sender_queue = client_sender_queue

            sender_thread = threading.Thread(target=self.sender_thread_func, args=(conn, client_sender_queue))
            sender_thread.start()

            # Inform the client about the current server state.
            client_sender_queue.put({"type": "status", "value": {"status": "connected"}})
            with self.pipeline_lock:
                pipeline_ready = self.pipeline_ready
                pipeline_starting = self.pipeline_starting
            if pipeline_ready:
                client_sender_queue.put({"type": "status", "value": {"status": "ready"}})
            elif pipeline_starting:
                client_sender_queue.put({"type": "status", "value": {"status": "starting_pipeline"}})

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
                        pipeline_ready = self.pipeline_ready

                    if pipeline is not None and pipeline_ready:
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
                                    can_start = self.pipeline is None and not self.pipeline_starting

                                if can_start:
                                    import json

                                    logger.info(f"Received pipeline settings:\n{json.dumps(pipeline_payload, indent=2)}")
                                    pipeline_settings = dataclass_from_dict(PipelineSettings, pipeline_payload)
                                    self._start_pipeline_async(pipeline_settings)

                            case "stop_pipeline":
                                with self.pipeline_lock:
                                    pipeline_ready = self.pipeline_ready
                                if pipeline_ready:
                                    self._stop_pipeline()
                                    client_sender_queue.put({"type": "status", "value": {"status": "connected"}})

                            case "start_sending_client_transcript":
                                with self.pipeline_lock:
                                    pipeline = self.pipeline
                                    pipeline_ready = self.pipeline_ready
                                if pipeline is not None and pipeline_ready:
                                    pipeline.start_sending_client_transcript()

                            case "stop_sending_client_transcript":
                                with self.pipeline_lock:
                                    pipeline = self.pipeline
                                    pipeline_ready = self.pipeline_ready
                                if pipeline is not None and pipeline_ready:
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
            if self.pipeline is not None or self.pipeline_starting:
                return

            self.pipeline_starting = True
            self.pipeline_ready = False
            self.pipeline_start_thread = threading.Thread(
                target=self._start_pipeline,
                args=(pipeline_settings,),
                daemon=True,
            )
            self.pipeline_start_thread.start()
            self.client_sender_queue.put({"type": "status", "value": {"status": "starting_pipeline"}})

    def _start_pipeline(self, pipeline_settings: PipelineSettings):
        if self.warmup_file:
            pipeline_settings.asr.warmup_file = self.warmup_file

        pipeline = WhisperPipeline(pipeline_settings, self._send_to_client)
        pipeline.wait_until_ready()

        with self.pipeline_lock:
            self.pipeline = pipeline
            self.pipeline_ready = True
            self.pipeline_starting = False
            self.pipeline_start_thread = None

        self._send_to_client({"type": "status", "value": {"status": "ready"}})

    def _stop_pipeline(self):
        with self.pipeline_lock:
            startup_thread = self.pipeline_start_thread
            self.pipeline_ready = False
            self.pipeline_starting = False

        if startup_thread is not None and startup_thread.is_alive():
            startup_thread.join()

        with self.pipeline_lock:
            pipeline = self.pipeline
            self.pipeline = None
            self.pipeline_ready = False

        if pipeline is not None:
            pipeline.stop()

    def _send_to_client(self, message: dict):
        with self.client_sender_queue_lock:
            sender_queue = self.client_sender_queue

        if sender_queue is not None:
            sender_queue.put(message)

    def sender_thread_func(self, conn: socket.socket, sender_queue: queue.Queue):
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
