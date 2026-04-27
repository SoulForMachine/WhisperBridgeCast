import logging
import queue
import socket
import threading
from typing import Callable

import numpy as np

from app.common import net_common as netc

logger = logging.getLogger(__name__)


class WhisperClient:
    def __init__(
            self,
            server_url: str,
            port: int,
            net_send_queue: queue.Queue,
            net_recv_queue: queue.Queue,
            notif_callback: Callable[[str, dict], None]=None):
        self.server_url = server_url
        self.port = port
        self.net_send_queue = net_send_queue
        self.net_recv_queue = net_recv_queue
        self.notif_callback = notif_callback if notif_callback else lambda et, d: None
        self.connected_event = None
        self.results_thread = None
        self.whisper_client_thread = None
        self.stop_event = None
        self.is_running = False

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.connected_event = threading.Event()
            self.stop_event = threading.Event()
            self.whisper_client_thread = threading.Thread(target=self.run)
            self.whisper_client_thread.start()

    def stop(self):
        if self.is_running:
            self.is_running = False
            self.net_send_queue.put(None)
            self.whisper_client_thread.join()
            self.whisper_client_thread = None
            self.connected_event = None
            self.stop_event = None

    def run(self):
        self.notif_callback("client_status", {"status": "connecting"})
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.server_url, self.port))
        except Exception as e:
            logger.error(f"Could not connect to the whisper server: {e}")
            self.notif_callback("client_status", {"status": "conn_error", "message": str(e)})
            return

        self.connected_event.set()
        self.notif_callback("client_status", {"status": "connected"})
        logger.info(f"Connected to whisper server at {self.server_url}:{self.port}.")

        # Start a thread to receive status, statistics and translation messages.
        self.results_thread = threading.Thread(target=self.listen_for_results, args=(sock,))
        self.results_thread.start()

        # Stream audio and control messages.
        try:
            while True:
                data = self.net_send_queue.get()
                if data is None:
                    # Tell the server we're done, results thread should receive shutdown confirmation
                    netc.send_json(sock, {
                        "type": "control",
                        "command": "stop"
                    })
                    sock.shutdown(socket.SHUT_WR)
                    self.notif_callback("client_status", {"status": "disconnecting"})
                    break

                if type(data) == np.ndarray:
                    netc.send_ndarray(sock, data)
                else:
                    netc.send_json(sock, data)

        except OSError as e:
            logger.error(f"Connection lost: {e}")
            self.notif_callback("client_status", {"status": "conn_lost", "source": "sender", "message": str(e)})
        except Exception as e:
            logger.error(f"Sender exception: {e}.")
            self.notif_callback("client_status", {"status": "conn_lost", "source": "sender", "message": str(e)})
        finally:
            # The results thread will receive a shutdown signal from the server,
            # but we also set a local stop event in case of connection loss to unblock it.
            self.results_thread.join(timeout=2.0)
            if self.results_thread.is_alive():
                self.stop_event.set()  # signal the results thread to stop
                self.results_thread.join()  # wait again for it to finish

            self.results_thread = None
            self.connected_event = None
            self.stop_event = None
            sock.close()

    def listen_for_results(self, sock):
        srv_status_strs = {
            "ready": "Server is ready to receive audio.",
            "conn_shutdown": "Connection has been shut down.",
            "translator_initializing": "Translation engine is initializing...",
            "translator_initialized": "Translation engine initialized.",
            "asr_initializing": "ASR engine is initializing...",
            "asr_initialized": "ASR engine initialized."
        }

        # Receive status, statistics and translation messages from the server.
        while not self.stop_event.is_set():
            try:
                msg_type, msg = netc.recv_message(sock)
            except OSError as e:
                logger.error(f"Connection lost: {e}.")
                self.notif_callback("client_status", {"status": "conn_lost", "source": "receiver", "message": str(e)})
                break
            except ValueError as e:  # JSON decode errors from corrupted/partial messages
                logger.error(f"Receiver JSON error: {e}.")
                continue
            except Exception as e:
                logger.error(f"Receiver exception: {e}.")
                self.notif_callback("client_status", {"status": "conn_lost", "source": "receiver", "message": str(e)})
                break

            if msg is None or msg_type != "json":
                continue

            msg_type = msg.get("type")
            if msg_type == "translation":
                text, complete = msg.get("text"), msg.get("complete")
                if text and complete is not None:
                    self.net_recv_queue.put((text, complete))
            elif msg_type == "statistics":
                values = msg.get("values", {})
                self.notif_callback("server_statistics", values)
            elif msg_type == "status":
                value = msg.get("value")
                self.notif_callback("server_status", value)

                status = value.get("status", "")
                status_message = srv_status_strs.get(status)
                if status_message:
                    logger.info(status_message)

                # Break the loop and end the thread if we receive a shutdown status from the server,
                # which is a confirmation that it received our stop signal and is closing the connection.
                if status == "conn_shutdown":
                    break
            else:
                logger.warning(f"Unknown message: {msg}")

    def wait_until_connected(self, timeout: float) -> bool:
        return self.connected_event.wait(timeout=timeout)

__all__ = ["WhisperClient"]

