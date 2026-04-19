import logging
import multiprocessing as mp
import socket
import threading

from app.common import net_common as netc

from app.server.pipeline import WhisperPipeline
from app.server.settings import ServerSettings, pipeline_settings_from_dict


logger = logging.getLogger(__name__)


class WhisperServer:
    def __init__(self, host="0.0.0.0", port=5000, warmup_file=None):
        self.conn = None
        self.server_thread = None
        self.is_running = False
        self.is_stopping = False
        self.host = host
        self.port = port
        self.listen_socket = None
        self.client_socket = None
        self.warmup_file = warmup_file

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
        pipeline = None
        wav_out = None
        clean_shutdown = False
        sender_queue = mp.Queue()
        import json

        try:
            # Start the sender thread.
            sender_thread = threading.Thread(target=self.sender_thread_func, args=(conn, sender_queue))
            sender_thread.start()

            # Receive pipeline settings from the client and initialize the pipeline.
            msg_type, pipeline_payload = netc.recv_message(conn)
            if pipeline_payload is None or msg_type != "json":
                logger.error("Failed to receive pipeline settings from the client.")
                self._close_socket(conn)
                return
            logger.info(f"Received pipeline settings:\n{json.dumps(pipeline_payload, indent=2)}")

            pipeline_settings = pipeline_settings_from_dict(pipeline_payload)
            if self.warmup_file:
                pipeline_settings.asr.warmup_file = self.warmup_file

            pipeline = WhisperPipeline(pipeline_settings, sender_queue)

            # Start the wav file writer if needed.
            if pipeline_settings.write_wav:
                from app.server import wav_writer
                from datetime import datetime

                filename = datetime.now().strftime("recording-%Y%m%d_%H%M%S.wav")
                wav_out = wav_writer.WavWriter(filename)

            # When the pipeline is ready, inform the client.
            pipeline.wait_until_ready()

            netc.send_json(conn, {"type": "status", "value": {"status": "ready"}})

            # Receive audio chunks and control messages.
            while True:
                msg_type, msg = netc.recv_message(conn)
                if msg is None:
                    logger.error("Invalid message received.")
                    break

                if msg_type == "audio":
                    if len(msg) == 0:
                        continue

                    # Send received chunk to the pipeline.
                    pipeline.process(msg)

                    if pipeline_settings.write_wav:
                        wav_out.write_chunk(msg)

                elif msg_type == "json":
                    if msg.get("type") == "control":
                        command = msg.get("command")
                        match command:
                            case "start_sending_client_transcript":
                                pipeline.start_sending_client_transcript()
                            case "stop_sending_client_transcript":
                                pipeline.stop_sending_client_transcript()
                            case "stop":
                                logger.info("Client gracefully disconnecting.")
                                clean_shutdown = True
                                break

        except OSError as e:
            if not self.is_stopping:
                logger.error(f"Connection lost (receiver): {e}.")
        except Exception as e:
            logger.error(f"Receiver exception: {e}.")
        finally:
            # Stop the pipeline. This will flush any remaining text.
            if pipeline:
                pipeline.stop()

            if clean_shutdown:
                # Confirm shutdown request to the client.
                sender_queue.put({"type": "status", "value": {"status": "conn_shutdown"}})

            # Stop the sender thread.
            sender_queue.put(None)
            sender_thread.join()

            # Close the connection.
            self._close_socket(conn)
            logger.info("Connection closed.")

            if wav_out:
                wav_out.close()

    def sender_thread_func(self, conn: socket.socket, sender_queue: mp.Queue):
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
