import logging
import multiprocessing as mp
import queue
import threading
from typing import Callable

from app.common.utils import MPCountingQueue
from app.server.asr import ASRProcessor
from app.server.senders import ClientCaptionSender, ZoomCaptionSender
from app.server.settings import PipelineSettings
from app.server.translation import Translator
from web_server import WebTranscriptServer


logger = logging.getLogger(__name__)


class WhisperPipeline:
    def __init__(self, pipeline_settings: PipelineSettings, sender_callback: Callable[[dict], None]):
        self.audio_queue = MPCountingQueue()
        self.asr_queue = MPCountingQueue()
        self.websrv_input_queue = queue.Queue()
        self.sender_callback = sender_callback
        self.asr_sender_queue = mp.Queue()
        self.asr_sender_thread = None
        self.client_caption_sender_lock = threading.Lock()

        zoom_url = pipeline_settings.zoom_url

        logger.info("Starting Translator thread...")
        self.translator = Translator(
            pipeline_settings.translation,
            self.asr_queue,
            [self.websrv_input_queue],
            sender_callback,
            only_complete_sent=bool(zoom_url),
        )
        self.translator.start()

        self.zoom_caption_sender = None
        self.zoom_caption_sender_queue = None
        if zoom_url:
            self.start_sending_zoom_transcript(zoom_url)

        self.client_caption_sender = None
        self.client_caption_sender_queue = None

        logger.info("Starting transcript web server...")

        self.websrv = WebTranscriptServer()
        self.websrv.start(self.websrv_input_queue)

        self.asr_sender_thread = threading.Thread(target=self._forward_asr_messages, daemon=True)
        self.asr_sender_thread.start()

        logger.info("Starting ASR thread...")
        self.asr_proc = ASRProcessor(
            pipeline_settings,
            self.audio_queue,
            self.asr_queue,
            self.asr_sender_queue
        )
        self.asr_proc.start()

    def process(self, arr):
        self.audio_queue.put(arr)

        self.sender_callback(
            {
                "type": "statistics",
                "values": {
                    "asr_in_q_size": self.audio_queue.qsize(),
                },
            }
        )

    def _forward_asr_messages(self):
        while True:
            msg = self.asr_sender_queue.get()
            if msg is None:
                break
            self.sender_callback(msg)

    def start_sending_client_transcript(self):
        with self.client_caption_sender_lock:
            if not self.client_caption_sender:
                logger.info("Starting client caption sender thread...")
                self.client_caption_sender_queue = queue.Queue()
                self.client_caption_sender = ClientCaptionSender(self.client_caption_sender_queue, self.sender_callback)
                self.client_caption_sender.start()
                self.translator.add_output_queue(self.client_caption_sender_queue)

    def stop_sending_client_transcript(self):
        with self.client_caption_sender_lock:
            if self.client_caption_sender:
                logger.info("Stopping client caption sender thread...")
                if self.translator:
                    self.translator.remove_output_queue(self.client_caption_sender_queue)
                self.client_caption_sender.stop()
                self.client_caption_sender = None
                self.client_caption_sender_queue = None

    def start_sending_zoom_transcript(self, zoom_url):
        if not self.zoom_caption_sender:
            logger.info("Starting Zoom caption sender thread...")
            zoom_url = zoom_url.strip()
            self.zoom_caption_sender_queue = queue.Queue()
            self.zoom_caption_sender_queue.put(("...", True))
            self.zoom_caption_sender = ZoomCaptionSender(self.zoom_caption_sender_queue, zoom_url)
            self.zoom_caption_sender.start()
            self.translator.add_output_queue(self.zoom_caption_sender_queue)

    def stop_sending_zoom_transcript(self):
        if self.zoom_caption_sender:
            logger.info("Stopping Zoom caption sender thread...")
            if self.translator:
                self.translator.remove_output_queue(self.zoom_caption_sender_queue)
            self.zoom_caption_sender.stop()
            self.zoom_caption_sender = None
            self.zoom_caption_sender_queue = None

    def stop(self):
        logger.info("Stopping all threads...")

        logger.info("ASR thread exiting...")
        self.asr_proc.stop()
        self.asr_proc = None
        self.asr_sender_queue.put(None)
        self.asr_sender_thread.join()
        self.asr_sender_thread = None

        logger.info("Translator thread exiting...")
        self.translator.stop()
        self.translator = None

        self.stop_sending_zoom_transcript()
        self.stop_sending_client_transcript()

        logger.info("Web server thread exiting...")
        self.websrv.stop()
        self.websrv = None

    def wait_until_ready(self, timeout: float = None) -> bool:
        for cmp in [self.translator, self.client_caption_sender, self.zoom_caption_sender, self.websrv, self.asr_proc]:
            if cmp is not None and not cmp.wait_until_ready(timeout=timeout):
                return False
        return True

__all__ = ["WhisperPipeline"]
