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
        self.pipeline_settings = pipeline_settings
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

    def update_settings(self, new_settings: PipelineSettings):
        old_settings = self.pipeline_settings
        self.pipeline_settings = new_settings

        # 1. Check if ASR needs restart
        asr_needs_restart = False
        if old_settings.asr != new_settings.asr:
            asr_needs_restart = True
        
        # Check VAC Restart requirements
        if old_settings.vac.enable != new_settings.vac.enable or \
           old_settings.vac.enable_whisper_internal_vad != new_settings.vac.enable_whisper_internal_vad:
            asr_needs_restart = True

        if asr_needs_restart:
            logger.info("Restarting ASR processor due to settings update...")
            self.asr_proc.stop()
            self.asr_proc = ASRProcessor(
                new_settings,
                self.audio_queue,
                self.asr_queue,
                self.asr_sender_queue
            )
            self.asr_proc.start()
            self.asr_proc.wait_until_ready()
        elif old_settings.vac != new_settings.vac:
            logger.info("Updating VAC settings on the fly...")
            self.audio_queue.put({"type": "update_vac_settings", "vac": new_settings.vac})

        # 2. Check if Translator needs restart
        translator_needs_restart = False
        old_trans = old_settings.translation
        new_trans = new_settings.translation

        if (old_trans.enable != new_trans.enable or
            old_trans.src_language != new_trans.src_language or
            old_trans.target_language != new_trans.target_language or
            old_trans.engine != new_trans.engine or
            old_trans.engine_params != new_trans.engine_params or
            old_trans.word_increment != new_trans.word_increment):
            translator_needs_restart = True
        
        if translator_needs_restart:
            logger.info("Restarting Translator thread due to settings update...")
            output_queues = self.translator.output_queues
            self.translator.stop()
            self.translator = Translator(
                new_settings.translation,
                self.asr_queue,
                output_queues,
                self.sender_callback,
                only_complete_sent=bool(new_settings.zoom_url)
            )
            self.translator.start()
            self.translator.wait_until_ready()
        elif old_trans.source_diff_enabled != new_trans.source_diff_enabled or old_trans.target_diff_enabled != new_trans.target_diff_enabled:
            logger.info("Updating Translation settings on the fly...")
            self.translator.source_diff_enabled = new_settings.translation.source_diff_enabled
            self.translator.target_diff_enabled = new_settings.translation.target_diff_enabled

        # 3. Zoom URL updates
        old_zoom = old_settings.zoom_url
        new_zoom = new_settings.zoom_url
        if old_zoom != new_zoom:
            if old_zoom:
                self.stop_sending_zoom_transcript()
            if new_zoom:
                self.start_sending_zoom_transcript(new_zoom)
                
            self.translator.only_complete_sent = bool(new_zoom)
