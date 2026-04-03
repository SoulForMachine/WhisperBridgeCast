import logging
import multiprocessing as mp
import queue
from typing import Any

from app.common.utils import MPCountingQueue
from app.server.asr import ASRProcessor
from app.server.senders import ClientCaptionSender, ZoomCaptionSender
from app.server.translation import Translator


logger = logging.getLogger(__name__)


class WhisperPipeline:
	def __init__(self, client_params: dict[str, Any], sender_queue: mp.Queue):
		self.audio_queue = MPCountingQueue()
		self.asr_queue = MPCountingQueue()
		self.websrv_input_queue = queue.Queue()
		self.sender_queue = sender_queue

		zoom_url = client_params.get("zoom_url")

		transl_params = client_params.get("translation_params")
		language = client_params.get("language")
		target_language = client_params.get("target_language")

		if client_params.get("enable_translation") is True:
			transl_engine = client_params.get("translation_engine", "MarianMT")
		else:
			transl_engine = "none"

		logger.info("Starting Translator thread...")
		self.translator = Translator(
			transl_engine,
			transl_params,
			language,
			target_language,
			self.asr_queue,
			[self.websrv_input_queue],
			sender_queue,
			only_complete_sent=bool(zoom_url),
		)
		self.translator.start()

		self.zoom_caption_sender = None
		self.zoom_caption_sender_queue = None
		if zoom_url:
			self.start_sending_zoom_transcript(zoom_url)

		self.client_caption_sender = None
		self.client_caption_sender_queue = None
		if client_params.get("send_transcript"):
			self.start_sending_client_transcript()

		logger.info("Starting transcript web server...")
		from web_server import WebTranscriptServer

		self.websrv = WebTranscriptServer()
		self.websrv.start(self.websrv_input_queue)

		logger.info("Starting ASR thread...")
		self.asr_proc = ASRProcessor(client_params, self.audio_queue, self.asr_queue, sender_queue)
		self.asr_proc.start()

	def process(self, arr):
		self.audio_queue.put(arr)

		self.sender_queue.put(
			{
				"type": "statistics",
				"values": {
					"asr_in_q_size": self.audio_queue.qsize(),
				},
			}
		)

	def start_sending_client_transcript(self):
		if not self.client_caption_sender:
			logger.info("Starting client caption sender thread...")
			self.client_caption_sender_queue = queue.Queue()
			self.client_caption_sender = ClientCaptionSender(self.client_caption_sender_queue, self.sender_queue)
			self.client_caption_sender.start()
			self.translator.add_output_queue(self.client_caption_sender_queue)

	def stop_sending_client_transcript(self):
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

		logger.info("Translator thread exiting...")
		self.translator.stop()
		self.translator = None

		self.stop_sending_zoom_transcript()
		self.stop_sending_client_transcript()

		logger.info("Web server thread exiting...")
		self.websrv.stop()
		self.websrv = None

	def wait_until_ready(self):
		for cmp in [self.translator, self.client_caption_sender, self.zoom_caption_sender, self.websrv, self.asr_proc]:
			if cmp is not None:
				cmp.wait_until_ready()

__all__ = ["WhisperPipeline"]
