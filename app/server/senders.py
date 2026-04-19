import json
import logging
import multiprocessing as mp
import os
import queue
import threading
from urllib.parse import parse_qs, urlparse

import requests


logger = logging.getLogger(__name__)


class ZoomCaptionSender:
    """Sends captions to Zoom via its caption API."""

    def __init__(self, source_queue, zoom_url):
        self.source_queue = source_queue
        self.zoom_url = zoom_url
        self.caption_thread = None
        self.is_running = False

        self.meeting_id = self._extract_meeting_id(zoom_url) if zoom_url else None
        if self.meeting_id:
            self.seq_map = self._load_sequence_map()
            self.sequence = self.seq_map.get(self.meeting_id, 0)
        else:
            self.seq_map = {}
            self.sequence = 0

    def start(self):
        if not self.is_running:
            self.caption_thread = threading.Thread(target=self.run)
            self.caption_thread.start()
            self.is_running = True

    def stop(self):
        if self.is_running:
            self.source_queue.put(None)
            self.caption_thread.join()
            self.caption_thread = None
            self.is_running = False

    def run(self):
        while True:
            message = self.source_queue.get()
            if message is None:
                break

            text = message.get("transl_text", "")
            complete = message.get("complete", True)
            lang_code = message.get("target_lang", "")
            if not text.strip():
                continue

            if self.zoom_url and complete:
                # Build URL with lang + sequence params.
                url = f"{self.zoom_url}&lang={lang_code}&seq={self.sequence}"

                headers = {
                    "Content-Type": "plain/text",
                    "Content-Length": str(len(text)),
                }

                try:
                    result = requests.post(url, headers=headers, data=text)
                    if result.ok:
                        self.sequence += 1
                        self.seq_map[self.meeting_id] = self.sequence
                        self._save_seq_map(self.seq_map)
                    else:
                        logger.error(f"[ZoomCaptioner] Error sending to Zoom: [{result.status_code}] {result.text}")
                except requests.RequestException as e:
                    logger.error(f"[ZoomCaptioner] Error sending to Zoom: {e}")

    def wait_until_ready(self, timeout: float = None) -> bool:
        return True  # Zoom caption sender is ready immediately.

    @classmethod
    def _extract_meeting_id(cls, zoom_url):
        parsed = urlparse(zoom_url)
        params = parse_qs(parsed.query)
        return params.get("id", ["unknown"])[0]

    SEQ_FILE = "zoom_seq_map.json"

    @classmethod
    def _load_sequence_map(cls):
        if os.path.exists(cls.SEQ_FILE):
            with open(cls.SEQ_FILE, "r") as f:
                return json.load(f)
        return {}

    @classmethod
    def _save_seq_map(cls, seq_map):
        with open(cls.SEQ_FILE, "w") as f:
            json.dump(seq_map, f)


class ClientCaptionSender:
    """Sends captions back to the client over the same TCP connection."""

    def __init__(self, source_queue: queue.Queue, sender_queue: mp.Queue):
        self.source_queue = source_queue
        self.sender_queue = sender_queue
        self.caption_thread = None
        self.is_running = False

    def start(self):
        if not self.is_running:
            self.caption_thread = threading.Thread(target=self.run)
            self.caption_thread.start()
            self.is_running = True

    def stop(self):
        if self.is_running:
            self.source_queue.put(None)
            self.caption_thread.join()
            self.caption_thread = None
            self.is_running = False

    def run(self):
        while True:
            message = self.source_queue.get()
            if message is None:
                break

            text = message.get("transl_text", "")
            complete = message.get("complete", True)
            lang_code = message.get("target_lang", "")
            if not text.strip():
                continue

            self.sender_queue.put(
                {
                    "type": "translation",
                    "lang": lang_code,
                    "text": text,
                    "complete": complete,
                }
            )

    def wait_until_ready(self, timeout: float = None) -> bool:
        return True  # Caption sender is ready immediately.

__all__ = ["ZoomCaptionSender", "ClientCaptionSender"]
