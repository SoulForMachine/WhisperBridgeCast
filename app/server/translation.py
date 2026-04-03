import logging
import multiprocessing as mp
import queue
import threading

from app.server.transl_backends import (
    TranslBase,
    discover_backend_classes,
)


logger = logging.getLogger(__name__)


def get_lang_code(lang: str) -> str:
    lang_code_map = {
        "English": "en",
        "German": "de",
        "Serbian": "sr",
        "Serbian Latin": "sr",
        "Serbian Cyrillic": "sr",
    }
    return lang_code_map.get(lang, "")


class Translator:
    BACKEND_CLASSES: tuple[type[TranslBase], ...] = discover_backend_classes()

    def __init__(self, engine_id, transl_params, src_lang, target_lang, source_queue, output_queues, sender_queue: mp.Queue, only_complete_sent: bool):
        self.engine_id = engine_id
        self.transl_params = transl_params or {}
        self.engine = None
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.src_lang_code = get_lang_code(src_lang)
        self.target_lang_code = get_lang_code(target_lang)
        self.source_queue = source_queue
        self.output_queues = output_queues
        self.sender_queue = sender_queue
        self.only_complete_sent = only_complete_sent
        self.current_text = ""
        self.unconfirmed_text = ""

        self.loop_thread = None
        self.translation_thread = None
        self.translation_queue = queue.Queue()
        self.is_running = False
        self.transl_ready_event = None
        self.shutdown_event = None

        self.next_text_id = 0
        self.pending_partial_id = None

        import spacy

        self.nlp = spacy.blank(self.src_lang_code)
        self.nlp.add_pipe("sentencizer")

    FLUSH_TIMEOUT = 6

    def _buffered_word_count(self):
        # Approximate word count by counting spaces. This is not exact but should be sufficient for statistics.
        return self.current_text.count(" ") + 1 if self.current_text else 0

    def _send_buffered_text_stats(self):
        self.sender_queue.put(
            {
                "type": "statistics",
                "values": {
                    "transl_buffer_word_count": self._buffered_word_count(),
                },
            }
        )

    def add_text(self, confirmed: str, unconfirmed: str):
        # Append new text to the current buffer.
        self.current_text += confirmed
        self.unconfirmed_text = unconfirmed

    def get_sentences(self) -> list[tuple[str, str, bool]]:
        doc = self.nlp(self.current_text)

        def fix_sent(text):
            text = text.strip()
            text = text[:1].upper() + text[1:] if text else text
            return text

        sentences = [(fix_sent(sent.text), "", True) for sent in doc.sents]
        added_unconfirmed = False

        def is_sentence_complete(sent):
            if not sent.endswith((".", "!", "?")):
                return False

            next_piece = self.unconfirmed_text.lstrip()
            if next_piece:
                ch = next_piece[0]
                if ch.islower() or ch.isdigit():
                    return False

            return True

        if sentences:
            last_sent = sentences[-1][0]
            if not is_sentence_complete(last_sent):
                # last sentence is partial
                sentences[-1] = (last_sent, self.unconfirmed_text, False)
                self.current_text = last_sent
                added_unconfirmed = True
            else:
                self.current_text = ""

        if not added_unconfirmed and self.unconfirmed_text != "":
            sentences.append(("", self.unconfirmed_text, False))

        return sentences

    def translate_and_send(self, text: tuple[str, str, bool]):
        confirmed_text, unconfirmed_text, complete = text
        text_id = self._resolve_text_id(complete)

        for out_q in self.output_queues.copy():
            out_q.put(
                {
                    "id": text_id,
                    "src_lang": self.src_lang_code,
                    "orig_text": confirmed_text,
                    "orig_unconfirmed_text": unconfirmed_text,
                    "complete": complete,
                }
            )

        if self.engine is not None:
            self.translation_queue.put(
                {
                    "id": text_id,
                    "text": confirmed_text + unconfirmed_text,
                    "complete": complete,
                }
            )

    def _next_id(self) -> int:
        self.next_text_id += 1
        return self.next_text_id

    def _resolve_text_id(self, complete: bool) -> int:
        if complete:
            if self.pending_partial_id is not None:
                text_id = self.pending_partial_id
                self.pending_partial_id = None
                return text_id
            return self._next_id()

        if self.pending_partial_id is None:
            self.pending_partial_id = self._next_id()
        return self.pending_partial_id

    def translation_thread_main(self):
        import time

        while True:
            job = self.translation_queue.get()
            if job is None:
                break

            text = job.get("text", "")
            if not text.strip():
                continue

            try:
                proc_start = time.perf_counter()
                transl_text = self.engine.translate_text(text)
                proc_end = time.perf_counter()
            except Exception as e:
                logger.error(f"[Translator] {e}")
                continue

            self.sender_queue.put(
                {
                    "type": "statistics",
                    "values": {
                        "last_transl_proc_time": proc_end - proc_start,
                    },
                }
            )

            for out_q in self.output_queues.copy():
                out_q.put(
                    {
                        "id": job["id"],
                        "target_lang": self.target_lang_code,
                        "transl_text": transl_text,
                        "complete": job.get("complete", True),
                    }
                )

    def start(self):
        if not self.is_running:
            self.transl_ready_event = threading.Event()
            self.shutdown_event = threading.Event()
            self.loop_thread = threading.Thread(target=self.run_thread_main)
            self.loop_thread.start()
            self.is_running = True

    def stop(self):
        if self.is_running:
            self.shutdown_event.set()
            self.source_queue.put((None, None))
            self.loop_thread.join()

            if self.translation_thread is not None:
                self.translation_queue.put(None)
                self.translation_thread.join()

            self.loop_thread = None
            self.translation_thread = None
            self.is_running = False
            self.transl_ready_event = None
            self.shutdown_event = None

    def initialize_engine(self):
        self.engine = None

        if self.src_lang != self.target_lang:
            try:
                backend_by_name = {backend_cls.get_name(): backend_cls for backend_cls in self.BACKEND_CLASSES}
                backend_cls = backend_by_name.get(self.engine_id)
                if backend_cls is None:
                    logger.error(f"[Translator] Unknown translation engine: {self.engine_id}")
                    return

                self.engine = backend_cls(self.transl_params, self.src_lang, self.target_lang)
            except Exception as e:
                logger.error(f"[Translator] Error initializing translation engine: {e}")
                self.engine = None

    def add_output_queue(self, q: queue.Queue):
        self.output_queues.append(q)

    def remove_output_queue(self, q: queue.Queue):
        if q in self.output_queues:
            self.output_queues.remove(q)

    def run_thread_main(self):
        self.sender_queue.put({"type": "status", "value": {"status": "translator_initializing"}})

        self.initialize_engine()

        if self.engine is not None:
            self.translation_thread = threading.Thread(target=self.translation_thread_main)
            self.translation_thread.start()

        self.sender_queue.put({"type": "status", "value": {"status": "translator_initialized"}})

        self.transl_ready_event.set()

        while not self.shutdown_event.is_set():
            # Get all available text from the queue. Block until we receive the first message.
            first_msg = True
            while True:
                try:
                    confirmed, unconfirmed = self.source_queue.get(block=first_msg)
                except queue.Empty:
                    break

                if confirmed is None:
                    return

                self.add_text(confirmed, unconfirmed)
                first_msg = False

            sentences = self.get_sentences()
            self._send_buffered_text_stats()

            for to_translate in sentences:
                self.translate_and_send(to_translate)

    def wait_until_ready(self, timeout: float = None) -> bool:
        return self.transl_ready_event.wait(timeout=timeout)


__all__ = ["Translator"]
