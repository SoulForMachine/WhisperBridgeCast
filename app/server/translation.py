from __future__ import annotations
import logging
import multiprocessing as mp
import queue
import threading

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from spacy.tokens import Token

from app.server.transl_backends import (
    TranslBase,
    discover_backend_classes,
)

from app.common.utils import MPCountingQueue
from app.server.settings import TranslationSettings

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

    def __init__(
        self,
        transl_settings: TranslationSettings,
        source_queue: MPCountingQueue,
        output_queues: list[queue.Queue],
        sender_queue: queue.Queue,
        only_complete_sent: bool
    ):
        self.transl_enabled = transl_settings.enable
        self.engine_id = transl_settings.engine
        self.transl_params = transl_settings.engine_params or {}
        self.engine = None
        self.src_lang = transl_settings.src_language
        self.target_lang = transl_settings.target_language
        self.src_lang_code = get_lang_code(self.src_lang)
        self.target_lang_code = get_lang_code(self.target_lang)
        self.source_queue = source_queue
        self.output_queues = output_queues
        self.sender_queue = sender_queue
        self.only_complete_sent = only_complete_sent
        self.confirmed_text = ""
        self.unconfirmed_text = ""

        self.loop_thread = None
        self.translation_thread = None
        self.translation_queue = queue.Queue()
        self.is_running = False
        self.transl_ready_event = None
        self.shutdown_event = None

        self.next_text_id = 0
        self.pending_partial_id = None
        self.source_diff_enabled = bool(self.transl_params.get("source_diff_enabled", False))
        self.target_diff_enabled = bool(self.transl_params.get("target_diff_enabled", False))
        self._prev_source_conf_tok_count = 0
        self._prev_source_unconf_tokens: list[Token] = []
        self._prev_target_tokens: list[Token] = []

        import spacy
        self.src_nlp = spacy.blank(self.src_lang_code)
        self.src_nlp.add_pipe("sentencizer")

        self.dest_nlp = spacy.blank(self.target_lang_code)
        self.dest_nlp.add_pipe("sentencizer")

    FLUSH_TIMEOUT = 6

    class Sentence:
        def __init__(self, confirmed: str = "", unconfirmed: str = "", num_tokens: int = 0, num_conf_tokens: int = 0, complete: bool = False):
            self.confirmed = confirmed
            self.unconfirmed = unconfirmed
            self.num_tokens = num_tokens
            self.num_conf_tokens = num_conf_tokens
            self.complete = complete

        def __bool__(self):
            return bool(self.confirmed.strip()) or bool(self.unconfirmed.strip())

    def _send_buffered_text_stats(self, sentences: list[Sentence]):
        # We send the token count of the last uncomplete sentence, that is token count of
        # the buffered confirmed + unconfirmed text.
        buf_token_count = 0
        if sentences:
            last_sent = sentences[-1]
            buf_token_count = last_sent.num_tokens if not last_sent.complete else 0

        self.sender_queue.put(
            {
                "type": "statistics",
                "values": {
                    "transl_buffer_token_count": buf_token_count,
                },
            }
        )

    def add_text(self, confirmed: str, unconfirmed: str):
        # Append new text to the current buffer.
        self.confirmed_text += confirmed
        self.unconfirmed_text = unconfirmed

    def get_sentences(self) -> list[Sentence]:
        doc_c = self.src_nlp(self.confirmed_text)
        doc_cu = self.src_nlp(self.confirmed_text + self.unconfirmed_text)

        def fix_sent(text: str) -> str:
            # Strip leading whitespace and capitalize first letter of the sentence.
            text = text.lstrip()
            text = text[:1].upper() + text[1:] if text else text
            return text

        sentences: list[Translator.Sentence] = []
        cf_sentences_tokens: list[Token] = []
        for sent in doc_c.sents:
            tokens = [t for t in sent]
            cf_sentences_tokens.append(tokens)
            s = self.Sentence(
                confirmed=fix_sent(sent.text),
                unconfirmed="",
                num_tokens=len(tokens),
                num_conf_tokens=len(tokens),
                complete=True
            )
            sentences.append(s)

        cf_uncf_sentences_tokens: list[Token] = []
        for sent in doc_cu.sents:
            cf_uncf_sentences_tokens.append([t for t in sent])

        added_unconfirmed = False
        if sentences:
            last_cf_sent = sentences[-1]
            cf_toks = [t.text for t in cf_sentences_tokens[-1]]
            last_idx = len(sentences) - 1
            assert last_idx < len(cf_uncf_sentences_tokens)
            cu_sent_tokens = cf_uncf_sentences_tokens[last_idx]
            cu_toks = [t.text for t in cu_sent_tokens]

            if cf_toks != cu_toks:
                # last sentence is partial, add the unconfirmed part as well and mark the sentence as incomplete
                cf_uncf_sent = self.Sentence(
                    confirmed=last_cf_sent.confirmed,
                    unconfirmed=self.unconfirmed_text,
                    num_tokens=len(cu_sent_tokens),
                    num_conf_tokens=last_cf_sent.num_conf_tokens,
                    complete=False
                )
                sentences[-1] = cf_uncf_sent
                self.confirmed_text = last_cf_sent.confirmed
                added_unconfirmed = True
            else:
                # since the last sentence is complete, we clear the confirmed text buffer
                self.confirmed_text = ""

        if not added_unconfirmed and self.unconfirmed_text !="":
            # The unconfirmed text isn't added either because there are no complete senteces or because the last sentence is complete.
            # In both cases we add the unconfirmed text as a separate incomplete sentence.
            unconfirmed_text = self.unconfirmed_text.lstrip()
            unc_doc = self.src_nlp(unconfirmed_text)
            unc_tokens = [t for t in unc_doc]
            unc_sent = self.Sentence(
                confirmed="",
                unconfirmed=unconfirmed_text,
                num_tokens=len(unc_tokens),
                num_conf_tokens=0,
                complete=False
            )
            sentences.append(unc_sent)

        return sentences

    @classmethod
    def _compute_diff_ops(
        cls,
        prev_text: list[Token],
        curr_text: list[Token],
        include_inserts: bool = True,
        include_middle_inserts: bool = True,
        include_trailing_inserts: bool = True,
    ) -> list[dict]:
        curr_offsets = [token.idx for token in curr_text]
        prev = [token.text for token in prev_text]
        curr = [token.text for token in curr_text]

        from difflib import SequenceMatcher
        matcher = SequenceMatcher(a=prev, b=curr, autojunk=False)
        ops: list[dict] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            match tag:
                case "insert":
                    if not include_inserts:
                        continue

                    is_trailing_insert = (i1 == i2 == len(prev)) and (j2 == len(curr))
                    if is_trailing_insert and not include_trailing_inserts:
                        continue
                    if not is_trailing_insert and not include_middle_inserts:
                        continue

                    start = curr_offsets[j1]
                    end = curr_offsets[j2 - 1] + len(curr_text[j2 - 1].text)
                    ops.append({"op": "+", "span": [start, end]})

                case "delete":
                    if not curr_text:
                        idx = 0
                    elif j1 == 0:
                        idx = curr_offsets[0]
                    elif j1 >= len(curr_text):
                        idx = curr_offsets[-1] + len(curr_text[-1].text)
                    else:
                        idx = curr_offsets[j1 - 1] + len(curr_text[j1 - 1].text)
                    ops.append({"op": "-", "idx": idx})

                case "replace":
                    start = curr_offsets[j1]
                    end = curr_offsets[j2 - 1] + len(curr_text[j2 - 1].text)
                    ops.append({"op": "~", "span": [start, end]})

        return ops

    def _compute_source_diff_ops(self, cur_sent: Sentence) -> list[dict]:
        if self._prev_source_unconf_tokens or not cur_sent.complete:
            doc = self.src_nlp(cur_sent.confirmed + cur_sent.unconfirmed)
            full_tokens = [t for t in doc]

        if self._prev_source_unconf_tokens:
            start = min(self._prev_source_conf_tok_count, len(full_tokens))
            end = min(start + len(self._prev_source_unconf_tokens), len(full_tokens))
            new_cmp_tokens = full_tokens[start:end]
            diff_ops = self._compute_diff_ops(
                self._prev_source_unconf_tokens,
                new_cmp_tokens,
                include_inserts=True,
                include_middle_inserts=True,
                include_trailing_inserts=True,
            )
        else:
            diff_ops: list[dict] = []

        if cur_sent.complete:
            self._prev_source_conf_tok_count = 0
            self._prev_source_unconf_tokens = []
        else:
            conf_tok_count = cur_sent.num_conf_tokens
            self._prev_source_conf_tok_count = conf_tok_count
            self._prev_source_unconf_tokens = full_tokens[conf_tok_count:]

        return diff_ops

    def _compute_target_diff_ops(self, transl_text: str, complete: bool) -> list[dict]:
        if self._prev_target_tokens or not complete:
            doc = self.dest_nlp(transl_text)
            transl_text_tokens = [t for t in doc]

        if self._prev_target_tokens:
            diff_ops = self._compute_diff_ops(
                self._prev_target_tokens,
                transl_text_tokens,
                include_inserts=True,
                include_middle_inserts=True,
                include_trailing_inserts=complete,
            )
        else:
            diff_ops: list[dict] = []

        if complete:
            self._prev_target_tokens = []
        else:
            self._prev_target_tokens = transl_text_tokens

        return diff_ops

    def translate_and_send(self, sentence: Sentence):
        confirmed = sentence.confirmed
        unconfirmed = sentence.unconfirmed
        complete = sentence.complete

        if self.source_diff_enabled:
            source_diff_ops = self._compute_source_diff_ops(sentence)
        else:
            source_diff_ops = []
        text_id = self._resolve_text_id(complete)

        for out_q in self.output_queues.copy():
            out_q.put(
                {
                    "id": text_id,
                    "src_lang": self.src_lang_code,
                    "orig_text": confirmed,
                    "orig_unconfirmed_text": unconfirmed,
                    "source_diff": source_diff_ops,
                    "complete": complete,
                }
            )

        if self.engine is not None:
            self.translation_queue.put(
                {
                    "id": text_id,
                    "text": confirmed + unconfirmed,
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

            complete = job.get("complete", True)

            if self.target_diff_enabled:
                target_diff_ops = self._compute_target_diff_ops(transl_text, complete)
            else:
                target_diff_ops = []

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
                        "target_diff": target_diff_ops,
                        "complete": complete,
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
            self._prev_source_conf_tok_count = 0
            self._prev_source_unconf_tokens.clear()
            self._prev_target_tokens.clear()

    def initialize_engine(self):
        self.engine = None

        if self.transl_enabled and self.src_lang != self.target_lang:
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
            self._send_buffered_text_stats(sentences)

            for to_translate in sentences:
                self.translate_and_send(to_translate)

    def wait_until_ready(self, timeout: float = None) -> bool:
        return self.transl_ready_event.wait(timeout=timeout)


__all__ = ["Translator"]
