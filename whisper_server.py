import socket

from pydot import Any
import net_common as netc
from urllib.parse import urlparse, parse_qs
import multiprocessing as mp

import argparse
import os
import logging
import queue
import threading
import requests
import json

logger = logging.getLogger(__name__)

class MPCountingQueue:
    def __init__(self):
        self.q = mp.Queue()
        self.counter = mp.Value('i', 0)

    def put(self, item):
        self.q.put(item)
        with self.counter.get_lock():
            self.counter.value += 1

    def get(self, block=True, timeout=None):
        item = self.q.get(block=block, timeout=timeout)
        with self.counter.get_lock():
            self.counter.value -= 1
        return item

    def qsize(self):
        return self.counter.value

def get_lang_code(lang: str) -> str:
    lang_code_map = {
        "English": "en",
        "German": "de",
        "Serbian": "sr",
        "Serbian Latin": "sr",
        "Serbian Cyrillic": "sr",
    }
    return lang_code_map.get(lang, "")

class WhisperServerParams:
    def __init__(self):
        # Zoom URL
        self.zoom_url = ""

        # Whisper model
        self.model = 'large-v2'
        self.model_cache_dir = None
        self.model_dir = None
        self.warmup_file = "data/samples_jfk.wav"

        # Language and task
        self.language = 'auto'
        self.task = 'transcribe'

        # Backend
        self.backend = 'faster-whisper'
        self.nsp_threshold = None

        # Voice activity detection
        self.vac = True
        self.vac_min_chunk_size = 1.0
        self.vac_dynamic_chunk_size = True
        self.vad_threshold = 0.5
        self.vad_min_silence_duration_ms = 1000
        self.vad_speech_pad_ms = 1000
        self.whisper_vad = False

        # Buffer trimming
        self.buffer_trimming = 'segment'
        self.buffer_trimming_sec = 15

        # Device and compute
        self.whisper_device = 'cuda'
        self.whisper_compute_type = 'float32'

        # Logging
        self.log_level = 'DEBUG'


######### WhisperOnline

class WhisperOnline:
    def __init__(self, client_params: dict, logger: logging.Logger):
        import whisper_online as wo

        params = WhisperServerParams()

        # Update params with client provided values
        for key, value in client_params.items():
            if hasattr(params, key):
                setattr(params, key, value)

        params.language = {
            "English": "en",
            "German": "de",
            "Serbian": "sr",
        }.get(client_params.get("language"), "auto")

        if (
            client_params.get("enable_translation") is True
            and client_params.get("target_language") == "English"
            and client_params.get("translation_engine") == "Whisper"
        ):
            params.task = "translate"
        else:
            params.task = "transcribe"

        wo.set_logging(logger.getEffectiveLevel())

        # Create whisper online processor object with params
        self.asr, self.asr_proc = wo.asr_factory(params)

        # warm up the ASR so first chunk isn’t slow
        if os.path.isfile(params.warmup_file):
            a = wo.load_audio_chunk(params.warmup_file, 0, 1)
            self.asr.transcribe(a)
            logger.info("Whisper has warmed up.")
        else:
            logger.warning("The warm up file is not available. Whisper is not warmed up. The first chunk processing may take longer.")

    # Call before using the object for new audio stream
    # Not used for now, as we create a new object for each client
    def clear(self):
        self.asr_proc.init()

    def has_vac(self) -> bool:
        return hasattr(self.asr_proc, "vac") and self.asr_proc.vac is not None

######### ASR processor

class ASRProcessor:
    def __init__(self, client_params: dict, audio_queue: MPCountingQueue, result_queue: MPCountingQueue, sender_queue: mp.Queue):
        self.client_params = client_params
        self.audio_queue = audio_queue
        self.result_queue = result_queue
        self.sender_queue = sender_queue

        self.asr_subproc = None
        self.asr_ready_event = mp.Event()
        self.shutdown_event = mp.Event()
        self.is_running = False

    def start(self):
        if not self.is_running:
            self.asr_subproc = mp.Process(
                target=asr_subprocess_main,
                args=(self.client_params, self.audio_queue, self.result_queue, self.sender_queue, self.asr_ready_event, self.shutdown_event),
                daemon=False
            )
            self.asr_subproc.start()
            self.is_running = True

    def stop(self):
        if self.is_running:
            self.shutdown_event.set()
            self.audio_queue.put(None)
            self.asr_subproc.join()
            self.asr_subproc = None
            self.asr_ready_event = None
            self.shutdown_event = None
            self.is_running = False

    def wait_until_ready(self, timeout: float = None) -> bool:
        return self.asr_ready_event.wait(timeout=timeout)


########## Translator

class Translator:
    class Engine:
        NONE = "none"
        MARIANMT = "MarianMT"
        NLLB = "NLLB"
        EUROLLM = "EuroLLM"
        GOOGLE_GEMINI = "Google Gemini"
        ONLINE_TRANSLATORS = "Online Translators"

    class OnlineProviders:
        GOOGLE = "Google"
        MYMEMORY = "MyMemory"
        DEEPL = "DeepL"
        MICROSOFT = "Microsoft"
        LIBRE = "Libre"
        CHATGPT = "ChatGpt"
        BAIDU = "Baidu"
        PAPAGO = "Papago"
        QCRI = "QCRI"
        YANDEX = "Yandex"

        API_KEY_REQUIRED = {DEEPL, MICROSOFT, CHATGPT, QCRI, YANDEX}

    LIBRE_MIRRORS = {
        "libretranslate.com": {
            "url": "https://libretranslate.com/",
            "api_key_required": True,
        },
    }

    class MarianMT:
        def __init__(self, src_lang: str, target_lang: str):
            from transformers import MarianTokenizer, MarianMTModel

            language_pairs = {
                ("English", "Serbian Cyrillic"): "Helsinki-NLP/opus-mt-tc-base-en-sh",
                ("English", "Serbian Latin"): "Helsinki-NLP/opus-mt-tc-base-en-sh",
                ("English", "German"): "Helsinki-NLP/opus-mt-en-de",
                ("German", "English"): "Helsinki-NLP/opus-mt-de-en",
            }
            model_name = language_pairs.get((src_lang, target_lang))
            if model_name is None:
                raise ValueError(f"MarianMT does not support translation from {src_lang} to {target_lang}.")

            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.translator = MarianMTModel.from_pretrained(model_name).to("cuda")
            self.target_lang_token = {
                "Serbian Cyrillic": "srp_Cyrl",
                "Serbian Latin": "srp_Latn",
            }.get(target_lang, "")

        def translate_text(self, text: str) -> str:
            text_to_translate = f">>{self.target_lang_token}<< {text}" if self.target_lang_token else text
            inputs = self.tokenizer(text_to_translate, return_tensors="pt", truncation=True).to("cuda")
            translated = self.translator.generate(**inputs)
            transl_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
            return transl_text

    class NLLB:
        def __init__(self, src_lang: str, target_lang: str):
            from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

            self.language_codes = {
                "English": "eng_Latn",
                "Serbian Cyrillic": "srp_Cyrl",
                "German": "deu_Latn",
            }
            self.target_lang_token = self.language_codes[target_lang]

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                #bnb_4bit_use_double_quant=True,
                #bnb_4bit_compute_dtype=torch.float16,
                #bnb_4bit_quant_type="nf4",
            )

            model_name = "facebook/nllb-200-distilled-600M"
            self.tokenizer = NllbTokenizer.from_pretrained(model_name, src_lang=self.language_codes[src_lang])
            self.forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.target_lang_token)
            self.translator = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
            #self.translator = self.translator.to("cuda")

        def translate_text(self, text: str) -> str:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            inputs = inputs.to(self.translator.device)
            #inputs = inputs.to("cuda")

            # Generate translation
            translated_tokens = self.translator.generate(
                **inputs,
                forced_bos_token_id=self.forced_bos_token_id
            )

            # Decode
            translation = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            return translation

    class EuroLLM:
        def __init__(self, api_key, src_lang: str, target_lang: str):
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from huggingface_hub import login

            login(token=api_key)

            model_id = "utter-project/EuroLLM-1.7B"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=True,
                quantization_config=bnb_config,
                device_map="auto"
            )
            #self.model = self.model.to("cuda")
            import torch
            self.model = torch.compile(self.model)

            self.src_lang = src_lang
            self.target_lang = target_lang

        def translate_text(self, text: str) -> str:
            # Ensure the text ends with punctuation, otherwise the model may not respond correctly.
            if not text.endswith(('.', '!', '?', ';')):
                text += '...'
            prompt = f"{self.src_lang}: {text} {self.target_lang}:"
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            #import torch
            #with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,      # Deterministic
                num_beams=1,          # Beam search
                use_cache=True,
                #early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            transl_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the translated part
            transl_text = transl_text[len(prompt):].strip()
            return transl_text

    class GoogleGemini:
        def __init__(self, api_key, src_lang: str, target_lang: str):
            from google import genai
            from google.genai import types
            self.genai = genai
            self.types = types

            # Set your key
            os.environ["GEMINI_API_KEY"] = api_key
            # Initialise client
            self.client = self.genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            self.src_lang = src_lang
            self.target_lang = target_lang

        def translate_text(self, text: str) -> str:
            try:
                prompt = f"Translate the following text in {self.src_lang} into {self.target_lang}:\n\"{text}\""
                max_output_tokens = max(2048, len(text.split()) * 10)
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=self.types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=max_output_tokens
                    )
                )
                transl_text = getattr(response, "text", None)
                if transl_text is None:
                    return f"[Translation Error]: {getattr(response.candidates[0], "finish_reason", None)}."
                return transl_text.strip()
            except Exception as e:
                return f"[Translation Exception]: {e}."

    class OnlineTranslators:
        def __init__(self, provider: str, transl_params: dict, src_lang: str, target_lang: str):
            from deep_translator import (
                BaiduTranslator,
                ChatGptTranslator,
                DeeplTranslator,
                GoogleTranslator,
                LibreTranslator,
                MicrosoftTranslator,
                MyMemoryTranslator,
                PapagoTranslator,
                QcriTranslator,
                YandexTranslator,
            )

            self.provider = provider or Translator.OnlineProviders.GOOGLE
            self.translator = None
            self.domain = None

            transl_params = transl_params or {}
            api_key = transl_params.get("api_key", "")
            api_secret = transl_params.get("api_secret", "")
            client_id = transl_params.get("client_id", "")
            region = transl_params.get("region", "")
            self.domain = transl_params.get("domain", "")
            libre_mirror = transl_params.get("libre_mirror", "libretranslate.com")

            src_code = self._lang_to_code(src_lang, self.provider)
            target_code = self._lang_to_code(target_lang, self.provider)

            match self.provider:
                case Translator.OnlineProviders.GOOGLE:
                    self.translator = GoogleTranslator(source=src_code, target=target_code)
                case Translator.OnlineProviders.MYMEMORY:
                    self.translator = MyMemoryTranslator(source=src_code, target=target_code)
                case Translator.OnlineProviders.DEEPL:
                    self.translator = DeeplTranslator(source=src_code, target=target_code, api_key=api_key)
                case Translator.OnlineProviders.MICROSOFT:
                    self.translator = MicrosoftTranslator(source=src_code, target=target_code, api_key=api_key, region=region or None)
                case Translator.OnlineProviders.LIBRE:
                    mirror_cfg = Translator.LIBRE_MIRRORS.get(libre_mirror, Translator.LIBRE_MIRRORS["libretranslate.com"])
                    mirror_requires_key = mirror_cfg["api_key_required"]
                    self.translator = LibreTranslator(
                        source=src_code,
                        target=target_code,
                        api_key=(api_key if mirror_requires_key else None),
                        custom_url=mirror_cfg["url"],
                        use_free_api=not mirror_requires_key,
                    )
                case Translator.OnlineProviders.CHATGPT:
                    self.translator = ChatGptTranslator(
                        source=self._lang_to_name(src_lang),
                        target=self._lang_to_name(target_lang),
                        api_key=api_key,
                    )
                case Translator.OnlineProviders.BAIDU:
                    self.translator = BaiduTranslator(source=src_code, target=target_code, appid=client_id, appkey=api_secret)
                case Translator.OnlineProviders.PAPAGO:
                    self.translator = PapagoTranslator(source=src_code, target=target_code, client_id=client_id, secret_key=api_secret)
                case Translator.OnlineProviders.QCRI:
                    self.translator = QcriTranslator(source=src_code, target=target_code, api_key=api_key)
                    if not self.domain:
                        self.domain = "general"
                case Translator.OnlineProviders.YANDEX:
                    self.translator = YandexTranslator(source=src_code, target=target_code, api_key=api_key)
                case _:
                    raise ValueError(f"Unknown online translator provider: {self.provider}")

        @staticmethod
        def _lang_to_code(lang: str, provider: str) -> str:
            match provider:
                case Translator.OnlineProviders.MYMEMORY:
                    return {
                        "English": "en-US",
                        "German": "de-DE",
                        "Serbian Latin": "sr-Latn-RS",
                        "Serbian Cyrillic": "sr-Cyrl-RS",
                        "Serbian": "sr-Latn-RS",
                    }.get(lang, "auto")
                case Translator.OnlineProviders.MICROSOFT:
                    return {
                        "English": "en",
                        "German": "de",
                        "Serbian Latin": "sr-latn",
                        "Serbian Cyrillic": "sr-cyrl",
                        "Serbian": "sr-latn",
                    }.get(lang, "auto")

            return {
                "English": "en",
                "German": "de",
                "Serbian": "sr",
                "Serbian Latin": "sr",
                "Serbian Cyrillic": "sr",
            }.get(lang, "auto")

        @staticmethod
        def _lang_to_name(lang: str) -> str:
            return {
                "English": "english",
                "German": "german",
                "Serbian": "serbian",
                "Serbian Latin": "serbian",
                "Serbian Cyrillic": "serbian",
            }.get(lang, "auto")

        def translate_text(self, text: str) -> str:
            if self.provider == Translator.OnlineProviders.QCRI:
                result = self.translator.translate(text, domain=self.domain)
            else:
                result = self.translator.translate(text)

            if isinstance(result, list):
                return "\n".join(str(x) for x in result)
            return result

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

    FLUSH_TIMEOUT = 6  # Flush partial sentences after this many seconds.

    def _buffered_word_count(self):
        # Approximate word count by counting spaces. This is not exact but should be sufficient for statistics.
        return self.current_text.count(" ") + 1 if self.current_text else 0

    def _word_count(self, text: str):
        return text.count(" ") + 1 if text else 0

    def _send_buffered_text_stats(self):
        self.sender_queue.put({
            "type": "statistics",
            "values": {
                "transl_buffer_word_count": self._buffered_word_count(),
            }
        })

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
            if not sent.endswith(('.', '!', '?')):
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
            out_q.put({
                "id": text_id,
                "src_lang": self.src_lang_code,
                "orig_text": confirmed_text,
                "orig_unconfirmed_text": unconfirmed_text,
                "complete": complete,
            })

        if self.engine is not None:
            self.translation_queue.put({
                "id": text_id,
                "text": confirmed_text + unconfirmed_text,
                "complete": complete,
            })

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

            self.sender_queue.put({
                "type": "statistics",
                "values": {
                    "last_transl_proc_time": proc_end - proc_start
                }
            })

            for out_q in self.output_queues.copy():
                out_q.put({
                    "id": job["id"],
                    "target_lang": self.target_lang_code,
                    "transl_text": transl_text,
                    "complete": job.get("complete", True),
                })

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
                match self.engine_id:
                    case Translator.Engine.MARIANMT:
                        self.engine = self.MarianMT(self.src_lang, self.target_lang)
                    case Translator.Engine.GOOGLE_GEMINI:
                        self.engine = self.GoogleGemini(self.transl_params["api_key"], self.src_lang, self.target_lang)
                    case Translator.Engine.NLLB:
                        self.engine = self.NLLB(self.src_lang, self.target_lang)
                    case Translator.Engine.EUROLLM:
                        self.engine = self.EuroLLM(self.transl_params["api_key"], self.src_lang, self.target_lang)
                    case Translator.Engine.ONLINE_TRANSLATORS:
                        provider = self.transl_params.get("provider", Translator.OnlineProviders.GOOGLE)
                        self.engine = self.OnlineTranslators(provider, self.transl_params, self.src_lang, self.target_lang)
            except Exception as e:
                logger.error(f"[Translator] Error initializing translation engine: {e}")
                self.engine = None

    def add_output_queue(self, q: queue.Queue):
        self.output_queues.append(q)

    def remove_output_queue(self, q: queue.Queue):
        if q in self.output_queues:
            self.output_queues.remove(q)

    def run_thread_main(self):
        self.sender_queue.put({
            "type": "status",
            "value": { "status": "translator_initializing" }
        })

        self.initialize_engine()

        if self.engine is not None:
            self.translation_thread = threading.Thread(target=self.translation_thread_main)
            self.translation_thread.start()

        self.sender_queue.put({
            "type": "status",
            "value": { "status": "translator_initialized" }
        })

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


########## Zoom caption sender

class ZoomCaptionSender:
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
                # Build URL with lang + sequence params
                url = f"{self.zoom_url}&lang={lang_code}&seq={self.sequence}"

                headers = {
                    "Content-Type": "plain/text",
                    "Content-Length": str(len(text))
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
        return True  # Zoom caption sender is ready immediately

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


########## Caption sender: sends captions to the client

class ClientCaptionSender:
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

            self.sender_queue.put({
                "type": "translation",
                "lang": lang_code,
                "text": text,
                "complete": complete,
            })

    def wait_until_ready(self, timeout: float = None) -> bool:
        return True  # Caption sender is ready immediately

########## WhisperPipeline

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
            transl_engine = client_params.get("translation_engine", Translator.Engine.MARIANMT)
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
            only_complete_sent=bool(zoom_url)
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

        self.sender_queue.put({
            "type": "statistics",
            "values": {
                "asr_in_q_size": self.audio_queue.qsize(),
            }
        })

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
            self.zoom_caption_sender_queue.put(("...", True))  # to warm up Zoom
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


class WhisperServer:
    def __init__(self, host="0.0.0.0", port=5000, write_wav=False, write_transcript=False):
        self.conn = None
        self.server_thread = None
        self.is_running = False
        self.is_stopping = False
        self.host = host
        self.port = port
        self.listen_socket = None
        self.client_socket = None
        self.write_wav = write_wav
        self.write_transcript = write_transcript

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

        try:
            # Start the sender thread
            sender_thread = threading.Thread(
                target=self.sender_thread_func,
                args=(conn, sender_queue)
            )
            sender_thread.start()

            # Receive parameters from the client and initialize the pipeline
            msg_type, params = netc.recv_message(conn)
            if params is None or msg_type != "json":
                logger.error("Failed to receive params from the client.")
                self._close_socket(conn)
                return
            logger.info(f"Received params: {params}")

            params["log_level"] = logging.getLevelName(logger.getEffectiveLevel())    # Easier way to pass the log level

            pipeline = WhisperPipeline(params, sender_queue)

            # Start the wav file writer if needed
            if self.write_wav:
                import wav_writer
                from datetime import datetime
                filename = datetime.now().strftime("recording-%Y%m%d_%H%M%S.wav")
                wav_out = wav_writer.WavWriter(filename)

            # When the pipeline is ready, inform the client
            pipeline.wait_until_ready()

            netc.send_json(conn, {
                "type": "status",
                "value": { "status": "ready" }
            })

            # Receive audio chunks and control messages
            while True:
                msg_type, msg = netc.recv_message(conn)
                if msg is None:
                    logger.error("Invalid message received.")
                    break

                if msg_type == "audio":
                    if len(msg) == 0:
                        continue

                    # Send received chunk to the pipeline
                    pipeline.process(msg)

                    if self.write_wav:
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
                # Confirm shutdown request
                sender_queue.put({
                    "type": "status",
                    "value": { "status": "conn_shutdown" }
                })

            # Stop the sender thread
            sender_queue.put(None)
            sender_thread.join()

            # Close the connection
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

"""
Runs in the subprocess: construct WhisperOnline here, then
read audio chunks from audio_queue and put results to asr_queue.
"""
def asr_subprocess_main(
        client_params: dict, 
        audio_queue: MPCountingQueue, 
        asr_queue: MPCountingQueue, 
        sender_queue: mp.Queue, 
        asr_ready_event,
        shutdown_event):
    logger = logging.getLogger("whisper_online_asr_subproc")
    logger.setLevel(client_params.get("log_level", "INFO"))
    logging.basicConfig(
        format='%(levelname)s\t%(message)s'
    )

    sender_queue.put({
        "type": "status",
        "value": { "status": "asr_initializing" }
    })

    whisper_online = WhisperOnline(client_params, logger)
    asr_ready_event.set()

    sender_queue.put({
        "type": "status",
        "value": { "status": "asr_initialized" }
    })

    has_vac = whisper_online.has_vac()
    last_vac_status = None

    import time

    try:
        while not shutdown_event.is_set():
            chunk = audio_queue.get()
            if chunk is None:
                break

            sender_queue.put({
                "type": "statistics",
                "values": {
                    "asr_in_q_size": audio_queue.qsize(),
                }
            })

            whisper_online.asr_proc.insert_audio_chunk(chunk)
            confirmed, unconfirmed, action = whisper_online.asr_proc.process_iter()

            if has_vac and whisper_online.asr_proc.status != last_vac_status:
                last_vac_status = whisper_online.asr_proc.status
                sender_queue.put({
                    "type": "statistics",
                    "values": {
                        "vac_voice_status": last_vac_status,
                    }
                })

            if confirmed[2] or unconfirmed[2]:
                asr_queue.put((confirmed[2], unconfirmed[2]))
                logger.info(f"[ASR] {confirmed[2]} | {unconfirmed[2]}")

                if action == "inference":
                    sender_queue.put({
                        "type": "statistics",
                        "values": {
                            "last_asr_proc_time": whisper_online.asr_proc.get_last_inference_time(),
                            "asr_roll_avg_proc_time": whisper_online.asr_proc.get_roll_avg_inference_time(),
                        }
                    })
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"ASR subprocess exception: {e}")

    # Flush any remaining audio
    whisper_online.asr_proc.finish()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    import sys

    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host adress. Default: 0.0.0.0")
    parser.add_argument("--port", type=int, default=5000, help="Port number. Default: 5000")
    parser.add_argument("--write-wav", action="store_true", help="Write received audio to a wav file.")
    parser.add_argument("--write-transcript", action="store_true", help="Write received transcript to a text file.")
    parser.add_argument("--warmup-file", type=str, default="data/samples_jfk.wav", help="Provide the audio file used to warm up the whisper model.")
    parser.add_argument("--log-level", type=str, default="INFO", choices="CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET".split(','), help="Logging level. Default: INFO")
    args = parser.parse_args(sys.argv[1:])

    logging.basicConfig(
        level=args.log_level,
        format='%(levelname)s\t%(message)s'
    )

    whisper_server = WhisperServer(args.host, args.port, args.write_wav, args.write_transcript)
    whisper_server.start()

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, stopping server...")
    finally:
        whisper_server.stop()
        os._exit(0)
