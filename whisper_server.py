import socket
import captioner_common as ccmn
from urllib.parse import urlparse, parse_qs
import multiprocessing as mp

import argparse
import os
import logging
import queue
import threading
import requests
import json
import re


def get_lang_code(lang: str) -> str:
    lang_code_map = {
        "English": "en",
        "German": "de",
        "Serbian": "sr",
        "Serbian Latin": "sr",
        "Serbian Cyrilic": "sr",
    }
    return lang_code_map.get(lang, "en")

class WhisperServerParams:
    def __init__(self):
        # Zoom URL
        self.zoom_url = ""

        # Audio processing
        self.min_chunk_size = 1.0

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
        self.vac = False
        self.vac_chunk_size = 0.04
        self.vad = False

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
    def __init__(self, client_params: dict):
        import whisper_online as wo

        logger = logging.getLogger(__name__)

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

        wo.set_logging(params, logger, other="")

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

######### ASR processor

class ASRProcessor:
    def __init__(self, client_params: dict, audio_queue: mp.Queue, result_queue: mp.Queue):
        self.client_params = client_params
        self.audio_queue = audio_queue
        self.result_queue = result_queue

        self.asr_subproc = None
        self.is_running = False

    def start(self):
        if not self.is_running:
            self.asr_subproc = mp.Process(
                target=asr_subprocess_main,
                args=(self.client_params, self.audio_queue, self.result_queue),
                daemon=False
            )
            self.asr_subproc.start()
            self.is_running = True

    def stop(self):
        if self.is_running:
            self.audio_queue.put(None)
            self.asr_subproc.join()
            self.asr_subproc = None
            self.is_running = False


########## Translator

class Translator:
    class Engine:
        NONE = "none"
        MARIANMT = "MarianMT"
        NLLB = "NLLB"
        EUROLLM = "EuroLLM"
        GOOGLE_GEMINI = "Google Gemini"

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
            self.translator = MarianMTModel.from_pretrained(model_name)
            self.target_lang_token = {
                "Serbian Cyrillic": "srp_Cyrl",
                "Serbian Latin": "srp_Latn",
            }.get(target_lang, "")

        def translate_text(self, text: str) -> str:
            text_to_translate = f">>{self.target_lang_token}<< {text}" if self.target_lang_token else text
            inputs = self.tokenizer(text_to_translate, return_tensors="pt", truncation=True)
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
                load_in_4bit=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=True,
                quantization_config=bnb_config,
                device_map="auto"
            )
            #self.model = self.model.to("cuda")

            self.src_lang = src_lang
            self.target_lang = target_lang

        def translate_text(self, text: str) -> str:
            # Ensure the text ends with punctuation, otherwise the model may not respond correctly.
            if not text.endswith(('.', '!', '?', ';')):
                text += '...'
            prompt = f"{self.src_lang}: {text} {self.target_lang}:"
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs.to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                #do_sample=False,      # Deterministic
                #num_beams=5,          # Beam search
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

    def __init__(self, engine_id, engine_params, src_lang, target_lang, source_queue, result_queue, only_complete_sent: bool):
        self.engine_id = engine_id
        self.engine_params = engine_params
        self.engine = None
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.source_queue = source_queue
        self.result_queue = result_queue
        self.only_complete_sent = only_complete_sent
        self.current_text = ""

        self.transl_thread = None
        self.is_running = False

        import spacy
        self.nlp = spacy.blank(get_lang_code(self.src_lang))
        self.nlp.add_pipe("sentencizer")

    FLUSH_TIMEOUT = 6  # seconds
    SEND_PARTIAL_LEN = 50  # Send partial text in increments of this many characters.

    def add_text(self, text: str):
        # Append new text to the current buffer.
        if text != "":
            self.current_text += text

    def get_sentences(self) -> list[tuple[str, bool]]:
        doc = self.nlp(self.current_text)
        sentences = [(sent.text.strip(), True) for sent in doc.sents]

        if sentences:
            last_sent = sentences[-1][0]
            if not last_sent.endswith(('.', '!', '?')):
                # last sentence is partial
                sentences[-1] = (last_sent, False)
                self.current_text = last_sent
            else:
                self.current_text = ""
        return sentences

    def translate_and_send(self, text: tuple[str, bool]):
        if self.engine is not None:
            transl_text = self.engine.translate_text(text[0])
            self.result_queue.put((transl_text, text[1]))
        else:
            self.result_queue.put(text)

    def start(self):
        if not self.is_running:
            self.transl_thread = threading.Thread(target=self.run)
            self.transl_thread.start()
            self.is_running = True

    def stop(self):
        if self.is_running:
            self.source_queue.put(None)
            self.transl_thread.join()
            self.transl_thread = None
            self.is_running = False

    def initialize_engine(self):
        self.engine = None

        if self.src_lang != self.target_lang:
            try:
                match self.engine_id:
                    case Translator.Engine.MARIANMT:
                        self.engine = self.MarianMT(self.src_lang, self.target_lang)
                    case Translator.Engine.GOOGLE_GEMINI:
                        self.engine = self.GoogleGemini(self.engine_params["api_key"], self.src_lang, self.target_lang)
                    case Translator.Engine.NLLB:
                        self.engine = self.NLLB(self.src_lang, self.target_lang)
                    case Translator.Engine.EUROLLM:
                        self.engine = self.EuroLLM(self.engine_params["api_key"], self.src_lang, self.target_lang)
            except Exception as e:
                print(f"[Translator] Error initializing translation engine: {e}", flush=True)
                self.engine = None

    def run(self):
        self.initialize_engine()
        last_partial_len = 0

        while True:
            try:
                timeout = self.FLUSH_TIMEOUT if self.current_text != "" else None
                text = self.source_queue.get(timeout=timeout)
            except queue.Empty:
                if self.current_text != "":
                    print("[Translator] Timeout reached, flushing all current text.", flush=True)
                    self.translate_and_send((self.current_text + "...", True))
                    self.current_text = ""
                    last_partial_len = 0
                continue

            if text is None:
                break

            self.add_text(text)
            sentences = self.get_sentences()

            for to_translate in sentences:
                text_to_transl = to_translate[0]
                complete_sentence = to_translate[1]

                if text_to_transl != "":
                    if complete_sentence:
                        self.translate_and_send(to_translate)
                        last_partial_len = 0
                    elif not self.only_complete_sent:
                        if (
                            self.engine is None
                            or len(text_to_transl) - last_partial_len >= self.SEND_PARTIAL_LEN
                        ):
                            last_partial_len = len(text_to_transl)
                            self.translate_and_send(to_translate)

########## Zoom caption sender

class ZoomCaptionSender:
    def __init__(self, source_queue, zoom_url, lang_code):
        self.source_queue = source_queue
        self.zoom_url = zoom_url
        self.lang_code = lang_code
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
            self.source_queue.put((None, None))
            self.caption_thread.join()
            self.caption_thread = None
            self.is_running = False

    def run(self):
        while True:
            text, complete = self.source_queue.get()
            if text is None:
                break
            elif not text.strip():
                continue

            #print(f"Sending Zoom caption: {text}", flush=True)

            if self.zoom_url and complete:
                # Build URL with lang + sequence params
                url = f"{self.zoom_url}&lang={self.lang_code}&seq={self.sequence}"

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
                        print(f"[ZoomCaptioner] Error sending to Zoom: {result.status_code} {result.text}")
                except requests.RequestException as e:
                    print(f"[ZoomCaptioner] Error sending to Zoom: {e}")

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

class CaptionSender:
    def __init__(self, source_queue, conn, lang_code):
        self.source_queue = source_queue
        self.conn = conn
        self.lang_code = lang_code
        self.caption_thread = None
        self.is_running = False

    def start(self):
        if not self.is_running:
            self.caption_thread = threading.Thread(target=self.run)
            self.caption_thread.start()
            self.is_running = True

    def stop(self):
        if self.is_running:
            self.source_queue.put((None, None))
            self.caption_thread.join()
            self.caption_thread = None
            self.is_running = False

    def run(self):
        while True:
            text, complete = self.source_queue.get()
            if text is None:
                break
            elif not text.strip():
                continue

            ccmn.send_json(self.conn, {
                "type": "translation",
                "lang": self.lang_code,
                "text": text,
                "complete": complete,
            })


########## WhisperPipeline

class WhisperPipeline:
    def __init__(self, client_params, conn):
        self.audio_queue = mp.Queue()
        self.asr_queue = mp.Queue()
        self.transl_queue = queue.Queue()

        zoom_url = client_params.get("zoom_url")

        transl_params = client_params.get("translation_params")
        language = client_params.get("language")
        target_language = client_params.get("target_language")

        if client_params.get("enable_translation") is True:
            transl_engine = client_params.get("translation_engine", Translator.Engine.MARIANMT)
            capt_lang = target_language
        else:
            transl_engine = "none"
            capt_lang = language

        print("Starting Translator thread...", flush=True)
        self.translator = Translator(
            transl_engine,
            transl_params,
            language,
            target_language,
            self.asr_queue,
            self.transl_queue,
            only_complete_sent=bool(zoom_url)
        )
        self.translator.start()

        print("Starting Caption sender thread...", flush=True)
        capt_lang_code = get_lang_code(capt_lang)
        if zoom_url:
            zoom_url = zoom_url.strip()
            self.transl_queue.put(("...", True))  # to warm up Zoom
            self.caption_sender = ZoomCaptionSender(self.transl_queue, zoom_url, capt_lang_code)
        else:
            self.caption_sender = CaptionSender(self.transl_queue, conn, capt_lang_code)
        self.caption_sender.start()

        print("Starting ASR thread...", flush=True)
        self.asr_proc = ASRProcessor(client_params, self.audio_queue, self.asr_queue)
        self.asr_proc.start()

    def process(self, arr):
        self.audio_queue.put(arr)

    def stop(self):
        print("Stopping all threads...", flush=True)

        print("ASR thread exiting...", flush=True)
        self.asr_proc.stop()

        print("Translator thread exiting...", flush=True)
        self.translator.stop()

        print("Caption sender thread exiting...", flush=True)
        self.caption_sender.stop()

        self.asr_proc = None
        self.translator = None
        self.caption_sender = None


class WhisperServer:
    def __init__(self, host="0.0.0.0", port=5000, write_wav=False, write_transcript=False):
        self.conn = None
        self.server_thread = None
        self.is_running = False
        self.host = host
        self.port = port
        self.write_wav = write_wav
        self.write_transcript = write_transcript

    def start(self):
        if not self.is_running:
            self.server_thread = threading.Thread(target=self.run, daemon=True)
            self.server_thread.start()
            self.is_running = True

    def run(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind((self.host, self.port))
        srv.listen(1)

        try:
            while True:
                print(f"Server listening on {self.host}:{self.port}")
                conn, addr = srv.accept()
                print(f"Connected by {addr}")
                self.handle_client(conn)
        except KeyboardInterrupt:
            print("Interrupted by user, stopping...")
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            print("Server shutting down.")
            srv.close()
            self.is_running = False

    def handle_client(self, conn: socket.socket):
        try:
            # Step 1: receive params
            params = ccmn.recv_json(conn)
            if params is None:
                print("Failed to receive params.")
                conn.close()
                return
            print("Received params:", params)

            pipeline = WhisperPipeline(params, conn)

            if self.write_wav:
                import wav_writer
                from datetime import datetime
                filename = datetime.now().strftime("recording-%Y%m%d_%H%M%S.wav")
                wav_out = wav_writer.WavWriter(filename)

            # Step 2: confirm initialization
            ccmn.send_json(conn, {"type": "status", "value": "ready"})

            # Step 3: receive audio chunks
            while True:
                chunk = ccmn.recv_ndarray(conn)
                if chunk is None:
                    print("Client disconnected unexpectedly.")
                    break

                # Empty array signals that the client is diconnecting.
                if len(chunk) == 0:
                    print("Client gracefully disconnecting.")

                    # Stop the pipeline. This will flush any remaining text.
                    pipeline.stop()

                    # Confirm shutdown
                    ccmn.send_json(conn, {
                        "type": "status",
                        "value": "shutdown"
                    })

                    conn.shutdown(socket.SHUT_WR)
                    break

                # Send received chunk to the pipeline
                pipeline.process(chunk)

                if self.write_wav:
                    wav_out.write_chunk(chunk)

        except (ConnectionResetError, OSError) as e:
            print(f"Connection lost: {e}.")
        finally:
            conn.close()
            print("Connection closed.")

            if self.write_wav:
                wav_out.close()


def asr_subprocess_main(client_params: dict, audio_queue: mp.Queue, asr_queue: mp.Queue):
    """
    Runs in the subprocess: construct WhisperOnline here, then
    read audio chunks from audio_queue and put results to asr_queue.
    """
    whisper_online = WhisperOnline(client_params)

    # loop until sentinel
    while True:
        chunk = audio_queue.get()
        if chunk is None:
            break

        whisper_online.asr_proc.insert_audio_chunk(chunk)
        result = whisper_online.asr_proc.process_iter()
        if result and result[2]:
            asr_queue.put(result[2])
            print(f"[ASR] {result[2]}", flush=True)

    # flush any remaining audio
    result = whisper_online.asr_proc.finish()
    if result and result[2]:
        asr_queue.put(result[2])


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    import sys

    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host adress. Default: 0.0.0.0")
    parser.add_argument("--port", type=int, default=5000, help="Port number. Default: 5000")
    parser.add_argument("--write-wav", action="store_true", help="Write received audio to a wav file.")
    parser.add_argument("--write-transcript", action="store_true", help="Write received transcript to a text file.")
    parser.add_argument("--warmup-file", type=str, default="data/samples_jfk.wav", help="Provide the audio file used to warm up the whisper model.")
    args = parser.parse_args(sys.argv[1:])

    whisper_server = WhisperServer(args.host, args.port, args.write_wav, args.write_transcript)
    whisper_server.start()

    while True:
        try:
            threading.Event().wait(1)
            if whisper_server.is_running is False:
                break
        except KeyboardInterrupt:
            print("Stopping server...")
            break
