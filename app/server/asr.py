import logging
import multiprocessing as mp
from pathlib import Path

from app.common.utils import MPCountingQueue


class WhisperServerParams:
    def __init__(self):
        # Zoom URL
        self.zoom_url = ""

        # Whisper model
        self.model = "large-v2"
        self.model_cache_dir = None
        self.model_dir = None
        self.warmup_file = "data/samples_jfk.wav"

        # Language and task
        self.language = "auto"
        self.task = "transcribe"

        # Backend
        self.backend = "faster-whisper"
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
        self.buffer_trimming = "segment"
        self.buffer_trimming_sec = 15

        # Device and compute
        self.whisper_device = "cuda"
        self.whisper_compute_type = "float32"

        # Logging
        self.log_level = "DEBUG"


class WhisperOnline:
    def __init__(self, client_params: dict, logger: logging.Logger):
        import app.server.whisper_streamer as ws

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

        ws.set_logging(logger.getEffectiveLevel())

        # Create whisper online processor object with params.
        self.asr, self.asr_proc = ws.asr_factory(params)

        # warm up the ASR so first chunk isn't slow
        warmup_file = self._resolve_warmup_file(params.warmup_file)
        if warmup_file:
            a = ws.load_audio_chunk(warmup_file, 0, 1)
            self.asr.transcribe(a)
            logger.info(f"Whisper has warmed up using: {warmup_file}")
        else:
            logger.warning(
                f"Warm up file not found: {params.warmup_file}. Whisper is not warmed up. The first chunk processing may take longer."
            )

    def clear(self):
        # Call before using this object for a new audio stream.
        # Not used for now because a new object is created per client.
        self.asr_proc.init()

    def has_vac(self) -> bool:
        return hasattr(self.asr_proc, "vac") and self.asr_proc.vac is not None

    def _resolve_warmup_file(self, path_str: str | None) -> str | None:
        if not path_str:
            return None

        p = Path(path_str)
        if p.is_absolute():
            return str(p) if p.is_file() else None

        server_dir = Path(__file__).resolve().parent
        repo_root = server_dir.parent.parent

        candidates = [
            server_dir / p,
            Path.cwd() / p,
            repo_root / p,
        ]

        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)

        return None


class ASRProcessor:
    def __init__(
        self,
        client_params: dict,
        audio_queue: MPCountingQueue,
        result_queue: MPCountingQueue,
        sender_queue: mp.Queue,
    ):
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
                args=(
                    self.client_params,
                    self.audio_queue,
                    self.result_queue,
                    self.sender_queue,
                    self.asr_ready_event,
                    self.shutdown_event,
                ),
                daemon=False,
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


def asr_subprocess_main(
    client_params: dict,
    audio_queue: MPCountingQueue,
    asr_queue: MPCountingQueue,
    sender_queue: mp.Queue,
    asr_ready_event,
    shutdown_event,
):
    """Run in subprocess: consume audio chunks and push ASR results."""
    logger = logging.getLogger("whisper_online_asr_subproc")
    logger.setLevel(client_params.get("log_level", "INFO"))
    logging.basicConfig(format="%(levelname)s\t%(message)s")

    sender_queue.put({"type": "status", "value": {"status": "asr_initializing"}})

    whisper_online = WhisperOnline(client_params, logger)
    asr_ready_event.set()

    sender_queue.put({"type": "status", "value": {"status": "asr_initialized"}})

    has_vac = whisper_online.has_vac()
    last_vac_status = None

    try:
        while not shutdown_event.is_set():
            chunk = audio_queue.get()
            if chunk is None:
                break

            sender_queue.put(
                {
                    "type": "statistics",
                    "values": {
                        "asr_in_q_size": audio_queue.qsize(),
                    },
                }
            )

            whisper_online.asr_proc.insert_audio_chunk(chunk)
            confirmed, unconfirmed, action = whisper_online.asr_proc.process_iter()

            if has_vac and whisper_online.asr_proc.status != last_vac_status:
                last_vac_status = whisper_online.asr_proc.status
                sender_queue.put(
                    {
                        "type": "statistics",
                        "values": {
                            "vac_voice_status": last_vac_status,
                        },
                    }
                )

            if confirmed[2] or unconfirmed[2]:
                asr_queue.put((confirmed[2], unconfirmed[2]))
                logger.info(f"[ASR] {confirmed[2]} | {unconfirmed[2]}")

                if action == "inference":
                    sender_queue.put(
                        {
                            "type": "statistics",
                            "values": {
                                "last_asr_proc_time": whisper_online.asr_proc.get_last_inference_time(),
                                "asr_roll_avg_proc_time": whisper_online.asr_proc.get_roll_avg_inference_time(),
                            },
                        }
                    )
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"ASR subprocess exception: {e}")

    whisper_online.asr_proc.finish()

__all__ = [
    "WhisperServerParams",
    "WhisperOnline",
    "ASRProcessor",
]
