import logging
import multiprocessing as mp
from pathlib import Path

from app.common.utils import MPCountingQueue
from app.server.settings import PipelineSettings

logger = logging.getLogger(__name__)


class WhisperOnline:
    def __init__(self, settings: PipelineSettings, logger: logging.Logger):
        import app.server.whisper_streamer as ws

        ws.set_logging(logger.getEffectiveLevel())

        # Create whisper online processor object with settings.
        self.asr, self.asr_proc = ws.asr_factory(settings)

        # warm up the ASR so first chunk isn't slow
        warmup_file = self._resolve_warmup_file(settings.asr.warmup_file)
        if warmup_file:
            a = ws.load_audio_chunk(warmup_file, 0, 1)
            self.asr.transcribe(a)
            logger.info(f"Whisper has warmed up using: {warmup_file}")
        else:
            logger.warning(
                f"Warm up file not found: {settings.asr.warmup_file}. Whisper is not warmed up. The first chunk processing may take longer."
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
        settings: PipelineSettings,
        audio_queue: MPCountingQueue,
        result_queue: MPCountingQueue,
        sender_queue: mp.Queue,
    ):
        self.settings = settings
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
                    self.settings,
                    self.audio_queue,
                    self.result_queue,
                    self.sender_queue,
                    self.asr_ready_event,
                    self.shutdown_event,
                    logging.getLevelName(logger.getEffectiveLevel()),
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
        if self.asr_ready_event is None:
            return False
        return self.asr_ready_event.wait(timeout=timeout)


def asr_subprocess_main(
    settings: PipelineSettings,
    audio_queue: MPCountingQueue,
    asr_queue: MPCountingQueue,
    sender_queue: mp.Queue,
    asr_ready_event,
    shutdown_event,
    log_level,
):
    """Run in subprocess: consume audio chunks and push ASR results."""
    logging.basicConfig(format="%(levelname)s\t%(message)s")
    logger = logging.getLogger("whisper_online_asr_subproc")
    logger.setLevel(log_level)

    sender_queue.put({"type": "status", "value": {"status": "asr_initializing"}})

    whisper_online = WhisperOnline(settings, logger)
    asr_ready_event.set()

    sender_queue.put({"type": "status", "value": {"status": "asr_initialized"}})

    has_vac = whisper_online.has_vac()
    last_vac_status = None

    try:
        while not shutdown_event.is_set():
            chunk = audio_queue.get()
            if chunk is None:
                break
                
            if isinstance(chunk, dict) and chunk.get("type") == "update_vac_settings":
                vac_settings = chunk["vac"]
                if hasattr(whisper_online.asr_proc, "vac") and whisper_online.asr_proc.vac is not None:
                    whisper_online.asr_proc.vac.start_threshold = vac_settings.start_threshold
                    whisper_online.asr_proc.vac.end_threshold = vac_settings.end_threshold
                    whisper_online.asr_proc.vac.min_silence_samples = whisper_online.asr_proc.vac.sampling_rate * vac_settings.min_silence_duration_ms / 1000
                    whisper_online.asr_proc.vac.speech_pad_start_samples = whisper_online.asr_proc.vac.sampling_rate * vac_settings.speech_pad_start_ms / 1000
                    whisper_online.asr_proc.vac.speech_pad_end_samples = whisper_online.asr_proc.vac.sampling_rate * vac_settings.speech_pad_end_ms / 1000
                    whisper_online.asr_proc.vac.hangover_chunks = vac_settings.hangover_chunks
                    
                whisper_online.asr_proc.online_chunk_size = vac_settings.min_chunk_size_s
                whisper_online.asr_proc.is_dynamic_chunk_size = vac_settings.is_dynamic_chunk_size
                continue

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
    "WhisperOnline",
    "ASRProcessor",
]
