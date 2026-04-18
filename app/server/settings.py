from dataclasses import asdict, dataclass, field, fields, is_dataclass


@dataclass
class VACSettings:
    enable: bool = True
    enable_whisper_internal_vad: bool = False
    min_chunk_size_s: float = 1.2
    is_dynamic_chunk_size: bool = True
    start_threshold: float = 0.5
    end_threshold: float = 0.35
    min_silence_duration_ms: int = 500
    speech_pad_start_ms: int = 800
    speech_pad_end_ms: int = 900
    hangover_chunks: int = 2


@dataclass
class ASRSettings:
    model: str = "distil-large-v3"
    model_cache_dir: str = None
    model_dir: str = None
    language: str = "en"
    task: str = "transcribe"
    backend: str = "faster-whisper"
    nsp_threshold: float = None
    device: str = "cuda"
    compute_type: str = "int8"
    buffer_trimming: str = "segment"
    buffer_trimming_sec: int = 15
    warmup_file: str = "data/samples_jfk.wav"


@dataclass
class TranslationSettings:
    enable: bool = True
    target_language: str = "sr"
    engine: str = "MarianMT"
    engine_params: dict = field(default_factory=dict)
    word_increment: int = 0
    source_diff_enabled: bool = True
    target_diff_enabled: bool = True


@dataclass
class PipelineSettings:
    zoom_url: str = None
    write_wav: bool = False
    write_transcript: bool = False

    vac: VACSettings = field(default_factory=VACSettings)
    asr: ASRSettings = field(default_factory=ASRSettings)
    translation: TranslationSettings = field(default_factory=TranslationSettings)


@dataclass
class ServerSettings:
    host: str = "0.0.0.0"
    port: int = 5000

    pipeline: PipelineSettings = field(default_factory=PipelineSettings)


def merge_dataclass_from_dict(instance, values: dict):
    """Recursively updates a dataclass instance from a nested dictionary."""
    if not isinstance(values, dict):
        return instance

    for f in fields(instance):
        if f.name not in values:
            continue

        incoming = values[f.name]
        current = getattr(instance, f.name)

        if is_dataclass(current) and isinstance(incoming, dict):
            merge_dataclass_from_dict(current, incoming)
        else:
            setattr(instance, f.name, incoming)

    return instance


def pipeline_settings_from_dict(values: dict | None) -> PipelineSettings:
    return merge_dataclass_from_dict(PipelineSettings(), values or {})


def server_settings_from_dict(values: dict | None) -> ServerSettings:
    return merge_dataclass_from_dict(ServerSettings(), values or {})


def settings_to_dict(instance) -> dict:
    return asdict(instance)


__all__ = [
    "ServerSettings",
    "VACSettings",
    "ASRSettings",
    "TranslationSettings",
    "PipelineSettings",
    "merge_dataclass_from_dict",
    "pipeline_settings_from_dict",
    "server_settings_from_dict",
    "settings_to_dict",
]
