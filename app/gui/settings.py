from dataclasses import dataclass, field

from app.common.utils import dataclass_from_dict, merge_dataclass_from_dict, settings_to_dict


@dataclass
class AudioDeviceSettings:
    name: str = ""
    host_api: str = ""
    block_duration_s: float = 0.0


@dataclass
class AudioSettings:
    chunk_size_s: float = 0.4
    use_second_device: bool = False

    device_1: AudioDeviceSettings = field(default_factory=AudioDeviceSettings)
    device_2: AudioDeviceSettings = field(default_factory=AudioDeviceSettings)


@dataclass
class CaptionsOverlaySettings:
    font_size: int = 24
    max_visible_lines: int = 4


@dataclass
class ClientSettings:
    host: str = ""
    port: int = 5000
    audio: AudioSettings = field(default_factory=AudioSettings)
    captions_overlay: CaptionsOverlaySettings = field(default_factory=CaptionsOverlaySettings)


__all__ = [
    "AudioDeviceSettings",
    "AudioSettings",
    "CaptionsOverlaySettings",
    "ClientSettings",
    "dataclass_from_dict",
    "merge_dataclass_from_dict",
    "settings_to_dict",
]