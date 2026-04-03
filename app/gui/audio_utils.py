import sys
from collections import defaultdict
from typing import Tuple

if sys.platform == "win32":
    import pyaudiowpatch as pyaudio
else:
    import pyaudio
WHISPER_SAMPLERATE: int = 16000

class InputDeviceCaps:
    def __init__(self, channels: int, samplerate: float, min_latency: float, max_latency: float, index: int):
        self.channels = channels
        self.samplerate = samplerate
        self.min_latency = min_latency
        self.max_latency = max_latency
        self.index = index

    def __repr__(self):
        return f"InputDeviceCaps(channels={self.channels}, samplerate={self.samplerate}, min_latency={self.min_latency}, max_latency={self.max_latency}, index={self.index})"

class InputDeviceInfo:
    def __init__(self, name: str, api: str, caps: InputDeviceCaps, is_loopback: bool = False):
        self.name = name
        self.api = api
        self.channels = caps.channels
        self.samplerate = caps.samplerate
        self.min_latency = caps.min_latency
        self.max_latency = caps.max_latency
        self.index = caps.index
        self.is_loopback = is_loopback

        self.downmix_needed = self.channels > 1
        self.resample_needed = self.samplerate != WHISPER_SAMPLERATE

        self.choose_safe_block_dur()

    def choose_safe_block_dur(
        self,
        min_block_dur_floor: float = 0.005,   # absolute min 5 ms
        preferred_upper_bound: float = 0.05   # don't pick > 50 ms for realtime
    ) -> Tuple[float, int, Tuple[float, float]]:
        """
        Return (block_dur_seconds, blocksize_frames, (min_allowed, max_allowed)).

        - Uses device low/high latencies as guidance.
        - Adjusts preferred block duration based on host API.
        - Enforces practical lower/upper limits for realtime streaming.
        """
        low = float(self.min_latency or 0.0)
        high = float(self.max_latency or 0.0)

        # Baseline allowed range
        min_allowed = max(low, min_block_dur_floor)
        max_allowed = max(high, min_allowed * 4)

        # Determine API-specific bias
        api_name = self.api.lower()
        if "wasapi" in api_name:
            api_bias = 1.0       # keep preferred near low latency
        elif "wdm-ks" in api_name:
            api_bias = 1.1       # slightly higher to handle jitter
        elif "directsound" in api_name:
            api_bias = 2.0       # bigger blocks for stability
        elif "mme" in api_name:
            api_bias = 2.5       # legacy API, even bigger blocks
        else:
            api_bias = 1.5       # fallback

        # Preferred block duration
        preferred = max(min_allowed * api_bias, 0.02)          # at least ~20ms
        preferred = min(preferred, preferred_upper_bound)      # don't exceed upper bound
        preferred = max(min_allowed, min(preferred, max_allowed)) # clamp to allowed range

        # Compute blocksize in frames
        samplerate: int = int(self.samplerate)
        blocksize = max(32, int(round(preferred * samplerate)))   # at least 32 frames
        preferred = blocksize / samplerate                        # adjust preferred to integer frames

        self.block_dur = preferred
        self.block_size = blocksize
        self.min_block_dur_allowed = min_allowed
        self.max_block_dur_allowed = max_allowed

    def set_block_dur(self, block_dur: float):
        self.block_dur = block_dur
        self.block_size = int(round(self.block_dur * self.samplerate))


def sort_api_by_preference(api_list):
    if sys.platform.startswith("win"):
        preferred_apis = ["Windows WASAPI", "Windows WDM-KS", "Windows DirectSound", "MME"]
    elif sys.platform == "darwin":
        preferred_apis = ["Core Audio"]
    else:  # Linux
        preferred_apis = ["ALSA", "PulseAudio", "JACK"]

    order = {item: i for i, item in enumerate(preferred_apis)}

    # Use a tuple as key: (index if in preferred_apis, else very large number to push to end)
    return sorted(api_list, key=lambda x: order.get(x, float('inf')))


def list_unique_input_devices():
    # Outer dict: device_name -> hostapi_name -> (InputDeviceCaps, is_loopback)
    devices_dict = defaultdict(dict)

    p = pyaudio.PyAudio()

    hostapis = []
    for i in range(p.get_host_api_count()):
        api = p.get_host_api_info_by_index(i)
        hostapis.append(api['name'])

    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)

        if dev['maxInputChannels'] > 0:
            hostapi_name = hostapis[dev['hostApi']]

            is_loopback = dev.get('isLoopbackDevice', False)

            caps = InputDeviceCaps(
                channels=int(dev['maxInputChannels']),
                samplerate=dev['defaultSampleRate'],
                min_latency=dev['defaultLowInputLatency'],
                max_latency=dev['defaultHighInputLatency'],
                index=dev['index']
            )

            devices_dict[dev['name']][hostapi_name] = (caps, is_loopback)

    p.terminate()

    return devices_dict


def default_input_device(device_map):
    p = pyaudio.PyAudio()
    def_dev_index = p.get_default_input_device_info()['index']
    p.terminate()

    for name, api_map in device_map.items():
        for api, (caps, is_loopback) in api_map.items():
            if caps.index == def_dev_index:
                return (name, api)
    return None

__all__ = ["WHISPER_SAMPLERATE", "InputDeviceCaps", "InputDeviceInfo", "sort_api_by_preference", "list_unique_input_devices", "default_input_device"]

