import sys
import socket
import queue
import threading
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, font
import logging
import captioner_common as ccmn
from collections import defaultdict
import captions_overlay
from typing import Tuple

logger = logging.getLogger(__name__)

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
    def __init__(self, name: str, api: str, caps: InputDeviceCaps):
        self.name = name
        self.api = api
        self.channels = caps.channels
        self.samplerate = caps.samplerate
        self.min_latency = caps.min_latency
        self.max_latency = caps.max_latency
        self.index = caps.index

        self.downmix_needed, self.resample_needed = self.check_needed_processing()
        self.target_channels = self.channels if self.downmix_needed else 1
        self.target_samplerate = self.samplerate if self.resample_needed else WHISPER_SAMPLERATE

        self.choose_safe_block_dur()

    def check_input_settings(self, device, channels, samplerate):
        settings = None
        if self.api == "Windows WASAPI":
            # For WASAPI, we want to allow other apps to access the mic, and to auto-convert to float32 if necessary.
            settings = sd.WasapiSettings(exclusive=False, auto_convert=True)

        try:
            sd.check_input_settings(
                device=device,
                channels=channels,
                samplerate=samplerate,
                dtype="float32",
                extra_settings=settings
            )
        except Exception:
            return False
        return True

    def check_needed_processing(self):
        # No processing needed if these settings are accepted
        if self.check_input_settings(self.index, 1, WHISPER_SAMPLERATE):
            return False, False

        # Only downmix needed if 16kHz samplerate is accepted
        if self.check_input_settings(self.index, self.channels, WHISPER_SAMPLERATE):
            return True, False

        # Only resample needed if mono is accepted
        if self.check_input_settings(self.index, 1, self.samplerate):
            return False, True

        # Both downmix and resample needed
        return True, True

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
        samplerate: int = int(self.target_samplerate)
        blocksize = max(32, int(round(preferred * samplerate)))   # at least 32 frames
        preferred = blocksize / samplerate                        # adjust preferred to integer frames

        self.block_dur = preferred
        self.block_size = blocksize
        self.min_block_dur_allowed = min_allowed
        self.max_block_dur_allowed = max_allowed

    def set_block_dur(self, block_dur: float):
        self.block_dur = block_dur
        self.block_size = int(round(self.block_dur * self.target_samplerate))


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
    # Outer dict: device_name -> hostapi_name -> InputDeviceCaps
    devices_dict = defaultdict(dict)

    all_devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    input_devices = []
    input_devices_mme = []

    for dev in all_devices:
        if dev['max_input_channels'] > 0:
            hostapi_name = hostapis[dev['hostapi']]['name']
            if hostapi_name == "MME":
                input_devices_mme.append(dev)
            else:
                input_devices.append(dev)

    # Convert to DeviceList
    input_devices = sd.DeviceList(input_devices)
    input_devices_mme = sd.DeviceList(input_devices_mme)

    for dev in input_devices:
        hostapi_name = hostapis[dev['hostapi']]['name']

        devices_dict[dev['name']][hostapi_name] = InputDeviceCaps(
            channels=dev['max_input_channels'],
            samplerate=dev['default_samplerate'],
            min_latency=dev['default_low_input_latency'],
            max_latency=dev['default_high_input_latency'],
            index=dev['index']
        )

    for dev in input_devices_mme:
        # If we already saved a device whose name starts with this devices name,
        # we assume that it is the same device - only with a cut-off name - and we
        # just add device caps for MME API to the saved device's API map.
        merged = False
        for saved_name, saved_api_map in devices_dict.items():
            if saved_name.startswith(dev['name']):
                if not saved_api_map.get("MME"):
                    saved_api_map["MME"] = InputDeviceCaps(
                        channels=dev['max_input_channels'],
                        samplerate=dev['default_samplerate'],
                        min_latency=dev['default_low_input_latency'],
                        max_latency=dev['default_high_input_latency'],
                        index=dev['index']
                    )
                    merged = True
                    break

        if not merged:
            devices_dict[dev['name']]["MME"] = InputDeviceCaps(
                channels=dev['max_input_channels'],
                samplerate=dev['default_samplerate'],
                min_latency=dev['default_low_input_latency'],
                max_latency=dev['default_high_input_latency'],
                index=dev['index']
            )

    return devices_dict


def default_input_device(device_map):
    def_dev_index = sd.default.device[0]
    for name, api_map in device_map.items():
        for api, caps in api_map.items():
            if caps.index == def_dev_index:
                return (name, api)
    return None

def str_to_float(s):
    try:
        f = float(s)
        return f, True  # conversion successful
    except ValueError:
        return None, False  # invalid float

def str_to_int(s):
    try:
        i = int(s)
        return i, True  # conversion successful
    except ValueError:
        return None, False  # invalid int

def clamp(x, lo, hi):
    return max(lo, min(x, hi))

def count_decimal_places(num: float) -> int:
    from decimal import Decimal

    d = Decimal(str(num))  # convert via string to preserve digits
    decimals = max(0, -d.as_tuple().exponent)
    return decimals

class CaptionerUI:
    def __init__(self):
        self.is_recording = False
        self.is_connected_to_server = False
        self.selected_device_1_info = None
        self.selected_device_2_info = None
        self.audio_listener = None
        self.audio_listener_2 = None
        self.audio_mixer = None
        self.audio_temp_queue_1 = None
        self.audio_temp_queue_2 = None
        self.audio_queue = None
        self.results_queue = None
        self.whisper_client = None
        self.captions_overlay = None
        self.setup_ui()

        self.gui_queue = queue.Queue()
        threading.Thread(target=self.update_gui, daemon=True).start()

    def update_gui(self):
        while True:
            try:
                lmbd = self.gui_queue.get()
                if lmbd is not None:
                    lmbd()
            except Exception:
                pass

    def setup_ui(self):
        root = tk.Tk()
        self.root_wnd = root
        self.root_wnd.title("Captioner")
        self.root_wnd.resizable(False, False)  # make window non-resizable
        self.row_index_map = {}  # to keep track of grid row index - wnd/frame -> index

        row_idx = self.next_row()
        url_frame = ttk.Frame(root)
        url_frame.grid(row=row_idx, column=0, sticky="ew", padx=5, pady=5)

        # make second column expand
        url_frame.columnconfigure(1, weight=1)

        # --- Whisper Server URL ---
        row_idx = self.next_row(url_frame)
        ttk.Label(url_frame, text="Whisper Server URL").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.server_url_var = tk.StringVar(value="galloway")
        self.server_url_entry = ttk.Entry(url_frame, textvariable=self.server_url_var)
        self.server_url_entry.grid(row=row_idx, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.server_port_var = tk.StringVar(value="5000")
        self.server_port_entry = ttk.Entry(url_frame, textvariable=self.server_port_var, width=6)
        self.server_port_entry.grid(row=row_idx, column=3, sticky="w", padx=5, pady=5)
        self.connect_btn = ttk.Button(url_frame, text="Connect", command=self.toggle_connection)
        self.connect_btn.grid(row=row_idx, column=4, sticky="e", padx=5, pady=5)

        # --- Zoom URL ---
        row_idx = self.next_row(url_frame)
        ttk.Label(url_frame, text="Zoom URL").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.zoom_url_var = tk.StringVar(value="")
        self.zoom_url_entry = ttk.Entry(url_frame, textvariable=self.zoom_url_var)
        self.zoom_url_entry.grid(row=row_idx, column=1, columnspan=3, sticky="ew", padx=5, pady=5)
        self.clear_btn = ttk.Button(url_frame, text="Clear", command=lambda: self.zoom_url_var.set(""))
        self.clear_btn.grid(row=row_idx, column=4, sticky="e", padx=5, pady=5)

        # --- Recording ---
        row_idx = self.next_row(url_frame)
        ttk.Label(url_frame, text="Audio sources").grid(row=row_idx, column=0, sticky="nw", padx=5, pady=5)
        self.dev_label = ttk.Label(url_frame, text="Dev1: \n Dev2: ", relief="solid", justify="left", anchor="nw", padding=(4, 4))
        self.dev_label.grid(row=row_idx, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        mute_frame = ttk.Frame(url_frame)
        mute_frame.grid(row=row_idx, column=3, sticky="ew")
        bigfont = font.Font(size=12)
        style = ttk.Style()
        style.configure("Icon.TButton", font=bigfont)
        self.mute_btn = ttk.Button(mute_frame, text="🔊", width=2, style="Icon.TButton", command=self.toggle_mute, state="disabled")
        self.mute_btn.pack(side="top", padx=0, pady=0)
        self.mute_btn_2 = ttk.Button(mute_frame, text="🔊", width=2, style="Icon.TButton", command=self.toggle_mute_2, state="disabled")
        self.mute_btn_2.pack(side="top", padx=0, pady=0)
        self.record_btn = ttk.Button(url_frame, text="Record", command=self.toggle_recording, state="disabled")
        self.record_btn.grid(row=row_idx, column=4, sticky="ne", padx=5, pady=5)

        # --- Settings ---

        row_idx = self.next_row()
        settings_frame = ttk.Frame(root)
        settings_frame.grid(row=row_idx, column=0, sticky="ew", padx=5, pady=5)

        settings_notebook = ttk.Notebook(settings_frame)
        settings_notebook.pack(fill="both", expand=True)

        # +++ Whisper tab +++
        whisper_tab = ttk.Frame(settings_notebook, padding=10)
        settings_notebook.add(whisper_tab, text="Whisper")

        # === Speech language ===
        row_idx = self.next_row(whisper_tab)
        ttk.Label(whisper_tab, text="Speech language").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.lang_var = tk.StringVar(value="English")
        self.lang_combo = ttk.Combobox(whisper_tab, textvariable=self.lang_var, values=["English", "German", "Serbian"], state="readonly")
        self.lang_combo.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)

        # === Whisper model ===
        row_idx = self.next_row(whisper_tab)
        ttk.Label(whisper_tab, text="Model").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.model_var = tk.StringVar(value="distil-large-v3")
        model_options = [
            "tiny.en", "tiny",
            "base.en", "base",
            "small.en", "distil-small.en", "small",
            "medium.en", "distil-medium.en", "medium",
            "large-v1", "large-v2", "distil-large-v2", "large-v3", "distil-large-v3", "distil-large-v3.5", "large", "large-v3-turbo", "turbo"
        ]
        self.model_combo = ttk.Combobox(whisper_tab, textvariable=self.model_var, values=model_options, state="readonly")
        self.model_combo.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)

        # === Whisper device ===
        row_idx = self.next_row(whisper_tab)
        ttk.Label(whisper_tab, text="Device").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.whisper_device_var = tk.StringVar(value="cuda")
        self.whisper_device_combo = ttk.Combobox(whisper_tab, textvariable=self.whisper_device_var, values=["cuda", "cpu"], state="readonly")
        self.whisper_device_combo.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)

        # === Whisper compute type ===
        row_idx = self.next_row(whisper_tab)
        ttk.Label(whisper_tab, text="Compute type").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.whisper_compute_type_var = tk.StringVar(value="int8")
        dtypes = ["int8", "int8_float16", "float16", "float32"]
        self.whisper_compute_type_combo = ttk.Combobox(whisper_tab, textvariable=self.whisper_compute_type_var, values=dtypes, state="readonly")
        self.whisper_compute_type_combo.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)

        # === Non-speech probability threshold ===
        row_idx = self.next_row(whisper_tab)
        ttk.Label(whisper_tab, text="Non-speech probability\nthreshold", justify="left", anchor="w").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.threshold_var = tk.DoubleVar(value=0.9)
        self.threshold_slider = tk.Scale(whisper_tab, from_=0.0, to=1.0, orient="horizontal", resolution=0.01, showvalue=False, variable=self.threshold_var,
                                         command=lambda val: self.threshold_label.config(text=f"{float(val):.2f}"))
        self.threshold_slider.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)
        self.threshold_label = ttk.Label(whisper_tab, text=f"{self.threshold_var.get()}", width=5, relief="flat", anchor="center")
        self.threshold_label.grid(row=row_idx, column=2, sticky="w", padx=5, pady=5)

        # === Minimum chunk size ===
        row_idx = self.next_row(whisper_tab)
        ttk.Label(whisper_tab, text="Minimum chunk size (sec)", justify="left", anchor="w").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.min_chunk_size_var = tk.DoubleVar(value=0.6)
        self.min_chunk_size_slider = tk.Scale(whisper_tab, from_=0.1, to=3.0, orient="horizontal", resolution=0.1, showvalue=False, variable=self.min_chunk_size_var,
                                              command=lambda val: self.min_chunk_size_label.config(text=f"{float(val):.1f}"))
        self.min_chunk_size_slider.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)
        self.min_chunk_size_label = ttk.Label(whisper_tab, text=f"{self.min_chunk_size_var.get()}", width=5, relief="flat", anchor="center")
        self.min_chunk_size_label.grid(row=row_idx, column=2, sticky="w", padx=5, pady=5)

        # === Checkboxes ===
        row_idx = self.next_row(whisper_tab)
        self.vac_var = tk.BooleanVar(value=True)
        self.vad_var = tk.BooleanVar(value=True)
        self.vac_check = ttk.Checkbutton(whisper_tab, text="Voice activity controller", variable=self.vac_var)
        self.vad_check = ttk.Checkbutton(whisper_tab, text="Voice activity detection", variable=self.vad_var)
        self.vac_check.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        row_idx = self.next_row(whisper_tab)
        self.vad_check.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)

        # +++ Translation tab +++
        translation_tab = ttk.Frame(settings_notebook, padding=10)
        settings_notebook.add(translation_tab, text="Translation")

        # === Target language ===
        row_idx = self.next_row(translation_tab)
        self.enable_translation_var = tk.BooleanVar(value=True)
        self.enable_translation_check = ttk.Checkbutton(translation_tab, text="Enable translation", variable=self.enable_translation_var, command=self.on_enable_translation_toggle)
        self.enable_translation_check.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)

        row_idx = self.next_row(translation_tab)
        ttk.Label(translation_tab, text="Target language").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.target_lang_var = tk.StringVar(value="Serbian Cyrillic")
        target_langs = ["Serbian Cyrillic", "Serbian Latin", "English", "German"]
        self.target_lang_combo = ttk.Combobox(translation_tab, textvariable=self.target_lang_var, values=target_langs, state="readonly")
        self.target_lang_combo.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)

        self.transl_engines = ["MarianMT", "NLLB", "EuroLLM", "Whisper", "Google Gemini"]
        self.transl_engines_with_params = ["EuroLLM", "Google Gemini"]
        row_idx = self.next_row(translation_tab)
        ttk.Label(translation_tab, text="Translation engine").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.transl_engine_var = tk.StringVar(value="MarianMT")
        self.transl_engine_combo = ttk.Combobox(translation_tab, textvariable=self.transl_engine_var, values=self.transl_engines, state="readonly")
        self.transl_engine_combo.bind("<<ComboboxSelected>>", self.on_transl_engine_selection_change)
        self.transl_engine_combo.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)

        row_idx = self.next_row(translation_tab)
        translation_tab.grid_columnconfigure(2, weight=1)
        self.engine_params_frame = ttk.Frame(translation_tab, padding=10)
        self.engine_params_frame.grid(row=row_idx, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        self.engine_params_frame.grid_columnconfigure(2, weight=1)
        row_idx = self.next_row(self.engine_params_frame)
        ttk.Label(self.engine_params_frame, text="API Key").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.transl_api_key_var = tk.StringVar(value="")
        self.transl_api_key_entry = ttk.Entry(self.engine_params_frame, textvariable=self.transl_api_key_var)
        self.transl_api_key_entry.grid(row=row_idx, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.transl_engine_combo.event_generate("<<ComboboxSelected>>")

        # +++ Audio tab +++
        audio_tab = ttk.Frame(settings_notebook, padding=10)
        settings_notebook.add(audio_tab, text="Audio")

        devices_notebook = ttk.Notebook(audio_tab)
        devices_notebook.pack(fill="both", expand=True)

        self.device_map = list_unique_input_devices()
        device_list = list(self.device_map.keys())
        dev_combo_width = max(len(v) for v in device_list)
        dev_name, api_name = default_input_device(self.device_map)

        # === Audio device 1 ===
        dev1_tab = ttk.Frame(devices_notebook, padding=10)
        devices_notebook.add(dev1_tab, text="Device 1")

        row_idx = self.next_row(dev1_tab)
        ttk.Label(dev1_tab, text="Name").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.audio_device_combo_1 = ttk.Combobox(dev1_tab, values=device_list, width=dev_combo_width, state="readonly")
        self.audio_device_combo_1.grid(row=row_idx, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.audio_device_combo_1.set(dev_name)

        row_idx = self.next_row(dev1_tab)
        ttk.Label(dev1_tab, text="Host API").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.audio_device_host_api_combo_1 = ttk.Combobox(dev1_tab, values=[], state="readonly")
        self.audio_device_host_api_combo_1.grid(row=row_idx, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.audio_device_host_api_combo_1.bind("<<ComboboxSelected>>", self.on_audio_device_host_api_1_selection_change)

        # Block duration / size
        row_idx = self.next_row(dev1_tab)
        ttk.Label(dev1_tab, text="Block").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)

        # Slider
        self.audio_device_block_dur_slider_1 = tk.Scale(dev1_tab, from_=0.0, to=1.0, orient="horizontal", resolution=0.01, showvalue=False, command=self.on_audio_device_1_block_dur_change)
        self.audio_device_block_dur_slider_1.grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")

        # Block size and duration label
        self.audio_device_block_dur_label_1 = ttk.Label(dev1_tab, text="Duration: \nSize: ", relief="flat", justify="left", anchor="w", padding=(4, 2))
        self.audio_device_block_dur_label_1.grid(row=row_idx, column=2, padx=5, pady=5, sticky="ew")

        # Use a Label widget for multiline read-only display
        row_idx = self.next_row(dev1_tab)
        self.input_dev_info_label_1 = ttk.Label(dev1_tab, text="", relief="solid", padding=(4, 2))
        self.input_dev_info_label_1.grid(row=row_idx, column=0, columnspan=3, sticky="new", padx=5, pady=5)

        # === Audio device 2 ===
        dev2_tab = ttk.Frame(devices_notebook, padding=10)
        devices_notebook.add(dev2_tab, text="Device 2")

        row_idx = self.next_row(dev2_tab)
        ttk.Label(dev2_tab, text="Name").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.audio_device_combo_2 = ttk.Combobox(dev2_tab, values=device_list, width=dev_combo_width, state="disabled")
        self.audio_device_combo_2.grid(row=row_idx, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.audio_device_combo_2.set(dev_name)

        # Checkbox for enabling the second device
        self.use_second_audio_dev_var = tk.BooleanVar(value=False)
        self.use_second_audio_dev_check = ttk.Checkbutton(dev2_tab, text="Use this device", variable=self.use_second_audio_dev_var, command=self.on_enable_second_audio_device)
        self.use_second_audio_dev_check.grid(row=row_idx, column=3, sticky="w", padx=5, pady=5)

        row_idx = self.next_row(dev2_tab)
        ttk.Label(dev2_tab, text="Host API").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.audio_device_host_api_combo_2 = ttk.Combobox(dev2_tab, values=[], state="disabled")
        self.audio_device_host_api_combo_2.grid(row=row_idx, column=1, columnspan=2, sticky="ew", padx=5, pady=5)
        self.audio_device_host_api_combo_2.bind("<<ComboboxSelected>>", self.on_audio_device_host_api_2_selection_change)

        # Block duration / size
        row_idx = self.next_row(dev2_tab)
        ttk.Label(dev2_tab, text="Block").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)

        # Slider
        self.audio_device_block_dur_slider_2 = tk.Scale(dev2_tab, from_=0.0, to=1.0, orient="horizontal", resolution=0.01, showvalue=False, command=self.on_audio_device_2_block_dur_change)
        self.audio_device_block_dur_slider_2.grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")

        # Block size and duration label
        self.audio_device_block_dur_label_2 = ttk.Label(dev2_tab, text="Duration: \nSize: ", relief="flat", justify="left", anchor="w", padding=(4, 2), state="disabled")
        self.audio_device_block_dur_label_2.grid(row=row_idx, column=2, padx=5, pady=5, sticky="ew")

        # Use a Label widget for multiline read-only display
        row_idx = self.next_row(dev2_tab)
        self.input_dev_info_label_2 = ttk.Label(dev2_tab, text="", relief="solid", padding=(4, 2), state="disabled")
        self.input_dev_info_label_2.grid(row=row_idx, column=0, columnspan=3, sticky="new", padx=5, pady=5)

        # --- Buttons ---
        row_idx = self.next_row()
        buttons_frame = ttk.Frame(root)
        buttons_frame.grid(row=row_idx, column=0, sticky="e", padx=5, pady=5)

        self.quit_btn = ttk.Button(buttons_frame, text="Quit", command=self.quit)
        self.quit_btn.grid(row=0, column=3, sticky="e", padx=5, pady=5)

        # Update block duration / size labels
        self.audio_device_combo_1.bind("<<ComboboxSelected>>", self.on_audio_device_1_selection_change)
        self.audio_device_combo_1.event_generate("<<ComboboxSelected>>")
        self.audio_device_combo_2.bind("<<ComboboxSelected>>", self.on_audio_device_2_selection_change)
        self.audio_device_combo_2.event_generate("<<ComboboxSelected>>")
        self.audio_device_block_dur_slider_2.config(state="disabled")   # Disable it here because its value needs to be set first

        root.protocol("WM_DELETE_WINDOW", self.quit)

    def show_modal_message(self, title, message, parent):
        # Create a modal Toplevel window
        win = tk.Toplevel(parent)
        win.title(title)
        win.transient(parent)   # Keep on top of parent
        win.grab_set()          # Make modal
        win.resizable(False, False)  # Disable resizing

        # Message text
        label = ttk.Label(win, text=message, padding=(20, 20))
        label.pack()

        # OK button
        btn = ttk.Button(win, text="OK", command=win.destroy, width=10)
        btn.pack(pady=10)
        btn.focus_set()

        # Center relative to parent
        win.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() // 2) - (win.winfo_width() // 2)
        y = parent.winfo_rooty() + (parent.winfo_height() // 2) - (win.winfo_height() // 2)
        win.geometry(f"+{x}+{y}")

        parent.wait_window(win)  # Wait until this window is closed

    def toggle_connection(self):
        if self.is_connected_to_server:
            if self.is_recording:
                self.stop_captioner()
                self.is_recording = False
            self.connect_btn.config(state="disabled")
            self.disconnect_from_server()
            self.connect_btn.config(state="normal", text="Connect")
            self.record_btn.config(state="disabled", text="Record")
        else:
            self.connect_btn.config(state="disabled")
            self.connect_btn.update_idletasks()
            if self.connect_to_server():
                self.connect_btn.config(text="Disconnect")
                self.record_btn.config(state="normal")
            self.connect_btn.config(state="normal")

    def toggle_recording(self):
        if self.is_recording:
            self.stop_captioner()
            self.record_btn.config(text="Record")
            self.mute_btn.config(state="disabled", text="🔊")
            self.mute_btn_2.config(state="disabled", text="🔊")
            self.is_recording = False
            self.update_selected_devices_label()
            print("Recording stopped.")
        else:
            if self.use_second_audio_dev_var.get() and (self.audio_device_combo_1.get() == self.audio_device_combo_2.get()):
                self.show_modal_message("Error", "When the second audio device is enabled, different audio devices must be selected.", self.root_wnd)
                return

            # Collect all values for debugging/demo purposes
            print("Zoom URL:", self.zoom_url_var.get())
            print("Audio device:", self.audio_device_combo_1.get())
            if self.use_second_audio_dev_var.get():
                print("Audio device 2:", self.audio_device_combo_2.get())
            print("Whisper model:", self.model_var.get())
            print("Whisper device:", self.whisper_device_var.get())
            print("Whisper compute type:", self.whisper_compute_type_var.get())
            print("Language:", self.lang_var.get())
            print("Enable translation:", self.enable_translation_var.get())
            print("Target language:", self.target_lang_var.get())
            print("Threshold:", self.threshold_var.get())
            print("Minimum chunk size:", self.min_chunk_size_var.get())
            print("VAC enabled:", self.vac_var.get())
            print("VAD enabled:", self.vad_var.get())

            print("Running Captioner...")
            if self.run_captioner():
                self.record_btn.config(text="Stop")
                self.mute_btn.config(state="normal")
                if self.use_second_audio_dev_var.get():
                    self.mute_btn_2.config(state="normal")
                self.is_recording = True

    def toggle_mute(self):
        if self.audio_listener:
            if self.audio_listener.is_paused():
                self.audio_listener.resume_stream()
                self.mute_btn.config(text="🔊")
            else:
                self.audio_listener.pause_stream()
                self.mute_btn.config(text="🔇")

    def toggle_mute_2(self):
        if self.audio_listener_2:
            if self.audio_listener_2.is_paused():
                self.audio_listener_2.resume_stream()
                self.mute_btn_2.config(text="🔊")
            else:
                self.audio_listener_2.pause_stream()
                self.mute_btn_2.config(text="🔇")

    def quit(self):
        if self.is_recording:
            self.stop_captioner()
        if self.is_connected_to_server:
            self.disconnect_from_server()
        self.root_wnd.quit()
        self.root_wnd.destroy()

    def on_enable_translation_toggle(self):
        if self.enable_translation_var.get():
            self.target_lang_combo.config(state="readonly")
            self.transl_engine_combo.config(state="readonly")
            for w in self.engine_params_frame.winfo_children():
                w.config(state="normal")
        else:
            self.target_lang_combo.config(state="disabled")
            self.transl_engine_combo.config(state="disabled")
            for w in self.engine_params_frame.winfo_children():
                w.config(state="disabled")

    def on_transl_engine_selection_change(self, event):
        transl_model = self.transl_engine_var.get()
        if transl_model in self.transl_engines_with_params:
            self.engine_params_frame.grid()
        else:
            self.engine_params_frame.grid_remove()

        if transl_model == "Google Gemini":
            self.transl_api_key_var.set("")
        elif transl_model == "EuroLLM":
            self.transl_api_key_var.set("")
        else:
            self.transl_api_key_var.set("")

    def on_enable_second_audio_device(self):
        if self.use_second_audio_dev_var.get():
            self.audio_device_combo_2.config(state="readonly")
            self.input_dev_info_label_2.config(state="normal")
            self.audio_device_host_api_combo_2.config(state="readonly")
            self.audio_device_block_dur_label_2.config(state="readonly")
            self.audio_device_block_dur_slider_2.config(state="normal")
        else:
            self.audio_device_combo_2.config(state="disabled")
            self.input_dev_info_label_2.config(state="disabled")
            self.audio_device_host_api_combo_2.config(state="disabled")
            self.audio_device_block_dur_label_2.config(state="disabled")
            self.audio_device_block_dur_slider_2.config(state="disabled")
        self.update_selected_devices_label()

    def on_audio_device_1_selection_change(self, event):
        dev_name = self.audio_device_combo_1.get()
        api_map = self.device_map[dev_name]
        self.audio_device_host_api_combo_1['values'] = sort_api_by_preference(api_map.keys())
        self.audio_device_host_api_combo_1.current(0)
        self.audio_device_host_api_combo_1.event_generate("<<ComboboxSelected>>")
        self.update_selected_devices_label()

    def format_device_info_text(self, dev_info: InputDeviceInfo):
        target_ch = 1
        target_sr = WHISPER_SAMPLERATE

        info_text = (
            f"Index: {dev_info.index}\n"
            f"Channels: {dev_info.channels}"
            + (
                f" ({'software' if dev_info.downmix_needed else 'system'} downmix → {target_ch}ch)"
                if dev_info.channels != target_ch else ""
            )
            + "\n"
            f"Samplerate: {int(dev_info.samplerate)} Hz"
            + (
                f" ({'software' if dev_info.resample_needed else 'system'} resample → {target_sr / 1000:.0f} kHz)"
                if dev_info.samplerate != target_sr else ""
            )
            + "\n"
            f"Latency range: {dev_info.min_latency * 1000:.0f} – {dev_info.max_latency * 1000:.0f} ms"
        )

        return info_text

    def on_audio_device_host_api_1_selection_change(self, event):
        dev_name = self.audio_device_combo_1.get()
        host_api = self.audio_device_host_api_combo_1.get()
        caps = self.device_map[dev_name][host_api]
        dev_info = InputDeviceInfo(dev_name, host_api, caps)
        self.selected_device_1_info = dev_info

        self.input_dev_info_label_1.config(text=self.format_device_info_text(dev_info))

        min_allowed = dev_info.min_block_dur_allowed
        max_allowed = dev_info.max_block_dur_allowed
        self.audio_device_block_dur_slider_1.config(
            from_=min_allowed,
            to=max_allowed,
            resolution=0.001
        )
        self.audio_device_block_dur_slider_1.set(dev_info.block_dur)

    def on_audio_device_2_selection_change(self, event):
        dev_name = self.audio_device_combo_2.get()
        api_map = self.device_map[dev_name]
        self.audio_device_host_api_combo_2['values'] = sort_api_by_preference(api_map.keys())
        self.audio_device_host_api_combo_2.current(0)
        self.audio_device_host_api_combo_2.event_generate("<<ComboboxSelected>>")
        self.update_selected_devices_label()

    def on_audio_device_host_api_2_selection_change(self, event):
        dev_name = self.audio_device_combo_2.get()
        host_api = self.audio_device_host_api_combo_2.get()
        caps = self.device_map[dev_name][host_api]
        dev_info = InputDeviceInfo(dev_name, host_api, caps)
        self.selected_device_2_info = dev_info

        self.input_dev_info_label_2.config(text=self.format_device_info_text(dev_info))

        min_allowed = dev_info.min_block_dur_allowed
        max_allowed = dev_info.max_block_dur_allowed
        self.audio_device_block_dur_slider_2.config(
            from_=min_allowed,
            to=max_allowed,
            resolution=0.001
        )
        self.audio_device_block_dur_slider_2.set(dev_info.block_dur)

    def on_audio_device_1_block_dur_change(self, value):
        dev_info = self.selected_device_1_info
        dev_info.set_block_dur(self.audio_device_block_dur_slider_1.get())
        info_text = (
            f"Duration: {dev_info.block_dur * 1000:.0f} ms\n"
            f"Size: {dev_info.block_size} frames"
        )
        self.audio_device_block_dur_label_1.config(text=info_text)

    def on_audio_device_2_block_dur_change(self, value):
        dev_info = self.selected_device_2_info
        dev_info.set_block_dur(self.audio_device_block_dur_slider_2.get())
        info_text = (
            f"Duration: {dev_info.block_dur * 1000:.0f} ms\n"
            f"Size: {int(dev_info.block_size)} frames"
        )
        self.audio_device_block_dur_label_2.config(text=info_text)

    def update_selected_devices_label(self):
        if not self.is_recording:
            dev1_name = self.audio_device_combo_1.get()
            dev2_name = self.audio_device_combo_2.get() if self.use_second_audio_dev_var.get() else "<none>"

            info_text = (
                f"Dev1: {dev1_name}\n\n"
                f"Dev2: {dev2_name}"
            )
            self.dev_label.config(text=info_text)

    # Helper to manage grid row indices
    def next_row(self, widget=None):
        if not widget:
            widget = self.root_wnd
        if widget not in self.row_index_map:
            self.row_index_map[widget] = 0

        i = self.row_index_map[widget]
        self.row_index_map[widget] += 1
        return i

    def connect_to_server(self):
        if self.is_connected_to_server:
            return False

        threshold = self.threshold_var.get()
        min_chunk_size = self.min_chunk_size_var.get()

        port, valid = str_to_int(self.server_port_var.get())
        if not valid or not (0 < port < 65536):
            print("Error: Invalid port number. Setting to default: 5000")
            port = 5000

        transl_engine = self.transl_engine_var.get()
        if transl_engine in self.transl_engines_with_params:
            transl_params = { "api_key": self.transl_api_key_var.get().strip() }
        else:
            transl_params = {}

        params = {
            "zoom_url": self.zoom_url_var.get().strip(),
            "model": self.model_var.get(),
            "whisper_device": self.whisper_device_var.get(),
            "whisper_compute_type": self.whisper_compute_type_var.get(),
            "language": self.lang_var.get(),
            "enable_translation": bool(self.enable_translation_var.get()),
            "target_language": self.target_lang_var.get(),
            "translation_engine": transl_engine,
            "translation_params": transl_params,
            "nsp_threshold": threshold,
            "min_chunk_size": min_chunk_size,
            "log_level": "INFO",
            "vac": self.vac_var.get(),
            "vad": self.vad_var.get(),
        }

        self.audio_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.whisper_client = WhisperClient(self.server_url_var.get().strip(), port, params, self.audio_queue, self.results_queue)
        self.whisper_client.start()
        self.is_connected_to_server = self.whisper_client.wait_until_connected(2.0)
        return self.is_connected_to_server

    def disconnect_from_server(self):
        if not self.is_connected_to_server:
            return

        if self.whisper_client:
            print("Stopping whisper client...", flush=True)
            self.whisper_client.stop()
            self.whisper_client = None
            self.audio_queue = None
            self.results_queue = None
            self.is_connected_to_server = False

    def run_captioner(self) -> bool:
        if self.is_connected_to_server:
            if self.create_audio_listener():
                if not self.zoom_url_var.get().strip():
                    self.run_captions_overlay()

                return True

        return False

    def run_captions_overlay(self):
        self.captions_overlay = CaptionsReceiver(self.root_wnd, self.results_queue, self.gui_queue)
        self.captions_overlay.start()

    def create_audio_listener(self) -> bool:
        min_chunk_size = self.min_chunk_size_var.get()
        use_second_dev = self.use_second_audio_dev_var.get()
        if use_second_dev:
            self.audio_temp_queue_1 = queue.Queue()
            self.audio_listener = AudioListener(min_chunk_size, self.selected_device_1_info, self.audio_temp_queue_1)
            self.audio_listener.start()

            self.audio_temp_queue_2 = queue.Queue()
            self.audio_listener_2 = AudioListener(min_chunk_size, self.selected_device_2_info, self.audio_temp_queue_2)
            self.audio_listener_2.start()

            self.audio_mixer = AudioMixer(self.audio_temp_queue_1, self.audio_temp_queue_2, self.audio_queue, int(self.audio_listener_2.min_chunk_size * self.audio_listener_2.WHISPER_SAMPLERATE))
            self.audio_mixer.start()
        else:
            # We use one device, puts the results directly to the audio_queue.
            self.audio_listener = AudioListener(min_chunk_size, self.selected_device_1_info, self.audio_queue)
            self.audio_listener.start()
            
        return True

    def stop_captioner(self):
        if self.audio_listener:
            print("Stopping audio listener...", flush=True)
            self.audio_listener.stop()
            self.audio_listener = None

        if self.audio_listener_2:
            print("Stopping audio listener 2...", flush=True)
            self.audio_listener_2.stop()
            self.audio_listener_2 = None

            print("Stopping audio mixer thread...", flush=True)
            self.audio_mixer.stop()
            self.audio_mixer = None

        if self.captions_overlay:
            print("Stopping captions overlay...", flush=True)
            self.captions_overlay.stop()
            self.captions_overlay = None

        self.audio_temp_queue_1 = None
        self.audio_temp_queue_2 = None

    def run_gui(self):
        self.root_wnd.mainloop()


class AudioListener:
    def __init__(self, min_chunk_size: float, input_device_info: InputDeviceInfo, result_queue: queue.Queue):
        self.min_chunk_size = min_chunk_size
        self.input_device_info = input_device_info

        self.device_channels = input_device_info.target_channels
        self.device_rate = int(input_device_info.target_samplerate)

        self.resample_stream = None

        self.in_stream_block_dur = input_device_info.block_dur
        self.blocksize = int(self.device_rate * self.in_stream_block_dur)
        self.stream = None

        self.audio_queue = queue.Queue()
        self.result_queue = result_queue

        self.stop_event = None
        self.audio_thread = None
        self.is_running = False

    def pause_stream(self):
        if self.stream and self.stream.active:
            self.stream.stop()

    def resume_stream(self):
        if self.stream and not self.stream.active:
            self.stream.start()

    def is_paused(self):
        if self.stream:
            return not self.stream.active
        return False

    def start(self):
        if not self.is_running:
            self.stop_event = threading.Event()
            self.audio_thread = threading.Thread(target=self.run)
            self.audio_thread.start()
            self.is_running = True

    def stop(self):
        if self.is_running:
            self.stop_event.set()
            self.audio_queue.put(None)  # Sentinel to unblock receive_audio_chunk()
            self.audio_thread.join()
            self.audio_thread = None
            self.is_running = False
            self.stream = None

    def downmix_mono(self, data: np.ndarray) -> np.ndarray:
        # Convert multi-channel audio to mono by averaging channels
        return data.mean(axis=1)

    def resample_to_whisper(self, data: np.ndarray) -> np.ndarray:
        return self.resample_stream.resample_chunk(data)

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio callback status: {status}")

        self.audio_queue.put(indata.copy())

    def receive_audio_chunk(self):
        out = []
        minlimit = int(self.min_chunk_size * self.device_rate)
        cur_size = 0
        while cur_size < minlimit:
            chunk = self.audio_queue.get()

            # Sentinel placed into the queue when stopping
            if chunk is None:
                return None

            out.append(chunk)
            cur_size += len(chunk)

        conc = np.concatenate(out)

        mono = self.downmix_mono(conc) if self.input_device_info.downmix_needed else conc
        mono16k = self.resample_to_whisper(mono) if self.input_device_info.resample_needed else mono

        return mono16k

    def run(self):
        print_info = (
            f"Listening to [{self.input_device_info.index}] {self.input_device_info.name}\n"
            f"  API: {self.input_device_info.api}\n"
            f"  Channels: {self.device_channels}\n"
            f"  Samplerate: {self.device_rate}\n"
            f"  Blocksize: {self.blocksize} frames (block duration: ~{self.in_stream_block_dur} s)"
        )
        print(print_info, flush=True)

        settings = None
        if self.input_device_info.api == "Windows WASAPI":
            import pythoncom
            pythoncom.CoInitialize()

            # For WASAPI, we want to allow other apps to access the mic, and to auto-convert to float32 if necessary.
            settings = sd.WasapiSettings(exclusive=False, auto_convert=True)

        try:
            with sd.InputStream(
                samplerate=self.device_rate,
                channels=self.device_channels,
                dtype="float32",
                blocksize=self.blocksize,
                callback=self.audio_callback,
                device=self.input_device_info.index,
                extra_settings=settings
            ) as stream:
                self.stream = stream

                if self.input_device_info.resample_needed:
                    import soxr
                    self.resample_stream = soxr.ResampleStream(self.device_rate, WHISPER_SAMPLERATE, num_channels=1, dtype='float32', quality='LQ')

                while not self.stop_event.is_set():
                    chunk = self.receive_audio_chunk()
                    if chunk is None or len(chunk) == 0:
                        continue

                    self.result_queue.put(chunk)

                if self.input_device_info.resample_needed:
                    tail = self.resample_stream.resample_chunk(np.empty((0,), dtype=np.float32), last=True)
                    self.result_queue.put(tail)
                    self.resample_stream = None
        except Exception as e:
            logger.error(f"Audio stream error: {e}")
        finally:
            if self.input_device_info.api == "Windows WASAPI":
                pythoncom.CoUninitialize()


class AudioMixer:
    """
    Mixes two audio sources (mic + VB-CABLE) into a single
    16 kHz mono stream, outputting uniform chunks.
    """
    def __init__(self, mic_queue: queue.Queue, cable_queue: queue.Queue, result_queue: queue.Queue, min_chunk_size: int):
        self.mic_queue = mic_queue
        self.cable_queue = cable_queue
        self.result_queue = result_queue
        self.stop_event = threading.Event()

        # Minimum chunk size in samples (e.g., 8000 = 0.5s @ 16kHz)
        self.min_chunk_size = min_chunk_size

        # Buffers to hold leftover samples until enough are collected
        self.buffer = np.zeros(0, dtype=np.float32)

        self.mixing_thread = None
        self.is_running = False

    def start(self):
        if not self.is_running:
            self.mixing_thread = threading.Thread(target=self.run)
            self.mixing_thread.start()
            self.is_running = True

    def stop(self):
        if self.is_running:
            self.stop_event.set()
            self.mic_queue.put(None)
            self.cable_queue.put(None)
            self.mixing_thread.join()
            self.mixing_thread = None
            self.is_running = False

    def receive_chunk_from_queue(self, q: queue.Queue):
        """Pull one chunk from a queue, or return None if empty."""
        try:
            return q.get(timeout=0.05)
        except queue.Empty:
            return None

    def run(self):
        while not self.stop_event.is_set():
            mic_chunk = self.receive_chunk_from_queue(self.mic_queue)
            cable_chunk = self.receive_chunk_from_queue(self.cable_queue)

            # If nothing arrived, loop again
            if mic_chunk is None and cable_chunk is None:
                continue

            # Replace missing chunks with zeros
            if mic_chunk is None and cable_chunk is not None:
                mic_chunk = np.zeros_like(cable_chunk)
            elif cable_chunk is None and mic_chunk is not None:
                cable_chunk = np.zeros_like(mic_chunk)

            # If still both None, skip
            if mic_chunk is None or cable_chunk is None:
                continue

            # Align lengths
            min_len = min(len(mic_chunk), len(cable_chunk))
            mixed = mic_chunk[:min_len] + cable_chunk[:min_len]

            # Prevent clipping
            mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32)

            # Append to buffer
            self.buffer = np.concatenate([self.buffer, mixed])

            # While enough samples, push out uniform chunks
            while len(self.buffer) >= self.min_chunk_size:
                out_chunk = self.buffer[:self.min_chunk_size]
                self.buffer = self.buffer[self.min_chunk_size:]
                self.result_queue.put(out_chunk)


class WhisperClient:
    def __init__(self, server_url: str, port: int, params: map, audio_queue: queue.Queue, results_queue: queue.Queue = None):
        self.server_url = server_url
        self.port = port
        self.params = params
        self.audio_queue = audio_queue
        self.results_queue = results_queue
        self.connected_event = threading.Event()
        self.results_thread = None
        self.whisper_client_thread = None
        self.is_running = False

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.whisper_client_thread = threading.Thread(target=self.run)
            self.whisper_client_thread.start()

    def stop(self):
        if self.is_running:
            self.is_running = False
            self.audio_queue.put(None)
            self.whisper_client_thread.join()
            self.whisper_client_thread = None

    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.server_url, self.port))
        except Exception as e:
            print(f"Could not connect to the whisper server: {e}")
            return

        self.connected_event.set()

        # Step 1: send params JSON
        ccmn.send_json(sock, self.params)

        # Step 2: start a thread to receive results
        self.results_thread = threading.Thread(target=self.listen_for_results, args=(sock,))
        self.results_thread.start()

        # Step 3: stream audio
        try:
            while True:
                chunk = self.audio_queue.get()
                if chunk is None:
                    # Tell the server we're done
                    ccmn.send_ndarray(sock, np.array([], dtype=np.float32))
                    sock.shutdown(socket.SHUT_WR)
                    self.results_thread.join()  # wait for results thread to finish, should receive shutdown confirmation
                    self.results_thread = None
                    break

                ccmn.send_ndarray(sock, chunk)

        except (BrokenPipeError, ConnectionResetError) as e:
            print(f"Server connection closed while streaming audio: {e}")
        finally:
            sock.close()

    def listen_for_results(self, sock):
        # Receive status and translation messages from the server.
        while True:
            try:
                msg = ccmn.recv_json(sock)
            except (ConnectionResetError, OSError) as e:
                print(f"Connection lost: {e}.")
                break

            if msg is None:
                continue

            if msg["type"] == "status":
                if msg["value"] == "ready":
                    # Here we handle the notification that the server is ready to receive audio.
                    pass
                if msg["value"] == "shutdown":
                    break
            elif msg["type"] == "translation":
                #print(f"[{msg['lang']}]{' (complete)' if msg['complete'] else ''}: {msg['text']}")
                if self.results_queue:
                    self.results_queue.put((msg['text'], msg['complete']))
            else:
                print("Unknown message: ", msg)

    def wait_until_connected(self, timeout: float) -> bool:
        return self.connected_event.wait(timeout=timeout)


class CaptionsReceiver:
    def __init__(self, root_wnd, source_queue, gui_queue):
        self.source_queue = source_queue
        self.gui_queue = gui_queue
        self.captions_thread = None
        self.is_running = False
        self.last_partial = False

        self.overlay = captions_overlay.CaptionsOverlay(root_wnd)

    def start(self):
        if not self.is_running:
            self.captions_thread = threading.Thread(target=self.run)
            self.captions_thread.start()
            self.is_running = True

    def run(self):
        while True:
            text, complete = self.source_queue.get()
            if text is None:
                break
            elif not text.strip():
                continue

            if complete:
                self.gui_queue.put(lambda t=text: self.overlay.overlay_wnd.after(0, lambda t=t: self.send_complete(text=t)))
            else:
                self.gui_queue.put(lambda t=text: self.overlay.overlay_wnd.after(0, lambda t=t: self.send_partial(text=t)))

    def send_complete(self, text):
        if self.overlay:
            if self.last_partial:
                self.overlay.update_last_text(text)
                self.last_partial = False
            else:
                self.overlay.add_text(text)

    def send_partial(self, text):
        if self.overlay:
            if self.last_partial:
                self.overlay.update_last_text(text)
            else:
                self.overlay.add_text(text)
                self.last_partial = True

    def stop(self):
        if self.is_running:
            self.source_queue.put((None, None))  # signal to stop
            self.captions_thread.join()
            self.captions_thread = None
            self.is_running = False
            self.overlay.destroy()
            self.overlay = None


if __name__ == "__main__":
    app = CaptionerUI()
    app.run_gui()
