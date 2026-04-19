import logging
import queue
import threading
import tkinter as tk
from dataclasses import asdict
from tkinter import font, ttk

from app.gui.widgets.graph_widget import GraphWidget
from app.gui.widgets.status_indicator import StatusIndicator

from app.gui.audio import AudioStreamProducer, AudioSwitcher
from app.gui.captions import CaptionsReceiver
from app.gui.audio_utils import (
    WHISPER_SAMPLERATE,
    InputDeviceInfo,
    default_input_device,
    list_unique_input_devices,
    sort_api_by_preference,
)
from app.gui.network import WhisperClient
from app.server.settings import PipelineSettings
from app.common.utils import str_to_int, clamp
from app.common.languages import (
    get_lang_code, get_lang_name, get_lang_name_list,
    get_target_lang_code, get_target_lang_name, get_target_lang_name_list
)

logger = logging.getLogger(__name__)


class Stats:
    def __init__(self):
        self.asr_proc_time_min = float("inf")
        self.asr_proc_time_max = 0.0
        self.transl_proc_time_min = float("inf")
        self.transl_proc_time_max = 0.0

    def update_asr_proc_time(self, proc_time: float):
        if proc_time < self.asr_proc_time_min:
            self.asr_proc_time_min = proc_time
        if proc_time > self.asr_proc_time_max:
            self.asr_proc_time_max = proc_time

    def update_transl_proc_time(self, proc_time: float):
        if proc_time < self.transl_proc_time_min:
            self.transl_proc_time_min = proc_time
        if proc_time > self.transl_proc_time_max:
            self.transl_proc_time_max = proc_time


class CaptionerUI:
    def __init__(self):
        self.pipeline_settings = PipelineSettings()
        self.is_recording = False
        self.is_connected_to_server = False
        self.selected_device_1_info = None
        self.selected_device_2_info = None
        self.audio_producer = None
        self.audio_producer_2 = None
        self.audio_switcher = None
        self.audio_producer_state_map = {}  # device_index -> "open" | "closed"
        self.audio_temp_queue_1 = None
        self.audio_temp_queue_2 = None
        self.net_send_queue = None
        self.net_recv_queue = None
        self.whisper_client = None
        self.captions_overlay = None
        self.stats: Stats = None
        self.setup_ui()

        self.gui_queue = queue.Queue()
        threading.Thread(target=self.update_gui, daemon=True).start()

    def update_gui(self):
        while True:
            try:
                lmbd = self.gui_queue.get()
                if lmbd is not None:
                    self.root_wnd.after(0, lmbd)
            except Exception:
                pass

    def setup_ui(self):
        root = tk.Tk()
        self.root_wnd = root
        self.root_wnd.title("Captioner")
        self.root_wnd.resizable(False, False)  # make window non-resizable
        self.row_index_map = {}  # to keep track of grid row index - wnd/frame -> index

        row_idx = self.next_row()
        captioner_frame = ttk.Frame(root)
        captioner_frame.grid(row=row_idx, column=0, sticky="ew", padx=5, pady=5)

        row_idx = self.next_row(captioner_frame)
        url_frame = ttk.Frame(captioner_frame)
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

        row_idx = self.next_row(captioner_frame)
        settings_frame = ttk.Frame(captioner_frame)
        settings_frame.grid(row=row_idx, column=0, sticky="ew", padx=5, pady=5)

        settings_notebook = ttk.Notebook(settings_frame)
        settings_notebook.pack(fill="both", expand=True)

        # +++ Whisper tab +++
        whisper_tab = ttk.Frame(settings_notebook, padding=10)
        settings_notebook.add(whisper_tab, text="Whisper")

        whisper_notebook = ttk.Notebook(whisper_tab)
        whisper_notebook.grid(row=0, column=0, sticky="ew")
        whisper_tab.columnconfigure(0, weight=1)

        whisper_general_tab = ttk.Frame(whisper_notebook, padding=10)
        whisper_vad_tab = ttk.Frame(whisper_notebook, padding=10)
        whisper_notebook.add(whisper_general_tab, text="General")
        whisper_notebook.add(whisper_vad_tab, text="VAD")
        whisper_general_tab.grid_columnconfigure(1, weight=1)
        whisper_vad_tab.grid_columnconfigure(1, weight=1)

        # === Speech language ===
        row_idx = self.next_row(whisper_general_tab)
        ttk.Label(whisper_general_tab, text="Speech language").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        default_lang = get_lang_name(self.pipeline_settings.asr.language, "English")
        self.lang_var = tk.StringVar(value=default_lang)
        self.lang_combo = ttk.Combobox(whisper_general_tab, textvariable=self.lang_var, values=get_lang_name_list(), state="readonly")
        self.lang_combo.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)

        # === Whisper model ===
        row_idx = self.next_row(whisper_general_tab)
        ttk.Label(whisper_general_tab, text="Model").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.model_var = tk.StringVar(value=self.pipeline_settings.asr.model)
        model_options = [
            "tiny.en", "tiny",
            "base.en", "base",
            "small.en", "distil-small.en", "small",
            "medium.en", "distil-medium.en", "medium",
            "large-v1", "large-v2", "distil-large-v2", "large-v3", "distil-large-v3", "distil-large-v3.5", "large", "large-v3-turbo", "turbo"
        ]
        self.model_combo = ttk.Combobox(whisper_general_tab, textvariable=self.model_var, values=model_options, state="readonly")
        self.model_combo.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)

        # === Whisper device ===
        row_idx = self.next_row(whisper_general_tab)
        ttk.Label(whisper_general_tab, text="Device").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.whisper_device_var = tk.StringVar(value=self.pipeline_settings.asr.device)
        self.whisper_device_combo = ttk.Combobox(whisper_general_tab, textvariable=self.whisper_device_var, values=["cuda", "cpu"], state="readonly")
        self.whisper_device_combo.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)

        # === Whisper compute type ===
        row_idx = self.next_row(whisper_general_tab)
        ttk.Label(whisper_general_tab, text="Compute type").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.whisper_compute_type_var = tk.StringVar(value=self.pipeline_settings.asr.compute_type)
        dtypes = ["int8", "int8_float16", "float16", "float32"]
        self.whisper_compute_type_combo = ttk.Combobox(whisper_general_tab, textvariable=self.whisper_compute_type_var, values=dtypes, state="readonly")
        self.whisper_compute_type_combo.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)

        # === Non-speech probability threshold ===
        row_idx = self.next_row(whisper_general_tab)
        ttk.Label(whisper_general_tab, text="Non-speech threshold", justify="left", anchor="w").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.threshold_var = tk.DoubleVar(value=self.pipeline_settings.asr.nsp_threshold)
        self.threshold_slider = tk.Scale(whisper_general_tab, from_=0.0, to=1.0, orient="horizontal", resolution=0.01, showvalue=False, variable=self.threshold_var,
                                         command=lambda val: self.threshold_label.config(text=f"{float(val):.2f}"))
        self.threshold_slider.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)
        self.threshold_label = ttk.Label(whisper_general_tab, text=f"{self.threshold_var.get()}", width=5, relief="flat", anchor="center")
        self.threshold_label.grid(row=row_idx, column=2, sticky="w", padx=5, pady=5)

        # === Buffer trimming ===
        row_idx = self.next_row(whisper_general_tab)
        ttk.Label(whisper_general_tab, text="Buffer trimming").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.buffer_trimming_var = tk.StringVar(value=self.pipeline_settings.asr.buffer_trimming)
        self.buffer_trimming_combo = ttk.Combobox(
            whisper_general_tab,
            textvariable=self.buffer_trimming_var,
            values=["segment", "sentence"],
            state="readonly"
        )
        self.buffer_trimming_combo.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)

        # === Buffer trimming time ===
        row_idx = self.next_row(whisper_general_tab)
        ttk.Label(whisper_general_tab, text="Buffer trimming time (s)", justify="left", anchor="w").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.buffer_trimming_sec_var = tk.DoubleVar(value=self.pipeline_settings.asr.buffer_trimming_sec)
        self.buffer_trimming_sec_slider = tk.Scale(
            whisper_general_tab,
            from_=5.0,
            to=30.0,
            orient="horizontal",
            resolution=0.5,
            showvalue=False,
            variable=self.buffer_trimming_sec_var,
            command=lambda val: self.buffer_trimming_sec_label.config(text=f"{float(val):.1f}")
        )
        self.buffer_trimming_sec_slider.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)
        self.buffer_trimming_sec_label = ttk.Label(whisper_general_tab, text=f"{self.buffer_trimming_sec_var.get():.1f}", width=5, relief="flat", anchor="center")
        self.buffer_trimming_sec_label.grid(row=row_idx, column=2, sticky="w", padx=5, pady=5)

        # === VAD/VAC ===
        row_idx = self.next_row(whisper_vad_tab)
        self.vac_var = tk.BooleanVar(value=self.pipeline_settings.vac.enable)
        self.vac_check = ttk.Checkbutton(whisper_vad_tab, text="Voice activity controller", variable=self.vac_var)
        self.vac_check.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)

        self.vad_var = tk.BooleanVar(value=self.pipeline_settings.vac.enable_whisper_internal_vad)
        self.vad_check = ttk.Checkbutton(whisper_vad_tab, text="Whisper's internal VAD", variable=self.vad_var)
        self.vad_check.grid(row=row_idx, column=1, sticky="w", padx=5, pady=5)

        row_idx = self.next_row(whisper_vad_tab)
        ttk.Label(whisper_vad_tab, text="Min. chunk size (s)").grid(row=row_idx, column=0, sticky="e", padx=5, pady=5)
        self.vac_min_chunk_size_var = tk.DoubleVar(value=self.pipeline_settings.vac.min_chunk_size_s)
        self.vac_min_chunk_size_slider = tk.Scale(whisper_vad_tab, from_=0.1, to=3.0, orient="horizontal", resolution=0.1, showvalue=False, variable=self.vac_min_chunk_size_var,
                                                  command=lambda val: self.vac_min_chunk_size_label.config(text=f"{float(val):.1f}"))
        self.vac_min_chunk_size_slider.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)
        self.vac_min_chunk_size_label = ttk.Label(whisper_vad_tab, text=f"{self.vac_min_chunk_size_var.get()}", width=5, relief="flat", anchor="center")
        self.vac_min_chunk_size_label.grid(row=row_idx, column=2, sticky="w", padx=5, pady=5)
        self.vac_is_dynamic_chunk_size_var = tk.BooleanVar(value=self.pipeline_settings.vac.is_dynamic_chunk_size)
        self.vac_is_dynamic_chunk_size_check = ttk.Checkbutton(whisper_vad_tab, text="Dynamic", variable=self.vac_is_dynamic_chunk_size_var)
        self.vac_is_dynamic_chunk_size_check.grid(row=row_idx, column=3, sticky="w", padx=5, pady=5)

        row_idx = self.next_row(whisper_vad_tab)
        ttk.Label(whisper_vad_tab, text="Speech start threshold").grid(row=row_idx, column=0, sticky="e", padx=5, pady=5)
        self.vad_start_threshold_var = tk.DoubleVar(value=self.pipeline_settings.vac.start_threshold)
        def start_threshold_slider_cmd(val):
            low_val = self.vad_end_threshold_var.get()
            if float(val) < low_val:
                self.vad_start_threshold_var.set(low_val)
                val = low_val
            self.vad_start_threshold_label.config(text=f"{float(val):.2f}")
        self.vad_start_threshold_slider = tk.Scale(whisper_vad_tab, from_=0.0, to=1.0, orient="horizontal", resolution=0.01, showvalue=False, variable=self.vad_start_threshold_var,
                                                   command=start_threshold_slider_cmd)
        self.vad_start_threshold_slider.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)
        self.vad_start_threshold_label = ttk.Label(whisper_vad_tab, text=f"{self.vad_start_threshold_var.get():.2f}", width=5, relief="flat", anchor="center")
        self.vad_start_threshold_label.grid(row=row_idx, column=2, sticky="w", padx=5, pady=5)

        row_idx = self.next_row(whisper_vad_tab)
        ttk.Label(whisper_vad_tab, text="Speech end threshold").grid(row=row_idx, column=0, sticky="e", padx=5, pady=5)
        self.vad_end_threshold_var = tk.DoubleVar(value=self.pipeline_settings.vac.end_threshold)
        def end_threshold_slider_cmd(val):
            high_val = self.vad_start_threshold_var.get()
            if float(val) > high_val:
                self.vad_end_threshold_var.set(high_val)
                val = high_val
            self.vad_end_threshold_label.config(text=f"{float(val):.2f}")
        self.vad_end_threshold_slider = tk.Scale(whisper_vad_tab, from_=0.0, to=1.0, orient="horizontal", resolution=0.01, showvalue=False, variable=self.vad_end_threshold_var,
                                                 command=end_threshold_slider_cmd)
        self.vad_end_threshold_slider.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)
        self.vad_end_threshold_label = ttk.Label(whisper_vad_tab, text=f"{self.vad_end_threshold_var.get():.2f}", width=5, relief="flat", anchor="center")
        self.vad_end_threshold_label.grid(row=row_idx, column=2, sticky="w", padx=5, pady=5)

        row_idx = self.next_row(whisper_vad_tab)
        ttk.Label(whisper_vad_tab, text="Min. silence duration (s)").grid(row=row_idx, column=0, sticky="e", padx=5, pady=5)
        self.vad_min_silence_duration_var = tk.DoubleVar(value=self.pipeline_settings.vac.min_silence_duration_ms / 1000.0)
        self.vad_min_silence_duration_slider = tk.Scale(whisper_vad_tab, from_=0.1, to=2.0, orient="horizontal", resolution=0.05, showvalue=False, variable=self.vad_min_silence_duration_var,
                                                        command=lambda val: self.vad_min_silence_duration_label.config(text=f"{float(val):.1f}"))
        self.vad_min_silence_duration_slider.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)
        self.vad_min_silence_duration_label = ttk.Label(whisper_vad_tab, text=f"{self.vad_min_silence_duration_var.get()}", width=5, relief="flat", anchor="center")
        self.vad_min_silence_duration_label.grid(row=row_idx, column=2, sticky="w", padx=5, pady=5)

        row_idx = self.next_row(whisper_vad_tab)
        ttk.Label(whisper_vad_tab, text="Speech pad start (s)").grid(row=row_idx, column=0, sticky="e", padx=5, pady=5)
        self.vad_speech_pad_start_var = tk.DoubleVar(value=self.pipeline_settings.vac.speech_pad_start_ms / 1000.0)
        self.vad_speech_pad_start_slider = tk.Scale(whisper_vad_tab, from_=0.1, to=2.0, orient="horizontal", resolution=0.05, showvalue=False, variable=self.vad_speech_pad_start_var,
                                                    command=lambda val: self.vad_speech_pad_start_label.config(text=f"{float(val):.1f}"))
        self.vad_speech_pad_start_slider.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)
        self.vad_speech_pad_start_label = ttk.Label(whisper_vad_tab, text=f"{self.vad_speech_pad_start_var.get()}", width=5, relief="flat", anchor="center")
        self.vad_speech_pad_start_label.grid(row=row_idx, column=2, sticky="w", padx=5, pady=5)

        row_idx = self.next_row(whisper_vad_tab)
        ttk.Label(whisper_vad_tab, text="Speech pad end (s)").grid(row=row_idx, column=0, sticky="e", padx=5, pady=5)
        self.vad_speech_pad_end_var = tk.DoubleVar(value=self.pipeline_settings.vac.speech_pad_end_ms / 1000.0)
        self.vad_speech_pad_end_slider = tk.Scale(whisper_vad_tab, from_=0.1, to=2.0, orient="horizontal", resolution=0.05, showvalue=False, variable=self.vad_speech_pad_end_var,
                                                  command=lambda val: self.vad_speech_pad_end_label.config(text=f"{float(val):.1f}"))
        self.vad_speech_pad_end_slider.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)
        self.vad_speech_pad_end_label = ttk.Label(whisper_vad_tab, text=f"{self.vad_speech_pad_end_var.get()}", width=5, relief="flat", anchor="center")
        self.vad_speech_pad_end_label.grid(row=row_idx, column=2, sticky="w", padx=5, pady=5)

        row_idx = self.next_row(whisper_vad_tab)
        ttk.Label(whisper_vad_tab, text="Hangover chunks").grid(row=row_idx, column=0, sticky="e", padx=5, pady=5)
        self.vad_hangover_chunks_var = tk.IntVar(value=self.pipeline_settings.vac.hangover_chunks)
        self.vad_hangover_chunks_slider = tk.Scale(whisper_vad_tab, from_=0, to=10, orient="horizontal", resolution=1, showvalue=False, variable=self.vad_hangover_chunks_var,
                                                   command=lambda val: self.vad_hangover_chunks_label.config(text=f"{int(float(val))}"))
        self.vad_hangover_chunks_slider.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)
        self.vad_hangover_chunks_label = ttk.Label(whisper_vad_tab, text=f"{self.vad_hangover_chunks_var.get()}", width=5, relief="flat", anchor="center")
        self.vad_hangover_chunks_label.grid(row=row_idx, column=2, sticky="w", padx=5, pady=5)

        # +++ Translation tab +++
        translation_tab = ttk.Frame(settings_notebook, padding=10)
        settings_notebook.add(translation_tab, text="Translation")

        # === Target language ===
        row_idx = self.next_row(translation_tab)
        self.enable_translation_var = tk.BooleanVar(value=self.pipeline_settings.translation.enable)
        self.enable_translation_check = ttk.Checkbutton(translation_tab, text="Enable translation", variable=self.enable_translation_var, command=self.on_enable_translation_toggle)
        self.enable_translation_check.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)

        row_idx = self.next_row(translation_tab)
        ttk.Label(translation_tab, text="Target language").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        default_target_lang = get_target_lang_name(
            self.pipeline_settings.translation.target_language,
            "Serbian Cyrillic")
        self.target_lang_var = tk.StringVar(value=default_target_lang)
        self.target_lang_combo = ttk.Combobox(translation_tab, textvariable=self.target_lang_var, values=get_target_lang_name_list(), state="readonly")
        self.target_lang_combo.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)

        self.transl_engines = ["MarianMT", "NLLB", "EuroLLM", "Whisper", "Google Gemini", "Online Translators"]
        self.transl_engines_with_params = ["EuroLLM", "Google Gemini"]
        self.online_translators = [
            "Google", "MyMemory", "DeepL", "Microsoft", "Libre", "ChatGpt",
            "Baidu", "Papago", "QCRI", "Yandex"
        ]
        self.libre_mirrors = [
            "libretranslate.com",
            "libretranslate.de",
            "libretranslate.org",
            "translate.cutie.dating", 
            "translate.terraprint.co",
            "translate.fedilab.app"]
        self.libre_mirror_key_required = {
            "libretranslate.com": True,
            "libretranslate.de": True,
            "libretranslate.org": False,
            "translate.cutie.dating": False,
            "translate.terraprint.co": False,
            "translate.fedilab.app": False,
        }
        self.online_provider_api_key_required = {"DeepL", "Microsoft", "ChatGpt", "QCRI", "Yandex"}
        self.online_provider_api_secret_required = {"Baidu", "Papago"}
        self.online_provider_client_id_required = {"Baidu", "Papago"}
        self.online_provider_domain_required = {"QCRI"}
        self.online_provider_region_supported = {"Microsoft"}
        row_idx = self.next_row(translation_tab)
        ttk.Label(translation_tab, text="Translation engine").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.transl_engine_var = tk.StringVar(value=self.pipeline_settings.translation.engine)
        self.transl_engine_combo = ttk.Combobox(translation_tab, textvariable=self.transl_engine_var, values=self.transl_engines, state="readonly")
        self.transl_engine_combo.bind("<<ComboboxSelected>>", self.on_transl_engine_selection_change)
        self.transl_engine_combo.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)
        self.online_translator_var = tk.StringVar(value=self.online_translators[0])
        self.online_translator_combo = ttk.Combobox(
            translation_tab,
            textvariable=self.online_translator_var,
            values=self.online_translators,
            state="readonly",
            width=14,
        )
        self.online_translator_combo.bind("<<ComboboxSelected>>", self.on_online_translator_selection_change)
        self.online_translator_combo.grid(row=row_idx, column=2, sticky="ew", padx=5, pady=5)

        row_idx = self.next_row(translation_tab)
        self.transl_word_increment_var = tk.IntVar(value=self.pipeline_settings.translation.word_increment)
        ttk.Label(translation_tab, text="Word increment").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.transl_word_increment_slider = tk.Scale(translation_tab, from_=0, to=15, orient="horizontal", resolution=1, showvalue=False, variable=self.transl_word_increment_var,
                                                     command=lambda val: self.transl_word_increment_label.config(text=f"{int(val)}"))
        self.transl_word_increment_slider.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)
        self.transl_word_increment_label = ttk.Label(translation_tab, text=f"{self.transl_word_increment_var.get()}", width=5, relief="flat", anchor="center")
        self.transl_word_increment_label.grid(row=row_idx, column=2, sticky="w", padx=5, pady=5)

        row_idx = self.next_row(translation_tab)
        ttk.Label(translation_tab, text="Send diff:").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.source_diff_enabled_var = tk.BooleanVar(value=self.pipeline_settings.translation.source_diff_enabled)
        self.source_diff_enabled_check = ttk.Checkbutton(translation_tab, text="source", variable=self.source_diff_enabled_var)
        self.source_diff_enabled_check.grid(row=row_idx, column=1, sticky="w", padx=5, pady=5)
        self.target_diff_enabled_var = tk.BooleanVar(value=self.pipeline_settings.translation.target_diff_enabled)
        self.target_diff_enabled_check = ttk.Checkbutton(translation_tab, text="target", variable=self.target_diff_enabled_var)
        self.target_diff_enabled_check.grid(row=row_idx, column=2, sticky="w", padx=5, pady=5)

        row_idx = self.next_row(translation_tab)
        translation_tab.grid_columnconfigure(2, weight=1)
        self.engine_params_frame = ttk.Frame(translation_tab, padding=10)
        self.engine_params_frame.grid(row=row_idx, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        self.engine_params_frame.grid_columnconfigure(2, weight=1)

        row_idx = self.next_row(self.engine_params_frame)
        self.libre_mirror_label = ttk.Label(self.engine_params_frame, text="Libre mirror")
        self.libre_mirror_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.libre_mirror_var = tk.StringVar(value=self.libre_mirrors[0])
        self.libre_mirror_combo = ttk.Combobox(self.engine_params_frame, textvariable=self.libre_mirror_var, values=self.libre_mirrors, state="readonly")
        self.libre_mirror_combo.grid(row=row_idx, column=1, columnspan=1, sticky="ew", padx=5, pady=5)
        self.libre_mirror_combo.bind("<<ComboboxSelected>>", self.on_online_translator_selection_change)

        row_idx = self.next_row(self.engine_params_frame)
        self.transl_api_key_label = ttk.Label(self.engine_params_frame, text="API Key")
        self.transl_api_key_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.transl_api_key_var = tk.StringVar(value="")
        self.transl_api_key_entry = ttk.Entry(self.engine_params_frame, textvariable=self.transl_api_key_var)
        self.transl_api_key_entry.grid(row=row_idx, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

        row_idx = self.next_row(self.engine_params_frame)
        self.transl_api_secret_label = ttk.Label(self.engine_params_frame, text="API Secret")
        self.transl_api_secret_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.transl_api_secret_var = tk.StringVar(value="")
        self.transl_api_secret_entry = ttk.Entry(self.engine_params_frame, textvariable=self.transl_api_secret_var)
        self.transl_api_secret_entry.grid(row=row_idx, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

        row_idx = self.next_row(self.engine_params_frame)
        self.transl_client_id_label = ttk.Label(self.engine_params_frame, text="Client/App ID")
        self.transl_client_id_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.transl_client_id_var = tk.StringVar(value="")
        self.transl_client_id_entry = ttk.Entry(self.engine_params_frame, textvariable=self.transl_client_id_var)
        self.transl_client_id_entry.grid(row=row_idx, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

        row_idx = self.next_row(self.engine_params_frame)
        self.transl_region_label = ttk.Label(self.engine_params_frame, text="Region")
        self.transl_region_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.transl_region_var = tk.StringVar(value="")
        self.transl_region_entry = ttk.Entry(self.engine_params_frame, textvariable=self.transl_region_var)
        self.transl_region_entry.grid(row=row_idx, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

        row_idx = self.next_row(self.engine_params_frame)
        self.transl_domain_label = ttk.Label(self.engine_params_frame, text="Domain")
        self.transl_domain_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.transl_domain_var = tk.StringVar(value="general")
        self.transl_domain_entry = ttk.Entry(self.engine_params_frame, textvariable=self.transl_domain_var)
        self.transl_domain_entry.grid(row=row_idx, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

        self.transl_engine_combo.event_generate("<<ComboboxSelected>>")

        # +++ Audio tab +++
        audio_tab = ttk.Frame(settings_notebook, padding=10)
        settings_notebook.add(audio_tab, text="Audio")

        devices_notebook = ttk.Notebook(audio_tab)
        row_idx = self.next_row(audio_tab)
        devices_notebook.grid(row=row_idx, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

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

        # === Common audio settings ===
        row_idx = self.next_row(audio_tab)
        ttk.Label(audio_tab, text="Audio chunk size (s)").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.audio_chunk_size_var = tk.DoubleVar(value=0.4)
        self.audio_chunk_size_slider = tk.Scale(audio_tab, from_=0.1, to=3.0, orient="horizontal", resolution=0.1, showvalue=False, variable=self.audio_chunk_size_var,
                                                  command=lambda val: self.audio_chunk_size_label.config(text=f"{float(val):.1f}"))
        self.audio_chunk_size_slider.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)
        self.audio_chunk_size_label = ttk.Label(audio_tab, text=f"{self.audio_chunk_size_var.get()}", width=5, relief="flat", anchor="center")
        self.audio_chunk_size_label.grid(row=row_idx, column=2, sticky="w", padx=5, pady=5)

        # +++ Captions tab +++
        captions_tab = ttk.Frame(settings_notebook, padding=10)
        settings_notebook.add(captions_tab, text="Captions overlay")
        row_idx = self.next_row(captions_tab)
        self.show_captions_overlay_btn = ttk.Button(captions_tab, text="Show overlay", state="disabled", command=self.toggle_show_captions_overlay)
        self.show_captions_overlay_btn.grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        row_idx = self.next_row(captions_tab)
        ttk.Label(captions_tab, text="Font size").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.captions_font_size_var = tk.IntVar(value=24)
        self.captions_font_size_slider = tk.Scale(captions_tab, from_=12, to=72, orient="horizontal", resolution=1, showvalue=False, variable=self.captions_font_size_var,
                                                  command=lambda val: self.captions_font_size_label.config(text=f"{int(val)}"))
        self.captions_font_size_slider.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)
        self.captions_font_size_label = ttk.Label(captions_tab, text=f"{self.captions_font_size_var.get()}", width=5, relief="flat", anchor="center")
        self.captions_font_size_label.grid(row=row_idx, column=2, sticky="w", padx=5, pady=5)
        row_idx = self.next_row(captions_tab)
        ttk.Label(captions_tab, text="Max visible lines").grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        self.captions_max_visible_lines_var = tk.IntVar(value=4)
        self.captions_max_visible_lines_slider = tk.Scale(captions_tab, from_=2, to=8, orient="horizontal", resolution=1, showvalue=False, variable=self.captions_max_visible_lines_var,
                                                  command=lambda val: self.captions_max_visible_lines_label.config(text=f"{int(val)}"))
        self.captions_max_visible_lines_slider.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)
        self.captions_max_visible_lines_label = ttk.Label(captions_tab, text=f"{self.captions_max_visible_lines_var.get()}", width=5, relief="flat", anchor="center")
        self.captions_max_visible_lines_label.grid(row=row_idx, column=2, sticky="w", padx=5, pady=5)

        # --- Stats ---
        separator = ttk.Separator(root, orient="vertical")
        separator.grid(row=0, column=1, sticky="ns", padx=0, pady=10)

        stats_frame = ttk.Frame(root)
        stats_frame.grid(row=0, column=2, rowspan=2, sticky="news", padx=5, pady=10)

        row_idx = self.next_row(stats_frame)
        net_server_label_frame = ttk.Frame(stats_frame)
        net_server_label_frame.grid(row=row_idx, column=0, sticky="ew", padx=5, pady=5)
        self.net_server_status_indicator = StatusIndicator(
            net_server_label_frame,
            states=[
                ("disconnected", "gray"),
                ("connected", "green")
            ],
            size=16
        )
        self.net_server_status_indicator.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        net_server_label = ttk.Label(net_server_label_frame, text="Server status:")
        net_server_label.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.net_server_status_label = ttk.Label(stats_frame, text="disconnected", anchor="w", justify="left")
        self.net_server_status_label.grid(row=row_idx, column=1, sticky="w", padx=5, pady=5)

        row_idx = self.next_row(stats_frame)
        net_server_asr_label_frame = ttk.Frame(stats_frame)
        net_server_asr_label_frame.grid(row=row_idx, column=0, sticky="ew", padx=5, pady=5)
        self.net_server_asr_status_indicator = StatusIndicator(
            net_server_asr_label_frame,
            states=[
                ("uninitialized", "gray"),
                ("initializing", "orange", 0.5),
                ("ready", "green")
            ],
            size=16
        )
        self.net_server_asr_status_indicator.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        net_server_asr_status_label = ttk.Label(net_server_asr_label_frame, text="ASR:")
        net_server_asr_status_label.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.net_server_asr_in_queue_progress = ttk.Progressbar(stats_frame, orient="horizontal", mode="determinate", maximum=10, value=0)
        self.net_server_asr_in_queue_progress.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)
        self.net_server_asr_in_queue_label = ttk.Label(stats_frame, text="chunks queued: --", width=20)
        self.net_server_asr_in_queue_label.grid(row=row_idx, column=2, sticky="w", padx=5, pady=5)

        row_idx = self.next_row(stats_frame)
        ttk.Label(stats_frame, text="Voice activity:").grid(row=row_idx, column=0, sticky="e", padx=5, pady=5)
        self.net_server_asr_vac_indicator = StatusIndicator(
            stats_frame,
            states=[
                ("nonvoice", "gray"),
                ("voice", "green")
            ],
            size=16
        )
        self.net_server_asr_vac_indicator.grid(row=row_idx, column=1, sticky="w", padx=5, pady=5)

        row_idx = self.next_row(stats_frame)
        net_server_asr_proc_t_label = ttk.Label(stats_frame, text="Inference time:")
        net_server_asr_proc_t_label.grid(row=row_idx, column=0, sticky="e", padx=5, pady=5)
        self.net_server_asr_proc_t_label = ttk.Label(stats_frame, text="last: --")
        self.net_server_asr_proc_t_label.grid(row=row_idx, column=1, sticky="w", padx=5, pady=5)
        self.net_server_asr_proc_t_graph = GraphWidget(stats_frame, width=100, height=50, line_color="blue", border_color="black", bg_color="lightgray", max_points=30)
        self.net_server_asr_proc_t_graph.grid(row=row_idx, column=2, rowspan=4, sticky="wn", padx=5, pady=5)
        row_idx = self.next_row(stats_frame)
        self.net_server_asr_proc_t_min_label = ttk.Label(stats_frame, text="min: --")
        self.net_server_asr_proc_t_min_label.grid(row=row_idx, column=1, sticky="w", padx=5, pady=5)
        row_idx = self.next_row(stats_frame)
        self.net_server_asr_proc_t_max_label = ttk.Label(stats_frame, text="max: --")
        self.net_server_asr_proc_t_max_label.grid(row=row_idx, column=1, sticky="w", padx=5, pady=5)
        row_idx = self.next_row(stats_frame)
        self.net_server_asr_proc_t_roll_avg_label = ttk.Label(stats_frame, text="roll avg: --")
        self.net_server_asr_proc_t_roll_avg_label.grid(row=row_idx, column=1, sticky="w", padx=5, pady=5)

        row_idx = self.next_row(stats_frame)
        net_server_transl_label_frame = ttk.Frame(stats_frame)
        net_server_transl_label_frame.grid(row=row_idx, column=0, sticky="ew", padx=5, pady=5)
        self.net_server_transl_status_indicator = StatusIndicator(
            net_server_transl_label_frame,
            states=[
                ("uninitialized", "gray"),
                ("initializing", "orange", 0.5),
                ("ready", "green")
            ],
            size=16
        )
        self.net_server_transl_status_indicator.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        net_server_transl_status_label = ttk.Label(net_server_transl_label_frame, text="Translation:")
        net_server_transl_status_label.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.net_server_transl_queue_progress = ttk.Progressbar(stats_frame, orient="horizontal", mode="determinate", maximum=100, value=0)
        self.net_server_transl_queue_progress.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=5)
        self.net_server_transl_queue_label = ttk.Label(stats_frame, text="tokens buffered: --", width=20)
        self.net_server_transl_queue_label.grid(row=row_idx, column=2, sticky="w", padx=5, pady=5)
        row_idx = self.next_row(stats_frame)
        net_server_transl_proc_t_label = ttk.Label(stats_frame, text="Inference time:")
        net_server_transl_proc_t_label.grid(row=row_idx, column=0, sticky="e", padx=5, pady=5)
        self.net_server_transl_proc_t_label = ttk.Label(stats_frame, text="last: --")
        self.net_server_transl_proc_t_label.grid(row=row_idx, column=1, sticky="w", padx=5, pady=5)
        self.net_server_transl_proc_t_graph = GraphWidget(stats_frame, width=100, height=50, line_color="blue", border_color="black", bg_color="lightgray", max_points=30)
        self.net_server_transl_proc_t_graph.grid(row=row_idx, column=2, rowspan=3, sticky="wn", padx=5, pady=5)
        row_idx = self.next_row(stats_frame)
        self.net_server_transl_proc_t_min_label = ttk.Label(stats_frame, text="min: --")
        self.net_server_transl_proc_t_min_label.grid(row=row_idx, column=1, sticky="w", padx=5, pady=5)
        row_idx = self.next_row(stats_frame)
        self.net_server_transl_proc_t_max_label = ttk.Label(stats_frame, text="max: --")
        self.net_server_transl_proc_t_max_label.grid(row=row_idx, column=1, sticky="w", padx=5, pady=5)

        # --- Buttons ---
        row_idx = self.next_row()
        buttons_frame = ttk.Frame(root)
        buttons_frame.grid(row=row_idx, column=0, columnspan=3, sticky="e", padx=5, pady=5)

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
            self.on_disconnect_from_server()
        else:
            self.on_connect_to_server()

    def toggle_recording(self):
        if self.is_recording:
            self.on_stop_recording()
        else:
            self.on_start_recording()

    def toggle_mute(self):
        if self.audio_producer:
            if self.audio_producer.is_paused():
                self.audio_producer.resume_stream()
                self.mute_btn.config(text="🔊")
            else:
                self.audio_producer.pause_stream()
                self.mute_btn.config(text="🔇")

    def toggle_mute_2(self):
        if self.audio_producer_2:
            if self.audio_producer_2.is_paused():
                self.audio_producer_2.resume_stream()
                self.mute_btn_2.config(text="🔊")
            else:
                self.audio_producer_2.pause_stream()
                self.mute_btn_2.config(text="🔇")

    def quit(self):
        self.stop_captions_overlay()
        self.stop_audio_producers()
        self.stop_whisper_client()
        self.root_wnd.quit()
        self.root_wnd.destroy()

    def on_connect_to_server(self):
        if not self.is_connected_to_server:
            self.connect_btn.config(state="disabled", text="Connecting...")
            self.connect_btn.update_idletasks()
            if not self.connect_to_server():
                self.connect_btn.config(state="normal", text="Connect")

    def on_disconnect_from_server(self):
        if self.is_connected_to_server:
            self.on_stop_recording()
            self.connect_btn.config(state="disabled")
            self.stop_whisper_client()
            self.connect_btn.config(state="normal", text="Connect")
            self.record_btn.config(state="disabled", text="Record")
            self.is_connected_to_server = False

    def on_start_recording(self):
        if self.is_recording or not self.is_connected_to_server:
            return

        if self.use_second_audio_dev_var.get() and (self.audio_device_combo_1.get() == self.audio_device_combo_2.get()):
            self.show_modal_message("Error", "When the second audio device is enabled, different audio devices must be selected.", self.root_wnd)
            return

        info_str = (
            f"Running AudioStreamProducer(s) and CaptionsReceiver...\n"
            f"\t  Audio device: {self.audio_device_combo_1.get()}\n"
            f"\t  Audio device 2: {self.audio_device_combo_2.get() if self.use_second_audio_dev_var.get() else 'N/A'}\n"
        )
        logger.info(info_str)

        # Succesfull start of AudioStreamProducer will start up the CaptionsReceiver as well, if necessary.
        self.create_audio_producer()

    def on_stop_recording(self):
        if self.is_recording:
            self.stop_captions_overlay()
            self.stop_audio_producers()
            logger.info("Recording stopped.")

    def on_enable_translation_toggle(self):
        if self.enable_translation_var.get():
            self.target_lang_combo.config(state="readonly")
            self.transl_engine_combo.config(state="readonly")
            self.target_diff_enabled_check.config(state="normal")
            self.on_transl_engine_selection_change(None)
        else:
            self.target_lang_combo.config(state="disabled")
            self.transl_engine_combo.config(state="disabled")
            self.online_translator_combo.config(state="disabled")
            self.target_diff_enabled_check.config(state="disabled")
            for w in self.engine_params_frame.winfo_children():
                w.config(state="disabled")

    def _set_param_row_visible(self, label_widget, input_widget, visible: bool):
        if visible:
            label_widget.grid()
            input_widget.grid()
        else:
            label_widget.grid_remove()
            input_widget.grid_remove()

    def _set_param_row_state(self, input_widget, state: str):
        if isinstance(input_widget, ttk.Combobox):
            input_widget.config(state="readonly" if state == "normal" else "disabled")
        else:
            input_widget.config(state=state)

    def on_transl_engine_selection_change(self, event):
        transl_model = self.transl_engine_var.get()
        uses_online_translator = transl_model == "Online Translators"
        provider = self.online_translator_var.get()

        if uses_online_translator:
            self.online_translator_combo.grid()
            if self.enable_translation_var.get():
                self.online_translator_combo.config(state="readonly")
        else:
            self.online_translator_combo.grid_remove()

        needs_api_key = transl_model in self.transl_engines_with_params
        if uses_online_translator:
            needs_api_key = (
                provider in self.online_provider_api_key_required
                or (provider == "Libre" and self.libre_mirror_key_required.get(self.libre_mirror_var.get(), False))
            )

        needs_api_secret = uses_online_translator and provider in self.online_provider_api_secret_required
        needs_client_id = uses_online_translator and provider in self.online_provider_client_id_required
        needs_domain = uses_online_translator and provider in self.online_provider_domain_required
        needs_region = uses_online_translator and provider in self.online_provider_region_supported
        needs_libre_mirror = uses_online_translator and provider == "Libre"

        self._set_param_row_visible(self.transl_api_key_label, self.transl_api_key_entry, needs_api_key)
        self._set_param_row_visible(self.transl_api_secret_label, self.transl_api_secret_entry, needs_api_secret)
        self._set_param_row_visible(self.transl_client_id_label, self.transl_client_id_entry, needs_client_id)
        self._set_param_row_visible(self.transl_domain_label, self.transl_domain_entry, needs_domain)
        self._set_param_row_visible(self.transl_region_label, self.transl_region_entry, needs_region)
        self._set_param_row_visible(self.libre_mirror_label, self.libre_mirror_combo, needs_libre_mirror)

        has_any_params = any([
            needs_api_key,
            needs_api_secret,
            needs_client_id,
            needs_domain,
            needs_region,
            needs_libre_mirror,
        ])

        if has_any_params:
            self.engine_params_frame.grid()
            api_state = "normal" if self.enable_translation_var.get() else "disabled"
            self._set_param_row_state(self.transl_api_key_entry, api_state)
            self._set_param_row_state(self.transl_api_secret_entry, api_state)
            self._set_param_row_state(self.transl_client_id_entry, api_state)
            self._set_param_row_state(self.transl_domain_entry, api_state)
            self._set_param_row_state(self.transl_region_entry, api_state)
            self._set_param_row_state(self.libre_mirror_combo, api_state)
        else:
            self.engine_params_frame.grid_remove()

    def on_online_translator_selection_change(self, event):
        self.on_transl_engine_selection_change(None)

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
            f"Index: {dev_info.index}  |  Type: {'loopback' if dev_info.is_loopback else 'input'}\n"
            f"Channels: {dev_info.channels}"
            + (
                f" (downmix -> {target_ch}ch)"
                if dev_info.downmix_needed else ""
            )
            + "\n"
            f"Samplerate: {int(dev_info.samplerate)} Hz"
            + (
                f" (resample -> {target_sr / 1000:.0f} kHz)" 
                if dev_info.resample_needed else ""
            )
            + "\n"
            f"Latency range: {dev_info.min_latency * 1000:.0f} - {dev_info.max_latency * 1000:.0f} ms"
        )

        return info_text

    def on_audio_device_host_api_1_selection_change(self, event):
        dev_name = self.audio_device_combo_1.get()
        host_api = self.audio_device_host_api_combo_1.get()
        caps, is_loopback = self.device_map[dev_name][host_api]
        dev_info = InputDeviceInfo(dev_name, host_api, caps, is_loopback)
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
        caps, is_loopback = self.device_map[dev_name][host_api]
        dev_info = InputDeviceInfo(dev_name, host_api, caps, is_loopback)
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

    def toggle_show_captions_overlay(self):
        if not self.net_send_queue:
            return

        if self.captions_overlay is None:
            self.start_captions_overlay()
            self.net_send_queue.put({
                "type": "control",
                "command": "start_sending_client_transcript"
            })
            self.show_captions_overlay_btn.config(text="Hide overlay")
        else:
            self.stop_captions_overlay()
            self.net_send_queue.put({
                "type": "control",
                "command": "stop_sending_client_transcript"
            })
            self.show_captions_overlay_btn.config(text="Show overlay")

    def update_selected_devices_label(self):
        if not self.is_recording:
            dev1_name = self.audio_device_combo_1.get()
            dev2_name = self.audio_device_combo_2.get() if self.use_second_audio_dev_var.get() else "<none>"

            info_text = (
                f"Dev1: {dev1_name}\n\n"
                f"Dev2: {dev2_name}"
            )
            self.dev_label.config(text=info_text)

    def clear_net_server_stats(self):
        self.net_server_status_label.config(text="disconnected")
        self.net_server_asr_in_queue_progress.config(value=0)
        self.net_server_asr_in_queue_label.config(text="chunks queued: --")
        self.net_server_asr_vac_indicator.set_state("nonvoice")
        self.net_server_asr_proc_t_label.config(text="last: --")
        self.net_server_asr_proc_t_min_label.config(text="min: --")
        self.net_server_asr_proc_t_max_label.config(text="max: --")
        self.net_server_asr_proc_t_roll_avg_label.config(text="roll avg: --")
        self.net_server_asr_proc_t_graph.clear()
        self.net_server_transl_queue_progress.config(value=0)
        self.net_server_transl_queue_label.config(text="tokens buffered: --")
        self.net_server_transl_proc_t_label.config(text="last: --")
        self.net_server_transl_proc_t_min_label.config(text="min: --")
        self.net_server_transl_proc_t_max_label.config(text="max: --")
        self.net_server_transl_proc_t_graph.clear()
        self.stats = None

    # Helper to manage grid row indices
    def next_row(self, widget=None):
        if not widget:
            widget = self.root_wnd
        if widget not in self.row_index_map:
            self.row_index_map[widget] = 0

        i = self.row_index_map[widget]
        self.row_index_map[widget] += 1
        return i

    def connect_to_server(self) -> bool:
        if self.is_connected_to_server:
            return False

        threshold = self.threshold_var.get()
        buffer_trimming = self.buffer_trimming_var.get()
        buffer_trimming_sec = self.buffer_trimming_sec_var.get()
        vac_min_chunk_size = self.vac_min_chunk_size_var.get()
        vac_is_dynamic_chunk_size = self.vac_is_dynamic_chunk_size_var.get()
        vad_start_threshold = self.vad_start_threshold_var.get()
        vad_end_threshold = self.vad_end_threshold_var.get()
        vad_min_silence_duration = self.vad_min_silence_duration_var.get()
        vad_speech_pad_start = self.vad_speech_pad_start_var.get()
        vad_speech_pad_end = self.vad_speech_pad_end_var.get()
        vad_hangover_chunks = int(self.vad_hangover_chunks_var.get())

        server_url = self.server_url_var.get().strip()
        if not server_url:
            return False

        port, valid = str_to_int(self.server_port_var.get())
        if not valid or not (0 < port < 65536):
            logger.error(f"Invalid port number: {port}.")
            return False

        transl_engine = self.transl_engine_var.get()
        transl_params = {}

        if transl_engine in self.transl_engines_with_params:
            transl_params["api_key"] = self.transl_api_key_var.get().strip()

        provider = ""
        if transl_engine == "Online Translators":
            provider = self.online_translator_var.get()
            transl_params["provider"] = provider
            transl_params["api_key"] = self.transl_api_key_var.get().strip()
            transl_params["api_secret"] = self.transl_api_secret_var.get().strip()
            transl_params["client_id"] = self.transl_client_id_var.get().strip()
            transl_params["domain"] = self.transl_domain_var.get().strip()
            transl_params["region"] = self.transl_region_var.get().strip()
            transl_params["libre_mirror"] = self.libre_mirror_var.get().strip() or self.libre_mirrors[0]

        if self.enable_translation_var.get():
            needs_key = transl_engine in self.transl_engines_with_params
            needs_secret = False
            needs_client_id = False
            needs_domain = False

            if transl_engine == "Online Translators":
                provider = self.online_translator_var.get()
                needs_key = (
                    provider in self.online_provider_api_key_required
                    or (provider == "Libre" and self.libre_mirror_key_required.get(self.libre_mirror_var.get(), False))
                )
                needs_secret = provider in self.online_provider_api_secret_required
                needs_client_id = provider in self.online_provider_client_id_required
                needs_domain = provider in self.online_provider_domain_required

            if needs_key and not transl_params.get("api_key"):
                logger.error("Translation backend requires API key, but the API key field is empty.")
                return False
            if needs_secret and not transl_params.get("api_secret"):
                logger.error("Translation backend requires API secret, but the API secret field is empty.")
                return False
            if needs_client_id and not transl_params.get("client_id"):
                logger.error("Translation backend requires Client/App ID, but the field is empty.")
                return False
            if needs_domain and not transl_params.get("domain"):
                logger.error("QCRI translator requires a domain, but the Domain field is empty.")
                return False

        pl_set = PipelineSettings()
        pl_set.zoom_url = self.zoom_url_var.get().strip()

        pl_set.asr.model = self.model_var.get()
        pl_set.asr.device = self.whisper_device_var.get()
        pl_set.asr.compute_type = self.whisper_compute_type_var.get()
        pl_set.asr.language = get_lang_code(self.lang_var.get(), "auto")
        pl_set.asr.nsp_threshold = threshold
        pl_set.asr.buffer_trimming = buffer_trimming
        pl_set.asr.buffer_trimming_sec = buffer_trimming_sec

        pl_set.vac.enable = bool(self.vac_var.get())
        pl_set.vac.enable_whisper_internal_vad = bool(self.vad_var.get())
        pl_set.vac.min_chunk_size_s = vac_min_chunk_size
        pl_set.vac.is_dynamic_chunk_size = vac_is_dynamic_chunk_size
        pl_set.vac.start_threshold = vad_start_threshold
        pl_set.vac.end_threshold = vad_end_threshold
        pl_set.vac.min_silence_duration_ms = int(vad_min_silence_duration * 1000)
        pl_set.vac.speech_pad_start_ms = int(vad_speech_pad_start * 1000)
        pl_set.vac.speech_pad_end_ms = int(vad_speech_pad_end * 1000)
        pl_set.vac.hangover_chunks = vad_hangover_chunks

        pl_set.translation.enable = bool(self.enable_translation_var.get())
        pl_set.translation.src_language = pl_set.asr.language
        pl_set.translation.target_language = get_target_lang_code(self.target_lang_var.get())
        pl_set.translation.engine = transl_engine
        pl_set.translation.engine_params = transl_params
        pl_set.translation.word_increment = self.transl_word_increment_var.get()
        pl_set.translation.source_diff_enabled = bool(self.source_diff_enabled_var.get())
        pl_set.translation.target_diff_enabled = bool(self.target_diff_enabled_var.get())

        # Set ASR task based on whether translation is enabled with Whisper and target language is English,
        # in which case we can use Whisper's built-in translation capability.
        if (
            pl_set.translation.enable 
            and pl_set.translation.engine == "Whisper"
            and pl_set.translation.target_language == "English"
        ):
            pl_set.asr.task = "translate"
        else:
            pl_set.asr.task = "transcribe"

        self.pipeline_settings = pl_set

        logger.info(f"Connecting to server {server_url}:{port}")

        self.net_send_queue = queue.Queue()
        self.net_recv_queue = queue.Queue()
        self.whisper_client = WhisperClient(
            server_url,
            port,
            asdict(pl_set),
            self.net_send_queue,
            self.net_recv_queue,
            self.whisper_client_callback
        )
        self.whisper_client.start()

        return True

    def stop_whisper_client(self):
        if self.whisper_client:
            logger.info("Stopping whisper client...")
            self.whisper_client.stop()
            self.whisper_client = None
            self.net_send_queue = None
            self.net_recv_queue = None

    def start_captions_overlay(self):
        if not self.captions_overlay:
            self.captions_overlay = CaptionsReceiver(
                self.root_wnd, 
                self.captions_font_size_var.get(), 
                self.captions_max_visible_lines_var.get(),
                self.net_recv_queue, 
                self.gui_queue
            )
            self.captions_overlay.start()

    def stop_captions_overlay(self):
        if self.captions_overlay:
            logger.info("Stopping captions overlay...")
            self.captions_overlay.stop()
            self.captions_overlay = None

    def create_audio_producer(self):
        audio_chunk_size = self.audio_chunk_size_var.get()
        use_second_dev = self.use_second_audio_dev_var.get()
        self.audio_producer_state_map.clear()

        if use_second_dev:
            self.audio_producer_state_map[self.selected_device_1_info.index] = "closed"
            self.audio_producer_state_map[self.selected_device_2_info.index] = "closed"

            self.audio_temp_queue_1 = queue.Queue()
            self.audio_producer = AudioStreamProducer(
                audio_chunk_size,
                self.selected_device_1_info,
                self.audio_temp_queue_1,
                self.audio_producer_callback
            )
            self.audio_producer.start()

            self.audio_temp_queue_2 = queue.Queue()
            self.audio_producer_2 = AudioStreamProducer(
                audio_chunk_size,
                self.selected_device_2_info,
                self.audio_temp_queue_2,
                self.audio_producer_callback
            )
            self.audio_producer_2.start()

            self.audio_switcher = AudioSwitcher(
                self.audio_temp_queue_1,
                self.audio_temp_queue_2,
                self.net_send_queue,
                int(self.audio_producer_2.min_chunk_size * WHISPER_SAMPLERATE)
            )
            self.audio_switcher.start()
        else:
            self.audio_producer_state_map[self.selected_device_1_info.index] = "closed"

            # We use one device, puts the results directly to the network send queue.
            self.audio_producer = AudioStreamProducer(
                audio_chunk_size,
                self.selected_device_1_info,
                self.net_send_queue,
                self.audio_producer_callback
            )
            self.audio_producer.start()

    def stop_audio_producers(self):
        if self.audio_producer:
            logger.info("Stopping audio producer...")
            self.audio_producer.stop()
            self.audio_producer = None

        if self.audio_producer_2:
            logger.info("Stopping audio producer 2...")
            self.audio_producer_2.stop()
            self.audio_producer_2 = None

        if self.audio_switcher:
            logger.info("Stopping audio switcher thread...")
            self.audio_switcher.stop()
            self.audio_switcher = None

        self.audio_temp_queue_1 = None
        self.audio_temp_queue_2 = None

    def whisper_client_callback(self, event_type: str, data: dict):

        def on_disconnect_from_server():
            self.on_disconnect_from_server()
            self.net_server_status_label.config(text="disconnected")
            self.net_server_status_indicator.set_state("disconnected")
            self.net_server_asr_status_indicator.set_state("uninitialized")
            self.net_server_transl_status_indicator.set_state("uninitialized")
            self.clear_net_server_stats()

        match event_type:
            case "server_statistics":
                for stat_name, stat_value in data.items():
                    match stat_name:
                        case "asr_in_q_size":
                            def upd_asr_q_size(stat_value=stat_value):
                                self.net_server_asr_in_queue_progress.config(value=min(stat_value, 10))
                                self.net_server_asr_in_queue_label.config(text=f"chunks queued: {stat_value}")
                            self.gui_queue.put(upd_asr_q_size)
                        case "transl_buffer_token_count":
                            def upd_transl_q_size(stat_value=stat_value):
                                self.net_server_transl_queue_progress.config(value=min(stat_value, 100))
                                self.net_server_transl_queue_label.config(text=f"tokens buffered: {stat_value}")
                            self.gui_queue.put(upd_transl_q_size)
                        case "last_asr_proc_time":
                            def upd_asr_proc_time(stat_value=stat_value):
                                self.net_server_asr_proc_t_label.config(text=f"last: {stat_value:.3f} s")
                                self.stats.update_asr_proc_time(stat_value)
                                self.net_server_asr_proc_t_min_label.config(text=f"min: {self.stats.asr_proc_time_min:.3f} s")
                                self.net_server_asr_proc_t_max_label.config(text=f"max: {self.stats.asr_proc_time_max:.3f} s")
                                self.net_server_asr_proc_t_graph.add_value(stat_value)
                            self.gui_queue.put(upd_asr_proc_time)
                        case "asr_roll_avg_proc_time":
                            def upd_asr_roll_avg_proc_time(stat_value=stat_value):
                                self.net_server_asr_proc_t_roll_avg_label.config(text=f"roll avg: {stat_value:.3f} s")
                            self.gui_queue.put(upd_asr_roll_avg_proc_time)

                            new_chunk_size = clamp(stat_value, 0.1, 3.0)
                            if self.audio_producer:
                                self.audio_producer.min_chunk_size = new_chunk_size
                            if self.audio_producer_2:
                                self.audio_producer_2.min_chunk_size = new_chunk_size
                        case "last_transl_proc_time":
                            def upd_transl_proc_time(stat_value=stat_value):
                                self.net_server_transl_proc_t_label.config(text=f"last: {stat_value:.3f} s")
                                self.stats.update_transl_proc_time(stat_value)
                                self.net_server_transl_proc_t_min_label.config(text=f"min: {self.stats.transl_proc_time_min:.3f} s")
                                self.net_server_transl_proc_t_max_label.config(text=f"max: {self.stats.transl_proc_time_max:.3f} s")
                                self.net_server_transl_proc_t_graph.add_value(stat_value)
                            self.gui_queue.put(upd_transl_proc_time)
                        case "vac_voice_status":
                            def upd_vac_state(stat_value=stat_value):
                                if stat_value is None:
                                    stat_value = "nonvoice"
                                self.net_server_asr_vac_indicator.set_state(stat_value)
                            self.gui_queue.put(upd_vac_state)

            case "client_status":
                match data.get("status", ""):
                    case "connecting":
                        def on_connecting():
                            self.net_server_status_label.config(text="connecting")
                        self.gui_queue.put(on_connecting)
                    case "connected":
                        def on_connected():
                            self.is_connected_to_server = True
                            self.connect_btn.config(text="Disconnect", state="normal")
                            self.stats = Stats()
                            self.net_server_status_label.config(text="connected")
                            self.net_server_status_indicator.set_state("connected")
                        self.gui_queue.put(on_connected)
                    case "conn_lost" | "params_send_error":
                        self.gui_queue.put(on_disconnect_from_server)
                    case "conn_error":
                        def on_conn_error():
                            self.stop_whisper_client()
                            self.connect_btn.config(text="Connect", state="normal")
                            self.record_btn.config(text="Record", state="disabled")
                            self.is_connected_to_server = False
                            self.net_server_status_label.config(text="disconnected")
                            self.net_server_status_indicator.set_state("disconnected")
                            self.net_server_asr_status_indicator.set_state("uninitialized")
                            self.net_server_transl_status_indicator.set_state("uninitialized")
                            self.clear_net_server_stats()
                        self.gui_queue.put(on_conn_error)
                    case "disconnecting":
                        def on_disconnecting():
                            self.net_server_status_label.config(text="disconnecting")
                        self.gui_queue.put(on_disconnecting)

            case "server_status":
                match data.get("status", ""):
                    case "ready":
                        def on_ready():
                            self.record_btn.config(text="Record", state="normal")
                            self.net_server_status_label.config(text="ready")
                        self.gui_queue.put(on_ready)
                    case "translator_initializing":
                        def on_transl_init():
                            self.net_server_transl_status_indicator.set_state("initializing")
                        self.gui_queue.put(on_transl_init)
                    case "translator_initialized":
                        def on_transl_initialized():
                            self.net_server_transl_status_indicator.set_state("ready")
                        self.gui_queue.put(on_transl_initialized)
                    case "asr_initializing":
                        def on_asr_init():
                            self.net_server_asr_status_indicator.set_state("initializing")
                        self.gui_queue.put(on_asr_init)
                    case "asr_initialized":
                        def on_asr_initialized():
                            self.net_server_asr_status_indicator.set_state("ready")
                        self.gui_queue.put(on_asr_initialized)
                    case "conn_shutdown":
                        self.gui_queue.put(on_disconnect_from_server)

    def audio_producer_callback(self, event_type: str, data: dict):
        match event_type:
            case "stream_open":
                def on_stream_open():
                    self.record_btn.config(text="Stop")
                    self.mute_btn.config(state="normal")
                    if self.use_second_audio_dev_var.get():
                        self.mute_btn_2.config(state="normal")
                    self.show_captions_overlay_btn.config(state="normal")
                    self.is_recording = True

                self.audio_producer_state_map[data["device_info"].index] = "open"
                # If second device is used, wait for both streams to be open to update the UI.
                if all(state == "open" for state in self.audio_producer_state_map.values()):
                    self.gui_queue.put(on_stream_open)

            case "stream_closed" | "stream_error":
                def on_stream_closed():
                    self.record_btn.config(text="Record")
                    self.mute_btn.config(state="disabled", text="🔊")
                    self.mute_btn_2.config(state="disabled", text="🔊")
                    self.show_captions_overlay_btn.config(state="disabled", text="Show overlay")
                    self.is_recording = False

                self.audio_producer_state_map[data["device_info"].index] = "closed"
                # If second device is used, wait for both streams to be closed to update the UI.
                if all(state == "closed" for state in self.audio_producer_state_map.values()):
                    self.gui_queue.put(on_stream_closed)

    def run_gui(self):
        self.root_wnd.mainloop()

__all__ = ["CaptionerUI"]

