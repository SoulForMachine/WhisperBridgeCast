import json
import sys
import types
from pathlib import Path

# Keep imports lightweight in environments without audio backends.
if sys.platform == "win32" and "pyaudiowpatch" not in sys.modules:
    pyaudiowpatch_stub = types.ModuleType("pyaudiowpatch")
    pyaudiowpatch_stub.PyAudio = object
    pyaudiowpatch_stub.paContinue = 0
    sys.modules["pyaudiowpatch"] = pyaudiowpatch_stub
if sys.platform != "win32" and "pyaudio" not in sys.modules:
    pyaudio_stub = types.ModuleType("pyaudio")
    pyaudio_stub.PyAudio = object
    pyaudio_stub.paContinue = 0
    sys.modules["pyaudio"] = pyaudio_stub

import app.gui.ui as ui_module
from app.common.utils import settings_to_dict
from app.gui.settings import ClientSettings
from app.gui.ui import CaptionerUI
from app.server.settings import PipelineSettings


class FakeVar:
    def __init__(self, value=None):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class FakeWidget:
    def __init__(self, value="", values=None):
        self._value = value
        self._values = list(values or [])
        self.state = "normal"
        self.text = ""

    def get(self):
        return self._value

    def set(self, value):
        self._value = value

    def current(self, idx: int):
        self._value = self._values[idx]

    def config(self, **kwargs):
        if "state" in kwargs:
            self.state = kwargs["state"]
        if "text" in kwargs:
            self.text = kwargs["text"]

    def event_generate(self, _event_name: str):
        return None

    def __setitem__(self, key: str, value):
        if key == "values":
            self._values = list(value)
        else:
            raise KeyError(key)

    def __getitem__(self, key: str):
        if key == "values":
            return self._values
        raise KeyError(key)


def _build_ui_test_double(settings_file: Path, monkeypatch) -> CaptionerUI:
    monkeypatch.setattr(ui_module, "default_input_device", lambda _dm: ("Mic A", "Windows WASAPI"))
    monkeypatch.setattr(ui_module, "sort_api_by_preference", lambda apis: sorted(list(apis)))

    ui = CaptionerUI.__new__(CaptionerUI)
    ui.settings_file = settings_file
    ui.client_settings = ClientSettings()
    ui.pipeline_settings = PipelineSettings()

    ui.transl_engines_with_params = ["EuroLLM", "Google Gemini"]
    ui.online_translators = ["Google", "Libre", "Microsoft"]
    ui.libre_mirrors = ["libretranslate.com", "libretranslate.de"]

    ui.device_map = {
        "Mic A": {"Windows WASAPI": object(), "MME": object()},
        "Mic B": {"MME": object()},
    }

    ui.on_enable_translation_toggle = lambda: None
    ui.on_transl_engine_selection_change = lambda _evt: None
    ui.on_enable_second_audio_device = lambda: None

    ui.server_url_var = FakeVar("")
    ui.server_port_var = FakeVar("5000")
    ui.zoom_url_var = FakeVar("")

    ui.model_var = FakeVar("distil-large-v3")
    ui.whisper_device_var = FakeVar("cuda")
    ui.whisper_compute_type_var = FakeVar("int8")
    ui.lang_var = FakeVar("English")

    ui.threshold_var = FakeVar(0.9)
    ui.threshold_label = FakeWidget()
    ui.buffer_trimming_var = FakeVar("segment")
    ui.buffer_trimming_sec_var = FakeVar(15.0)
    ui.buffer_trimming_sec_label = FakeWidget()

    ui.vac_var = FakeVar(True)
    ui.vad_var = FakeVar(False)
    ui.vac_min_chunk_size_var = FakeVar(1.2)
    ui.vac_min_chunk_size_label = FakeWidget()
    ui.vac_is_dynamic_chunk_size_var = FakeVar(True)
    ui.vad_start_threshold_var = FakeVar(0.5)
    ui.vad_start_threshold_label = FakeWidget()
    ui.vad_end_threshold_var = FakeVar(0.35)
    ui.vad_end_threshold_label = FakeWidget()
    ui.vad_min_silence_duration_var = FakeVar(0.5)
    ui.vad_min_silence_duration_label = FakeWidget()
    ui.vad_speech_pad_start_var = FakeVar(0.8)
    ui.vad_speech_pad_start_label = FakeWidget()
    ui.vad_speech_pad_end_var = FakeVar(0.9)
    ui.vad_speech_pad_end_label = FakeWidget()
    ui.vad_hangover_chunks_var = FakeVar(2)
    ui.vad_hangover_chunks_label = FakeWidget()

    ui.enable_translation_var = FakeVar(True)
    ui.target_lang_var = FakeVar("Serbian Cyrillic")
    ui.transl_engine_var = FakeVar("MarianMT")
    ui.transl_word_increment_var = FakeVar(0)
    ui.transl_word_increment_label = FakeWidget()
    ui.source_diff_enabled_var = FakeVar(True)
    ui.target_diff_enabled_var = FakeVar(True)

    ui.online_translator_var = FakeVar("Google")
    ui.transl_api_key_var = FakeVar("")
    ui.transl_api_secret_var = FakeVar("")
    ui.transl_client_id_var = FakeVar("")
    ui.transl_domain_var = FakeVar("general")
    ui.transl_region_var = FakeVar("")
    ui.libre_mirror_var = FakeVar("libretranslate.com")

    ui.audio_chunk_size_var = FakeVar(0.4)
    ui.audio_chunk_size_label = FakeWidget()
    ui.use_second_audio_dev_var = FakeVar(False)

    ui.audio_device_combo_1 = FakeWidget("")
    ui.audio_device_host_api_combo_1 = FakeWidget("")
    ui.audio_device_block_dur_slider_1 = FakeWidget(0.0)

    ui.audio_device_combo_2 = FakeWidget("")
    ui.audio_device_host_api_combo_2 = FakeWidget("")
    ui.audio_device_block_dur_slider_2 = FakeWidget(0.0)

    ui.captions_font_size_var = FakeVar(24)
    ui.captions_font_size_label = FakeWidget()
    ui.captions_max_visible_lines_var = FakeVar(4)
    ui.captions_max_visible_lines_label = FakeWidget()

    return ui


def _save_settings_blob(path: Path, client_settings: ClientSettings, pipeline_settings: PipelineSettings):
    data = {
        "client": settings_to_dict(client_settings),
        "server": settings_to_dict(pipeline_settings),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_ui_settings_full_cycle_roundtrip_with_mutations(tmp_path, monkeypatch):
    settings_file = tmp_path / "captioner-settings.json"

    initial_client = ClientSettings()
    initial_client.host = "localhost"
    initial_client.port = 5000
    initial_client.audio.chunk_size_s = 0.5
    initial_client.audio.use_second_device = True
    initial_client.audio.device_1.name = "Mic B"
    initial_client.audio.device_1.host_api = "MME"
    initial_client.audio.device_1.block_duration_s = 0.11
    initial_client.audio.device_2.name = "Mic A"
    initial_client.audio.device_2.host_api = "Windows WASAPI"
    initial_client.audio.device_2.block_duration_s = 0.22
    initial_client.captions_overlay.font_size = 30
    initial_client.captions_overlay.max_visible_lines = 6

    initial_pipeline = PipelineSettings()
    initial_pipeline.zoom_url = "https://zoom.us/j/123"
    initial_pipeline.asr.language = "de"
    initial_pipeline.translation.target_language = "de"
    initial_pipeline.translation.engine = "MarianMT"

    _save_settings_blob(settings_file, initial_client, initial_pipeline)

    ui = _build_ui_test_double(settings_file, monkeypatch)
    ui.load_settings_from_file()
    ui.apply_settings_to_ui()

    ui.server_url_var.set("127.0.0.1")
    ui.server_port_var.set("7001")
    ui.zoom_url_var.set("https://zoom.us/j/999")

    ui.model_var.set("small")
    ui.whisper_device_var.set("cpu")
    ui.whisper_compute_type_var.set("float32")
    ui.lang_var.set("German")
    ui.threshold_var.set(0.42)
    ui.buffer_trimming_var.set("sentence")
    ui.buffer_trimming_sec_var.set(9.5)

    ui.vac_var.set(False)
    ui.vad_var.set(True)
    ui.vac_min_chunk_size_var.set(0.9)
    ui.vac_is_dynamic_chunk_size_var.set(False)
    ui.vad_start_threshold_var.set(0.61)
    ui.vad_end_threshold_var.set(0.33)
    ui.vad_min_silence_duration_var.set(0.75)
    ui.vad_speech_pad_start_var.set(0.2)
    ui.vad_speech_pad_end_var.set(0.4)
    ui.vad_hangover_chunks_var.set(5)

    ui.enable_translation_var.set(True)
    ui.target_lang_var.set("English")
    ui.transl_engine_var.set("Whisper")
    ui.transl_word_increment_var.set(3)
    ui.source_diff_enabled_var.set(False)
    ui.target_diff_enabled_var.set(True)

    ui.audio_chunk_size_var.set(0.7)
    ui.use_second_audio_dev_var.set(True)
    ui.audio_device_combo_1.set("Mic A")
    ui.audio_device_host_api_combo_1.set("MME")
    ui.audio_device_block_dur_slider_1.set(0.15)
    ui.audio_device_combo_2.set("Mic B")
    ui.audio_device_host_api_combo_2.set("MME")
    ui.audio_device_block_dur_slider_2.set(0.25)

    ui.captions_font_size_var.set(28)
    ui.captions_max_visible_lines_var.set(5)

    ui.client_settings = ui.collect_client_settings_from_ui()
    ui.pipeline_settings = ui.collect_server_settings_from_ui()
    ui.save_settings_to_file()

    saved = _load_json(settings_file)
    assert saved["client"] == settings_to_dict(ui.client_settings)
    assert saved["server"] == settings_to_dict(ui.pipeline_settings)
    assert saved["server"]["asr"]["task"] == "transcribe"
    assert saved["server"]["vac"]["min_silence_duration_ms"] == 750

    ui_reloaded = _build_ui_test_double(settings_file, monkeypatch)
    ui_reloaded.load_settings_from_file()
    ui_reloaded.apply_settings_to_ui()

    reloaded_client = ui_reloaded.collect_client_settings_from_ui()
    reloaded_server = ui_reloaded.collect_server_settings_from_ui()
    assert settings_to_dict(reloaded_client) == settings_to_dict(ui.client_settings)
    assert settings_to_dict(reloaded_server) == settings_to_dict(ui.pipeline_settings)


def test_ui_settings_cycle_normalizes_invalid_values_and_fallbacks(tmp_path, monkeypatch):
    settings_file = tmp_path / "captioner-settings.json"

    initial_client = ClientSettings()
    initial_client.host = "example-host"
    initial_client.port = 6200
    initial_client.audio.device_1.name = "Missing Device"
    initial_client.audio.device_2.name = "Missing Device 2"

    initial_pipeline = PipelineSettings()
    initial_pipeline.translation.enable = True
    initial_pipeline.translation.engine = "Online Translators"
    initial_pipeline.translation.target_language = "de"
    initial_pipeline.translation.engine_params = {
        "provider": "Microsoft",
        "api_key": "old-key",
        "region": "westus",
    }

    _save_settings_blob(settings_file, initial_client, initial_pipeline)

    ui = _build_ui_test_double(settings_file, monkeypatch)
    ui.load_settings_from_file()
    ui.apply_settings_to_ui()

    assert ui.audio_device_combo_1.get() == "Mic A"
    assert ui.audio_device_combo_2.get() == "Mic A"
    assert ui.audio_device_host_api_combo_1.get() == "MME"

    ui.server_port_var.set("not-an-int")
    ui.enable_translation_var.set(True)
    ui.transl_engine_var.set("Online Translators")
    ui.online_translator_var.set("Libre")
    ui.transl_api_key_var.set("new-key")
    ui.transl_api_secret_var.set("secret")
    ui.transl_client_id_var.set("client-1")
    ui.transl_domain_var.set("news")
    ui.transl_region_var.set("eu")
    ui.libre_mirror_var.set("")

    ui.client_settings = ui.collect_client_settings_from_ui()
    ui.pipeline_settings = ui.collect_server_settings_from_ui()
    ui.save_settings_to_file()

    ui_reloaded = _build_ui_test_double(settings_file, monkeypatch)
    ui_reloaded.load_settings_from_file()
    ui_reloaded.apply_settings_to_ui()

    reloaded_client = ui_reloaded.collect_client_settings_from_ui()
    reloaded_server = ui_reloaded.collect_server_settings_from_ui()

    assert reloaded_client.port == 5000
    assert reloaded_server.translation.engine == "Online Translators"
    assert reloaded_server.translation.engine_params["provider"] == "Libre"
    assert reloaded_server.translation.engine_params["libre_mirror"] == "libretranslate.com"
    assert reloaded_server.asr.task == "transcribe"


def test_ui_settings_vad_and_engine_params_roundtrip_regression(tmp_path, monkeypatch):
    settings_file = tmp_path / "captioner-settings.json"

    initial_client = ClientSettings()
    initial_pipeline = PipelineSettings()
    initial_pipeline.vac.min_silence_duration_ms = 1230
    initial_pipeline.vac.speech_pad_start_ms = 450
    initial_pipeline.vac.speech_pad_end_ms = 980
    initial_pipeline.vac.hangover_chunks = 7
    initial_pipeline.translation.enable = True
    initial_pipeline.translation.engine = "Online Translators"
    initial_pipeline.translation.target_language = "de"
    initial_pipeline.translation.engine_params = {
        "provider": "Microsoft",
        "api_key": "old",
        "region": "old-region",
    }

    _save_settings_blob(settings_file, initial_client, initial_pipeline)

    ui = _build_ui_test_double(settings_file, monkeypatch)
    ui.load_settings_from_file()
    ui.apply_settings_to_ui()

    # Verify apply-settings conversion from ms storage to seconds widget values.
    assert ui.vad_min_silence_duration_var.get() == 1.23
    assert ui.vad_speech_pad_start_var.get() == 0.45
    assert ui.vad_speech_pad_end_var.get() == 0.98

    # Mutate widget values to exercise collect-settings conversion and engine params capture.
    ui.vad_min_silence_duration_var.set(1.57)
    ui.vad_speech_pad_start_var.set(0.33)
    ui.vad_speech_pad_end_var.set(0.44)
    ui.vad_hangover_chunks_var.set(9)

    ui.enable_translation_var.set(True)
    ui.transl_engine_var.set("Online Translators")
    ui.online_translator_var.set("Microsoft")
    ui.transl_api_key_var.set("ms-key")
    ui.transl_api_secret_var.set("ms-secret")
    ui.transl_client_id_var.set("client-42")
    ui.transl_domain_var.set("finance")
    ui.transl_region_var.set("northeurope")
    ui.libre_mirror_var.set("libretranslate.de")

    ui.client_settings = ui.collect_client_settings_from_ui()
    ui.pipeline_settings = ui.collect_server_settings_from_ui()
    ui.save_settings_to_file()

    saved = _load_json(settings_file)
    vac = saved["server"]["vac"]
    assert vac["min_silence_duration_ms"] == 1570
    assert vac["speech_pad_start_ms"] == 330
    assert vac["speech_pad_end_ms"] == 440
    assert vac["hangover_chunks"] == 9

    params = saved["server"]["translation"]["engine_params"]
    assert params["provider"] == "Microsoft"
    assert params["api_key"] == "ms-key"
    assert params["api_secret"] == "ms-secret"
    assert params["client_id"] == "client-42"
    assert params["domain"] == "finance"
    assert params["region"] == "northeurope"
    assert params["libre_mirror"] == "libretranslate.de"

    ui_reloaded = _build_ui_test_double(settings_file, monkeypatch)
    ui_reloaded.load_settings_from_file()
    ui_reloaded.apply_settings_to_ui()
    reloaded = ui_reloaded.collect_server_settings_from_ui()
    assert reloaded.vac.min_silence_duration_ms == 1570
    assert reloaded.vac.speech_pad_start_ms == 330
    assert reloaded.vac.speech_pad_end_ms == 440
    assert reloaded.translation.engine_params == params
