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

    ui.cl_host_var = FakeVar("")
    ui.cl_port_var = FakeVar("5000")
    ui.pl_zoom_url_var = FakeVar("")

    ui.pl_asr_model_var = FakeVar("distil-large-v3")
    ui.pl_asr_device_var = FakeVar("cuda")
    ui.pl_asr_compute_type_var = FakeVar("int8")
    ui.pl_asr_language_var = FakeVar("English")

    ui.pl_asr_nsp_threshold_var = FakeVar(0.9)
    ui.threshold_label = FakeWidget()
    ui.pl_asr_buffer_trimming_var = FakeVar("segment")
    ui.pl_asr_buffer_trimming_sec_var = FakeVar(15.0)
    ui.buffer_trimming_sec_label = FakeWidget()

    ui.pl_vac_enable_var = FakeVar(True)
    ui.pl_vac_enable_whisper_internal_vad_var = FakeVar(False)
    ui.pl_vac_min_chunk_size_s_var = FakeVar(1.2)
    ui.vac_min_chunk_size_label = FakeWidget()
    ui.pl_vac_is_dynamic_chunk_size_var = FakeVar(True)
    ui.pl_vac_start_threshold_var = FakeVar(0.5)
    ui.vad_start_threshold_label = FakeWidget()
    ui.pl_vac_end_threshold_var = FakeVar(0.35)
    ui.vad_end_threshold_label = FakeWidget()
    ui.pl_vac_min_silence_duration_ms_var = FakeVar(0.5)
    ui.vad_min_silence_duration_label = FakeWidget()
    ui.pl_vac_speech_pad_start_ms_var = FakeVar(0.8)
    ui.vad_speech_pad_start_label = FakeWidget()
    ui.pl_vac_speech_pad_end_ms_var = FakeVar(0.9)
    ui.vad_speech_pad_end_label = FakeWidget()
    ui.pl_vac_hangover_chunks_var = FakeVar(2)
    ui.vad_hangover_chunks_label = FakeWidget()

    ui.pl_translation_enable_var = FakeVar(True)
    ui.pl_translation_target_language_var = FakeVar("Serbian Cyrillic")
    ui.pl_translation_engine_var = FakeVar("MarianMT")
    ui.pl_translation_word_increment_var = FakeVar(0)
    ui.transl_word_increment_label = FakeWidget()
    ui.pl_translation_source_diff_enabled_var = FakeVar(True)
    ui.pl_translation_target_diff_enabled_var = FakeVar(True)

    ui.pl_translation_engine_params_provider_var = FakeVar("Google")
    ui.pl_translation_engine_params_api_key_var = FakeVar("")
    ui.pl_translation_engine_params_api_secret_var = FakeVar("")
    ui.pl_translation_engine_params_client_id_var = FakeVar("")
    ui.pl_translation_engine_params_domain_var = FakeVar("general")
    ui.pl_translation_engine_params_region_var = FakeVar("")
    ui.pl_translation_engine_params_libre_mirror_var = FakeVar("libretranslate.com")

    ui.cl_audio_chunk_size_s_var = FakeVar(0.4)
    ui.audio_chunk_size_label = FakeWidget()
    ui.cl_audio_use_second_device_var = FakeVar(False)

    ui.cl_audio_device_1_name_var = FakeVar("")
    ui.cl_audio_device_1_host_api_var = FakeVar("")
    ui.cl_audio_device_1_block_duration_s_var = FakeVar(0.0)

    ui.cl_audio_device_2_name_var = FakeVar("")
    ui.cl_audio_device_2_host_api_var = FakeVar("")
    ui.cl_audio_device_2_block_duration_s_var = FakeVar(0.0)

    ui.audio_device_combo_1 = FakeWidget("")
    ui.audio_device_host_api_combo_1 = FakeWidget("")
    ui.audio_device_block_dur_slider_1 = FakeWidget(0.0)

    ui.audio_device_combo_2 = FakeWidget("")
    ui.audio_device_host_api_combo_2 = FakeWidget("")
    ui.audio_device_block_dur_slider_2 = FakeWidget(0.0)

    ui.cl_captions_overlay_font_size_var = FakeVar(24)
    ui.captions_font_size_label = FakeWidget()
    ui.cl_captions_overlay_max_visible_lines_var = FakeVar(4)
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

    ui.cl_host_var.set("127.0.0.1")
    ui.cl_port_var.set("7001")
    ui.pl_zoom_url_var.set("https://zoom.us/j/999")

    ui.pl_asr_model_var.set("small")
    ui.pl_asr_device_var.set("cpu")
    ui.pl_asr_compute_type_var.set("float32")
    ui.pl_asr_language_var.set("German")
    ui.pl_asr_nsp_threshold_var.set(0.42)
    ui.pl_asr_buffer_trimming_var.set("sentence")
    ui.pl_asr_buffer_trimming_sec_var.set(9.5)

    ui.pl_vac_enable_var.set(False)
    ui.pl_vac_enable_whisper_internal_vad_var.set(True)
    ui.pl_vac_min_chunk_size_s_var.set(0.9)
    ui.pl_vac_is_dynamic_chunk_size_var.set(False)
    ui.pl_vac_start_threshold_var.set(0.61)
    ui.pl_vac_end_threshold_var.set(0.33)
    ui.pl_vac_min_silence_duration_ms_var.set(0.75)
    ui.pl_vac_speech_pad_start_ms_var.set(0.2)
    ui.pl_vac_speech_pad_end_ms_var.set(0.4)
    ui.pl_vac_hangover_chunks_var.set(5)

    ui.pl_translation_enable_var.set(True)
    ui.pl_translation_target_language_var.set("English")
    ui.pl_translation_engine_var.set("Whisper")
    ui.pl_translation_word_increment_var.set(3)
    ui.pl_translation_source_diff_enabled_var.set(False)
    ui.pl_translation_target_diff_enabled_var.set(True)

    ui.cl_audio_chunk_size_s_var.set(0.7)
    ui.cl_audio_use_second_device_var.set(True)
    ui.cl_audio_device_1_name_var.set("Mic A")
    ui.cl_audio_device_1_host_api_var.set("MME")
    ui.cl_audio_device_1_block_duration_s_var.set(0.15)
    ui.cl_audio_device_2_name_var.set("Mic B")
    ui.cl_audio_device_2_host_api_var.set("MME")
    ui.cl_audio_device_2_block_duration_s_var.set(0.25)

    ui.cl_captions_overlay_font_size_var.set(28)
    ui.cl_captions_overlay_max_visible_lines_var.set(5)

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

    assert ui.cl_audio_device_1_name_var.get() == "Mic A"
    assert ui.cl_audio_device_2_name_var.get() == "Mic A"
    assert ui.cl_audio_device_1_host_api_var.get() == "MME"

    ui.cl_port_var.set("not-an-int")
    ui.pl_translation_enable_var.set(True)
    ui.pl_translation_engine_var.set("Online Translators")
    ui.pl_translation_engine_params_provider_var.set("Libre")
    ui.pl_translation_engine_params_api_key_var.set("new-key")
    ui.pl_translation_engine_params_api_secret_var.set("secret")
    ui.pl_translation_engine_params_client_id_var.set("client-1")
    ui.pl_translation_engine_params_domain_var.set("news")
    ui.pl_translation_engine_params_region_var.set("eu")
    ui.pl_translation_engine_params_libre_mirror_var.set("")

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
    assert ui.pl_vac_min_silence_duration_ms_var.get() == 1.23
    assert ui.pl_vac_speech_pad_start_ms_var.get() == 0.45
    assert ui.pl_vac_speech_pad_end_ms_var.get() == 0.98

    # Mutate widget values to exercise collect-settings conversion and engine params capture.
    ui.pl_vac_min_silence_duration_ms_var.set(1.57)
    ui.pl_vac_speech_pad_start_ms_var.set(0.33)
    ui.pl_vac_speech_pad_end_ms_var.set(0.44)
    ui.pl_vac_hangover_chunks_var.set(9)

    ui.pl_translation_enable_var.set(True)
    ui.pl_translation_engine_var.set("Online Translators")
    ui.pl_translation_engine_params_provider_var.set("Microsoft")
    ui.pl_translation_engine_params_api_key_var.set("ms-key")
    ui.pl_translation_engine_params_api_secret_var.set("ms-secret")
    ui.pl_translation_engine_params_client_id_var.set("client-42")
    ui.pl_translation_engine_params_domain_var.set("finance")
    ui.pl_translation_engine_params_region_var.set("northeurope")
    ui.pl_translation_engine_params_libre_mirror_var.set("libretranslate.de")

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
