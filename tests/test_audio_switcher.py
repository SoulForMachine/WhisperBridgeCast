import queue
import sys
import time
import types


# Keep imports lightweight in environments without audio backends.
if "pyaudiowpatch" not in sys.modules:
    pyaudiowpatch_stub = types.ModuleType("pyaudiowpatch")
    pyaudiowpatch_stub.PyAudio = object
    pyaudiowpatch_stub.paContinue = 0
    sys.modules["pyaudiowpatch"] = pyaudiowpatch_stub

if "pyaudio" not in sys.modules:
    pyaudio_stub = types.ModuleType("pyaudio")
    pyaudio_stub.PyAudio = object
    pyaudio_stub.paContinue = 0
    sys.modules["pyaudio"] = pyaudio_stub

from app.gui.audio import AudioSwitcher


def _switcher() -> AudioSwitcher:
    return AudioSwitcher(
        audio1_queue=queue.Queue(),
        audio2_queue=queue.Queue(),
        output_queue=queue.Queue(),
    )


def test_audio_switcher_initializes_smoothed_rms_state() -> None:
    sw = _switcher()

    assert sw.rms1_smooth == 0.0
    assert sw.rms2_smooth == 0.0
    assert sw.snr_release_threshold < sw.snr_threshold


def test_compute_voice_metrics_keeps_both_when_above_thresholds() -> None:
    sw = _switcher()

    sw.snr1_smooth = 40.0
    sw.snr2_smooth = 12.0
    sw.rms1_smooth = 0.06
    sw.rms2_smooth = 0.05

    voice1, voice2 = sw.compute_voice_metrics()

    assert voice1 is True
    assert voice2 is True


def test_compute_voice_metrics_keeps_both_when_close_strength() -> None:
    sw = _switcher()

    sw.snr1_smooth = 18.0
    sw.snr2_smooth = 16.0
    sw.rms1_smooth = 0.03
    sw.rms2_smooth = 0.03

    voice1, voice2 = sw.compute_voice_metrics()

    assert voice1 is True
    assert voice2 is True


def test_update_voice_state_preserves_attack_candidate_across_short_dip() -> None:
    sw = _switcher()
    now = time.time()

    is_voice = False
    above_since = None
    below_since = None

    # First above-threshold sample starts attack timing.
    is_voice, above_since, below_since = sw.update_voice_state(
        sw.snr_threshold + 2.0,
        sw.min_voice_rms + 0.01,
        is_voice,
        above_since,
        below_since,
        now,
    )
    assert is_voice is False
    assert above_since is not None

    # A short dip before attack_time should not reset the candidate start.
    is_voice, above_since_after_dip, below_since = sw.update_voice_state(
        0.0,
        0.0,
        is_voice,
        above_since,
        below_since,
        now + sw.attack_time * 0.5,
    )
    assert is_voice is False
    assert above_since_after_dip == above_since

    # Another above-threshold sample after enough elapsed time should activate.
    is_voice, above_since, below_since = sw.update_voice_state(
        sw.snr_threshold + 2.0,
        sw.min_voice_rms + 0.01,
        is_voice,
        above_since_after_dip,
        below_since,
        now + sw.attack_time + 0.01,
    )
    assert is_voice is True


def test_update_voice_state_uses_release_hysteresis() -> None:
    sw = _switcher()
    now = time.time()

    # Already in voice state.
    is_voice = True
    above_since = None
    below_since = None

    # Above release threshold but below activation threshold should keep voice True.
    is_voice, above_since, below_since = sw.update_voice_state(
        sw.snr_release_threshold + 0.2,
        sw.min_voice_rms + 0.005,
        is_voice,
        above_since,
        below_since,
        now,
    )
    assert is_voice is True


def test_compute_idle_dominant_source_prefers_stronger_activity() -> None:
    sw = _switcher()

    sw.snr1_smooth = 0.9
    sw.rms1_smooth = 0.012
    sw.snr2_smooth = 5.2
    sw.rms2_smooth = 0.07

    assert sw.compute_idle_dominant_source() == 2
