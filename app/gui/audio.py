import sys

import logging
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

if sys.platform == "win32":
    import pyaudiowpatch as pyaudio
else:
    import pyaudio

from app.gui.audio_utils import InputDeviceInfo, WHISPER_SAMPLERATE

logger = logging.getLogger(__name__)

class AudioStreamProducer:
    def __init__(
        self,
        min_chunk_size: float,
        input_device_info: InputDeviceInfo,
        output_queue: queue.Queue,
        notif_callback: Callable[[str, dict], None]=None
    ):
        self.min_chunk_size = min_chunk_size
        self.dev_info = input_device_info

        self.resample_stream = None
        self.stream = None

        self.audio_queue = queue.Queue()
        self.output_queue = output_queue

        self.notif_callback = notif_callback if notif_callback else lambda et, d: None

        self.started_event = None
        self.stop_event = None
        self.audio_thread = None
        self.is_running = False

    def pause_stream(self):
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()

    def resume_stream(self):
        if self.stream and self.stream.is_stopped():
            self.stream.start_stream()

    def is_paused(self):
        if self.stream:
            return not self.stream.is_active()
        return False

    def start(self) -> bool:
        if not self.is_running:
            self.started_event = threading.Event()
            self.stop_event = threading.Event()
            self.audio_thread = threading.Thread(target=self.run)
            self.audio_thread.start()
            self.started_event.wait()
            return self.is_running

        return False

    def stop(self):
        if self.is_running:
            self.stop_event.set()
            self.audio_queue.put(None)  # Sentinel to unblock receive_audio_chunk()
            self.audio_thread.join()
            self.audio_thread = None
            self.is_running = False
            self.stream = None
            self.started_event = None
            self.stop_event = None

    def downmix_mono(self, data: np.ndarray) -> np.ndarray:
        # Downmix by summing and normalizing.
        # Ensure input is float32 (if it already is, this does nothing)
        data = data.astype(np.float32, copy=False)

        # Sum across channels
        mono = np.empty(data.shape[0], dtype=np.float32)
        np.add.reduce(data, axis=1, out=mono)

        # Scale to preserve RMS volume
        mono /= np.sqrt(data.shape[1])

        return mono

    def resample_to_whisper(self, data: np.ndarray) -> np.ndarray:
        return self.resample_stream.resample_chunk(data)

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio callback status: {status}")

        self.audio_queue.put(indata)

        return (None, pyaudio.paContinue)

    def receive_audio_chunk(self):
        out = []
        minlimit = int(self.min_chunk_size * self.dev_info.samplerate)
        cur_size = 0
        while cur_size < minlimit:
            chunk = self.audio_queue.get()

            # Sentinel placed into the queue when stopping
            if chunk is None:
                return None

            arr = np.frombuffer(chunk, dtype=np.float32).reshape(-1, self.dev_info.channels)
            out.append(arr)
            cur_size += len(arr)

        conc = np.concatenate(out)

        mono = self.downmix_mono(conc) if self.dev_info.downmix_needed else conc
        mono16k = self.resample_to_whisper(mono) if self.dev_info.resample_needed else mono

        return mono16k

    def run(self):
        info_str = (
            f"Listening to [{self.dev_info.index}] {self.dev_info.name}\n"
            f"\t  API: {self.dev_info.api}\n"
            f"\t  Channels: {self.dev_info.channels}\n"
            f"\t  Samplerate: {int(self.dev_info.samplerate)}\n"
            f"\t  Blocksize: {self.dev_info.block_size} frames (block duration: ~{self.dev_info.block_dur} s)"
        )
        logger.info(info_str)

        p = pyaudio.PyAudio()
        stream = None

        try:
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=self.dev_info.channels,
                rate=int(self.dev_info.samplerate),
                input=True,
                start=False,
                input_device_index=self.dev_info.index,
                frames_per_buffer=self.dev_info.block_size,
                stream_callback=self.audio_callback
            )

            self.stream = stream
            self.notif_callback("stream_open", {"device_info": self.dev_info})

            if self.dev_info.resample_needed:
                import soxr
                self.resample_stream = soxr.ResampleStream(self.dev_info.samplerate, WHISPER_SAMPLERATE, num_channels=1, dtype='float32', quality='LQ')

            self.is_running = True
            self.started_event.set()

            while not self.stop_event.is_set():
                chunk = self.receive_audio_chunk()
                if chunk is None or len(chunk) == 0:
                    continue

                self.output_queue.put(chunk)

            if self.dev_info.resample_needed:
                tail = self.resample_stream.resample_chunk(np.empty((0,), dtype=np.float32), last=True)
                self.output_queue.put(tail)
                self.resample_stream = None
        except Exception as e:
            logger.error(f"Audio stream error ({self.dev_info.name}): {e}")
            self.notif_callback("stream_error", {"message": str(e), "device_info": self.dev_info})
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            p.terminate()
            self.notif_callback("stream_closed", {"device_info": self.dev_info})
            self.is_running = False
            self.started_event.set()


class AudioSwitcher:
    """
    Professional-style automatic audio switcher.

    Features:
    - Adaptive per-source noise floor with startup floor guard
    - Smoothed SNR and RMS activity tracking per source
    - Voice activation with attack / release timing and dual thresholds
    - Dominance-based switching when both sources are voice-active
    - Idle recovery path when both voices are currently false
    - Cooldown-based switch stabilization to reduce chatter

    Algorithm overview:
     1) Poll one chunk per source each loop and skip only if both are absent.
     2) For each source with a chunk, compute RMS, update adaptive noise floor,
         derive instantaneous SNR, and exponentially smooth SNR/RMS.
         For a missing chunk, decay smoothed SNR/RMS toward zero.
     3) Update per-source voice state using:
         - Activation threshold (snr_threshold + min_voice_rms)
         - Release threshold (snr_release_threshold + relaxed RMS floor)
         - Attack/release timers to avoid flicker
         - Short attack-window preservation across brief dips
     4) Choose dominant source:
         - If only one source is voice-active, choose it.
         - If both are voice-active, use dominance_ratio on smoothed SNR.
         - If both are voice-inactive, use idle activity score
            (snr_smooth * rms_smooth) with idle_dominance_ratio.
     5) Apply switch_cooldown before committing a source change.
     6) Forward only the selected source chunk to output_queue when available.
    """

    def __init__(
        self,
        audio1_queue: queue.Queue,
        audio2_queue: queue.Queue,
        output_queue: queue.Queue,
        snr_threshold: float = 6.0,
        snr_release_threshold: Optional[float] = None,
        attack_time: float = 0.05,
        release_time: float = 0.3,
        switch_cooldown: float = 0.25,
        noise_alpha: float = 0.01,
        snr_smoothing: float = 0.5,
        dominance_ratio: float = 1.35,
        idle_dominance_ratio: float = 1.2,
        min_noise_floor: float = 1e-4,
        min_voice_rms: float = 0.01,
        snr_decay_no_data: float = 0.35,
    ):
        self.audio1_queue = audio1_queue
        self.audio2_queue = audio2_queue
        self.output_queue = output_queue

        self.snr_threshold = snr_threshold
        self.snr_release_threshold = (
            snr_release_threshold if snr_release_threshold is not None else snr_threshold * 0.6
        )
        self.attack_time = attack_time
        self.release_time = release_time
        self.switch_cooldown = switch_cooldown
        self.noise_alpha = noise_alpha
        self.snr_smoothing = snr_smoothing
        self.dominance_ratio = dominance_ratio
        self.idle_dominance_ratio = idle_dominance_ratio
        self.min_noise_floor = min_noise_floor
        self.min_voice_rms = min_voice_rms
        self.snr_decay_no_data = snr_decay_no_data
        self.stop_event = threading.Event()
        self.thread = None

        self.current_source = 1
        self.last_switch_time = 0.0

        # Noise floors
        self.noise1 = self.min_noise_floor
        self.noise2 = self.min_noise_floor

        # Smoothed SNR
        self.snr1_smooth = 0.0
        self.snr2_smooth = 0.0
        self.rms1_smooth = 0.0
        self.rms2_smooth = 0.0

        # Voice state tracking
        self.voice1 = False
        self.voice2 = False
        self.voice1_above_since: Optional[float] = None
        self.voice2_above_since: Optional[float] = None
        self.voice1_below_since: Optional[float] = None
        self.voice2_below_since: Optional[float] = None

    @staticmethod
    def rms(chunk: np.ndarray) -> float:
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)
        return float(np.sqrt(np.mean(chunk ** 2) + 1e-12))

    def update_noise_floor(self, rms: float, current_floor: float) -> float:
        # Track background energy with asymmetric smoothing so speech does not
        # inflate the baseline too quickly.
        if rms <= current_floor:
            alpha = min(1.0, self.noise_alpha * 2.0)
        else:
            alpha = max(1e-4, self.noise_alpha * 0.2)

        new_floor = (1 - alpha) * current_floor + alpha * rms
        return max(new_floor, self.min_noise_floor)

    def update_voice_state(
        self,
        snr_smooth: float,
        rms_smooth: float,
        is_voice: bool,
        above_since: Optional[float],
        below_since: Optional[float],
        now: float,
    ):
        above_on = snr_smooth > self.snr_threshold and rms_smooth > self.min_voice_rms
        above_off = snr_smooth > self.snr_release_threshold and rms_smooth > (self.min_voice_rms * 0.7)

        if above_on:
            below_since = None
            if is_voice:
                return True, above_since, below_since
            if above_since is None:
                return False, now, below_since
            if now - above_since >= self.attack_time:
                return True, above_since, below_since
            return False, above_since, below_since

        if is_voice and above_off:
            return True, None, None

        # If we are still in the attack window, keep the candidate start time.
        # This avoids missing voice activation when chunks alternate between sources.
        if not is_voice and above_since is not None and (now - above_since) < self.attack_time:
            return False, above_since, None

        above_since = None
        if not is_voice:
            return False, above_since, None
        if below_since is None:
            return True, above_since, now
        if now - below_since >= self.release_time:
            return False, above_since, below_since
        return True, above_since, below_since

    def compute_voice_metrics(self) -> tuple[bool, bool]:
        voice1_metric = (self.snr1_smooth > self.snr_threshold) and (self.rms1_smooth > self.min_voice_rms)
        voice2_metric = (self.snr2_smooth > self.snr_threshold) and (self.rms2_smooth > self.min_voice_rms)

        return voice1_metric, voice2_metric

    def compute_idle_dominant_source(self) -> Optional[int]:
        # Recovery path when both voice detectors are currently false.
        activity1 = self.snr1_smooth * max(self.rms1_smooth, 1e-9)
        activity2 = self.snr2_smooth * max(self.rms2_smooth, 1e-9)

        if activity1 > activity2 * self.idle_dominance_ratio:
            return 1
        if activity2 > activity1 * self.idle_dominance_ratio:
            return 2
        return None

    def start(self):
        if not self.thread or not self.thread.is_alive():
            self.stop_event.clear()
            self.thread = threading.Thread(target=self.run, name="AudioSwitcher")
            self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
            self.thread = None

    def run(self):
        while not self.stop_event.is_set():
            try:
                time.sleep(0.02)

                try:
                    chunk1 = self.audio1_queue.get_nowait()
                except queue.Empty:
                    chunk1 = None

                try:
                    chunk2 = self.audio2_queue.get_nowait()
                except queue.Empty:
                    chunk2 = None

                if chunk1 is None and chunk2 is None:
                    continue

                now = time.time()

                if chunk1 is not None:
                    rms1 = self.rms(chunk1)
                    self.noise1 = self.update_noise_floor(rms1, self.noise1)
                    snr1 = rms1 / max(self.noise1, self.min_noise_floor)
                    self.snr1_smooth = (1 - self.snr_smoothing) * self.snr1_smooth + self.snr_smoothing * snr1
                    self.rms1_smooth = (1 - self.snr_smoothing) * self.rms1_smooth + self.snr_smoothing * rms1
                else:
                    rms1 = 0.0
                    self.snr1_smooth *= (1.0 - self.snr_decay_no_data)
                    self.rms1_smooth *= (1.0 - self.snr_decay_no_data)

                if chunk2 is not None:
                    rms2 = self.rms(chunk2)
                    self.noise2 = self.update_noise_floor(rms2, self.noise2)
                    snr2 = rms2 / max(self.noise2, self.min_noise_floor)
                    self.snr2_smooth = (1 - self.snr_smoothing) * self.snr2_smooth + self.snr_smoothing * snr2
                    self.rms2_smooth = (1 - self.snr_smoothing) * self.rms2_smooth + self.snr_smoothing * rms2
                else:
                    rms2 = 0.0
                    self.snr2_smooth *= (1.0 - self.snr_decay_no_data)
                    self.rms2_smooth *= (1.0 - self.snr_decay_no_data)

                self.voice1, self.voice1_above_since, self.voice1_below_since = self.update_voice_state(
                    self.snr1_smooth,
                    self.rms1_smooth,
                    self.voice1,
                    self.voice1_above_since,
                    self.voice1_below_since,
                    now,
                )
                self.voice2, self.voice2_above_since, self.voice2_below_since = self.update_voice_state(
                    self.snr2_smooth,
                    self.rms2_smooth,
                    self.voice2,
                    self.voice2_above_since,
                    self.voice2_below_since,
                    now,
                )

                dominant = None
                if self.voice1 and not self.voice2:
                    dominant = 1
                elif self.voice2 and not self.voice1:
                    dominant = 2
                elif self.voice1 and self.voice2:
                    if self.snr1_smooth > self.snr2_smooth * self.dominance_ratio:
                        dominant = 1
                    elif self.snr2_smooth > self.snr1_smooth * self.dominance_ratio:
                        dominant = 2
                    else:
                        dominant = self.current_source
                else:
                    dominant = self.compute_idle_dominant_source()

                if (
                    dominant
                    and dominant != self.current_source
                    and now - self.last_switch_time >= self.switch_cooldown
                ):
                    self.current_source = dominant
                    self.last_switch_time = now
                    logger.debug(f"Switched to source {self.current_source}")

                selected = chunk1 if self.current_source == 1 else chunk2
                if selected is not None:
                    self.output_queue.put(selected)

                logger.debug(
                    f"RMS1={rms1:.5f} SNR1={self.snr1_smooth:.2f} "
                    f"RMS2={rms2:.5f} SNR2={self.snr2_smooth:.2f} "
                    f"Voice1={self.voice1} Voice2={self.voice2} "
                    f"Current={self.current_source}"
                )

            except Exception as e:
                logger.error(f"AudioSwitcher error: {e}")
                break

__all__ = ["AudioStreamProducer", "AudioSwitcher"]

