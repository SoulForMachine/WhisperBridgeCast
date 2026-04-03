import sys

import logging
import queue
import threading
import time
from typing import Callable

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

        self.stop_event = None
        self.audio_thread = None
        self.is_running = False

    def pause_stream(self):
        print(f"active: {self.stream.is_active()}, stopped: {self.stream.is_stopped()}")
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()

    def resume_stream(self):
        print(f"active: {self.stream.is_active()}, stopped: {self.stream.is_stopped()}")
        if self.stream and self.stream.is_stopped():
            self.stream.start_stream()

    def is_paused(self):
        if self.stream:
            return not self.stream.is_active()
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

        try:
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=self.dev_info.channels,
                rate=int(self.dev_info.samplerate),
                input=True,
                input_device_index=self.dev_info.index,
                frames_per_buffer=self.dev_info.block_size,
                stream_callback=self.audio_callback
            )

            self.stream = stream
            self.notif_callback("stream_open", {"device_info": self.dev_info})

            if self.dev_info.resample_needed:
                import soxr
                self.resample_stream = soxr.ResampleStream(self.dev_info.samplerate, WHISPER_SAMPLERATE, num_channels=1, dtype='float32', quality='LQ')

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
            logger.error(f"Audio stream error: {e}")
            self.notif_callback("stream_error", {"message": str(e), "device_info": self.dev_info})
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            self.notif_callback("stream_closed", {"device_info": self.dev_info})


class AudioSwitcher:
    """
    Professional-style automatic audio switcher.

    Features:
    - Adaptive per-source noise floor
    - SNR-based voice detection
    - Attack / release timing
    - Hysteresis
    - Stable switching (no chatter)
    """

    def __init__(
        self,
        audio1_queue: queue.Queue,
        audio2_queue: queue.Queue,
        output_queue: queue.Queue,
        snr_threshold: float = 2.0,
        attack_time: float = 0.05,
        release_time: float = 0.3,
        switch_cooldown: float = 0.15,
        noise_alpha: float = 0.01,
        snr_smoothing: float = 0.5,
    ):
        self.audio1_queue = audio1_queue
        self.audio2_queue = audio2_queue
        self.output_queue = output_queue

        self.snr_threshold = snr_threshold
        self.attack_time = attack_time
        self.release_time = release_time
        self.switch_cooldown = switch_cooldown
        self.noise_alpha = noise_alpha
        self.snr_smoothing = snr_smoothing

        self.stop_event = threading.Event()
        self.thread = None

        self.current_source = 1
        self.last_switch_time = 0.0

        # Noise floors
        self.noise1 = 1e-6
        self.noise2 = 1e-6

        # Smoothed SNR
        self.snr1_smooth = 0.0
        self.snr2_smooth = 0.0

        # Voice state tracking
        self.voice1 = False
        self.voice2 = False
        self.voice1_time = 0.0
        self.voice2_time = 0.0

    @staticmethod
    def rms(chunk: np.ndarray) -> float:
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)
        return float(np.sqrt(np.mean(chunk ** 2) + 1e-12))

    def update_noise_floor(self, rms, current_floor):
        # Update only when near noise level
        if rms < current_floor * 2:
            return (1 - self.noise_alpha) * current_floor + self.noise_alpha * rms
        return current_floor

    def update_voice_state(self, snr_smooth, is_voice, voice_time, now):
        if snr_smooth > self.snr_threshold:
            if not is_voice:
                if now - voice_time >= self.attack_time:
                    return True, now
                else:
                    return False, voice_time
            return True, voice_time
        else:
            if is_voice:
                if now - voice_time >= self.release_time:
                    return False, now
                else:
                    return True, voice_time
            return False, voice_time

    def start(self):
        if not self.thread or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.run)
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
                rms1 = rms2 = 0.0

                if chunk1 is not None:
                    rms1 = self.rms(chunk1)
                    self.noise1 = self.update_noise_floor(rms1, self.noise1)
                    snr1 = rms1 / (self.noise1 + 1e-9)

                    self.snr1_smooth = (
                        (1 - self.snr_smoothing) * self.snr1_smooth
                        + self.snr_smoothing * snr1
                    )

                    self.voice1, self.voice1_time = self.update_voice_state(
                        self.snr1_smooth, self.voice1, self.voice1_time, now
                    )


                if chunk2 is not None:
                    rms2 = self.rms(chunk2)
                    self.noise2 = self.update_noise_floor(rms2, self.noise2)
                    snr2 = rms2 / (self.noise2 + 1e-9)

                    self.snr2_smooth = (
                        (1 - self.snr_smoothing) * self.snr2_smooth
                        + self.snr_smoothing * snr2
                    )

                    self.voice2, self.voice2_time = self.update_voice_state(
                        self.snr2_smooth, self.voice2, self.voice2_time, now
                    )

                # Determine dominant source
                dominant = None
                if self.voice1 and not self.voice2:
                    dominant = 1
                elif self.voice2 and not self.voice1:
                    dominant = 2
                elif self.voice1 and self.voice2:
                    dominant = self.current_source

                # Switch if needed
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

