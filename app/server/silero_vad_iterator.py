import torch

# This is copied from silero-vad's vad_utils.py:
# https://github.com/snakers4/silero-vad/blob/94811cbe1207ec24bc0f5370b895364b8934936f/src/silero_vad/utils_vad.py#L398C1-L489C20
# (except changed defaults)

# Their licence is MIT, same as ours: https://github.com/snakers4/silero-vad/blob/94811cbe1207ec24bc0f5370b895364b8934936f/LICENSE

class VADIterator:
    def __init__(self,
                 model,
                 threshold: float = 0.5,
                 start_threshold: float | None = None,
                 end_threshold: float | None = None,
                 hysteresis_gap: float = 0.15,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 500,  # makes sense on one recording that I checked
                 speech_pad_ms: int = 100,            # symmetric pad kept for backward-compat
                 speech_pad_start_ms: int | None = None,
                 speech_pad_end_ms: int | None = None,
                 hangover_frames: int = 0,
                 ):

        """
        Class for stream imitation

        Parameters
        ----------
        model: preloaded .jit/.onnx silero VAD model

        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 sample rates

        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it

        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        """

        self.model = model

        if start_threshold is None:
            start_threshold = threshold
        if end_threshold is None:
            end_threshold = start_threshold - hysteresis_gap

        if not (0.0 <= end_threshold <= 1.0 and 0.0 <= start_threshold <= 1.0):
            raise ValueError("start_threshold and end_threshold must be in [0, 1]")
        if end_threshold > start_threshold:
            raise ValueError("end_threshold must be <= start_threshold")

        if speech_pad_start_ms is None:
            speech_pad_start_ms = speech_pad_ms
        if speech_pad_end_ms is None:
            speech_pad_end_ms = speech_pad_ms

        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.hangover_frames = max(0, int(hangover_frames))
        self.sampling_rate = sampling_rate

        if sampling_rate not in [8000, 16000]:
            raise ValueError('VADIterator does not support sampling rates other than [8000, 16000]')

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_start_samples = sampling_rate * speech_pad_start_ms / 1000
        self.speech_pad_end_samples = sampling_rate * speech_pad_end_ms / 1000
        self.reset_states()

    def reset_states(self):

        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.low_prob_frames = 0
        self.current_sample = 0

    @torch.no_grad()
    def __call__(self, x, return_seconds=False, time_resolution: int = 1):
        """
        x: torch.Tensor
            audio chunk (see examples in repo)

        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)

        time_resolution: int (default - 1)
            time resolution of speech coordinates when requested as seconds
        """

        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if (speech_prob >= self.start_threshold) and self.temp_end:
            self.temp_end = 0
            self.low_prob_frames = 0

        if (speech_prob >= self.start_threshold) and not self.triggered:
            self.triggered = True
            self.low_prob_frames = 0
            speech_start = max(0, self.current_sample - self.speech_pad_start_samples - window_size_samples)
            return {'start': int(speech_start) if not return_seconds else round(speech_start / self.sampling_rate, time_resolution)}

        if self.triggered and (speech_prob < self.end_threshold):
            self.low_prob_frames += 1
            if not self.temp_end:
                self.temp_end = self.current_sample

            if self.low_prob_frames <= self.hangover_frames:
                return None

            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                speech_end = self.temp_end + self.speech_pad_end_samples - window_size_samples
                self.temp_end = 0
                self.low_prob_frames = 0
                self.triggered = False
                return {'end': int(speech_end) if not return_seconds else round(speech_end / self.sampling_rate, time_resolution)}

        if self.triggered:
            self.low_prob_frames = 0

        return None

    def flush(self, return_seconds=False, time_resolution: int = 1, tail_samples: int = 0):
        """Force-close an active speech segment at stream end.

        tail_samples allows callers with internal buffering (e.g. FixedVADIterator)
        to include not-yet-processed trailing samples in the final end marker.
        """
        if tail_samples > 0:
            self.current_sample += int(tail_samples)

        if not self.triggered:
            return None

        speech_end = self.current_sample + self.speech_pad_end_samples
        self.temp_end = 0
        self.low_prob_frames = 0
        self.triggered = False

        return {
            'end': int(speech_end) if not return_seconds else round(speech_end / self.sampling_rate, time_resolution)
        }

#######################
# because Silero now requires exactly 512-sized audio chunks

import numpy as np
class FixedVADIterator(VADIterator):
    '''It fixes VADIterator by allowing to process any audio length, not only exactly 512 frames at once.
    If audio to be processed at once is long and multiple voiced segments detected,
    then __call__ returns the start of the first segment, and end (or middle, which means no end) of the last segment.
    '''

    def reset_states(self):
        super().reset_states()
        self.buffer = np.array([],dtype=np.float32)

    def __call__(self, x, return_seconds=False):
        self.buffer = np.append(self.buffer, x)
        ret = None
        while len(self.buffer) >= 512:
            r = super().__call__(self.buffer[:512], return_seconds=return_seconds)
            self.buffer = self.buffer[512:]
            if ret is None:
                ret = r
            elif r is not None:
                if 'end' in r:
                    ret['end'] = r['end']  # the latter end
                if 'start' in r and 'end' in ret:  # there is an earlier start.
                    # Remove end, merging this segment with the previous one.
                    del ret['end']
        return ret if ret != {} else None

    def flush(self, return_seconds=False, time_resolution: int = 1):
        tail_samples = len(self.buffer)
        self.buffer = np.array([], dtype=np.float32)
        return super().flush(
            return_seconds=return_seconds,
            time_resolution=time_resolution,
            tail_samples=tail_samples,
        )

if __name__ == "__main__":
    # test/demonstrate the need for FixedVADIterator:

    import torch
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad'
    )
    vac = FixedVADIterator(model)
#   vac = VADIterator(model)  # the second case crashes with this

    # this works: for both
    audio_buffer = np.array([0]*(512),dtype=np.float32)
    vac(audio_buffer)

    # this crashes on the non FixedVADIterator with
    # ops.prim.RaiseException("Input audio chunk is too short", "builtins.ValueError")
    audio_buffer = np.array([0]*(512-1),dtype=np.float32)
    vac(audio_buffer)
