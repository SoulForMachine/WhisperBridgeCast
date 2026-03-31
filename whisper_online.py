#!/usr/bin/env python3
import string
import sys
import numpy as np
from functools import lru_cache
import time
import logging

import io
import soundfile as sf
import math

logger = logging.getLogger(__name__)

@lru_cache(10**6)
def load_audio(fname):
    import librosa
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a

def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg*16000)
    end_s = int(end*16000)
    return audio[beg_s:end_s]


# Whisper backend

class ASRBase:

    sep = " "   # join transcribe words with this character (" " for whisper_timestamped,
                # "" for faster-whisper because it emits the spaces when neeeded)

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr, no_speech_prob_threshold=0.9, device="cuda", compute_type="float32"):
        self.logfile = logfile

        self.transcribe_kargs = {}
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan

        self.no_speech_prob_threshold = no_speech_prob_threshold
        self.device = device
        self.compute_type = compute_type

        self.model = self.load_model(modelsize, cache_dir, model_dir)

    def load_model(self, modelsize, cache_dir):
        raise NotImplemented("must be implemented in the child class")

    def transcribe(self, audio, init_prompt=""):
        raise NotImplemented("must be implemented in the child class")

    def use_vad(self, params=None):
        raise NotImplemented("must be implemented in the child class")


class WhisperTimestampedASR(ASRBase):
    """Uses whisper_timestamped library as the backend. Initially, we tested the code on this backend. It worked, but slower than faster-whisper.
    On the other hand, the installation for GPU could be easier.
    """

    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        import whisper
        import whisper_timestamped
        from whisper_timestamped import transcribe_timestamped
        self.transcribe_timestamped = transcribe_timestamped
        if model_dir is not None:
            logger.debug("ignoring model_dir, not implemented")
        return whisper.load_model(modelsize, download_root=cache_dir)

    def transcribe(self, audio, init_prompt=""):
        result = self.transcribe_timestamped(self.model,
                audio, language=self.original_language,
                initial_prompt=init_prompt, verbose=None,
                condition_on_previous_text=True, **self.transcribe_kargs)
        return result
 
    def ts_words(self,r):
        # return: transcribe result object to [(beg,end,"word1"), ...]
        o = []
        for s in r["segments"]:
            for w in s["words"]:
                t = (w["start"],w["end"],w["text"])
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s["end"] for s in res["segments"]]

    def use_vad(self, params=None):
        self.transcribe_kargs["vad"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class FasterWhisperASR(ASRBase):
    """Uses faster-whisper library as the backend. Works much faster, appx 4-times (in offline mode). For GPU, it requires installation with a specific CUDNN version.
    """

    sep = ""

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel
        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}. modelsize and cache_dir parameters are not used.")
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("modelsize or model_dir parameter must be set")

        import torch
        device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            logger.info("CUDA not available, using CPU")
            device = "cpu"
        model = WhisperModel(model_size_or_path, device=device, compute_type=self.compute_type, download_root=cache_dir)

        return model

    def transcribe(self, audio, init_prompt=""):
        # tested: beam_size=5 is faster and better than 1 (on one 200 second document from En ESIC, min chunk 0.01)
        segments, info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            **self.transcribe_kargs
        )

        return list(segments)

    def ts_words(self, segments):
        o = []
        for segment in segments:
            for word in segment.words:
                if segment.no_speech_prob > self.no_speech_prob_threshold:
                    logger.info(
                        f"Skipping word {word.word} "
                        f"(no_speech_prob={segment.no_speech_prob} > "
                        f"threshold={self.no_speech_prob_threshold}) "
                        "because it's in a no-speech segment"
                    )
                    continue
                # not stripping the spaces -- should not be merged with them!
                w = word.word
                t = (word.start, word.end, w)
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s.end for s in res]

    def use_vad(self, params=None):
        self.transcribe_kargs["vad_filter"] = True
        self.transcribe_kargs["vad_parameters"] = params

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"

class MLXWhisper(ASRBase):
    """
    Uses MLX Whisper library as the backend, optimized for Apple Silicon.
    Models available: https://huggingface.co/collections/mlx-community/whisper-663256f9964fbb1177db93dc
    Significantly faster than faster-whisper (without CUDA) on Apple M1. 
    """

    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        """
            Loads the MLX-compatible Whisper model.

            Args:
                modelsize (str, optional): The size or name of the Whisper model to load. 
                    If provided, it will be translated to an MLX-compatible model path using the `translate_model_name` method.
                    Example: "large-v3-turbo" -> "mlx-community/whisper-large-v3-turbo".
                cache_dir (str, optional): Path to the directory for caching models. 
                    **Note**: This is not supported by MLX Whisper and will be ignored.
                model_dir (str, optional): Direct path to a custom model directory. 
                    If specified, it overrides the `modelsize` parameter.
        """
        from mlx_whisper.transcribe import ModelHolder, transcribe
        import mlx.core as mx # Is installed with mlx-whisper
        
        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}. modelsize parameter is not used.")
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = self.translate_model_name(modelsize)
            logger.debug(f"Loading whisper model {modelsize}. You use mlx whisper, so {model_size_or_path} will be used.")
        
        self.model_size_or_path = model_size_or_path
        
        # Note: ModelHolder.get_model loads the model into a static class variable, 
        # making it a global resource. This means:
        # - Only one model can be loaded at a time; switching models requires reloading.
        # - This approach may not be suitable for scenarios requiring multiple models simultaneously,
        #   such as using whisper-streaming as a module with varying model sizes.
        dtype = mx.float16 # Default to mx.float16. In mlx_whisper.transcribe: dtype = mx.float16 if decode_options.get("fp16", True) else mx.float32
        ModelHolder.get_model(model_size_or_path, dtype) #Model is preloaded to avoid reloading during transcription
        
        return transcribe
    
    def translate_model_name(self, model_name):
        """
        Translates a given model name to its corresponding MLX-compatible model path.

        Args:
            model_name (str): The name of the model to translate.

        Returns:
            str: The MLX-compatible model path.
        """
        # Dictionary mapping model names to MLX-compatible paths
        model_mapping = {
            "tiny.en": "mlx-community/whisper-tiny.en-mlx",
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base.en": "mlx-community/whisper-base.en-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small.en": "mlx-community/whisper-small.en-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium.en": "mlx-community/whisper-medium.en-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large-v1": "mlx-community/whisper-large-v1-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "large": "mlx-community/whisper-large-mlx"
        }

        # Retrieve the corresponding MLX model path
        mlx_model_path = model_mapping.get(model_name)

        if mlx_model_path:
            return mlx_model_path
        else:
            raise ValueError(f"Model name '{model_name}' is not recognized or not supported.")
    
    def transcribe(self, audio, init_prompt=""):
        segments = self.model(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            word_timestamps=True,
            condition_on_previous_text=True,
            path_or_hf_repo=self.model_size_or_path,
            **self.transcribe_kargs
        )
        return segments.get("segments", [])


    def ts_words(self, segments):
        """
        Extract timestamped words from transcription segments and skips words with high no-speech probability.
        """
        return [
            (word["start"], word["end"], word["word"])
            for segment in segments
            for word in segment.get("words", [])
            if segment.get("no_speech_prob", 0) <= self.no_speech_prob_threshold
        ]
    
    def segments_end_ts(self, res):
        return [s['end'] for s in res]

    def use_vad(self, params=None):
        self.transcribe_kargs["vad_filter"] = True
        self.transcribe_kargs["vad_parameters"] = params

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"

class OpenaiApiASR(ASRBase):
    """Uses OpenAI's Whisper API for audio transcription."""

    def __init__(self, lan=None, temperature=0, logfile=sys.stderr, no_speech_prob_threshold=0.8):
        self.logfile = logfile

        self.modelname = "whisper-1"  
        self.original_language = None if lan == "auto" else lan # ISO-639-1 language code
        self.response_format = "verbose_json" 
        self.temperature = temperature

        self.load_model()

        self.use_vad_opt = False

        # reset the task in set_translate_task
        self.task = "transcribe"
        self.no_speech_prob_threshold = no_speech_prob_threshold

    def load_model(self, *args, **kwargs):
        from openai import OpenAI
        self.client = OpenAI()

        self.transcribed_seconds = 0  # for logging how many seconds were processed by API, to know the cost

    def ts_words(self, segments):
        no_speech_segments = []
        if self.use_vad_opt:
            for segment in segments.segments:
                # TODO: threshold can be set from outside
                if segment["no_speech_prob"] > self.no_speech_prob_threshold:
                    no_speech_segments.append((segment.get("start"), segment.get("end")))

        o = []
        for word in segments.words:
            start = word.start
            end = word.end
            if any(s[0] <= start <= s[1] for s in no_speech_segments):
                logger.info(f"Skipping word {word.word} because it's in a no-speech segment")
                continue
            o.append((start, end, word.word))
        return o

    def segments_end_ts(self, res):
        return [s.end for s in res.words]

    def transcribe(self, audio_data, prompt=None, *args, **kwargs):
        # Write the audio data to a buffer
        buffer = io.BytesIO()
        buffer.name = "temp.wav"
        sf.write(buffer, audio_data, samplerate=16000, format='WAV', subtype='PCM_16')
        buffer.seek(0)  # Reset buffer's position to the beginning

        self.transcribed_seconds += math.ceil(len(audio_data)/16000)  # it rounds up to the whole seconds

        params = {
            "model": self.modelname,
            "file": buffer,
            "response_format": self.response_format,
            "temperature": self.temperature,
            "timestamp_granularities": ["word", "segment"]
        }
        if self.task != "translate" and self.original_language:
            params["language"] = self.original_language
        if prompt:
            params["prompt"] = prompt

        if self.task == "translate":
            proc = self.client.audio.translations
        else:
            proc = self.client.audio.transcriptions

        # Process transcription/translation
        transcript = proc.create(**params)
        logger.debug(f"OpenAI API processed accumulated {self.transcribed_seconds} seconds")

        return transcript

    def use_vad(self, params=None):
        self.use_vad_opt = True

    def set_translate_task(self):
        self.task = "translate"


class HypothesisBuffer:

    def __init__(self, logfile=sys.stderr):
        self.committed_in_buffer = []
        self.buffer = []
        self.new = []

        self.last_committed_time = 0
        self.last_committed_word = None

        self.logfile = logfile

    def __repr__(self):
        return (
            f"committed_in_buf: {self.committed_in_buffer}  "
            f"buf: {self.buffer}  "
            f"new: {self.new}  "
            f"last_com_t: {self.last_committed_time}  "
            f"last_com_word: {self.last_committed_word}"
        )

    def insert(self, new, offset):
        # compare self.committed_in_buffer and new. It inserts only the words in new that extend the committed_in_buffer,
        # it means they are roughly behind last_committed_time and new in content the new tail is added to self.new

        new = [(a+offset,b+offset,t) for a,b,t in new]
        self.new = [(a,b,t) for a,b,t in new if a > self.last_committed_time-0.1]

        if len(self.new) >= 1:
            a,b,t = self.new[0]
            if abs(a - self.last_committed_time) < 1:
                if self.committed_in_buffer:
                    # it's going to search for 1, 2, ..., 5 consecutive words (n-grams) that are identical in committed and new. If they are, they're dropped.
                    cn = len(self.committed_in_buffer)
                    nn = len(self.new)
                    for i in range(1,min(min(cn,nn),5)+1):  # 5 is the maximum 
                        c = " ".join([self.committed_in_buffer[-j][2] for j in range(1,i+1)][::-1])
                        tail = " ".join(self.new[j-1][2] for j in range(1,i+1))
                        if c == tail:
                            words = []
                            for j in range(i):
                                words.append(repr(self.new.pop(0)))
                            words_msg = " ".join(words)
                            logger.debug(f"removing last {i} words: {words_msg}")
                            break

    def flush(self):
        # returns committed chunk = the longest common prefix of 2 last inserts. 

        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break

            if nt == self.buffer[0][2]:
                commit.append((na,nb,nt))
                self.last_committed_word = nt
                self.last_committed_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.committed_in_buffer.extend(commit)
        return commit

    def pop_committed(self, time):
        while self.committed_in_buffer and self.committed_in_buffer[0][1] <= time:
            self.committed_in_buffer.pop(0)

    def complete(self):
        return self.buffer

class OnlineASRProcessor:

    SAMPLING_RATE = 16000
    EMPTY_SEG = (None, None, "")

    def __init__(self, asr, tokenizer=None, buffer_trimming=("segment", 15), logfile=sys.stderr):
        """asr: WhisperASR object
        tokenizer: sentence tokenizer object for the target language. Must have a method *split* that behaves like the one of MosesTokenizer. It can be None, if "segment" buffer trimming option is used, then tokenizer is not used at all.
        ("segment", 15)
        buffer_trimming: a pair of (option, seconds), where option is either "sentence" or "segment", and seconds is a number. Buffer is trimmed if it is longer than "seconds" threshold. Default is the most recommended option.
        logfile: where to store the log. 
        """
        self.asr = asr
        self.tokenizer = tokenizer
        self.logfile = logfile
        self.last_inference_time = 0.0

        self.init()

        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming

    def init(self, offset=None):
        """run this when starting or restarting processing"""
        self.audio_buffer = np.array([],dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)
        self.buffer_time_offset = 0
        if offset is not None:
            self.buffer_time_offset = offset
        self.transcript_buffer.last_committed_time = self.buffer_time_offset
        self.committed = []

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        """Returns a tuple: (prompt, context), where "prompt" is a 200-character suffix of committed text that is inside of the scrolled away part of audio buffer. 
        "context" is the committed text that is inside the audio buffer. It is transcribed again and skipped. It is returned only for debugging and logging reasons.
        """
        k = max(0,len(self.committed)-1)
        while k > 0 and self.committed[k-1][1] > self.buffer_time_offset:
            k -= 1

        p = self.committed[:k]
        p = [t for _,_,t in p]
        prompt = []
        l = 0
        while p and l < 200:  # 200 characters prompt size
            x = p.pop(-1)
            l += len(x)+1
            prompt.append(x)
        non_prompt = self.committed[k:]
        return self.asr.sep.join(prompt[::-1]), self.asr.sep.join(t for _,_,t in non_prompt)

    def process_iter(self):
        """Runs on the current audio buffer.
        Returns: a tuple (beg_timestamp, end_timestamp, "text"), or (None, None, ""). 
        The non-emty text is confirmed (committed) partial transcript.
        """

        prompt, non_prompt = self.prompt()
        # Transcribing len(self.audio_buffer)/self.SAMPLING_RATE seconds from self.buffer_time_offset
        proc_start = time.perf_counter()
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)
        proc_end = time.perf_counter()
        self.last_inference_time = proc_end - proc_start

        # transform to [(beg,end,"word1"), ...]
        tsw = self.asr.ts_words(res)

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        out = self.transcript_buffer.flush()
        self.committed.extend(out)

        # there is a newly confirmed text

        if out and self.buffer_trimming_way == "sentence":  # trim the completed sentences
            if len(self.audio_buffer) / self.SAMPLING_RATE > self.buffer_trimming_sec:  # longer than this
                self.chunk_completed_sentence()

        if self.buffer_trimming_way == "segment":
            s = self.buffer_trimming_sec  # trim the completed segments longer than s,
        else:
            s = 30 # if the audio buffer is longer than 30s, trim it

        if len(self.audio_buffer) / self.SAMPLING_RATE > s:
            self.chunk_completed_segment(res)

        confirmed = self.to_flush(out)
        uc = self.to_flush(self.transcript_buffer.buffer)
        unconfirmed = (uc[0], uc[1], uc[2].rstrip(string.punctuation))
        return (confirmed, unconfirmed)

    def chunk_completed_sentence(self):
        if self.committed == []:
            return

        sents = self.words_to_sentences(self.committed)        
        if len(sents) < 2:
            return

        # we will continue with audio processing at this timestamp
        chunk_at = sents[-2][1]
        self.chunk_at(chunk_at)

    def chunk_completed_segment(self, res):
        if self.committed == []:
            return

        ends = self.asr.segments_end_ts(res)
        last_committed_end_time = self.committed[-1][1]

        if len(ends) > 1:
            e = ends[-2]+self.buffer_time_offset
            while len(ends) > 2 and e > last_committed_end_time:
                ends.pop(-1)
                e = ends[-2]+self.buffer_time_offset
            if e <= last_committed_end_time:
                self.chunk_at(e)

    def chunk_at(self, time):
        """trims the hypothesis and audio buffer at "time"
        """
        self.transcript_buffer.pop_committed(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds * self.SAMPLING_RATE):]
        self.buffer_time_offset = time

    def words_to_sentences(self, words):
        """Uses self.tokenizer for sentence segmentation of words.
        Returns: [(beg,end,"sentence 1"),...]
        """
        cwords = [w for w in words]
        t = self.asr.sep.join(o[2] for o in cwords)
        s = self.tokenizer.split(t)
        out = []
        while s:
            beg = None
            end = None
            sent = s.pop(0).strip()
            fsent = sent
            while cwords:
                b,e,w = cwords.pop(0)
                w = w.strip()
                if beg is None and sent.startswith(w):
                    beg = b
                elif end is None and sent == w:
                    end = e
                    out.append((beg,end,fsent))
                    break
                sent = sent[len(w):].strip()
        return out

    def finish(self):
        """Flush the incomplete text when the whole processing ends.
        Returns: the same format as self.process_iter()
        """
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        if f[2] and not f[2].endswith(('.', '!', '?')):
            f = (f[0], f[1], f[2] + "...")
        self.buffer_time_offset += len(self.audio_buffer) / self.SAMPLING_RATE
        return (f, self.EMPTY_SEG)


    def to_flush(self, sents, sep=None, offset=0, ):
        # concatenates the timestamped words or sentences into one sequence that is flushed in one line
        # sents: [(beg1, end1, "sentence1"), ...] or [] if empty
        # return: (beg1,end-of-last-sentence,"concatenation of sentences") or (None, None, "") if empty
        if sep is None:
            sep = self.asr.sep
        t = sep.join(s[2] for s in sents)
        if len(sents) == 0:
            b = None
            e = None
        else:
            b = offset + sents[0][0]
            e = offset + sents[-1][1]
        return (b,e,t)
    
    def get_last_inference_time(self):
        return self.last_inference_time

class VACOnlineASRProcessor(OnlineASRProcessor):
    '''Wraps OnlineASRProcessor with VAC (Voice Activity Controller). 

    It works the same way as OnlineASRProcessor: it receives chunks of audio (e.g. 0.04 seconds), 
    it runs VAD and continuously detects whether there is speech or not. 
    When it detects end of speech (non-voice for 500ms), it makes OnlineASRProcessor to end the utterance immediately.
    '''

    def __init__(self, online_chunk_size, vad_threshold, vad_min_silence_duration_ms, vad_speech_pad_ms, *a, **kw):
        self.online_chunk_size = online_chunk_size

        self.online = OnlineASRProcessor(*a, **kw)

        # VAC:
        import torch
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
        from silero_vad_iterator import FixedVADIterator
        self.vac = FixedVADIterator(model, threshold=vad_threshold, min_silence_duration_ms=vad_min_silence_duration_ms, speech_pad_ms=vad_speech_pad_ms)

        self.logfile = self.online.logfile
        self.init()

    def init(self):
        self.online.init()
        self.vac.reset_states()
        self.current_online_chunk_buffer_size = 0

        self.is_currently_final = False

        self.status = None  # or "voice" or "nonvoice"
        self.audio_buffer = np.array([],dtype=np.float32)
        self.buffer_offset = 0  # in frames

    def clear_buffer(self):
        self.buffer_offset += len(self.audio_buffer)
        self.audio_buffer = np.array([],dtype=np.float32)

    def insert_audio_chunk(self, audio):
        res = self.vac(audio)
        self.audio_buffer = np.append(self.audio_buffer, audio)

        if res is not None:
            frame = list(res.values())[0]-self.buffer_offset
            if 'start' in res and 'end' not in res:
                self.status = 'voice'
                send_audio = self.audio_buffer[frame:]
                self.online.init(offset=(frame+self.buffer_offset)/self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.clear_buffer()
            elif 'end' in res and 'start' not in res:
                self.status = 'nonvoice'
                send_audio = self.audio_buffer[:frame]
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
            else:
                beg = res["start"]-self.buffer_offset
                end = res["end"]-self.buffer_offset
                self.status = 'nonvoice'
                send_audio = self.audio_buffer[beg:end]
                self.online.init(offset=(beg+self.buffer_offset)/self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
        else:
            if self.status == 'voice':
                self.online.insert_audio_chunk(self.audio_buffer)
                self.current_online_chunk_buffer_size += len(self.audio_buffer)
                self.clear_buffer()
            else:
                # Keep at least the last second of silence, and up to configured VAD speech padding;
                # VAD may later find start of voice in it.
                # Trim anything older to avoid unbounded growth.
                pad_samples = max(self.SAMPLING_RATE, int(math.ceil(self.vac.speech_pad_samples)))
                trim = max(0, len(self.audio_buffer) - pad_samples)
                self.buffer_offset += trim
                self.audio_buffer = self.audio_buffer[-pad_samples:]

    def process_iter(self):
        if self.is_currently_final:
            return self.finish() + ("flush",)
        elif self.current_online_chunk_buffer_size >= self.SAMPLING_RATE * self.online_chunk_size:
            self.current_online_chunk_buffer_size = 0
            ret = self.online.process_iter()
            return ret + ("inference",)
        else:
            return (self.EMPTY_SEG, self.EMPTY_SEG, "vad_only")

    def finish(self):
        ret = self.online.finish()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        return ret

    def get_last_inference_time(self):
        return self.online.get_last_inference_time()



WHISPER_LANG_CODES = "af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he,hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk,ur,uz,vi,yi,yo,zh".split(",")

def create_tokenizer(lan):
    """returns an object that has split function that works like the one of MosesTokenizer"""

    assert lan in WHISPER_LANG_CODES, "language must be Whisper's supported lang code: " + " ".join(WHISPER_LANG_CODES)

    import spacy
    class SpacyTok:
        def __init__(self):
            nlp = self.get_nlp(lan)
            nlp.add_pipe("sentencizer")
            self.nlp = nlp

        def get_nlp(self, lang_code: str) -> spacy.language.Language:
            try:
                from spacy.util import get_lang_class
                get_lang_class(lang_code)
                return spacy.blank(lang_code)
            except ImportError:
                return spacy.blank("xx")  # fallback

        def split(self, sent):
            doc = self.nlp(sent)
            return [s.text for s in doc.sents]

    return SpacyTok()

def asr_factory(args, logfile=sys.stderr):
    """
    Creates and configures an ASR and ASR Online instance based on the specified backend and arguments.
    """
    backend = args.backend
    if backend == "openai-api":
        logger.debug("Using OpenAI API.")
        nsp_threshold = args.nsp_threshold if args.nsp_threshold else 0.8
        asr = OpenaiApiASR(lan=args.language, no_speech_prob_threshold=nsp_threshold)
    else:
        if backend == "faster-whisper":
            asr_cls = FasterWhisperASR
            nsp_threshold = args.nsp_threshold if args.nsp_threshold else 0.9
        elif backend == "mlx-whisper":
            asr_cls = MLXWhisper
            nsp_threshold = args.nsp_threshold if args.nsp_threshold else 0.9
        else:
            asr_cls = WhisperTimestampedASR

        # Only for FasterWhisperASR and WhisperTimestampedASR
        size = args.model
        logger.info(
            f"Loading Whisper {size} model...\n"
            f"\t  device: {args.whisper_device}\n"
            f"\t  compute type: {args.whisper_compute_type}\n"
            f"\t  nsp threshold: {nsp_threshold}\n"
            f"\t  language: {args.language}\n"
            f"\t  task: {args.task}"
        )
        t = time.time()
        asr = asr_cls(
            modelsize=size,
            lan=args.language,
            cache_dir=args.model_cache_dir,
            model_dir=args.model_dir,
            no_speech_prob_threshold=nsp_threshold,
            device=args.whisper_device,
            compute_type=args.whisper_compute_type
        )
        e = time.time()
        logger.info(f"done. It took {round(e-t,2)} seconds.")

    # Apply common configurations
    if getattr(args, 'whisper_vad', False):  # Checks if VAD argument is present and True
        logger.info("Setting VAD filter")
        asr.use_vad(dict(
            threshold=args.vad_threshold,
            neg_threshold=None,
            min_speech_duration_ms=0,
            max_speech_duration_s=float("inf"),
            min_silence_duration_ms=args.vad_min_silence_duration_ms,
            speech_pad_ms=args.vad_speech_pad_ms,
        ))

    language = args.language
    if args.task == "translate":
        asr.set_translate_task()
        tgt_language = "en"  # Whisper translates into English
    else:
        tgt_language = language  # Whisper transcribes in this language

    # Create the tokenizer
    if args.buffer_trimming == "sentence":
        tokenizer = create_tokenizer(tgt_language)
    else:
        tokenizer = None

    # Create the OnlineASRProcessor
    if args.vac:
        online = VACOnlineASRProcessor(
            args.vac_min_chunk_size,
            args.vad_threshold,
            args.vad_min_silence_duration_ms,
            args.vad_speech_pad_ms,
            asr,
            tokenizer,
            logfile=logfile,
            buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec)
        )
    else:
        online = OnlineASRProcessor(
            asr,
            tokenizer,
            logfile=logfile,
            buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec)
        )

    return asr, online

def set_logging(log_level):
    logger.setLevel(log_level)
