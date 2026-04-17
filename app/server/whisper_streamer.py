#!/usr/bin/env python3
import string
import sys
import numpy as np
from functools import lru_cache
import time
import logging

import math

from app.server.asr_backends import (
    FasterWhisperASR,
    MLXWhisper,
    OpenaiApiASR,
    WhisperTimestampedASR,
)
from app.common.utils import clamp

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
    ROLL_AVG_SIZE = 10

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
        self.roll_avg_inference_time = 0.0
        self.roll_avg_buffer = []

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

        self._update_inference_time(proc_end - proc_start)

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
        return (confirmed, unconfirmed, "inference")

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

    def get_roll_avg_inference_time(self):
        return self.roll_avg_inference_time

    def _update_inference_time(self, new_time):
        self.last_inference_time = new_time
        self.roll_avg_buffer.append(new_time)
        if len(self.roll_avg_buffer) > self.ROLL_AVG_SIZE:
            self.roll_avg_buffer.pop(0)
        self.roll_avg_inference_time = sum(self.roll_avg_buffer) / len(self.roll_avg_buffer)

class VACOnlineASRProcessor(OnlineASRProcessor):
    '''Wraps OnlineASRProcessor with VAC (Voice Activity Controller).

    It works the same way as OnlineASRProcessor: it receives chunks of audio (e.g. 0.04 seconds),
    it runs VAD and continuously detects whether there is speech or not.
    When it detects end of speech (non-voice for 500ms), it makes OnlineASRProcessor to end the utterance immediately.
    '''

    def __init__(
        self,
        online_chunk_size,
        is_dynamic_chunk_size,
        vad_start_threshold,
        vad_end_threshold,
        vad_min_silence_duration_ms,
        vad_speech_pad_start_ms,
        vad_speech_pad_end_ms,
        vad_hangover_chunks,
        *a,
        **kw,
    ):
        self.online_chunk_size = online_chunk_size
        self.is_dynamic_chunk_size = is_dynamic_chunk_size

        self.online = OnlineASRProcessor(*a, **kw)

        # VAC:
        import torch
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
        from app.server.silero_vad_iterator import FixedVADIterator
        self.vac = FixedVADIterator(
            model,
            start_threshold=vad_start_threshold,
            end_threshold=vad_end_threshold,
            min_silence_duration_ms=vad_min_silence_duration_ms,
            speech_pad_start_ms=vad_speech_pad_start_ms,
            speech_pad_end_ms=vad_speech_pad_end_ms,
            hangover_chunks=vad_hangover_chunks,
        )

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
            buf_len = len(self.audio_buffer)
            frame = int(list(res.values())[0] - self.buffer_offset)

            if 'start' in res and 'end' not in res:
                self.status = 'voice'
                start_idx = clamp(frame, 0, buf_len)
                send_audio = self.audio_buffer[start_idx:]
                self.online.init(offset=(start_idx + self.buffer_offset) / self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.clear_buffer()

            elif 'end' in res and 'start' not in res:
                self.status = 'nonvoice'
                end_idx = clamp(frame, 0, buf_len)
                send_audio = self.audio_buffer[:end_idx]
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()

            else:
                beg = int(res["start"] - self.buffer_offset)
                end = int(res["end"] - self.buffer_offset)
                beg = clamp(beg, 0, buf_len)
                end = clamp(end, beg, buf_len)

                self.status = 'nonvoice'
                send_audio = self.audio_buffer[beg:end]
                self.online.init(offset=(beg + self.buffer_offset) / self.SAMPLING_RATE)
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
                pad_samples = max(self.SAMPLING_RATE, int(math.ceil(self.vac.speech_pad_start_samples)))
                trim = max(0, len(self.audio_buffer) - pad_samples)
                self.buffer_offset += trim
                self.audio_buffer = self.audio_buffer[-pad_samples:]

    def process_iter(self):
        if self.is_currently_final:
            confirmed, unconfirmed = self._finalize_online_segment()
            return (confirmed, unconfirmed, "flush")
        elif self.current_online_chunk_buffer_size >= self.SAMPLING_RATE * self.online_chunk_size:
            self.current_online_chunk_buffer_size = 0
            ret = self.online.process_iter()

            if self.is_dynamic_chunk_size:
                roll_avg_t = self.online.get_roll_avg_inference_time()
                self.online_chunk_size = clamp(roll_avg_t, 0.1, 3.0)

            return ret
        else:
            return (self.EMPTY_SEG, self.EMPTY_SEG, "vad_only")

    def _merge_segments(self, left, right):
        if not left[2]:
            return right
        if not right[2]:
            return left

        beg = left[0] if left[0] is not None else right[0]
        end = right[1] if right[1] is not None else left[1]
        sep = self.online.asr.sep if hasattr(self.online, "asr") else " "
        text = f"{left[2]}{sep}{right[2]}"
        return (beg, end, text)

    def _finalize_online_segment(self):
        pre_finish_confirmed = self.EMPTY_SEG

        # Only infer if new audio arrived since the last inference cycle.
        has_pending_audio = self.current_online_chunk_buffer_size > 0
        if has_pending_audio:
            pre_finish_confirmed, _, _ = self.online.process_iter()

        finish_confirmed, _ = self.online.finish()
        combined_confirmed = self._merge_segments(pre_finish_confirmed, finish_confirmed)

        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False

        # Flush/finalize must not emit unconfirmed tail.
        return combined_confirmed, self.EMPTY_SEG

    def finish(self):
        end_marker = self.vac.flush()

        if end_marker and "end" in end_marker and len(self.audio_buffer) > 0:
            frame = int(end_marker["end"] - self.buffer_offset)
            frame = max(0, min(frame, len(self.audio_buffer)))
            if frame > 0:
                self.online.insert_audio_chunk(self.audio_buffer[:frame])
                self.current_online_chunk_buffer_size += frame
            self.clear_buffer()

        if self.vac.triggered or self.status == "voice" or self.current_online_chunk_buffer_size > 0 or len(self.online.audio_buffer) > 0:
            return self._finalize_online_segment()

        ret = self.online.finish()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        return ret

    def get_last_inference_time(self):
        return self.online.get_last_inference_time()

    def get_roll_avg_inference_time(self):
        return self.online.get_roll_avg_inference_time()



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
            threshold=args.vad_start_threshold,
            neg_threshold=args.vad_end_threshold,
            min_speech_duration_ms=0,
            max_speech_duration_s=float("inf"),
            min_silence_duration_ms=args.vad_min_silence_duration_ms,
            speech_pad_ms=max(args.vad_speech_pad_start_ms, args.vad_speech_pad_end_ms),
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
            args.vac_is_dynamic_chunk_size,
            args.vad_start_threshold,
            args.vad_end_threshold,
            args.vad_min_silence_duration_ms,
            args.vad_speech_pad_start_ms,
            args.vad_speech_pad_end_ms,
            args.vad_hangover_chunks,
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
