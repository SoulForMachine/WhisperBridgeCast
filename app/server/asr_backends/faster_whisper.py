import logging

from app.server.asr_backends.base import ASRBase


logger = logging.getLogger(__name__)


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
            **self.transcribe_kargs,
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
