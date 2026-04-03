import sys


class ASRBase:
    sep = " "   # join transcribe words with this character (" " for whisper_timestamped,
                # "" for faster-whisper because it emits the spaces when neeeded)

    def __init__(
        self,
        lan,
        modelsize=None,
        cache_dir=None,
        model_dir=None,
        logfile=sys.stderr,
        no_speech_prob_threshold=0.9,
        device="cuda",
        compute_type="float32",
    ):
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
