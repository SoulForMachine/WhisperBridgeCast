from app.server.transl_backends.base import TranslBase


class MarianMT(TranslBase):
    @classmethod
    def get_name(cls) -> str:
        return "MarianMT"

    def __init__(self, transl_params: dict, src_lang: str, target_lang: str):
        from transformers import MarianMTModel, MarianTokenizer

        language_pairs = {
            ("English", "Serbian Cyrillic"): "Helsinki-NLP/opus-mt-tc-base-en-sh",
            ("English", "Serbian Latin"): "Helsinki-NLP/opus-mt-tc-base-en-sh",
            ("English", "German"): "Helsinki-NLP/opus-mt-en-de",
            ("German", "English"): "Helsinki-NLP/opus-mt-de-en",
        }
        model_name = language_pairs.get((src_lang, target_lang))
        if model_name is None:
            raise ValueError(f"MarianMT does not support translation from {src_lang} to {target_lang}.")

        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.translator = MarianMTModel.from_pretrained(model_name).to("cuda")
        self.target_lang_token = {
            "Serbian Cyrillic": "srp_Cyrl",
            "Serbian Latin": "srp_Latn",
        }.get(target_lang, "")

    def translate_text(self, text: str) -> str:
        text_to_translate = f">>{self.target_lang_token}<< {text}" if self.target_lang_token else text
        inputs = self.tokenizer(text_to_translate, return_tensors="pt", truncation=True).to("cuda")
        translated = self.translator.generate(**inputs)
        transl_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        return transl_text
