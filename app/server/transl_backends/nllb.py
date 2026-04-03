from app.server.transl_backends.base import TranslBase


class NLLB(TranslBase):
    @classmethod
    def get_name(cls) -> str:
        return "NLLB"

    def __init__(self, transl_params: dict, src_lang: str, target_lang: str):
        from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, NllbTokenizer

        self.language_codes = {
            "English": "eng_Latn",
            "Serbian Cyrillic": "srp_Cyrl",
            "German": "deu_Latn",
        }
        self.target_lang_token = self.language_codes[target_lang]

        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        model_name = "facebook/nllb-200-distilled-600M"
        self.tokenizer = NllbTokenizer.from_pretrained(model_name, src_lang=self.language_codes[src_lang])
        self.forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.target_lang_token)
        self.translator = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

    def translate_text(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        inputs = inputs.to(self.translator.device)
        translated_tokens = self.translator.generate(**inputs, forced_bos_token_id=self.forced_bos_token_id)
        translation = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translation
