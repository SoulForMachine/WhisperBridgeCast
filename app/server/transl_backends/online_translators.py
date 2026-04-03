from app.server.transl_backends.base import TranslBase


class OnlineProviders:
    GOOGLE = "Google"
    MYMEMORY = "MyMemory"
    DEEPL = "DeepL"
    MICROSOFT = "Microsoft"
    LIBRE = "Libre"
    CHATGPT = "ChatGpt"
    BAIDU = "Baidu"
    PAPAGO = "Papago"
    QCRI = "QCRI"
    YANDEX = "Yandex"

    API_KEY_REQUIRED = {DEEPL, MICROSOFT, CHATGPT, QCRI, YANDEX}


LIBRE_MIRRORS = {
    "libretranslate.com": {
        "url": "https://libretranslate.com/",
        "api_key_required": True,
    },
    "libretranslate.de": {
        "url": "https://de.libretranslate.com/",
        "api_key_required": True,
    },
    "libretranslate.org": {
        "url": "https://libretranslate.org/",
        "api_key_required": False,
    },
    "translate.cutie.dating": {
        "url": "https://translate.cutie.dating/",
        "api_key_required": False,
    },
    "translate.terraprint.co": {
        "url": "https://translate.terraprint.co/",
        "api_key_required": False,
    },
    "translate.fedilab.app": {
        "url": "https://translate.fedilab.app/",
        "api_key_required": False,
    },
}


class OnlineTranslators(TranslBase):
    @classmethod
    def get_name(cls) -> str:
        return "Online Translators"

    def __init__(self, transl_params: dict, src_lang: str, target_lang: str):
        from deep_translator import (
            BaiduTranslator,
            ChatGptTranslator,
            DeeplTranslator,
            GoogleTranslator,
            LibreTranslator,
            MicrosoftTranslator,
            MyMemoryTranslator,
            PapagoTranslator,
            QcriTranslator,
            YandexTranslator,
        )

        provider = transl_params.get("provider", "") if transl_params else ""
        self.provider = provider or OnlineProviders.GOOGLE
        self.translator = None
        self.domain = None

        transl_params = transl_params or {}
        api_key = transl_params.get("api_key", "")
        api_secret = transl_params.get("api_secret", "")
        client_id = transl_params.get("client_id", "")
        region = transl_params.get("region", "")
        self.domain = transl_params.get("domain", "")
        libre_mirror = transl_params.get("libre_mirror", "libretranslate.com")

        src_code = self._lang_to_code(src_lang, self.provider)
        target_code = self._lang_to_code(target_lang, self.provider)

        match self.provider:
            case OnlineProviders.GOOGLE:
                self.translator = GoogleTranslator(source=src_code, target=target_code)
            case OnlineProviders.MYMEMORY:
                self.translator = MyMemoryTranslator(source=src_code, target=target_code)
            case OnlineProviders.DEEPL:
                self.translator = DeeplTranslator(source=src_code, target=target_code, api_key=api_key)
            case OnlineProviders.MICROSOFT:
                self.translator = MicrosoftTranslator(source=src_code, target=target_code, api_key=api_key, region=region or None)
            case OnlineProviders.LIBRE:
                mirror_cfg = LIBRE_MIRRORS.get(libre_mirror, LIBRE_MIRRORS["libretranslate.com"])
                mirror_requires_key = mirror_cfg["api_key_required"]
                self.translator = LibreTranslator(
                    source=src_code,
                    target=target_code,
                    api_key=(api_key if mirror_requires_key else "1234"),
                    custom_url=mirror_cfg["url"],
                    use_free_api=not mirror_requires_key,
                )
            case OnlineProviders.CHATGPT:
                self.translator = ChatGptTranslator(
                    source=self._lang_to_name(src_lang),
                    target=self._lang_to_name(target_lang),
                    api_key=api_key,
                )
            case OnlineProviders.BAIDU:
                self.translator = BaiduTranslator(source=src_code, target=target_code, appid=client_id, appkey=api_secret)
            case OnlineProviders.PAPAGO:
                self.translator = PapagoTranslator(source=src_code, target=target_code, client_id=client_id, secret_key=api_secret)
            case OnlineProviders.QCRI:
                self.translator = QcriTranslator(source=src_code, target=target_code, api_key=api_key)
                if not self.domain:
                    self.domain = "general"
            case OnlineProviders.YANDEX:
                self.translator = YandexTranslator(source=src_code, target=target_code, api_key=api_key)
            case _:
                raise ValueError(f"Unknown online translator provider: {self.provider}")

    @staticmethod
    def _lang_to_code(lang: str, provider: str) -> str:
        match provider:
            case OnlineProviders.MYMEMORY:
                return {
                    "English": "en-US",
                    "German": "de-DE",
                    "Serbian Latin": "sr-Latn-RS",
                    "Serbian Cyrillic": "sr-Cyrl-RS",
                    "Serbian": "sr-Latn-RS",
                }.get(lang, "auto")
            case OnlineProviders.MICROSOFT:
                return {
                    "English": "en",
                    "German": "de",
                    "Serbian Latin": "sr-latn",
                    "Serbian Cyrillic": "sr-cyrl",
                    "Serbian": "sr-latn",
                }.get(lang, "auto")

        return {
            "English": "en",
            "German": "de",
            "Serbian": "sr",
            "Serbian Latin": "sr",
            "Serbian Cyrillic": "sr",
        }.get(lang, "auto")

    @staticmethod
    def _lang_to_name(lang: str) -> str:
        return {
            "English": "english",
            "German": "german",
            "Serbian": "serbian",
            "Serbian Latin": "serbian",
            "Serbian Cyrillic": "serbian",
        }.get(lang, "auto")

    def translate_text(self, text: str) -> str:
        if self.provider == OnlineProviders.QCRI:
            result = self.translator.translate(text, domain=self.domain)
        else:
            result = self.translator.translate(text)

        if isinstance(result, list):
            return "\n".join(str(x) for x in result)
        return result
