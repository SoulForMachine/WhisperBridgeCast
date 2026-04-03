import os

from app.server.transl_backends.base import TranslBase


class GoogleGemini(TranslBase):
    @classmethod
    def get_name(cls) -> str:
        return "Google Gemini"

    def __init__(self, transl_params: dict, src_lang: str, target_lang: str):
        from google import genai
        from google.genai import types

        self.genai = genai
        self.types = types

        transl_params = transl_params or {}
        api_key = transl_params.get("api_key", "")
        os.environ["GEMINI_API_KEY"] = api_key
        self.client = self.genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.src_lang = src_lang
        self.target_lang = target_lang

    def translate_text(self, text: str) -> str:
        try:
            prompt = f"Translate the following text in {self.src_lang} into {self.target_lang}:\n\"{text}\""
            max_output_tokens = max(2048, len(text.split()) * 10)
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=self.types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=max_output_tokens,
                ),
            )
            transl_text = getattr(response, "text", None)
            if transl_text is None:
                reason = getattr(response.candidates[0], "finish_reason", None)
                return f"[Translation Error]: {reason}."
            return transl_text.strip()
        except Exception as e:
            return f"[Translation Exception]: {e}."
