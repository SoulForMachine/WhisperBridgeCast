from abc import ABC, abstractmethod


class TranslBase(ABC):
    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Return the canonical backend name used by translation_engine config."""

    @abstractmethod
    def translate_text(self, text: str) -> str:
        """Translate input text and return translated text."""
