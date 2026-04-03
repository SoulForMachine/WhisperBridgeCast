import importlib
import inspect
import pkgutil

from app.server.transl_backends.base import TranslBase
from app.server.transl_backends.euro_llm import EuroLLM
from app.server.transl_backends.google_gemini import GoogleGemini
from app.server.transl_backends.marian_mt import MarianMT
from app.server.transl_backends.nllb import NLLB
from app.server.transl_backends.online_translators import LIBRE_MIRRORS, OnlineProviders, OnlineTranslators


def discover_backend_classes() -> tuple[type[TranslBase], ...]:
    classes: list[type[TranslBase]] = []
    seen: set[type[TranslBase]] = set()

    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name.startswith("_"):
            continue

        module = importlib.import_module(f"{__name__}.{module_info.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj is TranslBase:
                continue
            if issubclass(obj, TranslBase) and obj not in seen:
                seen.add(obj)
                classes.append(obj)

    return tuple(classes)

__all__ = [
    "TranslBase",
    "MarianMT",
    "NLLB",
    "EuroLLM",
    "GoogleGemini",
    "OnlineProviders",
    "LIBRE_MIRRORS",
    "OnlineTranslators",
    "discover_backend_classes",
]
