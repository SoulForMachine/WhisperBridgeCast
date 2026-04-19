
LANG_NAME_TO_CODE = {
    "English": "en",
    "German": "de",
    "Serbian": "sr",
}
LANG_CODE_TO_NAME = {v: k for k, v in LANG_NAME_TO_CODE.items()}

def get_lang_name(lang_code: str, default: str = None) -> str:
    if default is None:
        default = lang_code
    return LANG_CODE_TO_NAME.get(lang_code, default)

def get_lang_code(lang_name: str, default: str = None) -> str:
    if default is None:
        default = lang_name
    return LANG_NAME_TO_CODE.get(lang_name, default)

def get_lang_name_list() -> list[str]:
    return list(LANG_NAME_TO_CODE.keys())

TARGET_LANG_NAME_TO_CODE = {
    "Serbian Cyrillic": "sr-Cyrl",
    "Serbian Latin": "sr-Latn",
    "English": "en",
    "German": "de",
}
TARGET_LANG_CODE_TO_NAME = {v: k for k, v in TARGET_LANG_NAME_TO_CODE.items()}

def get_target_lang_name(lang_code: str, default: str = None) -> str:
    if default is None:
        default = lang_code
    return TARGET_LANG_CODE_TO_NAME.get(lang_code, default)

def get_target_lang_code(lang_name: str, default: str = None) -> str:
    if default is None:
        default = lang_name
    return TARGET_LANG_NAME_TO_CODE.get(lang_name, default)

def get_target_lang_name_list() -> list[str]:
    return list(TARGET_LANG_NAME_TO_CODE.keys())
