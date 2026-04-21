import multiprocessing as mp
from dataclasses import asdict, fields, is_dataclass
from typing import TypeVar


T = TypeVar("T")


def str_to_float(s):
    try:
        f = float(s)
        return f, True  # conversion successful
    except ValueError:
        return None, False  # invalid float


def str_to_int(s):
    try:
        i = int(s)
        return i, True  # conversion successful
    except ValueError:
        return None, False  # invalid int


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def merge_dataclass_from_dict(instance, values: dict):
    """Recursively updates a dataclass instance from a nested dictionary."""
    if not isinstance(values, dict):
        return instance

    for f in fields(instance):
        if f.name not in values:
            continue

        incoming = values[f.name]
        current = getattr(instance, f.name)

        if is_dataclass(current) and isinstance(incoming, dict):
            merge_dataclass_from_dict(current, incoming)
        else:
            setattr(instance, f.name, incoming)

    return instance


def dataclass_from_dict(dataclass_type: type[T], values: dict | None) -> T:
    """Creates and populates a dataclass instance from a dictionary."""
    return merge_dataclass_from_dict(dataclass_type(), values or {})


def settings_to_dict(instance) -> dict:
    return asdict(instance)


class MPCountingQueue:
    def __init__(self):
        self.q = mp.Queue()
        self.counter = mp.Value("i", 0)

    def put(self, item):
        self.q.put(item)
        with self.counter.get_lock():
            self.counter.value += 1

    def get(self, block=True, timeout=None):
        item = self.q.get(block=block, timeout=timeout)
        with self.counter.get_lock():
            self.counter.value -= 1
        return item

    def qsize(self):
        return self.counter.value


__all__ = [
    "str_to_float",
    "str_to_int",
    "MPCountingQueue",
    "clamp",
    "merge_dataclass_from_dict",
    "dataclass_from_dict",
    "settings_to_dict",
]