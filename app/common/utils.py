import multiprocessing as mp


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


__all__ = ["str_to_float", "str_to_int", "MPCountingQueue", "clamp"]