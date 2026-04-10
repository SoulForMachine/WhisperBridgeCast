import queue
import threading
import time
from typing import Any

import pytest

from app.server.translation import Translator
from app.server.transl_backends.base import TranslBase


class DummyBackend(TranslBase):
    @classmethod
    def get_name(cls) -> str:
        return "Dummy"

    def __init__(self, transl_params: dict[str, Any], src_lang: str, target_lang: str):
        self.transl_params = transl_params
        self.src_lang = src_lang
        self.target_lang = target_lang

    def translate_text(self, text: str) -> str:
        return f"T {text}"


def _drain_queue(q: queue.Queue) -> list[dict]:
    items: list[dict] = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            return items


def _enqueue_stream(tr: Translator, stream: list[tuple[str, str]], timeout: float = 2.0) -> None:
    for confirmed, unconfirmed in stream:
        tr.source_queue.put((confirmed, unconfirmed))

        deadline = time.monotonic() + timeout
        while not tr.source_queue.empty():
            if time.monotonic() > deadline:
                raise TimeoutError("Translator did not consume source queue in time")
            time.sleep(0.005)


def _collect_output(stop_event: threading.Event, out_q: queue.Queue, sink: list[dict]) -> None:
    while True:
        if stop_event.is_set() and out_q.empty():
            break
        try:
            sink.append(out_q.get(timeout=0.05))
        except queue.Empty:
            continue


def _build_translator(*, source_diff_enabled: bool = False, with_engine: bool = False) -> tuple[Translator, queue.Queue]:
    out_q: queue.Queue = queue.Queue()
    tr = Translator(
        engine_id="Dummy",
        transl_params={"source_diff_enabled": source_diff_enabled},
        src_lang="English",
        target_lang="German",
        source_queue=queue.Queue(),
        output_queues=[out_q],
        sender_queue=queue.Queue(),
        only_complete_sent=False,
    )
    tr.BACKEND_CLASSES = (DummyBackend,)

    if with_engine:
        tr.engine = DummyBackend({}, "English", "German")

    return tr, out_q


def test_source_diff_disabled_by_default() -> None:
    tr, _ = _build_translator(source_diff_enabled=False)
    assert tr.source_diff_enabled is False


def test_compute_diff_ops_keeps_middle_insert_and_filters_trailing_insert() -> None:
    nlp = Translator("Dummy", {}, "English", "German", queue.Queue(), [], queue.Queue(), False).src_nlp

    prev = [tok for tok in nlp("alpha beta gamma")]
    curr_middle = [tok for tok in nlp("alpha x beta gamma")]
    curr_tail = [tok for tok in nlp("alpha beta gamma x")]

    middle_ops = Translator._compute_diff_ops(
        prev,
        curr_middle,
        include_inserts=True,
        include_middle_inserts=True,
        include_trailing_inserts=False,
    )
    tail_ops = Translator._compute_diff_ops(
        prev,
        curr_tail,
        include_inserts=True,
        include_middle_inserts=True,
        include_trailing_inserts=False,
    )

    assert any(op["op"] == "+" for op in middle_ops)
    assert not any(op["op"] == "+" for op in tail_ops)


def test_source_diff_window_alignment_replaces_previous_unconfirmed() -> None:
    tr, _ = _build_translator(source_diff_enabled=True)
    nlp = tr.src_nlp

    # Prime previous state with one unconfirmed token: "test".
    tokens=[tok for tok in nlp("This is test")],
    conf_tokens=[tok for tok in nlp("This is")],
    first_sentence = Translator.Sentence(
        confirmed="This is",
        unconfirmed=" test",
        num_tokens=len(tokens),
        num_conf_tokens=len(conf_tokens),
        complete=False,
    )
    first_ops = tr._compute_source_diff_ops(first_sentence)
    assert first_ops == []

    # The compared window stays aligned; "test" -> "testing" should be replace.
    second_sentence = Translator.Sentence(
        confirmed="This is test",
        unconfirmed="ing",
        num_tokens=len([tok for tok in nlp("This is testing")]),
        num_conf_tokens=len([tok for tok in nlp("This is test")]),
        complete=False,
    )
    second_ops = tr._compute_source_diff_ops(second_sentence)
    assert any(op["op"] == "~" for op in second_ops)


def test_get_sentences_marks_last_confirmed_as_complete_when_token_arrays_match() -> None:
    tr, _ = _build_translator(source_diff_enabled=False)

    tr.confirmed_text = "Hello world."
    tr.unconfirmed_text = ""
    sentences = tr.get_sentences()

    assert sentences
    assert sentences[-1].complete is True
    assert tr.confirmed_text == ""


def test_get_sentences_marks_last_confirmed_as_partial_when_tokens_grow() -> None:
    tr, _ = _build_translator(source_diff_enabled=False)

    tr.confirmed_text = "Hello"
    tr.unconfirmed_text = " world"
    sentences = tr.get_sentences()

    assert sentences
    assert sentences[-1].complete is False
    assert sentences[-1].unconfirmed == " world"


def test_get_sentences_index_guard_when_combined_sentence_index_missing() -> None:
    tr, _ = _build_translator(source_diff_enabled=False)

    class FakeToken:
        def __init__(self, text: str):
            self.text = text

    class FakeSpan:
        def __init__(self, token_texts: list[str]):
            self._tokens = [FakeToken(text) for text in token_texts]
            self.text = " ".join(token_texts)

        def __iter__(self):
            return iter(self._tokens)

    class FakeDoc:
        def __init__(self, spans: list[FakeSpan]):
            self.sents = spans

        def __iter__(self):
            return iter(t for s in self.sents for t in s)

    def fake_src_nlp(text: str):
        if text == "A.":
            return FakeDoc([FakeSpan(["A", "."])])
        if text == "A. b":
            return FakeDoc([FakeSpan(["A", "."]), FakeSpan(["b"])])
        if text == "b":
            return FakeDoc([FakeSpan(["b"])])

        return FakeDoc([])

    tr.src_nlp = fake_src_nlp
    tr.confirmed_text = "A."
    tr.unconfirmed_text = " b"

    sentences = tr.get_sentences()

    assert sentences
    assert sentences[-1].complete is False
    assert sentences[-1].unconfirmed == "b"
    assert sentences[-1].num_tokens == 1
    assert sentences[-1].num_conf_tokens == 0


def test_get_sentences_marks_partial_when_same_length_tokens_differ() -> None:
    tr, _ = _build_translator(source_diff_enabled=False)

    class FakeToken:
        def __init__(self, text: str):
            self.text = text

    class FakeSpan:
        def __init__(self, token_texts: list[str], text: str):
            self._tokens = [FakeToken(tok) for tok in token_texts]
            self.text = text

        def __iter__(self):
            return iter(self._tokens)

    class FakeDoc:
        def __init__(self, spans: list[FakeSpan]):
            self.sents = spans

    def fake_src_nlp(text: str):
        if text == "A B.":
            return FakeDoc([FakeSpan(["A", "B", "."], "A B.")])
        if text == "A B. x":
            # Same token count as confirmed sentence, but one token differs.
            return FakeDoc([FakeSpan(["A", "C", "."], "A C.")])
        return FakeDoc([])

    tr.src_nlp = fake_src_nlp
    tr.confirmed_text = "A B."
    tr.unconfirmed_text = " x"

    sentences = tr.get_sentences()

    assert sentences
    assert sentences[-1].complete is False


def test_get_sentences_buffer_transitions_partial_then_complete() -> None:
    tr, _ = _build_translator(source_diff_enabled=False)

    tr.confirmed_text = "Hello"
    tr.unconfirmed_text = " world"
    first = tr.get_sentences()
    assert first and first[-1].complete is False
    assert tr.confirmed_text == "Hello"

    tr.confirmed_text = "Hello world."
    tr.unconfirmed_text = ""
    second = tr.get_sentences()
    assert second and second[-1].complete is True
    assert tr.confirmed_text == ""


def test_unconfirmed_only_sentence_is_lstripped_and_source_diff_offsets_align() -> None:
    tr, out_q = _build_translator(source_diff_enabled=True)

    tr.confirmed_text = ""
    tr.unconfirmed_text = "   hello"
    sent1 = tr.get_sentences()
    assert sent1 and sent1[0].unconfirmed == "hello"
    tr.translate_and_send(sent1[0])

    tr.confirmed_text = ""
    tr.unconfirmed_text = "   hello there"
    sent2 = tr.get_sentences()
    assert sent2 and sent2[0].unconfirmed == "hello there"
    tr.translate_and_send(sent2[0])

    msgs = _drain_queue(out_q)
    assert len(msgs) >= 2
    latest = msgs[-1]
    text = latest["orig_unconfirmed_text"]
    assert text == "hello there"

    for op in latest.get("source_diff", []):
        if op.get("op") in {"+", "~"}:
            start, end = op["span"]
            assert 0 <= start <= end <= len(text)
        if op.get("op") == "-":
            assert 0 <= op["idx"] <= len(text)


def test_target_diff_filters_partial_trailing_insert_and_keeps_middle_insert() -> None:
    tr, _ = _build_translator(source_diff_enabled=False)

    # Initial partial sets baseline only.
    ops0 = tr._compute_target_diff_ops("alpha beta", complete=False)
    assert ops0 == []

    # Middle insert in partial update should be kept.
    ops1 = tr._compute_target_diff_ops("alpha x beta", complete=False)
    assert any(op["op"] == "+" for op in ops1)

    # Trailing append in partial update should be filtered.
    ops2 = tr._compute_target_diff_ops("alpha x beta tail", complete=False)
    assert not any(op["op"] == "+" for op in ops2)

    # Trailing append is allowed once complete.
    ops3 = tr._compute_target_diff_ops("alpha x beta tail .", complete=True)
    assert any(op["op"] == "+" for op in ops3)


def test_compute_diff_ops_delete_index_branches() -> None:
    tr, _ = _build_translator(source_diff_enabled=False)
    nlp = tr.src_nlp

    # Branch: curr_text empty.
    prev_only = [tok for tok in nlp("alpha")]
    ops_empty_curr = Translator._compute_diff_ops(prev_only, [])
    assert any(op["op"] == "-" and op["idx"] == 0 for op in ops_empty_curr)

    # Branch: j1 == 0 (delete at front).
    prev_front_delete = [tok for tok in nlp("x a")]
    curr_front_delete = [tok for tok in nlp("a")]
    ops_front = Translator._compute_diff_ops(prev_front_delete, curr_front_delete)
    assert any(op["op"] == "-" and op["idx"] == 0 for op in ops_front)

    # Branch: j1 >= len(curr_text) (delete at end).
    prev_end_delete = [tok for tok in nlp("a x")]
    curr_end_delete = [tok for tok in nlp("a")]
    ops_end = Translator._compute_diff_ops(prev_end_delete, curr_end_delete)
    assert any(op["op"] == "-" and op["idx"] == 1 for op in ops_end)


def test_compute_diff_ops_insert_filtering_combinations() -> None:
    tr, _ = _build_translator(source_diff_enabled=False)
    nlp = tr.src_nlp

    prev = [tok for tok in nlp("a b c")]
    curr_middle = [tok for tok in nlp("a x b c")]
    curr_tail = [tok for tok in nlp("a b c y")]
    curr_both = [tok for tok in nlp("a x b c y")]

    ops_no_inserts = Translator._compute_diff_ops(
        prev,
        curr_middle,
        include_inserts=False,
        include_middle_inserts=True,
        include_trailing_inserts=True,
    )
    assert not any(op["op"] == "+" for op in ops_no_inserts)

    ops_trailing_only_middle = Translator._compute_diff_ops(
        prev,
        curr_middle,
        include_inserts=True,
        include_middle_inserts=False,
        include_trailing_inserts=True,
    )
    ops_trailing_only_tail = Translator._compute_diff_ops(
        prev,
        curr_tail,
        include_inserts=True,
        include_middle_inserts=False,
        include_trailing_inserts=True,
    )
    assert not any(op["op"] == "+" for op in ops_trailing_only_middle)
    assert any(op["op"] == "+" for op in ops_trailing_only_tail)

    ops_middle_only = Translator._compute_diff_ops(
        prev,
        curr_both,
        include_inserts=True,
        include_middle_inserts=True,
        include_trailing_inserts=False,
    )
    plus_ops = [op for op in ops_middle_only if op["op"] == "+"]
    assert plus_ops
    curr_both_text = "a x b c y"
    assert all(op["span"][1] < len(curr_both_text) for op in plus_ops)


def test_compute_diff_ops_replace_span_boundaries_with_punctuation() -> None:
    tr, _ = _build_translator(source_diff_enabled=False)
    nlp = tr.src_nlp

    curr_text_1 = "hello !"
    ops1 = Translator._compute_diff_ops(
        [tok for tok in nlp("hello .")],
        [tok for tok in nlp(curr_text_1)],
    )
    replace_ops1 = [op for op in ops1 if op["op"] == "~"]
    assert replace_ops1
    for op in replace_ops1:
        start, end = op["span"]
        assert 0 <= start < end <= len(curr_text_1)

    curr_text_2 = "U.S.A"
    ops2 = Translator._compute_diff_ops(
        [tok for tok in nlp("U.S.")],
        [tok for tok in nlp(curr_text_2)],
    )
    replace_ops2 = [op for op in ops2 if op["op"] == "~"]
    assert replace_ops2
    for op in replace_ops2:
        start, end = op["span"]
        assert 0 <= start < end <= len(curr_text_2)


def test_source_diff_state_resets_after_complete() -> None:
    tr, _ = _build_translator(source_diff_enabled=True)
    nlp = tr.src_nlp

    s1 = Translator.Sentence(
        confirmed="This is",
        unconfirmed=" test",
        num_tokens=len([tok for tok in nlp("This is test")]),
        num_conf_tokens=len([tok for tok in nlp("This is")]),
        complete=False,
    )
    assert tr._compute_source_diff_ops(s1) == []
    assert tr._prev_source_unconf_tokens

    s2 = Translator.Sentence(
        confirmed="This is test.",
        unconfirmed="",
        num_tokens=len([tok for tok in nlp("This is test.")]),
        num_conf_tokens=len([tok for tok in nlp("This is test.")]),
        complete=True,
    )
    tr._compute_source_diff_ops(s2)
    assert tr._prev_source_unconf_tokens == []
    assert tr._prev_source_conf_tok_count == 0

    s3 = Translator.Sentence(
        confirmed="",
        unconfirmed="Next",
        num_tokens=len([tok for tok in nlp("Next")]),
        num_conf_tokens=0,
        complete=False,
    )
    ops3 = tr._compute_source_diff_ops(s3)
    assert ops3 == []


def test_target_diff_state_resets_after_complete() -> None:
    tr, _ = _build_translator(source_diff_enabled=False)

    assert tr._compute_target_diff_ops("alpha", complete=False) == []
    tr._compute_target_diff_ops("alpha beta", complete=False)
    tr._compute_target_diff_ops("alpha beta.", complete=True)
    assert tr._prev_target_tokens == []

    # After reset, a fresh partial should establish baseline without diff ops.
    assert tr._compute_target_diff_ops("new start", complete=False) == []


def test_transcription_only_pipeline_stream_emits_source_messages_without_target() -> None:
    out_q: queue.Queue = queue.Queue()
    tr = Translator(
        engine_id="Dummy",
        transl_params={"source_diff_enabled": False},
        src_lang="English",
        target_lang="English",  # same language => translation pipeline disabled
        source_queue=queue.Queue(),
        output_queues=[out_q],
        sender_queue=queue.Queue(),
        only_complete_sent=False,
    )

    stream = [
        ("", "This is Democracy Now"),
        ("This is Democracy Now.", " I'm Amy Goodman"),
        ("I'm Amy", " Goodman with Juan Gonzalez"),
        ("Goodman with Juan Gonzalez.", ""),
        ("", "As the U.S. and Israeli war in Iran"),
        ("As the U.S. and Israeli war in Iran", " enters its 37"),
        ("enters its", " 32nd day, we turn out to look at how"),
        ("32nd day, we turn out to look at how", " artificial intelligence"),
    ]

    tr.start()
    assert tr.wait_until_ready(timeout=2)

    _enqueue_stream(tr, stream)
    tr.stop()

    msgs = _drain_queue(out_q)
    assert msgs
    assert all("orig_text" in msg for msg in msgs)
    assert all("transl_text" not in msg for msg in msgs)
    assert all(msg.get("source_diff") == [] for msg in msgs)

    complete_ids = [msg["id"] for msg in msgs if msg["complete"]]
    partial_ids = [msg["id"] for msg in msgs if not msg["complete"]]
    assert complete_ids
    assert partial_ids


def test_translation_thread_pipeline_with_dummy_backend_emits_target_messages() -> None:
    tr, out_q = _build_translator(source_diff_enabled=False, with_engine=False)

    stream = [
        ("", "The military is largely"),
        ("The military", " has largely relied on an age"),
        ("has largely relied on an", " AI system known as"),
        ("AI system known", " as Project Maven to speak"),
        ("as Project Maven to", " speed up the process"),
        ("speed up the process of identifying targets.", ""),
    ]

    tr.start()
    assert tr.wait_until_ready(timeout=2)

    _enqueue_stream(tr, stream)
    tr.stop()

    msgs = _drain_queue(out_q)
    assert msgs

    source_msgs = [msg for msg in msgs if "orig_text" in msg]
    target_msgs = [msg for msg in msgs if "transl_text" in msg]

    assert source_msgs
    assert target_msgs
    assert all(msg["target_lang"] == "de" for msg in target_msgs)
    assert any(msg["complete"] for msg in target_msgs)

    # Ensure translated payload corresponds to backend output format.
    assert all(msg["transl_text"].startswith("T ") for msg in target_msgs)


def test_runtime_with_output_consumer_thread_collects_source_and_target_events() -> None:
    tr, out_q = _build_translator(source_diff_enabled=False, with_engine=False)

    stream = [
        ("", "The Pentagon is now investigating a project"),
        ("The Pentagon is now investigating a", " project Maven played a role in the"),
        ("project Maven played a role in the", " U.S. strike on the"),
        ("U.S. strike on the", " Iranian girl"),
        ("Iranian", " Girls School that"),
    ]

    collected: list[dict] = []
    stop_collect = threading.Event()
    consumer = threading.Thread(target=_collect_output, args=(stop_collect, out_q, collected))
    consumer.start()

    tr.start()
    assert tr.wait_until_ready(timeout=2)

    _enqueue_stream(tr, stream)

    # Let translation thread flush queued work before teardown.
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if tr.translation_queue.empty():
            break
        time.sleep(0.01)

    tr.stop()

    stop_collect.set()
    consumer.join(timeout=5)
    assert not consumer.is_alive()

    assert collected
    source_msgs = [msg for msg in collected if "orig_text" in msg]
    target_msgs = [msg for msg in collected if "transl_text" in msg]

    assert source_msgs
    assert target_msgs
    assert all(msg["src_lang"] == "en" for msg in source_msgs)
    assert all(msg["target_lang"] == "de" for msg in target_msgs)


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ([(False, 1), (False, 1), (True, 1), (True, 2)], [1, 1, 1, 2]),
        ([(True, 1), (False, 2), (True, 2)], [1, 2, 2]),
    ],
)
def test_text_id_lifecycle(updates: list[tuple[bool, int]], expected: list[int]) -> None:
    tr, _ = _build_translator(source_diff_enabled=False)
    actual: list[int] = []

    for complete, _ in updates:
        actual.append(tr._resolve_text_id(complete))

    assert actual == expected
