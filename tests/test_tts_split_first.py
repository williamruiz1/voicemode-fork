"""First audio must not wait on message length: the splitter picks a quick
first piece, and streaming plays it, then the prefetched rest, with no gap and
with Pause / failure behaviour intact."""

import asyncio
import time

import numpy as np
import pytest

import voice_mode.audio_player as ap
from voice_mode import streaming
from voice_mode.tts_split import split_first_piece
from tests.test_streaming_barge_in_wiring import _FakeOutputStream


def _long(first: str, rest: str = "And then the rest of the message continues for a while longer here.") -> str:
    return f"{first} {rest}"


class TestSplitter:
    def test_short_message_unchanged(self):
        assert split_first_piece("Done. All good.") == ["Done. All good."]
        assert split_first_piece("") == [""]

    def test_splits_at_first_sentence(self):
        first = "The build finished and all of the checks passed on the first try."
        pieces = split_first_piece(_long(first))
        assert pieces[0] == first
        assert pieces[1].startswith("And then")

    def test_question_and_exclamation_end_sentences(self):
        first = "Did the deploy actually go out to every machine in the fleet?"
        assert split_first_piece(_long(first))[0] == first
        first = "The deploy actually went out to every machine in the fleet!"
        assert split_first_piece(_long(first))[0] == first

    def test_abbreviations_do_not_end_sentences(self):
        text = ("I asked Dr. Smith about it, e.g. the config file and so on. "
                "Then we moved on to the rest of the plan, which is longer.")
        assert split_first_piece(text)[0] == (
            "I asked Dr. Smith about it, e.g. the config file and so on."
        )

    def test_decimal_numbers_do_not_end_sentences(self):
        text = ("Version 3.5 is now installed on the machine and ready to go. "
                "Next we will check the other machines in the group today.")
        assert split_first_piece(text)[0] == (
            "Version 3.5 is now installed on the machine and ready to go."
        )

    def test_initials_do_not_end_sentences(self):
        text = ("Ask J. Smith about the release schedule for the next quarter please. "
                "He knows the details of the whole plan and the owners.")
        assert split_first_piece(text)[0].endswith("next quarter please.")

    def test_long_first_sentence_cut_at_clause(self):
        text = ("This first sentence just keeps going and going without stopping, "
                "then carries on past ninety characters with more words until it finally ends.")
        pieces = split_first_piece(text)
        assert len(pieces) == 2
        assert pieces[0] == "This first sentence just keeps going and going without stopping,"
        assert len(pieces[0]) <= 90
        assert " ".join(pieces) == text

    def test_no_boundary_unchanged(self):
        text = "word " * 40
        assert split_first_piece(text) == [text]

    def test_tiny_tail_not_split(self):
        text = "x" * 70 + " and then it goes on for a bit more text here. Ok."
        assert split_first_piece(text)[0] == text or len(split_first_piece(text)) == 1

    def test_pieces_rejoin_to_original_words(self):
        text = _long("The build finished and all of the checks passed on the first try.")
        assert " ".join(split_first_piece(text)) == text


class _RecordingClient:
    """Fake TTS server: per-input chunks, with a log of requests, and optional
    delay before the first chunk of a given input."""

    def __init__(self, by_input, delays=None, fail_on=None):
        self.by_input = by_input
        self.delays = delays or {}
        self.fail_on = fail_on
        self.requests = []
        self.audio = self
        self.speech = self
        self.with_streaming_response = self

    def create(self, **kwargs):
        self.requests.append(kwargs["input"])
        return _Response(self, kwargs["input"])


class _Response:
    def __init__(self, client, text):
        self.client, self.text = client, text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def iter_bytes(self, chunk_size=None):
        await asyncio.sleep(self.client.delays.get(self.text, 0))
        if self.client.fail_on == self.text:
            raise RuntimeError("tts down")
        for c in self.client.by_input[self.text]:
            yield c
            await asyncio.sleep(0)


def _chunks(value, n=3):
    return [np.full(2400, value, dtype=np.int16).tobytes() for _ in range(n)]


FIRST = "The build finished and all of the checks passed on the first try."
REST = "And then the rest of the message continues for a while longer here."
TEXT = f"{FIRST} {REST}"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(ap, "convomode_paused", lambda: False)
    ap.reset_barge_in_event()
    while ap.is_tts_speaking():
        ap._speaking_dec()
    yield
    while ap.is_tts_speaking():
        ap._speaking_dec()


async def _run(client, monkeypatch, text=TEXT):
    holder = {}

    def factory(*a, **kw):
        holder["s"] = _FakeOutputStream(*a, **kw)
        return holder["s"]

    monkeypatch.setattr(streaming.sd, "OutputStream", factory)
    ok, m = await streaming.stream_pcm_audio(
        text=text, openai_client=client, request_params={"response_format": "pcm", "input": text},
    )
    return ok, m, holder["s"]


class TestSplitPlayback:
    async def test_pieces_play_in_order_without_gap(self, monkeypatch):
        client = _RecordingClient({FIRST: _chunks(100), REST: _chunks(200)})
        ok, m, stream = await _run(client, monkeypatch)
        assert ok and not m.truncated
        assert client.requests == [FIRST, REST]
        levels = [int(w[0]) for w in stream.written]
        assert levels == [100] * 3 + [200] * 3
        assert m.chunks_played == 6

    async def test_rest_is_prefetched_while_first_is_playing(self, monkeypatch):
        """The remainder is requested as soon as the first piece's first chunk
        arrives, not after the first piece has finished."""
        client = _RecordingClient({FIRST: _chunks(100, 8), REST: _chunks(200)})
        seen_at_write = []
        orig = _FakeOutputStream.write

        def write(self, samples):
            seen_at_write.append(len(client.requests))
            orig(self, samples)

        monkeypatch.setattr(_FakeOutputStream, "write", write)
        await _run(client, monkeypatch)
        assert seen_at_write[0] == 1
        assert max(seen_at_write[:8]) == 2  # rest already requested mid-first-piece

    async def test_first_audio_does_not_wait_for_rest(self, monkeypatch):
        """A slow remainder must not delay the first chunk or TTS_FIRST_AUDIO."""
        client = _RecordingClient({FIRST: _chunks(100), REST: _chunks(200)}, delays={REST: 0.3})
        events = []

        class _EL:
            TTS_FIRST_AUDIO = "first"
            TTS_PLAYBACK_START = "start"
            TTS_PLAYBACK_END = "end"

            def log_event(self, name, data=None):
                events.append(name)

        monkeypatch.setattr(streaming, "get_event_logger", lambda: _EL())
        ok, m, stream = await _run(client, monkeypatch)
        assert ok
        assert events.count("first") == 1 and events.count("end") == 1
        assert events.index("first") < events.index("end")
        assert m.generation_time < 0.2  # not the 0.3s remainder delay

    async def test_disabled_sends_one_request(self, monkeypatch):
        import voice_mode.config as cfg
        monkeypatch.setattr(cfg, "TTS_SPLIT_FIRST", False)
        client = _RecordingClient({TEXT: _chunks(100)})
        ok, m, stream = await _run(client, monkeypatch)
        assert ok and client.requests == [TEXT]

    async def test_short_message_sends_one_request(self, monkeypatch):
        client = _RecordingClient({"Hello there.": _chunks(100)})
        ok, m, stream = await _run(client, monkeypatch, text="Hello there.")
        assert ok and client.requests == ["Hello there."]

    async def test_rest_failure_never_replays_whole_message(self, monkeypatch):
        """Remainder fails after the first piece played: still ok, remainder
        retried once, tail recorded undelivered, first piece not repeated."""
        client = _RecordingClient({FIRST: _chunks(100), REST: _chunks(200)}, fail_on=REST)
        ok, m, stream = await _run(client, monkeypatch)
        assert ok, "must not return failure (core.py would replay the full text)"
        assert client.requests == [FIRST, REST, REST]  # remainder retried once
        assert FIRST not in client.requests[1:]
        assert [int(w[0]) for w in stream.written] == [100] * 3  # first piece once, no replay
        assert m.truncated and not m.paused
        assert m.undelivered_text == REST
        assert m.delivered_text == FIRST
        assert m.truncation_confidence == "tts-failure"

    async def test_rest_retry_succeeds(self, monkeypatch):
        client = _RecordingClient({FIRST: _chunks(100), REST: _chunks(200)})
        calls = {"n": 0}
        orig = _Response.iter_bytes

        async def flaky(self, chunk_size=None):
            if self.text == REST:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("blip")
            async for c in orig(self, chunk_size):
                yield c

        monkeypatch.setattr(_Response, "iter_bytes", flaky)
        ok, m, stream = await _run(client, monkeypatch)
        assert ok and not m.truncated
        assert client.requests == [FIRST, REST, REST]
        assert [int(w[0]) for w in stream.written] == [100] * 3 + [200] * 3

    async def test_pause_during_first_piece_stops_and_cancels_prefetch(self, monkeypatch):
        client = _RecordingClient({FIRST: _chunks(100, 10), REST: _chunks(200, 10)},
                                  delays={REST: 5})
        paused = {"on": False}
        monkeypatch.setattr(ap, "convomode_paused", lambda: paused["on"])
        orig = _FakeOutputStream.write
        count = {"n": 0}

        def write(self, samples):
            count["n"] += 1
            if count["n"] == 2:
                paused["on"] = True
            orig(self, samples)

        monkeypatch.setattr(_FakeOutputStream, "write", write)
        tasks = []
        orig_start = streaming._SplitSpeechResponse._start_prefetch

        def spy(self, chunk_size):
            q = orig_start(self, chunk_size)
            tasks.append(self._prefetch)
            return q

        monkeypatch.setattr(streaming._SplitSpeechResponse, "_start_prefetch", spy)
        t0 = time.perf_counter()
        ok, m, stream = await asyncio.wait_for(_run(client, monkeypatch), timeout=2)
        assert time.perf_counter() - t0 < 1.0, "close() waited out the slow prefetch"
        assert len(tasks) == 1 and tasks[0].cancelled(), "prefetch task was not cancelled"
        assert ok and m.paused and m.truncated
        assert len(stream.written) == 2
        assert m.undelivered_text
        # Nothing from the remainder was played, and the 5s prefetch did not hold us up.
        assert all(int(w[0]) == 100 for w in stream.written)
