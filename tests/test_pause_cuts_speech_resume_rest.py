"""Pause must cut streamed TTS (B18: Kokoro/pcm played to the end) and the
cut message's remainder must be replayed after Proceed, before any listen."""

import numpy as np
import pytest

import voice_mode.audio_player as ap
from voice_mode import pause_resume, streaming
from tests.test_streaming_barge_in_wiring import (
    _FakeAudioSegment,
    _FakeOpenAIClient,
    _FakeOutputStream,
)

TEXT = "First sentence here. Second sentence here. Third sentence here. Fourth one."


_PAUSED = {"on": False}


def _set_paused(on: bool):
    _PAUSED["on"] = on


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    # The real watcher thread mirrors the (absent) flag file into _pause_event
    # every 50ms and would clear a hand-set event, so drive the accessor itself.
    _PAUSED["on"] = False
    monkeypatch.setattr(ap, "convomode_paused", lambda: _PAUSED["on"])
    ap.reset_barge_in_event()
    while ap.is_tts_speaking():
        ap._speaking_dec()
    yield
    while ap.is_tts_speaking():
        ap._speaking_dec()


class _PausingStream(_FakeOutputStream):
    """Records writes; sets the pause event after N writes (user taps Pause)."""

    def __init__(self, *a, pause_after=2, **kw):
        super().__init__(*a, **kw)
        self.writes = 0
        self.pause_after = pause_after
        self.aborted = False

    def write(self, samples):
        self.writes += 1
        if self.writes == self.pause_after:
            _set_paused(True)

    def abort(self):
        self.aborted = True


def _pcm_chunks(n):
    return [np.full(2400, 1000, dtype=np.int16).tobytes() for _ in range(n)]


class TestPcmPathStopsOnPause:
    async def test_pause_mid_playback_stops_stream(self, monkeypatch):
        holder = {}

        def factory(*a, **kw):
            holder["s"] = _PausingStream(*a, **kw)
            return holder["s"]

        monkeypatch.setattr(streaming.sd, "OutputStream", factory)
        ok, m = await streaming.stream_pcm_audio(
            text=TEXT, openai_client=_FakeOpenAIClient(_pcm_chunks(20)),
            request_params={"response_format": "pcm"},
        )
        assert ok and m.paused and m.truncated
        assert holder["s"].writes == 2, "kept writing after Pause"
        assert holder["s"].aborted
        assert m.undelivered_text  # something left to replay

    async def test_no_pause_plays_everything(self, monkeypatch):
        holder = {}

        def factory(*a, **kw):
            holder["s"] = _PausingStream(*a, pause_after=10**9, **kw)
            return holder["s"]

        monkeypatch.setattr(streaming.sd, "OutputStream", factory)
        ok, m = await streaming.stream_pcm_audio(
            text=TEXT, openai_client=_FakeOpenAIClient(_pcm_chunks(5)),
            request_params={"response_format": "pcm"},
        )
        assert ok and not m.paused and not m.truncated
        assert holder["s"].writes == 5


class TestBufferedPathStopsOnPause:
    async def test_pause_mid_decoded_buffer(self, monkeypatch):
        holder = {}

        def factory(*a, **kw):
            holder["s"] = _PausingStream(*a, pause_after=3, **kw)
            return holder["s"]

        monkeypatch.setattr(streaming.sd, "OutputStream", factory)
        seg = _FakeAudioSegment(24000, 2, np.full(24000 * 5, 1000, dtype=np.int16))
        monkeypatch.setattr(streaming.AudioSegment, "from_file", lambda *a, **kw: seg)
        ok, m = await streaming.stream_with_buffering(
            text=TEXT, openai_client=_FakeOpenAIClient([b"\x00" * 100]),
            request_params={"response_format": "mp3"},
        )
        assert ok and m.paused
        assert holder["s"].writes == 3, "wrote the whole 5s buffer despite Pause"


class TestResumeAfterPause:
    async def test_waits_for_proceed_then_replays_remainder_once(self):
        _set_paused(True)
        events = []

        async def replay(text):
            events.append(("replay", text, ap.convomode_paused()))
            return {"paused": False}

        import asyncio
        task = asyncio.create_task(pause_resume.resume_after_pause(
            TEXT, {"paused": True, "undelivered_text": "Third. Fourth."}, replay,
            max_seconds=5))
        await asyncio.sleep(0.35)
        assert events == [], "replayed while still paused"
        _set_paused(False)
        out = await task
        assert events == [("replay", "Third. Fourth.", False)]
        assert out.replayed and not out.timed_out

    async def test_unknown_split_replays_whole_message(self):
        got = []

        async def replay(text):
            got.append(text)
            return {"paused": False}

        await pause_resume.resume_after_pause(TEXT, {"paused": True}, replay, max_seconds=1)
        assert got == [TEXT]

    async def test_paused_again_during_replay_waits_again(self):
        calls = []

        async def replay(text):
            calls.append(text)
            return {"paused": len(calls) == 1, "undelivered_text": "Last bit."}

        await pause_resume.resume_after_pause(
            TEXT, {"paused": True, "undelivered_text": "A. B."}, replay, max_seconds=1)
        assert calls == ["A. B.", "Last bit."]

    async def test_timeout_never_replays(self):
        _set_paused(True)
        called = []

        async def replay(text):
            called.append(text)

        out = await pause_resume.resume_after_pause(
            TEXT, {"paused": True, "undelivered_text": "x."}, replay, max_seconds=0.25)
        assert out.timed_out and called == []

    async def test_send_during_pause_is_consumed_and_skips_replay(self, tmp_path):
        sig = tmp_path / "turn-end.signal"
        sig.write_text("")
        called = []

        async def replay(text):
            called.append(text)

        out = await pause_resume.resume_after_pause(
            TEXT, {"paused": True, "undelivered_text": "x."}, replay,
            turn_end_signal_path=str(sig), max_seconds=1)
        assert out.skipped_by_send and called == [] and not sig.exists()

    async def test_not_paused_is_noop(self):
        assert await pause_resume.resume_after_pause(TEXT, {"truncated": True}, None) is None
        assert await pause_resume.resume_after_pause(TEXT, None, None) is None
