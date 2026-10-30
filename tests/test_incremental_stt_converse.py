"""Incremental STT wired into converse: tail-only transcription after Send (B27).
Fake STT, synthetic audio; no mic, no speakers, no network."""

import asyncio
import json
import time
from unittest.mock import patch

import numpy as np
import pytest

from voice_mode import live_transcript as lt
from voice_mode.tools import converse as cv

SR = cv.SAMPLE_RATE


def talk(total_s, burst=4.0, gap=0.7):
    rng = np.random.default_rng(1)
    parts, t = [], 0.0
    while t < total_s:
        parts += [(rng.uniform(-0.3, 0.3, int(burst * SR)) * 32767).astype(np.int16),
                  np.zeros(int(gap * SR), dtype=np.int16)]
        t += burst + gap
    return np.concatenate(parts)[: int(total_s * SR)]


def fake_stt_factory(per_audio_second, log):
    """Async stand-in for speech_to_text: sleeps proportionally to audio length."""
    async def fake(audio, save_audio=False, audio_dir=None, transport="local", archive_audio=None):
        dur = len(audio) / SR
        t0 = time.perf_counter()
        await asyncio.sleep(dur * per_audio_second)
        log.append((dur, time.perf_counter() - t0, archive_audio is not None))
        return {"text": f"[{dur:.0f}s]", "provider": "fake", "metrics": {}}
    return fake


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_90s_held_recording_post_send_is_one_tail_not_whole(monkeypatch):
    """A2: 90 s held recording; STT cost 0.02 s per audio second. Whole-file
    would take ~1.8 s after Send; with incremental only the tail remains."""
    monkeypatch.setenv("VOICEMODE_INCREMENTAL_STT", "true")
    log = []
    fake = fake_stt_factory(0.02, log)
    audio = talk(90)

    # Segments are transcribed on worker threads through the real wrapper, which
    # calls speech_to_text via asyncio.run -> patch it with the fake.
    with patch.object(cv, "speech_to_text", fake):
        inc = cv._make_incremental("conv_test", "local")
        assert inc is not None
        inc.begin(None)
        chunk = int(0.1 * SR)
        start = time.perf_counter()
        for i in range(0, len(audio), chunk):          # "recording": 90 s of audio over ~3 s wall
            inc.feed(audio[i:i + chunk])
            time.sleep(0.0033)
        record_wall = time.perf_counter() - start
        log.clear()                                     # only post-Send STT counts from here
        t_send = time.perf_counter()
        result = run(cv._speech_to_text_incremental(audio, inc, False, None, "local"))
        post_send = time.perf_counter() - t_send

    whole_cost = 90 * 0.02
    tail_durations = [d for d, _, _ in log]
    print(f"\nA2: recorded 90s in {record_wall:.2f}s wall; whole-file STT would cost {whole_cost:.2f}s; "
          f"post-Send wall {post_send:.3f}s; post-Send STT calls (audio s): {tail_durations}")
    assert len(log) == 1                                # exactly one tail transcription after Send
    assert tail_durations[0] < 15                       # at most one max-length segment
    assert post_send < whole_cost * 0.35                # not the whole recording
    assert log[0][2] is True                            # whole audio still handed over for archival
    assert result["text"].startswith("[") and result["provider"] == "fake"


def test_segment_failure_falls_back_to_whole_recording(monkeypatch):
    log = []
    audio = talk(40)
    calls = {"n": 0}

    async def flaky(a, save_audio=False, audio_dir=None, transport="local", archive_audio=None):
        calls["n"] += 1
        if len(a) < len(audio) and calls["n"] == 1:      # first segment fails
            return {"error_type": "connection_failed"}
        log.append(len(a))
        return {"text": "ok", "provider": "fake"}

    with patch.object(cv, "speech_to_text", flaky):
        inc = cv._make_incremental("conv_test", "local")
        inc.begin(None)
        for i in range(0, len(audio), int(0.1 * SR)):
            inc.feed(audio[i:i + int(0.1 * SR)])
        result = run(cv._speech_to_text_incremental(audio, inc, False, None, "local"))
    assert log[-1] == len(audio)                         # whole-recording transcription ran
    assert result["text"] == "ok"


def test_disabled_gate_never_builds_transcriber(monkeypatch):
    monkeypatch.setenv("VOICEMODE_INCREMENTAL_STT", "false")
    assert cv._make_incremental("conv_test") is None
    log = []
    audio = talk(10)
    with patch.object(cv, "speech_to_text", fake_stt_factory(0, log)):
        run(cv._speech_to_text_incremental(audio, None, False, None, "local"))
    assert [round(d) for d, _, _ in log] == [10]         # single whole-recording call


def test_partials_reach_feed_with_turn_id(monkeypatch, tmp_path):
    feed = tmp_path / "f.jsonl"
    monkeypatch.setattr(lt, "feed_path", lambda: feed)
    monkeypatch.setattr(lt, "resolve_client_pid", lambda: 11)
    audio = talk(30)
    with patch.object(cv, "speech_to_text", fake_stt_factory(0, [])):
        inc = cv._make_incremental("conv_77", "local")
        inc.begin(None)
        for i in range(0, len(audio), int(0.1 * SR)):
            inc.feed(audio[i:i + int(0.1 * SR)])
        inc.collect(audio)
    rows = [json.loads(l) for l in feed.read_text().splitlines()]
    assert rows and all(r["kind"] == "user_partial" and r["turn_id"] == "conv_77" and r["client_pid"] == 11 for r in rows)
    assert rows[-1]["text"].count("[") == len(rows)      # cumulative: one more segment per partial


def test_recorders_feed_incremental(monkeypatch):
    """The held no-VAD recorder feeds every chunk to the transcriber (mic mocked)."""
    import queue

    class FakeInc:
        def __init__(self): self.began, self.fed = 0, 0
        def begin(self, pre): self.began += 1
        def feed(self, c): self.fed += len(c)

    chunk = np.ones(int(SR * cv.VAD_CHUNK_DURATION_MS / 1000), dtype=np.int16)

    class FakeStream:
        def __init__(self, *a, callback=None, **k): self.cb = callback
        def __enter__(self):
            for _ in range(10):
                self.cb(chunk.reshape(-1, 1), len(chunk), None, None)
            return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(cv.sd, "InputStream", FakeStream)
    monkeypatch.setattr(cv, "_bluetooth_input_active", lambda: False)
    monkeypatch.setattr(cv, "hold_max_seconds", lambda: 0.25)
    monkeypatch.setattr(cv, "_clear_stale_send", lambda: None)
    inc = FakeInc()
    audio, ok = cv._record_held_until_send(5.0, {}, inc)
    assert ok and inc.began == 1 and inc.fed == len(audio) > 0


def test_default_vad_recorder_feeds_incremental(monkeypatch):
    """The default (webrtcvad) recorder, driven through a mocked queue with real
    committed audio, begins and feeds the transcriber; no mic is opened."""
    from pathlib import Path
    from unittest.mock import MagicMock
    from scipy.io import wavfile

    wav = Path(__file__).parent / "fixtures" / "endpointing" / "clean_speech_then_pause.wav"
    if not wav.exists():
        pytest.skip("endpointing fixtures not generated")
    sr, data = wavfile.read(wav)
    assert sr == SR
    n = int(SR * cv.VAD_CHUNK_DURATION_MS / 1000)
    full = np.concatenate([np.asarray(data, dtype=np.int16).reshape(-1), np.zeros(5 * SR, dtype=np.int16)])
    chunks = [full[i:i + n].reshape(-1, 1) for i in range(0, len(full) - n + 1, n)]

    class FakeInc:
        def __init__(self): self.began, self.fed = [], 0
        def begin(self, pre): self.began.append(pre)
        def feed(self, c): self.fed += len(c)

    q = MagicMock()
    q.get.side_effect = list(chunks)
    inc = FakeInc()
    with patch.object(cv, "ENDPOINTING_ENABLED", False), \
         patch.object(cv, "sd") as sd, \
         patch("queue.Queue", return_value=q):
        sd.InputStream.return_value.__enter__.return_value = MagicMock()
        sd.InputStream.return_value.__exit__.return_value = False
        audio, speech = cv.record_audio_with_silence_detection(8.0, incremental=inc)
    assert speech and len(audio) > 0
    assert inc.began == [None]
    assert inc.fed == len(audio)         # every recorded sample was offered to the transcriber
