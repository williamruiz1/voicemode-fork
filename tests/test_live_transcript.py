"""Live transcript feed + incremental STT segmentation (B27). No mic, no network:
audio is synthetic and STT is a fake function."""

import json
import os
import time
from unittest.mock import patch

import threading

import numpy as np
import pytest

from voice_mode import live_transcript as lt

SR = 24000


@pytest.fixture
def feed(tmp_path, monkeypatch):
    path = tmp_path / "live-transcript.jsonl"
    monkeypatch.setattr(lt, "feed_path", lambda: path)
    monkeypatch.setattr(lt, "resolve_client_pid", lambda: 4242)
    return path


def lines(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


# ---- feed writer ----

def test_feed_line_schema(feed):
    assert lt.append_event("agent", "conv_1", "hello there")
    (row,) = lines(feed)
    assert set(row) == {"ts", "kind", "turn_id", "server_pid", "client_pid", "text"}
    assert row["kind"] == "agent" and row["turn_id"] == "conv_1" and row["text"] == "hello there"
    assert row["server_pid"] == os.getpid() and row["client_pid"] == 4242
    assert isinstance(row["ts"], float)


def test_feed_appends_and_allows_empty_final(feed):
    lt.append_event("user_partial", "conv_1", "he")
    lt.append_event("user_final", "conv_1", "")
    assert [r["kind"] for r in lines(feed)] == ["user_partial", "user_final"]
    assert lines(feed)[1]["text"] == ""


def test_feed_unknown_client_pid_is_null(feed, monkeypatch):
    monkeypatch.setattr(lt, "resolve_client_pid", lambda: None)
    lt.append_event("agent", "conv_1", "x")
    assert lines(feed)[0]["client_pid"] is None


def test_feed_truncates_over_1mb(feed):
    feed.write_text("x" * (lt.FEED_MAX_BYTES + 1))
    lt.append_event("agent", "conv_2", "fresh")
    (row,) = lines(feed)
    assert row["turn_id"] == "conv_2"


def test_feed_truncates_on_new_local_day(feed):
    lt.append_event("agent", "conv_old", "old")
    yesterday = time.time() - 36 * 3600
    os.utime(feed, (yesterday, yesterday))
    lt.append_event("agent", "conv_new", "new")
    assert [r["turn_id"] for r in lines(feed)] == ["conv_new"]


def test_feed_same_day_small_file_is_kept(feed):
    lt.append_event("agent", "a", "1")
    lt.append_event("agent", "b", "2")
    assert [r["turn_id"] for r in lines(feed)] == ["a", "b"]


def test_feed_fail_open(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setattr(lt, "feed_path", lambda: blocker / "sub" / "feed.jsonl")  # parent is a file
    assert lt.append_event("agent", "conv_1", "x") is False  # no raise


def test_feed_default_path_under_base_dir():
    from voice_mode import config
    assert lt.feed_path() == config.BASE_DIR / lt.FEED_FILENAME if hasattr(config.BASE_DIR, "__truediv__") \
        else str(lt.feed_path()).startswith(str(config.BASE_DIR))


# ---- client pid ----

def test_client_pid_daemon_uses_bound_identity(monkeypatch):
    from voice_mode import client_identity
    tok = client_identity.bind(client_identity.ClientIdentity("pid:777", 777))
    try:
        assert lt.resolve_client_pid() == 777
    finally:
        client_identity.unbind(tok)


def test_client_pid_shared_mode_unbound_is_none(monkeypatch):
    from voice_mode import client_identity
    monkeypatch.setattr(client_identity, "SHARED_MODE", True)
    assert lt.resolve_client_pid() is None


def test_client_pid_stdio_walks_to_claude_ancestor(monkeypatch):
    from voice_mode import client_identity
    monkeypatch.setattr(client_identity, "SHARED_MODE", False)
    lt._claude_pid_cache.clear()
    table = {100: (200, "/usr/bin/node"), 200: (300, "/opt/claude"), 300: (1, "zsh")}
    monkeypatch.setattr(lt, "_ps_ppid_comm", lambda pid: table.get(pid))
    monkeypatch.setattr(os, "getppid", lambda: 100)
    try:
        assert lt.resolve_client_pid() == 200
    finally:
        lt._claude_pid_cache.clear()


def test_client_pid_stdio_no_claude_is_none(monkeypatch):
    from voice_mode import client_identity
    monkeypatch.setattr(client_identity, "SHARED_MODE", False)
    lt._claude_pid_cache.clear()
    table = {100: (200, "node"), 200: (1, "zsh")}
    monkeypatch.setattr(lt, "_ps_ppid_comm", lambda pid: table.get(pid))
    monkeypatch.setattr(os, "getppid", lambda: 100)
    try:
        assert lt.resolve_client_pid() is None
    finally:
        lt._claude_pid_cache.clear()


def test_client_pid_ps_failure_is_none(monkeypatch):
    from voice_mode import client_identity
    monkeypatch.setattr(client_identity, "SHARED_MODE", False)
    lt._claude_pid_cache.clear()
    monkeypatch.setattr(lt, "_ps_ppid_comm", lambda pid: None)
    try:
        assert lt.resolve_client_pid() is None
    finally:
        lt._claude_pid_cache.clear()


# ---- segmentation ----

def speech(seconds, amp=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.uniform(-amp, amp, int(seconds * SR)) * 32767).astype(np.int16)


def quiet(seconds):
    return np.zeros(int(seconds * SR), dtype=np.int16)


def talk_with_pauses(total_s, burst=4.0, gap=0.7):
    parts, t, i = [], 0.0, 0
    while t < total_s:
        parts += [speech(burst, seed=i), quiet(gap)]
        t += burst + gap
        i += 1
    return np.concatenate(parts)


def feed_all(inc, audio, chunk_s=0.1):
    n = int(chunk_s * SR)
    for i in range(0, len(audio), n):
        inc.feed(audio[i:i + n])


class FakeSTT:
    """Returns 'w<start>' words keyed by segment length so joins are checkable."""
    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    def __call__(self, seg):
        self.calls.append(len(seg))
        if self.fail_on is not None and len(self.calls) - 1 == self.fail_on:
            raise RuntimeError("boom")
        return f"seg{len(self.calls)}"


def make(stt, partials=None):
    return lt.IncrementalTranscriber(stt, SR, on_partial=(partials.append if partials is not None else None),
                                     min_segment_s=6.0, max_segment_s=15.0)


def test_cuts_at_pauses_and_joins_with_tail():
    stt, partials = FakeSTT(), []
    inc = make(stt, partials)
    audio = talk_with_pauses(20)
    inc.begin(None)
    feed_all(inc, audio)
    prefix, tail = inc.collect(audio)
    n = len(stt.calls)
    assert n >= 2
    assert prefix == " ".join(f"seg{i + 1}" for i in range(n))
    assert sum(stt.calls) + len(tail) == len(audio)  # segments + tail cover the audio exactly once
    assert all(c >= 6 * SR for c in stt.calls)
    assert partials[-1] == prefix and partials[0] == "seg1"  # cumulative, growing


def test_segments_end_in_a_pause():
    stt = FakeSTT()
    inc = make(stt)
    audio = talk_with_pauses(20)
    inc.begin(None)
    feed_all(inc, audio)
    inc.collect(audio)
    consumed = 0
    for c in stt.calls:
        consumed += c
        edge = audio[consumed - int(0.3 * SR):consumed]
        assert np.abs(edge).max() == 0


def test_continuous_speech_forced_cut_at_max_segment():
    stt = FakeSTT()
    inc = make(stt)
    audio = speech(40)  # no pauses at all
    inc.begin(None)
    feed_all(inc, audio)
    prefix, tail = inc.collect(audio)
    assert len(stt.calls) >= 2
    assert all(c <= 15 * SR + SR for c in stt.calls)
    assert sum(stt.calls) + len(tail) == len(audio)


def test_short_recording_makes_no_segments():
    stt = FakeSTT()
    inc = make(stt)
    audio = talk_with_pauses(4)
    inc.begin(None)
    feed_all(inc, audio)
    assert inc.segments_started == 0 and stt.calls == []
    prefix, tail = inc.collect(audio)
    assert prefix == "" and len(tail) == len(audio)


def test_segment_failure_means_fallback():
    inc = make(FakeSTT(fail_on=0))
    audio = talk_with_pauses(20)
    inc.begin(None)
    feed_all(inc, audio)
    assert inc.collect(audio) is None


def test_later_segment_failure_means_fallback():
    inc = make(FakeSTT(fail_on=1))
    audio = talk_with_pauses(30)
    inc.begin(None)
    feed_all(inc, audio)
    assert inc.collect(audio) is None


def test_mismatched_final_audio_means_fallback():
    inc = make(FakeSTT())
    audio = talk_with_pauses(20)
    inc.begin(None)
    feed_all(inc, audio)
    assert inc.collect(speech(20, seed=99)) is None


def test_no_speech_segment_is_empty_text_not_failure():
    stt = lambda seg: ""
    inc = make(stt)
    audio = np.concatenate([quiet(8), speech(3)])
    inc.begin(None)
    feed_all(inc, audio)
    prefix, tail = inc.collect(audio)
    assert prefix == "" and len(tail) > 0


def test_begin_keeps_segments_when_preroll_matches_prefix():
    """Device-recovery reopen passes everything captured so far as pre_roll."""
    stt = FakeSTT()
    inc = make(stt)
    audio = talk_with_pauses(20)
    cut_at = int(14 * SR)
    inc.begin(None)
    feed_all(inc, audio[:cut_at])
    made = inc.segments_started
    assert made >= 1
    inc.begin(audio[:cut_at])  # restart seeded with the same audio
    feed_all(inc, audio[cut_at:])
    prefix, tail = inc.collect(audio)
    assert sum(stt.calls) + len(tail) == len(audio)
    assert len(stt.calls) >= made and len(set(stt.calls)) >= 1


def test_begin_with_unrelated_preroll_resets():
    stt = FakeSTT()
    inc = make(stt)
    inc.begin(None)
    feed_all(inc, talk_with_pauses(14))
    other = talk_with_pauses(10)
    inc.begin(other)
    feed_all(inc, other[:0])
    prefix, tail = inc.collect(other)
    assert sum(stt.calls[-inc.segments_started:] if inc.segments_started else []) + len(tail) == len(other)


def test_feed_never_raises_and_degrades_to_fallback():
    inc = make(FakeSTT())
    inc.begin(None)
    inc._maybe_cut = lambda: (_ for _ in ()).throw(RuntimeError("x"))
    inc.feed(speech(7))
    assert inc.collect(speech(7)) is None or inc.segments_started == 0


def test_env_gate(monkeypatch):
    monkeypatch.delenv("VOICEMODE_INCREMENTAL_STT", raising=False)
    assert lt.incremental_stt_enabled()
    monkeypatch.setenv("VOICEMODE_INCREMENTAL_STT", "false")
    assert not lt.incremental_stt_enabled()


def test_stale_segment_cannot_write_after_begin_reset():
    """A segment still running from an earlier recording must not touch the new state."""
    import threading
    gate, started = threading.Event(), threading.Event()
    seen = []

    def slow_then_text(seg):
        started.set()
        gate.wait(5)
        return "stale-text"

    partials = []
    inc = lt.IncrementalTranscriber(slow_then_text, SR, on_partial=partials.append, min_segment_s=6.0)
    inc.begin(None)
    feed_all(inc, talk_with_pauses(14))
    assert started.wait(5)
    inc.begin(None)            # restart with unrelated audio -> reset while segment 0 is mid-flight
    gen_after = inc._gen
    gate.set()
    inc._pool.shutdown(wait=True)
    assert inc._texts == {} and not inc._failed and partials == []
    assert inc._gen == gen_after


def test_stale_segment_failure_does_not_poison_new_recording():
    import threading
    gate, started = threading.Event(), threading.Event()

    def boom(seg):
        started.set()
        gate.wait(5)
        raise RuntimeError("late failure")

    inc = lt.IncrementalTranscriber(boom, SR, min_segment_s=6.0)
    inc.begin(None)
    feed_all(inc, talk_with_pauses(14))
    assert started.wait(5)
    inc.begin(None)
    gate.set()
    inc._pool.shutdown(wait=True)
    assert not inc._failed


# -- words while talking --------------------------------------------------------

LOUD = (np.ones(SR // 2, dtype=np.int16) * 8000)
QUIET = np.zeros(SR // 2, dtype=np.int16)


def rolling(stt, partials, **kw):
    inc = lt.IncrementalTranscriber(stt, SR, on_partial=partials.append, rolling_s=100, **kw)
    inc.begin(None)
    return inc


def test_words_show_before_a_segment_is_cut():
    partials = []
    inc = rolling(lambda a: f"{len(a) / SR:.0f}s", partials)
    inc.feed(LOUD), inc.feed(LOUD)
    inc._roll_once()
    assert partials == ["1s"]
    assert inc.segments_started == 0
    inc.close()


def test_no_new_audio_means_no_new_request():
    calls = []
    inc = rolling(lambda a: calls.append(1) or "x", [])
    inc.feed(LOUD), inc.feed(LOUD)
    inc._roll_once(), inc._roll_once()
    assert len(calls) == 1
    inc.close()


def test_quiet_after_speech_sends_nothing_to_whisper():
    calls = []
    inc = rolling(lambda a: calls.append(1) or "x", [])
    inc.feed(LOUD), inc.feed(LOUD)
    inc._roll_once()
    inc.feed(QUIET), inc.feed(QUIET)
    inc._roll_once()
    assert len(calls) == 1
    inc.close()


def test_invented_words_from_a_quiet_room_never_show():
    """B9: nothing is published until the recorder has heard speech."""
    partials = []
    inc = rolling(lambda a: "Blink", partials, wait_for_speech=True, min_segment_s=1.0, max_segment_s=4.0)
    inc.feed(LOUD), inc.feed(LOUD), inc.feed(QUIET)  # a pause: this segment is cut
    assert inc.segments_started == 1
    inc._roll_once()
    for f in inc._futures:
        f.result()
    assert partials == []
    inc.mark_speech()
    inc._transcribe_rolling = lambda a: "hello there"
    inc.feed(LOUD), inc.feed(LOUD)
    inc._roll_once()
    assert partials[-1] == "hello there", "segments from before speech stay off the display"
    inc.close()


def test_final_text_ignores_provisional_text():
    partials = []
    inc = rolling(lambda a: "provisional" if len(a) < SR else "segment", partials, min_segment_s=1.0, max_segment_s=2.0)
    inc.feed(LOUD)
    inc.feed(QUIET[:SR // 4])  # under a second: no cut yet
    inc._roll_once = lt.IncrementalTranscriber._roll_once.__get__(inc)
    inc.feed(LOUD), inc.feed(LOUD), inc.feed(QUIET), inc.feed(QUIET)
    audio = np.concatenate([LOUD, QUIET[:SR // 4], LOUD, LOUD, QUIET, QUIET])
    prefix, tail = inc.collect(audio)
    assert "provisional" not in prefix and prefix.startswith("segment")


def test_a_segment_finishing_keeps_the_words_after_it():
    partials = []
    release = threading.Event()

    def stt(a):
        if len(a) >= 2 * SR:
            release.wait(5)
            return "SEG"
        return "TAIL"
    inc = rolling(stt, partials, min_segment_s=2.0, max_segment_s=3.0)
    for _ in range(4):
        inc.feed(LOUD)
    inc.feed(QUIET)  # pause: the segment is cut
    assert inc.segments_started == 1
    inc._roll_once()  # the segment is still at Whisper: goes first, no request
    assert partials == []
    release.set()
    inc._futures[0].result()
    inc.feed(LOUD), inc.feed(LOUD)
    inc._roll_once()
    assert partials[-1] == "SEG TAIL"
    inc.close()


def test_a_cut_during_a_rolling_request_drops_its_result():
    partials = []
    inc = rolling(lambda a: inc._consumed or inc.feed(QUIET) or "stale", partials, min_segment_s=1.0, max_segment_s=2.0)
    inc._consumed = 0
    inc.feed(LOUD), inc.feed(LOUD)

    def cut_then_answer(a):
        inc._consumed += 1
        return "stale"
    inc._transcribe = cut_then_answer
    inc._roll_once()
    assert partials == []
    inc.close()


def test_a_rolling_failure_ends_rolling_but_not_the_turn():
    inc = lt.IncrementalTranscriber(lambda a: (_ for _ in ()).throw(RuntimeError("no")), SR,
                                    on_partial=lambda t: None, rolling_s=0.01)
    inc.begin(None)
    inc.feed(LOUD), inc.feed(LOUD)
    inc._rolling_thread.join(2)
    assert not inc._rolling_thread.is_alive()
    assert inc._failed is False
    inc.close()


def test_close_stops_the_rolling_thread():
    inc = lt.IncrementalTranscriber(lambda a: "x", SR, on_partial=lambda t: None, rolling_s=0.01)
    inc.begin(None)
    thread = inc._rolling_thread
    inc.close()
    thread.join(2)
    assert not thread.is_alive()


def test_off_by_default_when_built_directly():
    inc = lt.IncrementalTranscriber(lambda a: "x", SR, on_partial=lambda t: None)
    inc.begin(None)
    assert inc._rolling_thread is None
    inc.close()


def test_rolling_never_uses_the_failover_path():
    """Failover can reach a paid provider; a display update every 2 seconds must not."""
    from unittest.mock import MagicMock, patch
    from voice_mode.tools import converse as cv

    with patch.object(cv, "STT_BASE_URLS", ["https://api.openai.com/v1"]), patch.object(cv.httpx, "post") as post:
        with pytest.raises(RuntimeError):
            cv._rolling_local_stt(np.zeros(SR, dtype=np.int16))
        post.assert_not_called()
    answer = MagicMock()
    answer.json.return_value = {"text": "hi"}
    with patch.object(cv, "STT_BASE_URLS", ["https://api.openai.com/v1", "http://127.0.0.1:2022/v1"]), \
         patch.object(cv.httpx, "post", return_value=answer) as post, \
         patch.object(cv, "speech_to_text") as failover:
        assert cv._rolling_local_stt(np.zeros(SR, dtype=np.int16)) == "hi"
    assert post.call_args[0][0] == "http://127.0.0.1:2022/v1/audio/transcriptions"
    failover.assert_not_called()


def test_rolling_stops_when_the_listen_ends():
    inc = lt.IncrementalTranscriber(lambda a: "x", SR, on_partial=lambda t: None, rolling_s=0.01)
    inc.begin(None)
    thread = inc._rolling_thread
    inc.stop_rolling()
    thread.join(2)
    assert not thread.is_alive()
    inc.close()

