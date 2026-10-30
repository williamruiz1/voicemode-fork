"""Live transcript feed + incremental STT segmentation (B27). No mic, no network:
audio is synthetic and STT is a fake function."""

import json
import os
import time
from unittest.mock import patch

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
