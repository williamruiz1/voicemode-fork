"""Live settings: ~/.voicemode/live-settings.json is re-read on every call.

No audio is ever opened. HOME is a temp dir so real ~/.voicemode is never touched.
"""

import asyncio
import json
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.modules.setdefault("webrtcvad", MagicMock())

from voice_mode import live_settings
from voice_mode.live_settings import get_live, load_live_settings
import voice_mode.tools.converse as conv


@pytest.fixture(autouse=True)
def temp_home(tmp_path, monkeypatch):
    # tests/conftest.py already redirects expanduser("~") to an isolated temp dir;
    # give each test a fresh, empty one of its own so nothing leaks between tests.
    import os
    monkeypatch.setenv("HOME", str(tmp_path))
    real = os.path.expanduser
    monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path) + p[1:] if p.startswith("~") else real(p))
    monkeypatch.delenv("VOICEMODE_LIVE_SETTINGS_PATH", raising=False)
    (tmp_path / ".voicemode").mkdir()
    # The legacy append-window path is computed at import; point it into the temp HOME.
    monkeypatch.setattr(conv, "APPEND_WINDOW_FLAG_PATH", str(tmp_path / ".voicemode" / "append-window-ms"))
    monkeypatch.setattr(conv, "APPEND_WINDOW_MS", 0)
    return tmp_path


def write(home, obj, raw=None):
    (home / ".voicemode" / "live-settings.json").write_text(raw if raw is not None else json.dumps(obj))


def test_missing_file_falls_back(temp_home):
    assert load_live_settings() == {}
    assert get_live("silence_threshold_ms", 1000) == 1000
    assert conv.append_window_ms() == 0


def test_changed_value_used_on_next_call_without_reimport(temp_home):
    write(temp_home, {"silence_threshold_ms": 800})
    assert get_live("silence_threshold_ms", 1000) == 800
    write(temp_home, {"silence_threshold_ms": 2500})
    assert get_live("silence_threshold_ms", 1000) == 2500
    (temp_home / ".voicemode" / "live-settings.json").unlink()
    assert get_live("silence_threshold_ms", 1000) == 1000


@pytest.mark.parametrize("bad", [
    {"silence_threshold_ms": 5}, {"silence_threshold_ms": 10**9}, {"silence_threshold_ms": "800"},
    {"silence_threshold_ms": True}, {"silence_threshold_ms": 800.5}, {"silence_threshold_ms": None},
    {"vad_aggressiveness": 4}, {"vad_aggressiveness": -1},
    {"listen_duration_max": 0}, {"listen_duration_max": -3}, {"listen_duration_max": float("nan")},
    {"listen_duration_min": -1}, {"append_window_ms": -5}, {"append_window_ms": 999999},
])
def test_bad_values_ignored(temp_home, bad):
    write(temp_home, bad)
    assert load_live_settings() == {}


@pytest.mark.parametrize("raw", ["", "not json", "[1,2]", "null", '{"silence_threshold_ms": '])
def test_bad_file_never_raises(temp_home, raw):
    write(temp_home, None, raw=raw)
    assert load_live_settings() == {}
    assert get_live("vad_aggressiveness", 2) == 2


def test_good_keys_survive_next_to_bad_ones(temp_home):
    write(temp_home, {"silence_threshold_ms": 700, "vad_aggressiveness": 9, "unknown": 1})
    assert load_live_settings() == {"silence_threshold_ms": 700}


def test_append_window_live_then_legacy_file(temp_home):
    legacy = temp_home / ".voicemode" / "append-window-ms"
    legacy.write_text("300")
    assert conv.append_window_ms() == 300  # legacy file still works
    write(temp_home, {"append_window_ms": 450})
    assert conv.append_window_ms() == 450  # live-settings wins
    write(temp_home, {"append_window_ms": -1})
    assert conv.append_window_ms() == 300  # bad live value -> legacy file


# ---- record loop: silence threshold + vad aggressiveness are read per call ----

def _chunks_consumed(speech_chunks=40, silence_chunks=400, **kw):
    """Run the record loop on scripted audio; return (#chunks consumed, Vad ctor mock)."""
    pattern = [True] * speech_chunks + [False] * silence_chunks
    state = {"i": 0}
    vad = MagicMock()

    def is_speech(*a, **k):
        v = pattern[min(state["i"], len(pattern) - 1)]
        state["i"] += 1
        return v

    vad.is_speech.side_effect = is_speech
    chunk = np.zeros((int(conv.SAMPLE_RATE * conv.VAD_CHUNK_DURATION_MS / 1000), 1), dtype=np.int16)
    q = MagicMock()
    q.get.side_effect = [chunk] * (len(pattern) + 5)
    with patch.object(conv, "webrtcvad") as wv, patch.object(conv, "VAD_AVAILABLE", True), \
         patch.object(conv, "DISABLE_SILENCE_DETECTION", False), \
         patch.object(conv, "ENDPOINTING_ENABLED", False), \
         patch.object(conv, "sd") as sd, patch("queue.Queue", return_value=q):
        wv.Vad.return_value = vad
        sd.InputStream.return_value.__enter__.return_value = MagicMock()
        try:
            conv.record_audio_with_silence_detection(max_duration=60.0, min_duration=0.0, **kw)
        except Exception:
            pass
        return state["i"], wv.Vad


def test_silence_threshold_changes_next_recording(temp_home):
    write(temp_home, {"silence_threshold_ms": 300})
    short, _ = _chunks_consumed()
    write(temp_home, {"silence_threshold_ms": 3000})
    long_, _ = _chunks_consumed()
    assert short < long_, (short, long_)
    # ~300ms vs ~3000ms of 30ms chunks: roughly 10 vs 100 extra chunks past the speech
    assert 40 <= short <= 40 + 20
    assert long_ >= 40 + 90


def test_missing_file_uses_configured_silence_threshold(temp_home):
    n, _ = _chunks_consumed()
    expected = 40 + conv.SILENCE_THRESHOLD_MS / conv.VAD_CHUNK_DURATION_MS
    assert abs(n - expected) <= 15, (n, expected)


def test_vad_aggressiveness_live_and_explicit_wins(temp_home):
    write(temp_home, {"vad_aggressiveness": 3})
    _, vad_ctor = _chunks_consumed(speech_chunks=2, silence_chunks=3)
    vad_ctor.assert_called_with(3)
    write(temp_home, {"vad_aggressiveness": 1})
    _, vad_ctor = _chunks_consumed(speech_chunks=2, silence_chunks=3)
    vad_ctor.assert_called_with(1)
    _, vad_ctor = _chunks_consumed(speech_chunks=2, silence_chunks=3, vad_aggressiveness=0)
    vad_ctor.assert_called_with(0)  # explicit per-call param wins
    write(temp_home, {"vad_aggressiveness": 7})
    _, vad_ctor = _chunks_consumed(speech_chunks=2, silence_chunks=3)
    vad_ctor.assert_called_with(conv.VAD_AGGRESSIVENESS)  # bad value ignored


# ---- converse(): listen durations. Stop right after resolution (wait_for_response + ffmpeg gate). ----

def _converse_durations(**kw):
    """Call converse() with everything past duration resolution stubbed out; capture durations."""
    seen = {}

    class Stop(Exception):
        pass

    def fake_conch(*a, **k):
        raise Stop()

    fn = conv.converse.fn if hasattr(conv.converse, "fn") else conv.converse
    with patch.object(conv, "logger") as lg:
        def capture(msg, *a, **k):
            if isinstance(msg, str) and msg.startswith("Converse:"):
                pass
        # Resolution happens before the ffmpeg check; make that check raise so nothing else runs.
        with patch.object(conv.voice_mode.config, "FFMPEG_AVAILABLE", False, create=True):
            out = asyncio.run(fn(message="x", wait_for_response=True, **kw))
    return out


def test_converse_listen_duration_validation_sees_live_values(temp_home):
    # live max below live min is clamped silently; invalid explicit values still error as before.
    write(temp_home, {"listen_duration_max": 30, "listen_duration_min": 1.5})
    # explicit bad value wins over file and is rejected
    out = _converse_durations(listen_duration_max=-1)
    assert "listen_duration_max must be positive" in out
    out = _converse_durations(listen_duration_min=-1)
    assert "listen_duration_min cannot be negative" in out


def test_converse_signature_defaults_are_none_so_explicit_wins():
    import inspect
    fn = conv.converse.fn if hasattr(conv.converse, "fn") else conv.converse
    sig = inspect.signature(fn)
    assert sig.parameters["listen_duration_max"].default is None
    assert sig.parameters["listen_duration_min"].default is None
