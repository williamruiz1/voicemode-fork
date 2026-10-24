"""B43: Pause (pause.flag) must suspend an OPEN listen, not only TTS.

Incident 2026-10-24 12:14: pressing Pause while the agent was listening did
nothing -- recording continued and silence still ended the turn. PR #30 only
cut TTS. Now, while paused: incoming audio is discarded and no stop condition
runs; Proceed resumes the SAME turn with a fresh silence clock; Send ends the
turn with what was captured before the pause; a pause past
VOICEMODE_PAUSE_MAX_SECONDS ends the listen with end_reason "paused_timeout".

Each test drives the real loop with a fake sounddevice on a daemon thread with
a hard join timeout, so a regression fails fast instead of hanging the suite.
"""

import collections
import json
import sys
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault("webrtcvad", MagicMock())

import voice_mode.tools.converse as C
from voice_mode.tools.converse import record_audio_with_silence_detection, recording_end_event_data
from voice_mode.config import SAMPLE_RATE, VAD_CHUNK_DURATION_MS


CHUNK_SAMPLES = int(SAMPLE_RATE * VAD_CHUNK_DURATION_MS / 1000)
SILENCE_MS = 300
SILENCE_CHUNKS = SILENCE_MS // VAD_CHUNK_DURATION_MS


def _chunk(loud):
    if loud:
        return np.random.randint(-8000, 8000, size=(CHUNK_SAMPLES, 1), dtype=np.int16)
    return np.zeros((CHUNK_SAMPLES, 1), dtype=np.int16)


class _ScriptedStream:
    """Delivers queued chunks (feed), then `fill` ('loud' / 'quiet' / None =
    deliver nothing) forever, one chunk per ~1ms."""

    feed = collections.deque()
    fill = None

    def __init__(self, *, callback, **_kw):
        self.callback = callback
        self._stop = threading.Event()

    def __enter__(self):
        def pump():
            while not self._stop.is_set():
                if _ScriptedStream.feed:
                    loud = _ScriptedStream.feed.popleft()
                elif _ScriptedStream.fill is not None:
                    loud = _ScriptedStream.fill == "loud"
                else:
                    time.sleep(0.001)
                    continue
                self.callback(_chunk(loud), CHUNK_SAMPLES, None, None)
                time.sleep(0.001)
        threading.Thread(target=pump, daemon=True).start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        return False


class _EnergyVad:
    def __init__(self, *a):
        pass

    def is_speech(self, chunk_bytes, rate):
        arr = np.frombuffer(chunk_bytes, dtype=np.int16).astype(float)
        return bool(arr.size) and float(np.sqrt(np.mean(arr ** 2))) > 500


@pytest.fixture
def sim(monkeypatch, isolate_home_directory):
    home = isolate_home_directory / ".voicemode"
    home.mkdir(parents=True, exist_ok=True)
    fake_sd = MagicMock()
    fake_sd.InputStream = _ScriptedStream
    fake_sd.Stream = _ScriptedStream
    fake_sd.PortAudioError = RuntimeError
    monkeypatch.setattr(C, "sd", fake_sd)
    monkeypatch.setattr(C, "_bluetooth_input_active", lambda: False)
    monkeypatch.setattr(C, "VAD_AVAILABLE", True)
    monkeypatch.setattr(C, "DISABLE_SILENCE_DETECTION", False)
    monkeypatch.setattr(C, "ENDPOINTING_ENABLED", False)
    monkeypatch.setattr(C, "VAD_ENERGY_THRESHOLD", 0)
    monkeypatch.setattr(C, "SILENCE_THRESHOLD_MS", SILENCE_MS)
    monkeypatch.setattr(C, "MIN_RECORDING_DURATION", 0.1)
    monkeypatch.setattr(C, "STEP_AWAY_ENV", False)
    monkeypatch.setattr(C, "STEP_AWAY_FLAG_PATH", str(home / "nonexistent-step-away"))
    monkeypatch.setattr(C, "APPEND_WINDOW_MS", 0)
    monkeypatch.setattr(C, "APPEND_WINDOW_FLAG_PATH", str(home / "nonexistent-append"))
    monkeypatch.setattr(C, "record_audio", MagicMock(side_effect=AssertionError("fixed-length record_audio used")))
    fake_webrtc = MagicMock()
    fake_webrtc.Vad = _EnergyVad
    monkeypatch.setattr(C, "webrtcvad", fake_webrtc)
    monkeypatch.setenv("VOICEMODE_HOLD_MAX_SECONDS", "8")
    monkeypatch.setenv("VOICEMODE_STREAM_STALL_SECONDS", "30")
    monkeypatch.delenv("VOICEMODE_PAUSE_MAX_SECONDS", raising=False)
    _ScriptedStream.feed = collections.deque()
    _ScriptedStream.fill = None
    return home


def _listen_async(paused, **kwargs):
    box = {"stats": {}}

    def run():
        try:
            box["result"] = record_audio_with_silence_detection(
                listen_stats=box["stats"], listen_pause_check=paused.is_set, **kwargs)
        except BaseException as e:
            box["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, box


def _wait(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


def _listening(home):
    try:
        return json.loads((home / "listening.json").read_text())
    except (OSError, ValueError):
        return {}


def _pause(box, paused):
    paused.set()
    assert _wait(lambda: box["stats"].get("pause_count", 0) >= 1), box


def _resume(box, paused):
    paused.clear()
    assert _wait(lambda: box["stats"].get("paused_seconds", 0) > 0), box


def _finish(t, box, timeout=3.0):
    t.join(timeout)
    assert not t.is_alive(), f"listen did not end: {box}"
    assert "error" not in box, box.get("error")
    return box["result"]


def test_pause_stops_capture_and_suspends_silence_stop(sim):
    paused = threading.Event()
    _ScriptedStream.feed.extend([True] * 20)
    t, box = _listen_async(paused, max_duration=30.0)
    assert _wait(lambda: box["stats"].get("chunks_processed", 0) == 20)
    _pause(box, paused)
    at_pause = box["stats"]["chunks_processed"]

    state = _listening(sim)
    assert state.get("paused") is True and state.get("paused_at")

    _ScriptedStream.fill = "quiet"  # would end the turn in ~10 chunks if not paused
    time.sleep(0.6)
    assert t.is_alive(), f"silence ended a paused listen: {box}"
    assert box["stats"]["chunks_processed"] == at_pause  # nothing accumulated
    assert box["stats"].get("chunks_discarded_paused", 0) > 3 * SILENCE_CHUNKS

    (sim / "turn-end.signal").touch()  # end the turn
    audio, speech = _finish(t, box)
    assert box["stats"]["end_reason"] == "send"
    assert speech is True
    assert len(audio) == at_pause * CHUNK_SAMPLES  # only pre-pause audio


def test_proceed_resumes_same_turn_with_fresh_silence_clock(sim):
    paused = threading.Event()
    # Speech, then trailing silence just short of the stop threshold.
    near = SILENCE_CHUNKS - 3
    _ScriptedStream.feed.extend([True] * 20 + [False] * near)
    t, box = _listen_async(paused, max_duration=30.0)
    assert _wait(lambda: box["stats"].get("chunks_processed", 0) == 20 + near)
    _pause(box, paused)

    _ScriptedStream.fill = "loud"  # talk while paused: must be discarded
    time.sleep(0.3)
    _ScriptedStream.fill = None
    time.sleep(0.05)
    discarded = box["stats"].get("chunks_discarded_paused", 0)
    assert discarded > 20
    _resume(box, paused)
    assert t.is_alive()
    assert _listening(sim).get("paused") is False
    assert _listening(sim).get("paused_seconds", 0) >= 0.3

    _ScriptedStream.fill = "quiet"
    audio, speech = _finish(t, box)
    stats = box["stats"]
    assert stats["end_reason"] == "silence"
    assert speech is True
    after_resume = stats["chunks_processed"] - (20 + near)
    # Fresh clock: a full threshold of new silence, not the 3 chunks left over.
    assert after_resume >= SILENCE_CHUNKS
    # One turn: pre-pause audio + post-resume audio, nothing from the pause.
    assert len(audio) == stats["chunks_processed"] * CHUNK_SAMPLES
    assert stats["paused_seconds"] >= 0.3


def test_send_while_paused_held_ends_turn_with_captured_audio(sim):
    (sim / "hold-mode").touch()
    paused = threading.Event()
    _ScriptedStream.feed.extend([True] * 10)
    t, box = _listen_async(paused, max_duration=30.0)
    assert _wait(lambda: box["stats"].get("chunks_processed", 0) == 10)
    _pause(box, paused)
    assert _listening(sim).get("held") is True and _listening(sim).get("paused") is True
    _ScriptedStream.fill = "loud"
    time.sleep(0.1)
    sent = time.monotonic()
    (sim / "turn-end.signal").touch()
    audio, speech = _finish(t, box, timeout=1.0)
    assert time.monotonic() - sent < 1.0
    assert box["stats"]["end_reason"] == "send"
    assert box["stats"]["held"] is True
    assert speech is True
    assert len(audio) == 10 * CHUNK_SAMPLES
    assert not (sim / "turn-end.signal").exists()  # consumed


def test_pause_timeout_ends_listen_with_paused_timeout(sim, monkeypatch):
    monkeypatch.setenv("VOICEMODE_PAUSE_MAX_SECONDS", "0.3")
    paused = threading.Event()
    _ScriptedStream.feed.extend([True] * 10)
    t, box = _listen_async(paused, max_duration=30.0)
    assert _wait(lambda: box["stats"].get("chunks_processed", 0) == 10)
    _pause(box, paused)
    _ScriptedStream.fill = "loud"
    audio, speech = _finish(t, box, timeout=2.0)
    stats = box["stats"]
    assert stats["end_reason"] == "paused_timeout"
    assert stats["paused_seconds"] >= 0.3
    assert speech is True  # speech before the pause is still delivered
    assert len(audio) == 10 * CHUNK_SAMPLES


def test_pause_deadline_from_caller_ends_listen(sim):
    paused = threading.Event()
    paused.set()  # paused before the listen even starts
    t, box = _listen_async(paused, max_duration=30.0, pause_deadline=time.monotonic() + 0.3)
    _ScriptedStream.fill = "loud"
    audio, speech = _finish(t, box, timeout=2.0)
    assert box["stats"]["end_reason"] == "paused_timeout"
    assert speech is False and len(audio) == 0


def test_paused_time_excluded_from_held_wall_clock_ceiling(sim, monkeypatch):
    monkeypatch.setenv("VOICEMODE_HOLD_MAX_SECONDS", "0.5")  # wall backstop at 1.5s
    (sim / "hold-mode").touch()
    paused = threading.Event()
    _ScriptedStream.feed.extend([True] * 5)
    t, box = _listen_async(paused, max_duration=30.0)
    assert _wait(lambda: box["stats"].get("chunks_processed", 0) == 5)
    _pause(box, paused)
    time.sleep(1.7)  # longer than the whole wall-clock ceiling
    _resume(box, paused)
    time.sleep(0.1)
    assert t.is_alive(), f"paused time counted toward the hold ceiling: {box}"
    (sim / "turn-end.signal").touch()
    _finish(t, box)
    assert box["stats"]["end_reason"] == "send"


def test_step_away_keeps_its_idle_pause(sim):
    """With step-away on, an idle pause stays step-away's: it keeps hearing him."""
    paused = threading.Event()
    paused.set()
    _ScriptedStream.fill = "quiet"
    t, box = _listen_async(paused, max_duration=0.3, step_away_enabled_override=True,
                           pause_check=paused.is_set)
    time.sleep(0.3)
    assert box["stats"].get("chunks_processed", 0) > 0
    assert "pause_count" not in box["stats"]
    (sim / "turn-end.signal").touch()
    _finish(t, box)


def test_recording_end_reports_paused_seconds_and_is_not_starved():
    data, starved = recording_end_event_data(
        30.0, int(2.0 * SAMPLE_RATE),
        {"end_reason": "send", "wall_seconds": 30.0, "paused_seconds": 27.5})
    assert data["paused_seconds"] == 27.5
    assert starved is False
    _, starved = recording_end_event_data(
        30.0, int(2.0 * SAMPLE_RATE), {"end_reason": "send", "wall_seconds": 30.0})
    assert starved is True


def test_default_check_is_the_real_pause_flag(sim):
    """No listen_pause_check passed: pause.flag (what the Pause button,
    AirPods and convomode-floor.py write) suspends the listen."""
    import os
    from voice_mode import audio_player
    flag = audio_player.PAUSE_FLAG_PATH
    os.makedirs(os.path.dirname(flag), exist_ok=True)
    box = {"stats": {}}

    def run():
        box["result"] = record_audio_with_silence_detection(max_duration=30.0, listen_stats=box["stats"])

    _ScriptedStream.feed.extend([True] * 10)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    try:
        assert _wait(lambda: box["stats"].get("chunks_processed", 0) == 10)
        open(flag, "w").close()
        assert _wait(lambda: box["stats"].get("pause_count", 0) == 1)
        _ScriptedStream.fill = "quiet"
        time.sleep(0.4)
        assert t.is_alive() and box["stats"]["chunks_processed"] == 10
        os.remove(flag)
        audio, speech = _finish(t, box)
        assert box["stats"]["end_reason"] == "silence" and speech is True
    finally:
        if os.path.exists(flag):
            os.remove(flag)
