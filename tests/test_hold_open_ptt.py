"""Tests for hold-open push-to-talk in record_audio_with_silence_detection().

Signal files (all under a temp HOME's ~/.voicemode):
  turn-end.signal  SEND -- ends the listen.
  hold-open.signal per-turn HOLD -- silence never ends the listen once seen;
                   removed when the listen ends; cleared on entry only if stale.
  hold-mode        persistent HOLD MODE -- every listen held; never removed.

Uses the same fake mic + fake webrtcvad harness as test_pause_stepaway_append.py:
a daemon thread feeds scripted chunks into the loop's callback, then silence
forever, so the loop can only end via VAD/silence, a signal, or a ceiling.
"""

import os
import sys
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault("webrtcvad", MagicMock())

import voice_mode.tools.converse as C
from voice_mode.tools.converse import record_audio_with_silence_detection
from voice_mode.config import SAMPLE_RATE, VAD_CHUNK_DURATION_MS


CHUNK_SAMPLES = int(SAMPLE_RATE * VAD_CHUNK_DURATION_MS / 1000)
CHUNK_S = VAD_CHUNK_DURATION_MS / 1000
SILENCE_MS = 20 * VAD_CHUNK_DURATION_MS   # pinned so a local env override can't skew timing


def _speech_chunk():
    return (np.random.randint(-8000, 8000, size=CHUNK_SAMPLES, dtype=np.int16)
            .reshape(-1, 1))


def _silence_chunk():
    return np.zeros((CHUNK_SAMPLES, 1), dtype=np.int16)


class _FakeInputStream:
    """sd.InputStream stand-in. Feeds `script`, then silence forever. After
    each delivered chunk, runs `on_chunk(n)` (n = chunks delivered so far) so a
    test can drop signal files at a precise point in the recording."""

    script = []
    on_chunk = None

    def __init__(self, *, samplerate, channels, dtype, callback, blocksize):
        self.callback = callback
        self._stop = threading.Event()

    def __enter__(self):
        def pump():
            n = 0
            script = list(_FakeInputStream.script)
            while not self._stop.is_set():
                ch = script.pop(0) if script else _silence_chunk()
                self.callback(ch, len(ch), None, None)
                n += 1
                if _FakeInputStream.on_chunk is not None:
                    _FakeInputStream.on_chunk(n)
                time.sleep(0.0005)
        threading.Thread(target=pump, daemon=True).start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        return False


class _FakeVad:
    """Loud -> speech. Also counts chunks the loop has actually PROCESSED and
    runs `on_processed(n)` -- the VAD-loop tests drop signal files from here,
    because the pump runs ahead of the consumer and a pump-side trigger would
    fire before the loop has reached that point in the recording."""

    on_processed = None
    count = 0

    def __init__(self, *a):
        pass

    def is_speech(self, chunk_bytes, rate):
        _FakeVad.count += 1
        if _FakeVad.on_processed is not None:
            _FakeVad.on_processed(_FakeVad.count)
        arr = np.frombuffer(chunk_bytes, dtype=np.int16).astype(float)
        return bool(np.sqrt(np.mean(arr ** 2)) > 500) if arr.size else False


@pytest.fixture
def home(isolate_home_directory, monkeypatch):
    """conftest's autouse fixture already points expanduser("~") at a temp
    home; the signal files live in its .voicemode."""
    monkeypatch.delenv("VOICEMODE_HOLD_MAX_SECONDS", raising=False)
    return isolate_home_directory / ".voicemode"


@pytest.fixture
def sim(monkeypatch, home):
    fake_sd = MagicMock()
    fake_sd.InputStream = _FakeInputStream
    fake_sd.PortAudioError = RuntimeError
    monkeypatch.setattr(C, "sd", fake_sd)
    monkeypatch.setattr(C, "_bluetooth_input_active", lambda: False)
    monkeypatch.setattr(C, "VAD_AVAILABLE", True)
    monkeypatch.setattr(C, "DISABLE_SILENCE_DETECTION", False)
    monkeypatch.setattr(C, "ENDPOINTING_ENABLED", False)
    monkeypatch.setattr(C, "VAD_ENERGY_THRESHOLD", 0)
    monkeypatch.setattr(C, "SILENCE_THRESHOLD_MS", SILENCE_MS)
    monkeypatch.setattr(C, "STEP_AWAY_ENV", False)
    monkeypatch.setattr(C, "STEP_AWAY_FLAG_PATH", str(home / "nonexistent-step-away"))
    monkeypatch.setattr(C, "APPEND_WINDOW_MS", 0)
    monkeypatch.setattr(C, "APPEND_WINDOW_FLAG_PATH", str(home / "nonexistent-append"))
    # record_audio() is the fixed-length path; it must never run while held.
    monkeypatch.setattr(C, "record_audio", MagicMock(side_effect=AssertionError("fixed-length record_audio used")))
    fake_webrtc = MagicMock()
    fake_webrtc.Vad = _FakeVad
    monkeypatch.setattr(C, "webrtcvad", fake_webrtc)
    _FakeInputStream.script = []
    _FakeInputStream.on_chunk = None
    _FakeVad.on_processed = None
    _FakeVad.count = 0
    yield home
    _FakeInputStream.on_chunk = None
    _FakeVad.on_processed = None


def _speech_then_silence(speech_n, silence_n=0):
    return [_speech_chunk() for _ in range(speech_n)] + [_silence_chunk() for _ in range(silence_n)]


def _touch_at(path, at_chunk):
    def on_chunk(n):
        if n == at_chunk:
            path.touch()
    return on_chunk


def _seconds(audio):
    return len(audio) / SAMPLE_RATE


# Silence threshold in chunks, and a "well past it" point to send at.
SILENCE_CHUNKS = SILENCE_MS // VAD_CHUNK_DURATION_MS
SPEECH_N = 30                                   # 0.9s of speech
SEND_AT = SPEECH_N + 4 * SILENCE_CHUNKS         # 4x the silence threshold later


class TestHeldVadLoop:
    def test_a_hold_mode_survives_silence_and_ends_on_send(self, sim):
        (sim / "hold-mode").touch()
        (sim / "hold-open.signal").touch()
        _FakeInputStream.script = _speech_then_silence(SPEECH_N)
        _FakeVad.on_processed = _touch_at(sim / "turn-end.signal", SEND_AT)

        audio, speech = record_audio_with_silence_detection(max_duration=30.0)

        assert speech is True
        # Kept recording through 4x the silence threshold, stopped near Send.
        assert _seconds(audio) >= SEND_AT * CHUNK_S - 0.05
        assert _seconds(audio) < SEND_AT * CHUNK_S + 1.0
        assert not (sim / "turn-end.signal").exists()      # consumed
        assert not (sim / "hold-open.signal").exists()     # per-turn hold cleared
        assert (sim / "hold-mode").exists()                # persistent mode kept

    def test_b_not_held_stops_on_silence(self, sim):
        _FakeInputStream.script = _speech_then_silence(SPEECH_N)
        # A Send far in the future must not be what ends this listen.
        _FakeVad.on_processed = _touch_at(sim / "turn-end.signal", SEND_AT)

        audio, speech = record_audio_with_silence_detection(max_duration=30.0)

        assert speech is True
        expected = (SPEECH_N + SILENCE_CHUNKS) * CHUNK_S
        assert abs(_seconds(audio) - expected) <= 3 * CHUNK_S
        assert _seconds(audio) < SEND_AT * CHUNK_S - 1.0

    def test_c_hold_open_mid_recording_suppresses_silence_stop(self, sim):
        _FakeInputStream.script = _speech_then_silence(SPEECH_N)

        def on_chunk(n):
            if n == SPEECH_N - 10:           # while still speaking
                (sim / "hold-open.signal").touch()
            if n == SEND_AT:
                (sim / "turn-end.signal").touch()
        _FakeVad.on_processed = on_chunk

        audio, speech = record_audio_with_silence_detection(max_duration=30.0)

        assert speech is True
        assert _seconds(audio) >= SEND_AT * CHUNK_S - 0.05
        assert not (sim / "hold-open.signal").exists()

    def test_e_stale_hold_open_cleared_and_not_held(self, sim):
        stale = sim / "hold-open.signal"
        stale.touch()
        old = time.time() - (C.HOLD_OPEN_STALE_SECONDS + 60)
        os.utime(stale, (old, old))
        _FakeInputStream.script = _speech_then_silence(SPEECH_N)

        audio, speech = record_audio_with_silence_detection(max_duration=30.0)

        expected = (SPEECH_N + SILENCE_CHUNKS) * CHUNK_S
        assert abs(_seconds(audio) - expected) <= 3 * CHUNK_S   # silence ended it
        assert not stale.exists()

    def test_e_fresh_hold_open_is_honored(self, sim):
        (sim / "hold-open.signal").touch()   # pressed during TTS, before listening
        _FakeInputStream.script = _speech_then_silence(SPEECH_N)
        _FakeVad.on_processed = _touch_at(sim / "turn-end.signal", SEND_AT)

        audio, speech = record_audio_with_silence_detection(max_duration=30.0)

        assert _seconds(audio) >= SEND_AT * CHUNK_S - 0.05
        assert not (sim / "hold-open.signal").exists()

    def test_f_hold_ceiling_stops_recording(self, sim, monkeypatch):
        monkeypatch.setenv("VOICEMODE_HOLD_MAX_SECONDS", "1")
        (sim / "hold-mode").touch()
        _FakeInputStream.script = _speech_then_silence(SPEECH_N)

        t0 = time.monotonic()
        audio, speech = record_audio_with_silence_detection(max_duration=30.0)

        assert abs(_seconds(audio) - 1.0) <= 2 * CHUNK_S
        assert time.monotonic() - t0 < 10

    def test_held_ceiling_replaces_listen_max(self, sim, monkeypatch):
        monkeypatch.setenv("VOICEMODE_HOLD_MAX_SECONDS", "3")
        (sim / "hold-mode").touch()
        _FakeInputStream.script = _speech_then_silence(SPEECH_N)

        audio, _ = record_audio_with_silence_detection(max_duration=1.5)

        assert _seconds(audio) >= 3.0 - 2 * CHUNK_S

    def test_send_before_min_duration_is_deferred_not_dropped(self, sim):
        (sim / "hold-mode").touch()
        _FakeVad.on_processed = _touch_at(sim / "turn-end.signal", 2)   # at 0.06s

        audio, speech = record_audio_with_silence_detection(max_duration=30.0, min_duration=20.0)

        # Held: listen_duration_min (20s) does not delay Send; only the tiny
        # MIN_RECORDING_DURATION floor applies.
        assert C.MIN_RECORDING_DURATION - 2 * CHUNK_S <= _seconds(audio) <= C.MIN_RECORDING_DURATION + 0.5
        assert speech is True

    def test_held_yield_request_ignored(self, sim, monkeypatch):
        monkeypatch.setattr(C, "CONCH_YIELD_GRACE_SECONDS", 0.1)
        (sim / "hold-mode").touch()
        _FakeVad.on_processed = _touch_at(sim / "turn-end.signal", 60)
        state = {"yielded": False}

        audio, _ = record_audio_with_silence_detection(
            max_duration=30.0, yield_check=lambda: True, yield_state=state)

        assert state["yielded"] is False
        assert _seconds(audio) >= 60 * CHUNK_S - 0.05

    def test_hold_open_removed_even_when_recording_raises(self, sim, monkeypatch):
        (sim / "hold-open.signal").touch()
        monkeypatch.setattr(C, "_record_audio_with_silence_detection",
                            MagicMock(side_effect=RuntimeError("boom")))
        with pytest.raises(RuntimeError):
            record_audio_with_silence_detection(max_duration=1.0)
        assert not (sim / "hold-open.signal").exists()


class _FakeEndpointer:
    """Silero Endpointer stand-in: speech_started from the first frame,
    endpointed (end of turn) from frame `endpoint_at` on. Counts processed
    frames for on_processed like _FakeVad does."""

    endpoint_at = SPEECH_N

    def __init__(self, **kw):
        self.n = 0

    def reset(self):
        self.n = 0

    def update(self, frame):
        self.n += 1
        if _FakeVad.on_processed is not None:
            _FakeVad.on_processed(self.n)
        return MagicMock(speech_started=True, endpointed=self.n >= _FakeEndpointer.endpoint_at,
                         speech_prob=0.0)


class TestHeldSileroEndpointing:
    @pytest.fixture
    def silero(self, sim, monkeypatch):
        monkeypatch.setattr(C, "ENDPOINTING_ENABLED", True)
        monkeypatch.setattr(C, "SILERO_AVAILABLE", True)
        monkeypatch.setattr(C, "Endpointer", _FakeEndpointer)
        return sim

    def test_not_held_endpoint_stops(self, silero):
        audio, _ = record_audio_with_silence_detection(max_duration=30.0)
        assert abs(_seconds(audio) - SPEECH_N * CHUNK_S) <= 3 * CHUNK_S

    def test_held_endpoint_suppressed_until_send(self, silero):
        (silero / "hold-mode").touch()
        _FakeVad.on_processed = _touch_at(silero / "turn-end.signal", SEND_AT)

        audio, speech = record_audio_with_silence_detection(max_duration=30.0)

        assert speech is True
        assert _seconds(audio) >= SEND_AT * CHUNK_S - 0.05


class TestHeldWithoutVad:
    def test_d_disable_silence_detection_held_ends_on_send(self, sim):
        (sim / "hold-mode").touch()
        _FakeInputStream.script = _speech_then_silence(SPEECH_N)
        _FakeInputStream.on_chunk = _touch_at(sim / "turn-end.signal", SEND_AT)

        audio, speech = record_audio_with_silence_detection(
            max_duration=30.0, disable_silence_detection=True)

        assert speech is True
        assert SEND_AT * CHUNK_S - 0.05 <= _seconds(audio) < SEND_AT * CHUNK_S + 1.0
        assert not (sim / "turn-end.signal").exists()
        assert (sim / "hold-mode").exists()

    def test_d_disable_silence_detection_held_not_cut_at_max_duration(self, sim):
        (sim / "hold-mode").touch()
        send_at = int(2.0 / CHUNK_S)                     # Send at 2s ...
        _FakeInputStream.on_chunk = _touch_at(sim / "turn-end.signal", send_at)

        audio, _ = record_audio_with_silence_detection(
            max_duration=1.0, disable_silence_detection=True)   # ... past the 1s max

        assert _seconds(audio) >= 2.0 - 0.05

    def test_disable_silence_detection_not_held_uses_fixed_length(self, sim, monkeypatch):
        fixed = MagicMock(return_value=np.zeros(10, dtype=np.int16))
        monkeypatch.setattr(C, "record_audio", fixed)

        audio, speech = record_audio_with_silence_detection(
            max_duration=2.0, disable_silence_detection=True)

        fixed.assert_called_once_with(2.0)
        assert speech is True

    def test_held_no_vad_ceiling(self, sim, monkeypatch):
        monkeypatch.setenv("VOICEMODE_HOLD_MAX_SECONDS", "1")
        (sim / "hold-open.signal").touch()

        audio, _ = record_audio_with_silence_detection(
            max_duration=30.0, disable_silence_detection=True)

        assert abs(_seconds(audio) - 1.0) <= 2 * CHUNK_S
        assert not (sim / "hold-open.signal").exists()


class TestHelpers:
    def test_hold_max_seconds(self, monkeypatch):
        monkeypatch.delenv("VOICEMODE_HOLD_MAX_SECONDS", raising=False)
        assert C.hold_max_seconds() == 600.0
        monkeypatch.setenv("VOICEMODE_HOLD_MAX_SECONDS", "42")
        assert C.hold_max_seconds() == 42.0
        for bad in ("abc", "0", "-5"):
            monkeypatch.setenv("VOICEMODE_HOLD_MAX_SECONDS", bad)
            assert C.hold_max_seconds() == 600.0

    def test_hold_active(self, home):
        assert C.hold_active() is False
        (home / "hold-mode").touch()
        assert C.hold_active() is True
        (home / "hold-mode").unlink()
        (home / "hold-open.signal").touch()
        assert C.hold_active() is True
