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
        # Kept recording through 4x the silence threshold. (No upper bound:
        # on Send every chunk the fake mic already delivered is kept, and the
        # fake mic runs ahead of the loop.)
        assert _seconds(audio) >= SEND_AT * CHUNK_S - 0.05
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
        # Ended right after the floor -- nowhere near the 20s listen_duration_min
        # (the upper slack covers delivered-but-unprocessed audio kept on Send).
        assert C.MIN_RECORDING_DURATION - 2 * CHUNK_S <= _seconds(audio) < 5.0
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
        assert _seconds(audio) >= SEND_AT * CHUNK_S - 0.05
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

        fixed.assert_called_once_with(2.0, honor_send=True)
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


# --------------------------------------------------------------------------- #
# Send keeps delivered audio; listening.json; RECORDING_END diagnostics       #
# --------------------------------------------------------------------------- #

class _SlowVad(_FakeVad):
    """A consumer that falls behind the mic (CPU-starved server): each
    processed chunk costs 5ms while the fake mic delivers one per 0.5ms."""

    def is_speech(self, chunk_bytes, rate):
        time.sleep(0.005)
        return super().is_speech(chunk_bytes, rate)


class TestSendKeepsDeliveredAudio:
    def _run(self, sim, monkeypatch):
        monkeypatch.setattr(C.webrtcvad, "Vad", _SlowVad)
        (sim / "hold-mode").touch()
        stats = {}
        _FakeVad.on_processed = _touch_at(sim / "turn-end.signal", 40)
        audio, _ = record_audio_with_silence_detection(max_duration=30.0, listen_stats=stats)
        return audio, stats

    def test_backlog_is_kept_on_send(self, sim, monkeypatch):
        audio, stats = self._run(sim, monkeypatch)
        assert stats["end_reason"] == "send"
        assert stats["chunks_drained"] > 0
        assert stats["chunks_delivered"] > stats["chunks_processed"]
        # Every chunk the mic delivered before Send is in the result.
        assert len(audio) >= (stats["chunks_processed"] + stats["chunks_drained"]) * CHUNK_SAMPLES

    def test_without_drain_the_backlog_is_lost(self, sim, monkeypatch):
        """On/off control for the test above: with the drain disabled, a slow
        consumer returns only what it processed -- the pre-fix behavior."""
        monkeypatch.setattr(C, "_drain_delivered_audio", lambda q, chunks: 0)
        audio, stats = self._run(sim, monkeypatch)
        assert stats["chunks_delivered"] > stats["chunks_processed"]
        assert len(audio) == stats["chunks_processed"] * CHUNK_SAMPLES


class TestListeningState:
    def _state(self, sim):
        import json
        path = sim / "listening.json"
        return json.loads(path.read_text()) if path.exists() else None

    def test_written_while_listening_and_removed_after(self, sim):
        seen = []
        (sim / "hold-mode").touch()

        def on_processed(n):
            if n == 5:
                seen.append(self._state(sim))
            if n == 20:
                (sim / "turn-end.signal").touch()
        _FakeVad.on_processed = on_processed
        before = time.time()

        record_audio_with_silence_detection(max_duration=30.0)

        state = seen[0]
        assert state["held"] is True
        assert state["pid"] == os.getpid()
        assert before - 1 <= state["started_at"] <= time.time()
        assert not (sim / "listening.json").exists()

    def test_held_flips_true_when_hold_open_appears(self, sim):
        seen = {}
        _FakeInputStream.script = _speech_then_silence(SPEECH_N)

        def on_processed(n):
            if n == 3:
                seen["before"] = self._state(sim)
                (sim / "hold-open.signal").touch()
            if n == 10:
                seen["after"] = self._state(sim)
            if n == 30:
                (sim / "turn-end.signal").touch()
        _FakeVad.on_processed = on_processed

        record_audio_with_silence_detection(max_duration=30.0)

        assert seen["before"]["held"] is False
        assert seen["after"]["held"] is True
        assert seen["after"]["started_at"] == seen["before"]["started_at"]
        assert not (sim / "listening.json").exists()

    def test_removed_when_recording_raises(self, sim, monkeypatch):
        monkeypatch.setattr(C, "_record_audio_with_silence_detection",
                            MagicMock(side_effect=RuntimeError("boom")))
        with pytest.raises(RuntimeError):
            record_audio_with_silence_detection(max_duration=1.0)
        assert not (sim / "listening.json").exists()


class TestEndReason:
    def _stats(self, **kwargs):
        stats = {}
        record_audio_with_silence_detection(listen_stats=stats, **kwargs)
        return stats

    def test_send(self, sim):
        _FakeVad.on_processed = _touch_at(sim / "turn-end.signal", 25)
        stats = self._stats(max_duration=30.0)
        assert stats["end_reason"] == "send"
        assert stats["held"] is False

    def test_silence(self, sim):
        _FakeInputStream.script = _speech_then_silence(SPEECH_N)
        assert self._stats(max_duration=30.0)["end_reason"] == "silence"

    def test_no_speech(self, sim):
        assert self._stats(max_duration=0.3)["end_reason"] == "no_speech"

    def test_max_duration(self, sim):
        _FakeInputStream.script = _speech_then_silence(200)
        assert self._stats(max_duration=1.0)["end_reason"] == "max_duration"

    def test_hold_ceiling(self, sim, monkeypatch):
        monkeypatch.setenv("VOICEMODE_HOLD_MAX_SECONDS", "0.5")
        (sim / "hold-mode").touch()
        stats = self._stats(max_duration=30.0)
        assert stats["end_reason"] == "hold_ceiling"
        assert stats["held"] is True

    def test_yield(self, sim, monkeypatch):
        monkeypatch.setattr(C, "CONCH_YIELD_GRACE_SECONDS", 0.1)
        stats = self._stats(max_duration=30.0, yield_check=lambda: True, yield_state={"yielded": False})
        assert stats["end_reason"] == "yield"

    def test_supersede(self, sim):
        stats = self._stats(max_duration=30.0, supersede_check=lambda: True, yield_state={"yielded": False})
        assert stats["end_reason"] == "supersede"

    def test_endpoint(self, sim, monkeypatch):
        monkeypatch.setattr(C, "ENDPOINTING_ENABLED", True)
        monkeypatch.setattr(C, "SILERO_AVAILABLE", True)
        monkeypatch.setattr(C, "Endpointer", _FakeEndpointer)
        assert self._stats(max_duration=30.0)["end_reason"] == "endpoint"

    def test_device_error(self, sim, monkeypatch):
        class _BrokenStream(_FakeInputStream):
            def __init__(self, **kw):
                # Not one of the "recoverable" strings, so no reopen attempts.
                raise RuntimeError("stream open failed")
        C.sd.InputStream = _BrokenStream
        monkeypatch.setattr(C, "record_audio", MagicMock(return_value=np.zeros(10, dtype=np.int16)))
        assert self._stats(max_duration=1.0)["end_reason"] == "device_error"

    def test_held_no_vad_send(self, sim):
        (sim / "hold-mode").touch()
        _FakeInputStream.on_chunk = _touch_at(sim / "turn-end.signal", 30)
        stats = self._stats(max_duration=30.0, disable_silence_detection=True)
        assert stats["end_reason"] == "send"
        assert stats["held"] is True

    def test_stats_carry_samples_and_seconds(self, sim):
        _FakeInputStream.script = _speech_then_silence(SPEECH_N)
        stats = {}
        audio, _ = record_audio_with_silence_detection(max_duration=30.0, listen_stats=stats)
        assert stats["samples"] == len(audio)
        assert stats["audio_seconds"] == pytest.approx(len(audio) / SAMPLE_RATE)
        assert stats["wall_seconds"] > 0


class TestRecordingEndEventData:
    def test_fields(self):
        data, starved = C.recording_end_event_data(
            2.5, 48000, {"end_reason": "send", "held": True, "wall_seconds": 2.4,
                         "chunks_delivered": 70, "chunks_processed": 66, "chunks_drained": 4})
        assert data == {
            "duration": 2.5, "samples": 48000, "audio_seconds": 2.0, "wall_seconds": 2.4,
            "end_reason": "send", "held": True, "paused_seconds": 0.0, "send_source": "unknown",
            "chunks_delivered": 70, "chunks_processed": 66, "chunks_drained": 4,
        }
        assert starved is False

    def test_live_failure_signature_is_starved(self):
        # The 2026-10-23 20:31 turn: 12240 samples (0.51s) in 23.6s.
        data, starved = C.recording_end_event_data(23.637, 12240, {"wall_seconds": 23.6})
        assert data["audio_seconds"] == 0.51
        assert starved is True

    def test_short_listens_never_flag(self):
        _, starved = C.recording_end_event_data(0.4, 0, {"wall_seconds": 0.4, "end_reason": "supersede"})
        assert starved is False

    def test_missing_stats_fall_back(self):
        data, _ = C.recording_end_event_data(3.0, 72000, None)
        assert data["end_reason"] == "other"
        assert data["held"] is False
        assert data["wall_seconds"] == 3.0
