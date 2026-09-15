"""Tests for push-to-talk hold (W3d push-to-talk dispatch, 2026-09-15).

Covers the ONE addition to the LISTEN side of convomode this dispatch makes:
while ~/.voicemode/ptt-hold.flag exists (a Hammerspoon key-down writes it, key-up
removes it AND drops the pre-existing turn-end.signal -- see
convomode-turn-end.sh's `hold`/`release` subcommands, and converse.py's
"Honor a manual turn-end signal first" block, both untouched by this dispatch),
the silence-based stop decision is suppressed entirely so a natural mid-thought
pause can never end the turn early. INERT BY DEFAULT: with the flag absent
(the default), the loop behaves byte-for-byte as before -- proven by
TestRecordLoopInertByDefault below, mirroring tests/test_pause_stepaway_append.py's
own acceptance-test shape for the sibling step-away/append features.
"""

import os
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# webrtcvad is an optional C dep; mock it so the module imports everywhere.
import sys
sys.modules.setdefault("webrtcvad", MagicMock())

import voice_mode.tools.converse as C
from voice_mode.tools.converse import (
    ptt_hold_active,
    record_audio_with_silence_detection,
)
from voice_mode.config import SAMPLE_RATE, VAD_CHUNK_DURATION_MS, SILENCE_THRESHOLD_MS

CHUNK_SAMPLES = int(SAMPLE_RATE * VAD_CHUNK_DURATION_MS / 1000)


# --------------------------------------------------------------------------- #
# Pure helper                                                                  #
# --------------------------------------------------------------------------- #

class TestPttHoldActive:
    def test_default_off_no_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(C, "PTT_HOLD_FLAG_PATH", str(tmp_path / "nope.flag"))
        assert ptt_hold_active() is False

    def test_true_while_flag_file_exists(self, monkeypatch, tmp_path):
        flag = tmp_path / "ptt-hold.flag"
        monkeypatch.setattr(C, "PTT_HOLD_FLAG_PATH", str(flag))
        assert ptt_hold_active() is False
        flag.write_text("")
        assert ptt_hold_active() is True
        flag.unlink()
        assert ptt_hold_active() is False

    def test_fail_safe_to_not_holding_on_error(self, monkeypatch):
        # An unreadable/racy check must fail to False (not holding) -- a stuck
        # True would silently disable silence-based stopping forever.
        def _boom(_path):
            raise OSError("simulated permission error")
        monkeypatch.setattr(C.os.path, "exists", _boom)
        assert ptt_hold_active() is False


# --------------------------------------------------------------------------- #
# Record-loop simulation harness (fake mic + fake VAD) -- mirrors             #
# tests/test_pause_stepaway_append.py's harness, kept self-contained here     #
# because that file defines it as a local fixture, not a shared one.          #
# --------------------------------------------------------------------------- #

def _speech_chunk():
    return (np.random.randint(-8000, 8000, size=CHUNK_SAMPLES, dtype=np.int16)
            .reshape(-1, 1))


def _silence_chunk():
    return np.zeros((CHUNK_SAMPLES, 1), dtype=np.int16)


class _FakeInputStream:
    """Stands in for sd.InputStream: on __enter__ spawns a daemon thread that
    calls the loop's callback with scripted chunks, then feeds silence forever
    (the loop ends via VAD/silence-threshold/max-duration, never on us).

    `on_index` (optional, {chunk_index: callable}) fires just before delivering
    that chunk -- lets a test pin an action (e.g. releasing the PTT-hold flag)
    to chunk PROGRESS instead of racing it against wall-clock time, which does
    not track the loop's internal (chunk-count-derived) recording_duration.
    """

    on_index = {}

    def __init__(self, *, samplerate, channels, dtype, callback, blocksize):
        self.callback = callback
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        def pump():
            for i, ch in enumerate(_FakeInputStream.script):
                if self._stop.is_set():
                    return
                fn = _FakeInputStream.on_index.get(i)
                if fn is not None:
                    try:
                        fn()
                    except Exception:
                        pass
                self.callback(ch, len(ch), None, None)
                time.sleep(0.001)
            while not self._stop.is_set():
                self.callback(_silence_chunk(), CHUNK_SAMPLES, None, None)
                time.sleep(0.001)
        self._thread = threading.Thread(target=pump, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        return False


class _FakeVad:
    """is_speech() decides from the chunk's byte energy -- loud -> speech."""
    def __init__(self, *a):
        pass

    def is_speech(self, chunk_bytes, rate):
        arr = np.frombuffer(chunk_bytes, dtype=np.int16).astype(float)
        return bool(np.sqrt(np.mean(arr ** 2)) > 500) if arr.size else False


@pytest.fixture
def sim_env(monkeypatch):
    """Wire the fakes into the record loop. Yields a helper to set the script."""
    fake_sd = MagicMock()
    fake_sd.InputStream = _FakeInputStream
    fake_sd.PortAudioError = RuntimeError
    monkeypatch.setattr(C, "sd", fake_sd)
    monkeypatch.setattr(C, "VAD_AVAILABLE", True)
    monkeypatch.setattr(C, "DISABLE_SILENCE_DETECTION", False)
    # Force the webrtcvad path deterministically regardless of this machine's
    # VOICEMODE_ENDPOINTING env var (some dev shells default it on) -- this
    # harness's _FakeVad only wires into that path. Mirrors the same
    # environment-independence tests/test_pause_stepaway_append.py's own
    # harness relies on implicitly.
    monkeypatch.setattr(C, "ENDPOINTING_ENABLED", False)
    fake_webrtc = MagicMock()
    fake_webrtc.Vad = _FakeVad
    monkeypatch.setattr(C, "webrtcvad", fake_webrtc)

    _FakeInputStream.on_index = {}

    def set_script(chunks):
        _FakeInputStream.script = chunks
    return set_script


def _script_speech_then_silence(speech_n, silence_n):
    return [_speech_chunk() for _ in range(speech_n)] + \
           [_silence_chunk() for _ in range(silence_n)]


# --------------------------------------------------------------------------- #
# Inert-by-default acceptance test                                            #
# --------------------------------------------------------------------------- #

class TestRecordLoopInertByDefault:
    def test_no_ptt_flag_matches_baseline_stop(self, sim_env, monkeypatch, tmp_path):
        # PTT_HOLD_FLAG_PATH points at a file that will never exist -> fully inert.
        monkeypatch.setattr(C, "PTT_HOLD_FLAG_PATH", str(tmp_path / "nope.flag"))
        sim_env(_script_speech_then_silence(20, 200))
        audio, speech = record_audio_with_silence_detection(max_duration=8.0)
        assert speech is True
        # Stopped on silence well before the 8s ceiling (baseline VAD behavior,
        # unchanged by this dispatch).
        assert len(audio) < 8.0 * SAMPLE_RATE


# --------------------------------------------------------------------------- #
# PTT hold suppresses the silence-based stop                                  #
# --------------------------------------------------------------------------- #

class TestPttHoldSuppressesSilenceStop:
    def test_held_throughout_runs_to_max_duration(self, sim_env, monkeypatch, tmp_path):
        # Sized off the LIVE SILENCE_THRESHOLD_MS (this can be env-overridden
        # per machine/dev-shell, e.g. VOICEMODE_SILENCE_THRESHOLD_MS) rather
        # than a hardcoded constant, mirroring how
        # tests/test_pause_stepaway_append.py's TestAppendToTurn sizes its own
        # gap off the same live config value.
        speech_n = 20
        stop_point_ms = speech_n * VAD_CHUNK_DURATION_MS + SILENCE_THRESHOLD_MS
        max_duration = (stop_point_ms / 1000.0) * 3  # generous ceiling past the stop point
        silence_n = int(max_duration * 1000 / VAD_CHUNK_DURATION_MS) + 20  # covers the whole ceiling
        script = _script_speech_then_silence(speech_n, silence_n)

        # Baseline: no hold -> stops early at the silence threshold.
        monkeypatch.setattr(C, "PTT_HOLD_FLAG_PATH", str(tmp_path / "nope.flag"))
        sim_env(script)
        baseline_audio, baseline_speech = record_audio_with_silence_detection(max_duration=max_duration)

        # Held throughout: same script, flag present the whole call -> silence
        # never trips the stop, so the loop runs all the way to max_duration.
        flag = tmp_path / "ptt-hold.flag"
        flag.write_text("")
        monkeypatch.setattr(C, "PTT_HOLD_FLAG_PATH", str(flag))
        sim_env(script)
        held_audio, held_speech = record_audio_with_silence_detection(max_duration=max_duration)

        assert baseline_speech is True and held_speech is True
        assert len(baseline_audio) < 0.6 * max_duration * SAMPLE_RATE   # stopped meaningfully early
        assert len(held_audio) >= 0.9 * max_duration * SAMPLE_RATE      # ran to the ceiling

    def test_release_mid_call_lets_silence_stop_fire(self, sim_env, monkeypatch, tmp_path):
        # silence_duration_ms accumulates unconditionally even while held (only
        # the STOP action is suppressed -- see the gate added around the
        # `stop_recording = True` line) -- so by the time PTT is released well
        # past the silence threshold, the threshold is ALREADY satisfied and the
        # very next loop iteration stops almost immediately, no further wait.
        # Release is pinned to CHUNK PROGRESS via on_index (not a wall-clock
        # sleep, which the fake pump's far-faster-than-real-time delivery would
        # race against and make flaky). The companion turn-end.signal mechanism
        # that makes a real key-release end the turn INSTANTLY is pre-existing,
        # untouched by this dispatch, and out of scope for this harness -- it
        # lives outside ptt_hold_active()'s suppression gate.
        flag = tmp_path / "ptt-hold.flag"
        flag.write_text("")
        monkeypatch.setattr(C, "PTT_HOLD_FLAG_PATH", str(flag))

        speech_n = 20
        held_silence_chunks = int((SILENCE_THRESHOLD_MS / VAD_CHUNK_DURATION_MS) * 3)  # 3x past threshold
        tail_silence_chunks = held_silence_chunks + 200  # plenty of runway after release
        script = _script_speech_then_silence(speech_n, held_silence_chunks + tail_silence_chunks)
        sim_env(script)
        release_index = speech_n + held_silence_chunks - 1
        _FakeInputStream.on_index = {release_index: flag.unlink}

        max_duration = ((speech_n + held_silence_chunks + tail_silence_chunks)
                        * VAD_CHUNK_DURATION_MS / 1000.0)
        audio, speech = record_audio_with_silence_detection(max_duration=max_duration)

        assert speech is True
        expected_stop_samples = (speech_n + held_silence_chunks) * CHUNK_SAMPLES
        # Stopped at (essentially) the release point, not the far-away ceiling --
        # generous tolerance for scheduling jitter around exactly which chunk
        # the stop check lands on.
        assert len(audio) < expected_stop_samples + 20 * CHUNK_SAMPLES
        assert len(audio) < 0.5 * max_duration * SAMPLE_RATE
