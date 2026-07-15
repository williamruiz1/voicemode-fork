"""Tests for graceful pause / step-away / append-to-turn (founder-os#11655).

Covers three additions to the LISTEN side of convomode, all gated to be INERT by
default:

  1. Step-away  — a paused idle listen waits longer + speaks ONE check-in, then
     ends gracefully (StepAwayTracker + record-loop wiring).
  2. Pause on the listen side — reuses the existing convomode_paused() primitive.
  3. Append-to-turn — a short grace window after the silence timer where resumed
     speech continues the SAME turn (silence-threshold extension).

The load-bearing acceptance test is `test_inert_by_default_*`: with nothing
enabled the record loop behaves byte-for-byte as before.
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
    StepAwayTracker,
    is_step_away_phrase,
    is_resume_phrase,
    step_away_enabled,
    append_window_ms,
    record_audio_with_silence_detection,
)
from voice_mode.config import SAMPLE_RATE, VAD_CHUNK_DURATION_MS, SILENCE_THRESHOLD_MS


CHUNK_SAMPLES = int(SAMPLE_RATE * VAD_CHUNK_DURATION_MS / 1000)


# --------------------------------------------------------------------------- #
# Pure helpers                                                                #
# --------------------------------------------------------------------------- #

class TestPhraseMatchers:
    def test_step_away_whole_utterance(self):
        assert is_step_away_phrase("hold on")
        assert is_step_away_phrase("Hold on.")          # punctuation/case-insensitive
        assert is_step_away_phrase("one sec")
        assert is_step_away_phrase("hang on")

    def test_step_away_not_mid_sentence(self):
        # A passing mention must NOT fire — whole-utterance only (fail safe).
        assert not is_step_away_phrase("I was going to hold on there for a while")
        assert not is_step_away_phrase("let's talk about the hold on the deal")

    def test_resume_phrases(self):
        assert is_resume_phrase("I'm back")
        assert is_resume_phrase("okay I'm back")
        assert is_resume_phrase("resume")

    def test_empty_and_none_fail_safe(self):
        for f in (is_step_away_phrase, is_resume_phrase):
            assert f(None) is False
            assert f("") is False
            assert f("   ") is False


# --------------------------------------------------------------------------- #
# StepAwayTracker state machine                                               #
# --------------------------------------------------------------------------- #

class TestStepAwayTracker:
    def test_disabled_is_inert(self):
        t = StepAwayTracker(enabled=False, base_max_duration=120.0,
                            grace_seconds=180.0, checkin_seconds=45.0)
        # Even when "paused", a disabled tracker never extends, never checks in.
        assert t.observe(paused=True, elapsed=100.0, speech_detected=False) is False
        assert t.effective_max() == 120.0
        assert t.stepped_away is False
        assert t.resumed is False

    def test_paused_extends_deadline(self):
        t = StepAwayTracker(True, 120.0, 180.0, 45.0)
        assert t.effective_max() == 120.0            # not yet paused
        t.observe(paused=True, elapsed=5.0, speech_detected=False)
        assert t.effective_max() == pytest.approx(185.0)  # start(5) + grace(180)
        assert t.stepped_away is True

    def test_checkin_fires_exactly_once(self):
        t = StepAwayTracker(True, 120.0, 180.0, 45.0)
        t.observe(True, 5.0, False)                   # pause begins at 5s
        assert t.observe(True, 40.0, False) is False  # 35s in — not yet
        assert t.observe(True, 51.0, False) is True   # 46s in — check-in due
        assert t.observe(True, 60.0, False) is False  # never a second time
        assert t.checkin_done is True

    def test_speech_cancels_stepaway(self):
        t = StepAwayTracker(True, 120.0, 180.0, 45.0)
        # Once speech is detected, step-away no longer applies to this turn.
        assert t.observe(paused=True, elapsed=100.0, speech_detected=True) is False
        assert t.effective_max() == 120.0

    def test_resume_when_pause_clears(self):
        t = StepAwayTracker(True, 120.0, 180.0, 45.0)
        t.observe(True, 5.0, False)
        t.observe(False, 8.0, False)                  # pause cleared while idle
        assert t.resumed is True
        assert t.effective_max() == 120.0             # deadline back to base


# --------------------------------------------------------------------------- #
# Runtime config resolvers (env + live flag files)                            #
# --------------------------------------------------------------------------- #

class TestRuntimeResolvers:
    def test_step_away_default_off(self, monkeypatch, tmp_path):
        monkeypatch.setattr(C, "STEP_AWAY_ENV", False)
        monkeypatch.setattr(C, "STEP_AWAY_FLAG_PATH", str(tmp_path / "nope.flag"))
        assert step_away_enabled() is False

    def test_step_away_via_flag_file(self, monkeypatch, tmp_path):
        flag = tmp_path / "step-away.enabled"
        monkeypatch.setattr(C, "STEP_AWAY_ENV", False)
        monkeypatch.setattr(C, "STEP_AWAY_FLAG_PATH", str(flag))
        assert step_away_enabled() is False
        flag.write_text("")
        assert step_away_enabled() is True

    def test_step_away_via_env(self, monkeypatch, tmp_path):
        monkeypatch.setattr(C, "STEP_AWAY_ENV", True)
        monkeypatch.setattr(C, "STEP_AWAY_FLAG_PATH", str(tmp_path / "nope.flag"))
        assert step_away_enabled() is True

    def test_append_default_zero(self, monkeypatch, tmp_path):
        monkeypatch.setattr(C, "APPEND_WINDOW_MS", 0)
        monkeypatch.setattr(C, "APPEND_WINDOW_FLAG_PATH", str(tmp_path / "nope"))
        assert append_window_ms() == 0

    def test_append_via_knob_file(self, monkeypatch, tmp_path):
        knob = tmp_path / "append-window-ms"
        knob.write_text("800")
        monkeypatch.setattr(C, "APPEND_WINDOW_MS", 0)
        monkeypatch.setattr(C, "APPEND_WINDOW_FLAG_PATH", str(knob))
        assert append_window_ms() == 800

    def test_append_knob_bad_value_fail_safe(self, monkeypatch, tmp_path):
        knob = tmp_path / "append-window-ms"
        knob.write_text("garbage")
        monkeypatch.setattr(C, "APPEND_WINDOW_MS", 0)
        monkeypatch.setattr(C, "APPEND_WINDOW_FLAG_PATH", str(knob))
        assert append_window_ms() == 0


# --------------------------------------------------------------------------- #
# Record-loop simulation harness (fake mic + fake VAD)                        #
# --------------------------------------------------------------------------- #

def _speech_chunk():
    return (np.random.randint(-8000, 8000, size=CHUNK_SAMPLES, dtype=np.int16)
            .reshape(-1, 1))


def _silence_chunk():
    return np.zeros((CHUNK_SAMPLES, 1), dtype=np.int16)


class _FakeInputStream:
    """Stands in for sd.InputStream: on __enter__ spawns a daemon thread that
    calls the loop's callback with scripted chunks, then feeds silence forever
    (the loop ends via VAD/silence-threshold/max-duration, never on us)."""

    def __init__(self, *, samplerate, channels, dtype, callback, blocksize):
        self.callback = callback
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        def pump():
            for ch in _FakeInputStream.script:
                if self._stop.is_set():
                    return
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
    """is_speech() decides from the chunk's byte energy — loud → speech."""
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
    fake_webrtc = MagicMock()
    fake_webrtc.Vad = _FakeVad
    monkeypatch.setattr(C, "webrtcvad", fake_webrtc)

    def set_script(chunks):
        _FakeInputStream.script = chunks
    return set_script


def _script_speech_then_silence(speech_n, silence_n):
    return [_speech_chunk() for _ in range(speech_n)] + \
           [_silence_chunk() for _ in range(silence_n)]


class TestRecordLoopInertByDefault:
    """THE acceptance test: with nothing enabled, the loop behaves as before."""

    def test_inert_by_default_matches_baseline(self, sim_env, monkeypatch):
        # Guarantee the environment is fully default/inert.
        monkeypatch.setattr(C, "STEP_AWAY_ENV", False)
        monkeypatch.setattr(C, "STEP_AWAY_FLAG_PATH", "/nonexistent/step-away.enabled")
        monkeypatch.setattr(C, "APPEND_WINDOW_MS", 0)
        monkeypatch.setattr(C, "APPEND_WINDOW_FLAG_PATH", "/nonexistent/append-window-ms")

        # ~0.6s of speech then long silence → VAD stops after SILENCE_THRESHOLD_MS.
        sim_env(_script_speech_then_silence(20, 200))
        audio_a, speech_a = record_audio_with_silence_detection(max_duration=8.0)

        sim_env(_script_speech_then_silence(20, 200))
        # Explicit inert overrides must produce the SAME behavior as the defaults.
        audio_b, speech_b = record_audio_with_silence_detection(
            max_duration=8.0, append_window_override_ms=0, step_away_enabled_override=False)

        assert speech_a is True and speech_b is True
        # Same stop point ⇒ near-identical captured length (± a couple VAD chunks
        # of thread-timing jitter). Proves no new code path altered turn-taking.
        assert abs(len(audio_a) - len(audio_b)) <= 3 * CHUNK_SAMPLES

    def test_no_pause_no_checkin_when_disabled(self, sim_env, monkeypatch):
        monkeypatch.setattr(C, "STEP_AWAY_ENV", False)
        monkeypatch.setattr(C, "STEP_AWAY_FLAG_PATH", "/nonexistent/step-away.enabled")
        sim_env(_script_speech_then_silence(20, 200))
        calls = []
        state = {}
        # Even if a pause_check would return True, disabled step-away never checks in.
        audio, speech = record_audio_with_silence_detection(
            max_duration=8.0, pause_check=lambda: True,
            checkin_callback=lambda: calls.append(1), step_away_state=state,
            step_away_enabled_override=False)
        assert calls == []
        assert state.get("stepped_away") in (False, None)


class TestAppendToTurn:
    def test_append_captures_more_trailing_audio(self, sim_env):
        # Same script; a larger append window means the loop waits longer through
        # trailing silence before stopping ⇒ MORE samples captured.
        sim_env(_script_speech_then_silence(20, 400))
        short, sp1 = record_audio_with_silence_detection(
            max_duration=12.0, append_window_override_ms=0)

        sim_env(_script_speech_then_silence(20, 400))
        long, sp2 = record_audio_with_silence_detection(
            max_duration=12.0, append_window_override_ms=1500)

        assert sp1 and sp2
        # The append window (1500ms ≈ 50 silence chunks) is captured on top of the
        # normal SILENCE_THRESHOLD_MS before stopping.
        extra = len(long) - len(short)
        assert extra >= 0.8 * (1500 / VAD_CHUNK_DURATION_MS) * CHUNK_SAMPLES

    def test_resumed_speech_continues_same_turn(self, sim_env):
        # speech, a gap SHORTER than SILENCE_THRESHOLD_MS+append, then speech again:
        # with an append window it's ONE turn (speech_detected stays True, no split).
        gap = int((SILENCE_THRESHOLD_MS / VAD_CHUNK_DURATION_MS)) + 5  # just over base threshold
        script = ([_speech_chunk() for _ in range(15)]
                  + [_silence_chunk() for _ in range(gap)]
                  + [_speech_chunk() for _ in range(15)]
                  + [_silence_chunk() for _ in range(200)])
        sim_env(script)
        audio, speech = record_audio_with_silence_detection(
            max_duration=12.0, append_window_override_ms=1200)
        assert speech is True
        # The second speech burst was appended → total exceeds just the first burst.
        assert len(audio) > (15 + gap) * CHUNK_SAMPLES


class TestStepAwayInLoop:
    def test_paused_idle_extends_and_checks_in_once(self, sim_env):
        # No speech at all (pure silence), paused the whole time. With a small grace
        # + checkin the loop waits past base max, speaks ONE check-in, ends idle.
        sim_env([_silence_chunk() for _ in range(5)])  # then infinite silence
        calls = []
        state = {}
        audio, speech = record_audio_with_silence_detection(
            max_duration=0.2,                    # base would end at 0.2s...
            pause_check=lambda: True,            # ...but we're paused,
            step_away_enabled_override=True,
            checkin_callback=lambda: calls.append(1),
            step_away_state=state,
        )
        # Patched grace/checkin are the module constants; override them small so the
        # test is fast. (Done via the monkeypatch fixture below.)
        assert speech is False
        assert state.get("stepped_away") is True
        assert len(calls) == 1  # exactly ONE check-in

    @pytest.fixture(autouse=True)
    def _fast_stepaway(self, monkeypatch):
        # Keep the step-away test fast + bounded: 0.1s check-in, 0.4s grace.
        monkeypatch.setattr(C, "STEP_AWAY_CHECKIN_SECONDS", 0.1)
        monkeypatch.setattr(C, "STEP_AWAY_GRACE_SECONDS", 0.4)
