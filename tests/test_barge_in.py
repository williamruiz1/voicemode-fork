"""Tests for the natural-mode barge-in listener (voice_mode/barge_in.py).

No real audio hardware or webrtcvad speech classification involved -- the mic
InputStream and the VAD decision are both mocked, so these validate the STATE
MACHINE (sustained post-AEC "speech" during playback -> trigger; not during
playback -> no trigger; VAD unavailable -> graceful no-op) rather than real
acoustic behavior. Real on-device tuning is a separate, explicitly-flagged
live-trial task (see the natural-voice-mode research doc).
"""

import time
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import voice_mode.barge_in as bi


@pytest.fixture
def exit_stack():
    """Module-level so every test class below can share it (moved out of
    TestBargeInListenerStateMachine, which originally owned it alone)."""
    with ExitStack() as stack:
        yield stack


def _make_chunk(value: int = 5000, n: int = bi.CHUNK_SAMPLES_MIC) -> np.ndarray:
    """A synthetic mono int16 mic chunk, shaped like sounddevice's callback
    indata (frames, channels)."""
    return np.full((n, 1), value, dtype=np.int16)


class TestNaturalModeEnabled:
    def test_false_when_flag_file_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bi, "NATURAL_MODE_FLAG_PATH", str(tmp_path / "natural-mode.flag"))
        assert bi.natural_mode_enabled() is False

    def test_true_when_flag_file_present(self, tmp_path, monkeypatch):
        flag = tmp_path / "natural-mode.flag"
        flag.touch()
        monkeypatch.setattr(bi, "NATURAL_MODE_FLAG_PATH", str(flag))
        assert bi.natural_mode_enabled() is True

    def test_survives_a_broken_path(self, monkeypatch):
        # A path that raises on os.path.exists (e.g. permission weirdness)
        # must not crash the caller -- default to turn mode (the safe side).
        monkeypatch.setattr(bi.os.path, "exists", lambda p: (_ for _ in ()).throw(OSError("boom")))
        assert bi.natural_mode_enabled() is False


class TestFitLength:
    def test_pads_short_array(self):
        out = bi._fit_length(np.array([1.0, 2.0]), 5)
        assert len(out) == 5
        assert list(out) == [1.0, 2.0, 0.0, 0.0, 0.0]

    def test_truncates_long_array(self):
        out = bi._fit_length(np.array([1.0, 2.0, 3.0, 4.0]), 2)
        assert list(out) == [1.0, 2.0]

    def test_exact_length_unchanged(self):
        arr = np.array([1.0, 2.0, 3.0])
        out = bi._fit_length(arr, 3)
        assert list(out) == list(arr)


class TestBargeInListenerDegradesGracefully:
    def test_start_noop_when_vad_unavailable(self, monkeypatch):
        monkeypatch.setattr(bi, "VAD_AVAILABLE", False)
        listener = bi.BargeInListener()
        listener.start()
        assert listener._thread is None
        assert listener._stream is None
        assert listener._error == "webrtcvad unavailable"
        result = listener.stop()
        assert result.triggered is False

    def test_start_noop_when_input_stream_fails_to_open(self, monkeypatch):
        with patch.object(bi, "sd") as mock_sd:
            mock_sd.InputStream.side_effect = RuntimeError("no such device")
            listener = bi.BargeInListener()
            listener.start()
            assert listener._stream is None
            assert "no such device" in listener._error
            result = listener.stop()
            assert result.triggered is False


class TestBargeInListenerStateMachine:
    """Drive the watcher loop with mocked audio hardware + a mocked VAD
    decision, feeding chunks directly into the internal queue (bypassing the
    real InputStream callback) for deterministic timing."""

    def _arm(self, stack: ExitStack, monkeypatch, *, tts_speaking: bool, vad_says_speech: bool):
        """Patch out real audio hardware + the VAD decision, start a listener,
        and return (listener, triggered_calls). `stack` keeps the patches
        alive for the caller's whole test (closed automatically by the
        `exit_stack` fixture below)."""
        triggered_calls = []
        monkeypatch.setattr(bi.audio_player, "is_tts_speaking", lambda: tts_speaking)
        monkeypatch.setattr(
            bi.audio_player, "get_reference_audio",
            lambda n, delay_samples=0: np.zeros(n, dtype=np.float32),
        )
        monkeypatch.setattr(bi.audio_player, "trigger_barge_in", lambda: triggered_calls.append(True))
        monkeypatch.setattr(bi.audio_player, "reset_barge_in_event", lambda: None)

        mock_vad = MagicMock()
        mock_vad.is_speech.return_value = vad_says_speech
        mock_webrtcvad = stack.enter_context(patch.object(bi, "webrtcvad"))
        mock_webrtcvad.Vad.return_value = mock_vad
        mock_sd = stack.enter_context(patch.object(bi, "sd"))
        mock_sd.InputStream.return_value = MagicMock()

        listener = bi.BargeInListener()
        listener.start()
        return listener, triggered_calls

    def test_triggers_on_sustained_post_aec_speech_during_playback(self, monkeypatch, exit_stack):
        listener, triggered_calls = self._arm(exit_stack, monkeypatch, tts_speaking=True, vad_says_speech=True)

        # Enough 30ms chunks to exceed BARGE_IN_TRIGGER_MS (default 300ms).
        n_chunks = (bi.BARGE_IN_TRIGGER_MS // bi.CHUNK_MS) + 3
        for _ in range(n_chunks):
            listener._audio_queue.put(_make_chunk())

        listener._thread.join(timeout=3.0)
        assert not listener._thread.is_alive(), "watcher thread should have stopped itself on trigger"

        result = listener.stop()
        assert result.triggered is True
        assert triggered_calls == [True]
        assert result.pre_roll is not None
        assert len(result.pre_roll) > 0

    def test_no_trigger_when_tts_not_playing(self, monkeypatch, exit_stack):
        listener, triggered_calls = self._arm(exit_stack, monkeypatch, tts_speaking=False, vad_says_speech=True)

        n_chunks = (bi.BARGE_IN_TRIGGER_MS // bi.CHUNK_MS) + 5
        for _ in range(n_chunks):
            listener._audio_queue.put(_make_chunk())

        time.sleep(0.4)  # let the watcher thread drain the queue
        result = listener.stop()

        assert result.triggered is False
        assert triggered_calls == []

    def test_no_trigger_when_vad_says_no_speech(self, monkeypatch, exit_stack):
        listener, triggered_calls = self._arm(exit_stack, monkeypatch, tts_speaking=True, vad_says_speech=False)

        n_chunks = (bi.BARGE_IN_TRIGGER_MS // bi.CHUNK_MS) + 5
        for _ in range(n_chunks):
            listener._audio_queue.put(_make_chunk())

        time.sleep(0.4)
        result = listener.stop()

        assert result.triggered is False
        assert triggered_calls == []

    def test_speech_run_resets_when_tts_stops_mid_burst(self, monkeypatch):
        """If TTS stops playing partway through what would have been a
        sustained speech run, the run counter must reset -- a burst that
        happens to straddle "TTS just finished naturally" shouldn't count
        toward a barge-in that's no longer interrupting anything."""
        speaking_flags = [True] * 3 + [False] * 20  # stops being "speaking" after 3 chunks
        call_count = {"n": 0}

        def fake_is_speaking():
            i = min(call_count["n"], len(speaking_flags) - 1)
            call_count["n"] += 1
            return speaking_flags[i]

        monkeypatch.setattr(bi.audio_player, "is_tts_speaking", fake_is_speaking)
        monkeypatch.setattr(
            bi.audio_player, "get_reference_audio",
            lambda n, delay_samples=0: np.zeros(n, dtype=np.float32),
        )
        triggered_calls = []
        monkeypatch.setattr(bi.audio_player, "trigger_barge_in", lambda: triggered_calls.append(True))
        monkeypatch.setattr(bi.audio_player, "reset_barge_in_event", lambda: None)

        mock_vad = MagicMock()
        mock_vad.is_speech.return_value = True

        with patch.object(bi, "webrtcvad") as mock_webrtcvad, patch.object(bi, "sd") as mock_sd:
            mock_webrtcvad.Vad.return_value = mock_vad
            mock_sd.InputStream.return_value = MagicMock()

            listener = bi.BargeInListener()
            listener.start()

            for _ in range(15):
                listener._audio_queue.put(_make_chunk())

            time.sleep(0.5)
            result = listener.stop()

        assert result.triggered is False
        assert triggered_calls == []


class TestBargeInLifecycleEvents:
    """The 2026-07-14 evidence-trail addition: a live trial with NO event
    logged either way is exactly what happened on 2026-07-13 (barge_in.py's
    logger.* calls only reach stderr; nothing was persisted). These lock in
    that the four lifecycle events fire at the right moments regardless of
    whether VOICEMODE_BARGE_IN_TRACE is set."""

    def test_armed_and_disarmed_logged_on_clean_stop(self, monkeypatch):
        with ExitStack() as stack:
            armed_calls, disarmed_calls = [], []
            monkeypatch.setattr(bi, "log_barge_in_armed", lambda vad_aggr: armed_calls.append(vad_aggr))
            monkeypatch.setattr(bi, "log_barge_in_disarmed",
                                 lambda triggered, frames, elapsed: disarmed_calls.append((triggered, frames)))
            monkeypatch.setattr(bi.audio_player, "reset_barge_in_event", lambda: None)
            mock_sd = stack.enter_context(patch.object(bi, "sd"))
            mock_sd.InputStream.return_value = MagicMock()

            listener = bi.BargeInListener()
            listener.start()
            result = listener.stop()

        assert armed_calls == [listener._vad_aggressiveness]
        assert disarmed_calls == [(False, 0)]
        assert result.triggered is False

    def test_unavailable_logged_when_vad_missing(self, monkeypatch):
        monkeypatch.setattr(bi, "VAD_AVAILABLE", False)
        calls = []
        monkeypatch.setattr(bi, "log_barge_in_unavailable", lambda reason: calls.append(reason))

        listener = bi.BargeInListener()
        listener.start()

        assert calls == ["webrtcvad unavailable"]

    def test_unavailable_logged_when_stream_open_fails(self, monkeypatch):
        calls = []
        monkeypatch.setattr(bi, "log_barge_in_unavailable", lambda reason: calls.append(reason))
        with patch.object(bi, "sd") as mock_sd:
            mock_sd.InputStream.side_effect = RuntimeError("no such device")
            listener = bi.BargeInListener()
            listener.start()

        assert len(calls) == 1
        assert "no such device" in calls[0]

    def test_triggered_logged_on_a_real_trigger(self, monkeypatch):
        calls = []
        monkeypatch.setattr(bi, "log_barge_in_triggered", lambda speech_run_ms, elapsed: calls.append(speech_run_ms))
        monkeypatch.setattr(bi.audio_player, "is_tts_speaking", lambda: True)
        monkeypatch.setattr(bi.audio_player, "get_reference_audio",
                             lambda n, delay_samples=0: np.zeros(n, dtype=np.float32))
        monkeypatch.setattr(bi.audio_player, "trigger_barge_in", lambda: None)
        monkeypatch.setattr(bi.audio_player, "reset_barge_in_event", lambda: None)

        mock_vad = MagicMock()
        mock_vad.is_speech.return_value = True
        with patch.object(bi, "webrtcvad") as mock_webrtcvad, patch.object(bi, "sd") as mock_sd:
            mock_webrtcvad.Vad.return_value = mock_vad
            mock_sd.InputStream.return_value = MagicMock()

            listener = bi.BargeInListener()
            listener.start()
            n_chunks = (bi.BARGE_IN_TRIGGER_MS // bi.CHUNK_MS) + 3
            for _ in range(n_chunks):
                listener._audio_queue.put(_make_chunk())
            listener._thread.join(timeout=3.0)
            listener.stop()

        assert len(calls) == 1
        assert calls[0] >= bi.BARGE_IN_TRIGGER_MS


class TestEnergyMarginGate:
    """config.BARGE_IN_ENERGY_MARGIN (default 0 = disabled). These validate
    the asymmetric echo-floor tracker directly, independent of real acoustic
    hardware -- see scripts/barge_in_acoustic_test.py for the real-hardware
    validation this gate was added, tuned, and evaluated against on
    2026-07-14 (net finding: no single margin value on that hardware avoided
    BOTH false-positives and false-negatives at once -- see config.py's
    BARGE_IN_ENERGY_MARGIN docstring)."""

    def test_default_zero_is_a_pure_noop(self, monkeypatch, exit_stack):
        """Margin=0 (the shipped default) must behave IDENTICALLY to no gate
        at all -- this is the backward-compatibility guarantee that lets the
        gate exist in the codebase without changing anyone's behavior."""
        monkeypatch.setattr(bi, "BARGE_IN_ENERGY_MARGIN", 0)
        listener, triggered_calls = TestBargeInListenerStateMachine()._arm(
            exit_stack, monkeypatch, tts_speaking=True, vad_says_speech=True
        )
        n_chunks = (bi.BARGE_IN_TRIGGER_MS // bi.CHUNK_MS) + 3
        for _ in range(n_chunks):
            listener._audio_queue.put(_make_chunk())
        listener._thread.join(timeout=3.0)
        result = listener.stop()
        assert result.triggered is True  # unchanged from the no-gate state machine test

    def test_flat_unchanging_signal_never_clears_the_margin(self, monkeypatch, exit_stack):
        """A CONSTANT-amplitude 'echo residual' (VAD says speech every frame,
        but the level never actually rises above its own recent floor) must
        NOT sustain a speech run once the gate is enabled -- this is the
        self-interruption-off-a-flat-echo scenario the gate exists to catch."""
        monkeypatch.setattr(bi, "BARGE_IN_ENERGY_MARGIN", 2.0)
        listener, triggered_calls = TestBargeInListenerStateMachine()._arm(
            exit_stack, monkeypatch, tts_speaking=True, vad_says_speech=True
        )
        # Many more chunks than the trigger threshold would need -- if the
        # gate is broken (e.g. the floor never calibrates), this alone would
        # trigger well before we stop it.
        n_chunks = (bi.BARGE_IN_TRIGGER_MS // bi.CHUNK_MS) * 5
        for _ in range(n_chunks):
            listener._audio_queue.put(_make_chunk(value=5000))
        time.sleep(0.5)
        result = listener.stop()
        assert result.triggered is False
        assert triggered_calls == []

    def test_genuine_amplitude_jump_still_triggers(self, monkeypatch, exit_stack):
        """A real jump well above the calibrated floor must still clear the
        gate and trigger -- proves the gate isn't just permanently closed."""
        monkeypatch.setattr(bi, "BARGE_IN_ENERGY_MARGIN", 2.0)
        listener, triggered_calls = TestBargeInListenerStateMachine()._arm(
            exit_stack, monkeypatch, tts_speaking=True, vad_says_speech=True
        )
        # Calibrate the floor on a low, flat level first...
        for _ in range(20):
            listener._audio_queue.put(_make_chunk(value=500))
        # ...then a sustained, much louder run that should clear a 2.0x margin.
        n_chunks = (bi.BARGE_IN_TRIGGER_MS // bi.CHUNK_MS) + 5
        for _ in range(n_chunks):
            listener._audio_queue.put(_make_chunk(value=20000))
        listener._thread.join(timeout=3.0)
        result = listener.stop()
        assert result.triggered is True
        assert triggered_calls == [True]
