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

    @pytest.fixture
    def exit_stack(self):
        with ExitStack() as stack:
            yield stack

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
