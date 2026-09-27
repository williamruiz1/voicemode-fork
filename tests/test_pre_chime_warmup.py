"""Tests for the pre-chime mic warm-up added to fix recording-start cutoff
(2026-09-27): William's spoken replies on AirPods were losing their first
~2 words. A diagnostic harness (chime_capture_harness.py, run standalone,
not part of this test suite) confirmed the mic stream opening AFTER the
"listening" chime leaves a ~0.6-0.7s dead-zero window on Bluetooth (the
A2DP->HFP profile switch) landing exactly where he starts talking.

_start_pre_chime_capture()/_stop_pre_chime_capture() open the listen-mode
stream BEFORE the chime instead, and hand its captured audio back as a
`pre_roll` array -- the SAME mechanism natural-mode barge-in already uses
(see test_pre_roll_seeding.py), so no new consumption path is introduced.

Mocks sd directly (per the pattern in test_vad_aggressiveness.py /
test_pre_roll_seeding.py) rather than opening a real audio device.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock webrtcvad before importing voice_mode modules, matching this file's
# siblings -- converse.py imports it at module scope.
sys.modules.setdefault("webrtcvad", MagicMock())

import voice_mode.tools.converse as C
from voice_mode.config import SAMPLE_RATE


@pytest.fixture
def mock_sd_input_only():
    """Non-Bluetooth path: sd.InputStream, callback captured for manual firing."""
    with patch.object(C, "_bluetooth_input_active", return_value=False), \
         patch("voice_mode.tools.converse.sd") as mock_sd:
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        yield mock_sd, mock_stream


@pytest.fixture
def mock_sd_bluetooth():
    """Bluetooth path: sd.Stream (full duplex), callback captured for manual firing."""
    with patch.object(C, "_bluetooth_input_active", return_value=True), \
         patch.object(C, "_duplex_device_pair", return_value=(1, 2)), \
         patch("voice_mode.tools.converse.sd") as mock_sd:
        mock_stream = MagicMock()
        mock_sd.Stream.return_value = mock_stream
        yield mock_sd, mock_stream


class TestPreChimeCaptureDisabled:
    def test_disabled_by_config_returns_none_none_without_touching_sd(self, monkeypatch):
        monkeypatch.setattr(C, "PRE_CHIME_WARMUP_ENABLED", False)
        with patch("voice_mode.tools.converse.sd") as mock_sd:
            stream, chunks = C._start_pre_chime_capture()
        assert stream is None
        assert chunks is None
        mock_sd.InputStream.assert_not_called()
        mock_sd.Stream.assert_not_called()


class TestPreChimeCaptureOpenFailure:
    def test_stream_open_exception_returns_none_none(self, monkeypatch):
        monkeypatch.setattr(C, "PRE_CHIME_WARMUP_ENABLED", True)
        with patch.object(C, "_bluetooth_input_active", return_value=False), \
             patch("voice_mode.tools.converse.sd") as mock_sd:
            mock_sd.InputStream.side_effect = Exception("device busy")
            stream, chunks = C._start_pre_chime_capture()
        assert stream is None
        assert chunks is None


class TestPreChimeCaptureNonBluetooth:
    def test_opens_input_only_stream_and_starts_it(self, monkeypatch, mock_sd_input_only):
        monkeypatch.setattr(C, "PRE_CHIME_WARMUP_ENABLED", True)
        mock_sd, mock_stream = mock_sd_input_only
        stream, chunks = C._start_pre_chime_capture()
        assert stream is mock_stream
        assert chunks == []
        mock_stream.start.assert_called_once()
        mock_sd.Stream.assert_not_called()  # non-BT never opens the duplex path

    def test_callback_appends_chunks_and_stop_concatenates(self, monkeypatch, mock_sd_input_only):
        monkeypatch.setattr(C, "PRE_CHIME_WARMUP_ENABLED", True)
        _, mock_stream = mock_sd_input_only
        stream, chunks = C._start_pre_chime_capture()

        # Fire the callback the stream was constructed with, as PortAudio would.
        callback = C.sd.InputStream.call_args.kwargs["callback"]
        chunk_a = np.full((10, 1), 111, dtype=np.int16)
        chunk_b = np.full((10, 1), 222, dtype=np.int16)
        callback(chunk_a, 10, None, None)
        callback(chunk_b, 10, None, None)

        result = C._stop_pre_chime_capture(stream, chunks)
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert result is not None
        assert result.shape == (20, 1)
        assert np.array_equal(result[:10].flatten(), chunk_a.flatten())
        assert np.array_equal(result[10:].flatten(), chunk_b.flatten())


class TestPreChimeCaptureBluetooth:
    def test_opens_full_duplex_stream_with_silent_output(self, monkeypatch, mock_sd_bluetooth):
        monkeypatch.setattr(C, "PRE_CHIME_WARMUP_ENABLED", True)
        mock_sd, mock_stream = mock_sd_bluetooth
        stream, chunks = C._start_pre_chime_capture()
        assert stream is mock_stream
        mock_stream.start.assert_called_once()
        mock_sd.InputStream.assert_not_called()  # BT never takes the input-only path
        _, kwargs = mock_sd.Stream.call_args
        assert kwargs["device"] == (1, 2)

    def test_duplex_callback_fills_silent_output_and_appends_input(self, monkeypatch, mock_sd_bluetooth):
        monkeypatch.setattr(C, "PRE_CHIME_WARMUP_ENABLED", True)
        _, mock_stream = mock_sd_bluetooth
        stream, chunks = C._start_pre_chime_capture()

        callback = C.sd.Stream.call_args.kwargs["callback"]
        indata = np.full((10, 1), 77, dtype=np.int16)
        outdata = np.full((10, 1), 999, dtype=np.int16)  # non-zero, to prove it gets zeroed
        callback(indata, outdata, 10, None, None)

        assert np.all(outdata == 0)  # output must stay silent (this is a warm-up, not playback)
        result = C._stop_pre_chime_capture(stream, chunks)
        assert result is not None
        assert np.array_equal(result.flatten(), indata.flatten())


class TestPreChimeCaptureStopEdgeCases:
    def test_stop_with_no_chunks_returns_none(self):
        mock_stream = MagicMock()
        result = C._stop_pre_chime_capture(mock_stream, [])
        assert result is None
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

    def test_stop_with_none_stream_and_none_chunks_returns_none(self):
        # The disabled/open-failure return shape from _start_pre_chime_capture.
        result = C._stop_pre_chime_capture(None, None)
        assert result is None

    def test_stop_close_exception_is_swallowed(self):
        mock_stream = MagicMock()
        mock_stream.close.side_effect = Exception("already closed")
        chunk = np.full((5, 1), 1, dtype=np.int16)
        result = C._stop_pre_chime_capture(mock_stream, [chunk])
        assert result is not None  # close() failing must not lose the captured audio

    def test_capped_to_max_seconds(self, monkeypatch):
        monkeypatch.setattr(C, "PRE_CHIME_WARMUP_MAX_SECONDS", 0.001)  # ~24 samples at 24kHz
        big_chunk = np.arange(SAMPLE_RATE, dtype=np.int16).reshape(-1, 1)
        result = C._stop_pre_chime_capture(MagicMock(), [big_chunk])
        max_samples = int(0.001 * SAMPLE_RATE)
        assert len(result) == max_samples
        # Capped from the FRONT (a ring buffer keeps the most RECENT audio --
        # the samples right before the chime, not the oldest ones).
        assert np.array_equal(result.flatten(), big_chunk[-max_samples:].flatten())


class TestPreChimeCaptureFeedsPreRollIntoRecording:
    """End-to-end-ish: the array _stop_pre_chime_capture returns is exactly
    what record_audio_with_silence_detection's pre_roll parameter expects
    (see test_pre_roll_seeding.py) -- same shape, same dtype, prepended
    verbatim."""

    def test_result_shape_matches_pre_roll_contract(self, monkeypatch, mock_sd_input_only):
        monkeypatch.setattr(C, "PRE_CHIME_WARMUP_ENABLED", True)
        _, mock_stream = mock_sd_input_only
        stream, chunks = C._start_pre_chime_capture()
        callback = C.sd.InputStream.call_args.kwargs["callback"]
        chunk = np.full((SAMPLE_RATE // 2, 1), 5000, dtype=np.int16)  # 0.5s
        callback(chunk, len(chunk), None, None)
        pre_roll = C._stop_pre_chime_capture(stream, chunks)

        with patch("voice_mode.tools.converse.sd") as mock_sd_record:
            mock_record_stream = MagicMock()
            mock_sd_record.InputStream.return_value.__enter__.return_value = mock_record_stream
            result, speech_detected = C.record_audio_with_silence_detection(
                max_duration=0.4, pre_roll=pre_roll
            )
        assert speech_detected is True
        assert np.array_equal(result, pre_roll)
