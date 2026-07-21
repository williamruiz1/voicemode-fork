"""Tests for the natural-mode barge-in `pre_roll` parameter added to
record_audio_with_silence_detection() -- the mic audio a barge-in listener
already captured seeds the next turn's recording, so an interruption becomes
the start of the turn instead of being discarded and re-prompted.

Mocks sd.InputStream directly (per the pattern in test_vad_aggressiveness.py)
rather than exercising the full VAD loop, which the pre-existing
test_silence_detection.py notes is prone to hanging under mocks -- these
tests instead validate ONLY the seeding behavior added for barge-in, using a
tiny max_duration so the loop body never needs to run at all (the seeded
duration alone already satisfies the exit condition).
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock webrtcvad before importing voice_mode modules, matching the existing
# test convention in this file's siblings.
sys.modules.setdefault("webrtcvad", MagicMock())

from voice_mode.tools.converse import record_audio_with_silence_detection
from voice_mode.config import SAMPLE_RATE


@pytest.fixture
def mock_no_new_audio():
    """sd.InputStream opens fine but yields no new frames -- isolates the
    pre_roll seeding behavior from the (separately tested) VAD loop."""
    with patch("voice_mode.tools.converse.sd") as mock_sd:
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value.__enter__.return_value = mock_stream
        yield mock_sd


class TestPreRollSeeding:
    def test_pre_roll_is_prepended_to_result(self, mock_no_new_audio):
        pre_roll = np.full(SAMPLE_RATE // 2, 12345, dtype=np.int16)  # 0.5s
        # max_duration <= pre_roll's own duration -> the while-loop condition
        # (recording_duration < max_duration) is false from the first check,
        # so the function returns immediately with just the seeded chunks.
        result, speech_detected = record_audio_with_silence_detection(
            max_duration=0.4, pre_roll=pre_roll
        )
        assert speech_detected is True
        assert len(result) == len(pre_roll)
        assert np.array_equal(result, pre_roll)

    def test_no_pre_roll_behaves_as_before(self, mock_no_new_audio):
        # With no pre_roll and max_duration effectively zero, there is no
        # seeded speech and no chunks -- matches the pre-existing "no audio
        # chunks recorded" fallback path.
        result, speech_detected = record_audio_with_silence_detection(
            max_duration=0.0, pre_roll=None
        )
        assert speech_detected is False
        assert len(result) == 0

    def test_empty_pre_roll_treated_as_no_pre_roll(self, mock_no_new_audio):
        result, speech_detected = record_audio_with_silence_detection(
            max_duration=0.0, pre_roll=np.array([], dtype=np.int16)
        )
        assert speech_detected is False
        assert len(result) == 0
