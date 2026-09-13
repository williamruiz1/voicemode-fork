"""Integration tests for Silero-VAD endpointing wired into the record loop
(voice_mode/tools/converse.py::record_audio_with_silence_detection).

ROOT CAUSE under test: webrtcvad's binary is_speech decision + an
accumulating silence_duration_ms timer reads steady road/room noise and a
trailing breath as "speech", so the timer never reaches SILENCE_THRESHOLD_MS
and the mic hangs open. voice_mode/silero_vad.py's Endpointer (frozen
interface, built + validated separately) fixes this with a continuous
P(speech) threshold instead of a binary energy-driven decision. This module
wires that Endpointer into record_audio_with_silence_detection behind the
VOICEMODE_ENDPOINTING toggle (config.ENDPOINTING_ENABLED) and proves the fix
END-TO-END through the real record loop -- not just the Endpointer in
isolation (that's tests/test_silero_vad.py's job).

Integration seam (no real microphone): record_audio_with_silence_detection
creates `audio_queue = queue.Queue()` via a LOCAL `import queue` inside the
function, then reads from it with `audio_queue.get(timeout=0.1)` inside the
`with stream_ctx:` block. Patching `queue.Queue` (module-level, the same
pattern already established by tests/test_vad_aggressiveness.py) makes that
local `import queue; queue.Queue()` call return a MagicMock whose `.get()`
side_effect is a pre-built list of fixture-derived audio chunks -- so the
loop consumes deterministic, real committed audio with no timing races and
no real PortAudio stream. `sd.InputStream`/`sd.Stream` are separately mocked
so no real device is opened. Since `recording_duration` inside the loop is a
NOMINAL per-chunk counter (chunk_duration_s added once per processed chunk),
not a wall-clock read, this all runs at test speed, not real time.

Termination safety: fixture chunks are padded with several seconds of
true-silence chunks. If endpointing correctly fires, the loop exits long
before those padding chunks are ever consumed. If it does NOT fire (a
regression), the mocked queue's side_effect list eventually runs out and
`.get()` raises StopIteration, which the loop's own
`except Exception as e: logger.error(...); break` catches -- so a broken
implementation fails its assertion instead of hanging the test suite.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.io import wavfile

from voice_mode.tools.converse import record_audio_with_silence_detection
from voice_mode.config import SAMPLE_RATE, VAD_CHUNK_DURATION_MS
from voice_mode.silero_vad import SILERO_AVAILABLE

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "endpointing"
CHUNK_SAMPLES = int(SAMPLE_RATE * VAD_CHUNK_DURATION_MS / 1000)  # 720 @ 24kHz/30ms

pytestmark = pytest.mark.skipif(
    not SILERO_AVAILABLE,
    reason="onnxruntime/vendored Silero model unavailable -- install the `silero` extra",
)


def _fixture_duration_s(name: str) -> float:
    sr, data = wavfile.read(FIXTURES_DIR / name)
    assert sr == SAMPLE_RATE, f"fixture {name} is {sr}Hz, expected {SAMPLE_RATE}Hz"
    return len(np.asarray(data).reshape(-1)) / sr


def _fixture_chunks(name: str, pad_seconds: float = 5.0) -> list:
    """Split a committed fixture .wav into CHUNK_SAMPLES-sized int16 pieces
    shaped (CHUNK_SAMPLES, 1) -- exactly what sounddevice's InputStream
    callback hands audio_callback as `indata`. Padded with trailing
    true-silence chunks (see module docstring's Termination safety note)."""
    if not (FIXTURES_DIR / name).exists():
        pytest.skip(f"{name} not generated -- run tests/fixtures/endpointing/generate_fixtures.py")
    sr, data = wavfile.read(FIXTURES_DIR / name)
    assert sr == SAMPLE_RATE, f"fixture {name} is {sr}Hz, expected {SAMPLE_RATE}Hz"
    data = np.asarray(data, dtype=np.int16).reshape(-1)

    pad = np.zeros(int(pad_seconds * SAMPLE_RATE), dtype=np.int16)
    full = np.concatenate([data, pad])

    chunks = []
    for i in range(0, len(full) - CHUNK_SAMPLES + 1, CHUNK_SAMPLES):
        chunks.append(full[i : i + CHUNK_SAMPLES].reshape(-1, 1))
    return chunks


def _run_with_fixture(fixture_name: str, max_duration: float = 8.0, **kwargs):
    """Drive record_audio_with_silence_detection with fixture-derived chunks
    fed through a mocked queue.Queue(), endpointing forced ON. No real
    microphone/PortAudio stream is opened."""
    chunks = _fixture_chunks(fixture_name)

    mock_queue_instance = MagicMock()
    mock_queue_instance.get.side_effect = list(chunks)

    with patch("voice_mode.tools.converse.ENDPOINTING_ENABLED", True), \
         patch("voice_mode.tools.converse.sd") as mock_sd, \
         patch("queue.Queue", return_value=mock_queue_instance):
        mock_sd.InputStream.return_value.__enter__.return_value = MagicMock()
        mock_sd.InputStream.return_value.__exit__.return_value = False
        return record_audio_with_silence_detection(max_duration=max_duration, **kwargs)


class TestEndpointingRecordLoopFixesTheHang:
    """D1/D2/D3: the actual bug (mic hangs open) is fixed, proven through the
    real record loop against real committed audio -- not a scripted stand-in."""

    def test_d1_clean_speech_then_pause_stops_well_before_max_duration(self):
        """Baseline: real speech then true silence. Recording must stop
        roughly min_endpoint_ms after speech ends, nowhere near max_duration."""
        fixture_duration = _fixture_duration_s("clean_speech_then_pause.wav")
        max_duration = 8.0

        audio, speech_detected = _run_with_fixture(
            "clean_speech_then_pause.wav", max_duration=max_duration
        )

        recorded_duration = len(audio) / SAMPLE_RATE
        assert speech_detected is True
        # Stopped before ever reaching the padding appended after the fixture
        # -- i.e. it did NOT hang open past the fixture's own baked-in pause.
        assert recorded_duration < fixture_duration, (
            f"recording ran {recorded_duration:.2f}s, at/past the fixture's own "
            f"{fixture_duration:.2f}s -- endpointing did not fire on the built-in pause"
        )
        # And nowhere near max_duration (the old bug: hangs open until the
        # listen's outer ceiling, not a real end-of-turn signal).
        assert recorded_duration < max_duration * 0.6, (
            f"recording ran {recorded_duration:.2f}s, too close to max_duration="
            f"{max_duration}s -- looks like it hung open instead of endpointing"
        )

    def test_d2_breath_tail_does_not_hold_the_mic_open(self):
        """The core bug: a high-ENERGY, low-Silero-probability breath tail
        immediately following speech must NOT hold the recording open --
        this is exactly what defeats plain energy-gate VAD."""
        fixture_duration = _fixture_duration_s("speech_then_breath_tail.wav")
        max_duration = 8.0

        audio, speech_detected = _run_with_fixture(
            "speech_then_breath_tail.wav", max_duration=max_duration
        )

        recorded_duration = len(audio) / SAMPLE_RATE
        assert speech_detected is True
        assert recorded_duration < fixture_duration, (
            f"recording ran {recorded_duration:.2f}s, at/past the fixture's own "
            f"{fixture_duration:.2f}s -- the breath tail held the mic open"
        )
        assert recorded_duration < max_duration * 0.6

    def test_d3_background_chatter_stops_after_foreground_speech_ends(self):
        """Foreground speech ends but a continuous quieter background bed
        keeps playing -- the mic must still stop once the FOREGROUND speech
        is done, not keep listening because the room is never truly silent."""
        fixture_duration = _fixture_duration_s("speech_with_background_chatter.wav")
        max_duration = 8.0

        audio, speech_detected = _run_with_fixture(
            "speech_with_background_chatter.wav", max_duration=max_duration
        )

        recorded_duration = len(audio) / SAMPLE_RATE
        assert speech_detected is True
        assert recorded_duration < fixture_duration, (
            f"recording ran {recorded_duration:.2f}s, at/past the fixture's own "
            f"{fixture_duration:.2f}s -- continuing background chatter held the mic open"
        )
        assert recorded_duration < max_duration * 0.6


class TestD5DefaultPathUnchanged:
    """D5: the toggle is OFF by default, and OFF whenever Silero isn't
    available -- the pre-existing webrtcvad + silence-timer path must be
    taken byte-for-byte, proven by asserting Endpointer is never constructed."""

    def _run_webrtcvad_path(self, **endpointing_patch_kwargs):
        rng = np.random.default_rng(0)
        speech_chunk = (rng.standard_normal(CHUNK_SAMPLES) * 12000).astype(np.int16).reshape(-1, 1)
        silence_chunk = np.zeros(CHUNK_SAMPLES, dtype=np.int16).reshape(-1, 1)
        # a few "speech" chunks then plenty of true silence -- enough for the
        # existing webrtcvad silence-timer to run its own course, bounded so
        # the test can't hang even if something regresses.
        chunks = [speech_chunk] * 10 + [silence_chunk] * 80

        mock_queue_instance = MagicMock()
        mock_queue_instance.get.side_effect = list(chunks)

        with patch("voice_mode.tools.converse.Endpointer") as mock_endpointer_cls, \
             patch("voice_mode.tools.converse.sd") as mock_sd, \
             patch("queue.Queue", return_value=mock_queue_instance), \
             patch.multiple("voice_mode.tools.converse", **endpointing_patch_kwargs):
            mock_sd.InputStream.return_value.__enter__.return_value = MagicMock()
            mock_sd.InputStream.return_value.__exit__.return_value = False
            record_audio_with_silence_detection(max_duration=5.0)

        return mock_endpointer_cls

    def test_endpointer_not_constructed_when_endpointing_disabled(self):
        """VOICEMODE_ENDPOINTING unset/false (default) -- Endpointer must
        never be constructed, regardless of Silero availability."""
        mock_endpointer_cls = self._run_webrtcvad_path(
            ENDPOINTING_ENABLED=False, SILERO_AVAILABLE=True
        )
        mock_endpointer_cls.assert_not_called()

    def test_endpointer_not_constructed_when_silero_unavailable(self):
        """VOICEMODE_ENDPOINTING=true but Silero itself isn't available --
        must still fall back to webrtcvad, not construct an Endpointer that
        would immediately raise on .update()."""
        mock_endpointer_cls = self._run_webrtcvad_path(
            ENDPOINTING_ENABLED=True, SILERO_AVAILABLE=False
        )
        mock_endpointer_cls.assert_not_called()
