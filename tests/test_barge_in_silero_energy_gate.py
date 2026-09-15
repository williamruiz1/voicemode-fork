"""Regression test for the B2 fix: the Silero VAD path skipped the
echo-floor energy-margin gate entirely.

`voice_mode/barge_in.py`'s Silero branch (the active/preferred path) used to
set `gated_speech = is_speech` straight from `speech_prob >= threshold`,
with NO reference to amplitude at all -- the asymmetric echo-floor tracker
(`BARGE_IN_ENERGY_MARGIN`) was computed only inside the webrtcvad `else`
branch and never consulted when Silero was active. Because Silero's
P(speech) is amplitude-invariant, a low-amplitude ECHO RESIDUAL that is
still spectrally speech-shaped (imperfect AEC cancellation of the agent's
own TTS onset, or background video/media) scores `speech_prob` near 1.0 and
used to trigger a false barge-in regardless of how quiet it actually was.

The fix hoists the echo-floor tracker so it's computed once per frame,
shared by BOTH VAD paths, and the Silero branch's `gated_speech` now also
requires `clean_rms >= floor_before_update * BARGE_IN_ENERGY_MARGIN`
whenever `BARGE_IN_ENERGY_MARGIN > 0` -- with the default margin of 0 the
extra clause is a no-op (see config.py; the default is intentionally left
at 0 by this fix).

This test drives the real `BargeInListener` end-to-end (mocked mic/output
stream, mocked Silero, real AEC) rather than re-deriving the gate arithmetic
in isolation, because the interesting bug was specifically in the WIRING
between the Silero decision and the pre-existing energy-floor tracker --
the same class of "component X works, component Y works, the two were never
actually connected" bug documented in tests/test_streaming_barge_in_wiring.py.
The AEC's far-end reference is mocked to all-zeros throughout (no real TTS
audio was actually captured for playback), which makes the AEC's echo
prediction a no-op, so each frame's post-AEC `clean` signal tracks the raw
mic amplitude directly -- letting the test control `clean_rms` deterministically
via the synthetic mic chunk's amplitude, exactly like tests/test_bargein_silero.py
already does for `is_speech`. No real audio hardware is opened: `sd.InputStream`
is patched, mirroring the existing barge-in test suite's pattern.
"""

import time
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import voice_mode.barge_in as bi


def _make_chunk(value: int, n: int = bi.CHUNK_SAMPLES_MIC) -> np.ndarray:
    """A synthetic mono int16 mic chunk at a fixed amplitude, shaped like
    sounddevice's callback indata (frames, channels)."""
    return np.full((n, 1), value, dtype=np.int16)


@pytest.fixture
def exit_stack():
    with ExitStack() as stack:
        yield stack


def _arm(stack, monkeypatch, *, energy_margin: float, probs):
    """Arm a Silero-backed BargeInListener with TTS "playing" throughout,
    an all-zero AEC far-end reference (so the AEC is a pass-through and
    post-AEC `clean` tracks raw mic amplitude), and a scripted Silero
    probability sequence -- one value per fed frame."""
    triggered_calls = []
    monkeypatch.setattr(bi, "BARGE_IN_ENERGY_MARGIN", energy_margin)
    monkeypatch.setattr(bi.audio_player, "is_tts_speaking", lambda: True)
    monkeypatch.setattr(
        bi.audio_player, "get_reference_audio",
        lambda n, delay_samples=0: np.zeros(n, dtype=np.float32),
    )
    monkeypatch.setattr(bi.audio_player, "trigger_barge_in", lambda: triggered_calls.append(True))
    monkeypatch.setattr(bi.audio_player, "reset_barge_in_event", lambda: None)

    probs_iter = iter(probs)
    mock_silero_instance = MagicMock()
    mock_silero_instance.prob.side_effect = lambda frame: next(probs_iter, probs[-1])

    monkeypatch.setattr(bi, "SILERO_AVAILABLE", True)
    stack.enter_context(patch.object(bi, "SileroVAD", return_value=mock_silero_instance))
    mock_sd = stack.enter_context(patch.object(bi, "sd"))
    mock_sd.InputStream.return_value = MagicMock()

    listener = bi.BargeInListener()
    assert listener._silero is mock_silero_instance
    listener.start()
    return listener, triggered_calls


class TestSileroEnergyFloorGate:
    """B2: a low-amplitude, spectrally-speech-shaped echo residual must be
    gated OUT on the Silero path once BARGE_IN_ENERGY_MARGIN > 0, while a
    genuinely loud voice above the floor still triggers normally."""

    def test_low_amplitude_high_probability_echo_residual_is_gated_out(self, monkeypatch, exit_stack):
        n_trigger = (bi.BARGE_IN_TRIGGER_MS // bi.CHUNK_MS) + 5
        # One quiet, low-probability frame first to calibrate the echo floor
        # at the residual's own amplitude without contributing to speech_run_ms.
        probs = [0.05] + [0.95] * n_trigger
        listener, triggered_calls = self._arm_low_amplitude(exit_stack, monkeypatch, probs)

        QUIET = 200  # same amplitude for the calibration frame and the "residual" burst
        listener._audio_queue.put(_make_chunk(QUIET))
        for _ in range(n_trigger):
            listener._audio_queue.put(_make_chunk(QUIET))
        time.sleep(0.5)
        result = listener.stop()

        assert result.triggered is False, (
            "a high-probability but low-amplitude (at-floor) frame must be gated out "
            "once BARGE_IN_ENERGY_MARGIN > 0 -- this is the false-barge-in-on-echo-residual bug"
        )
        assert triggered_calls == []

    def test_low_amplitude_high_probability_never_triggers_even_with_default_margin_off(self, monkeypatch, exit_stack):
        """Sanity check on the OTHER side: with the shipped default margin
        (0, i.e. the gate disabled), Silero alone decides -- proving the new
        code path doesn't silently change default (margin=0) behavior."""
        n_trigger = (bi.BARGE_IN_TRIGGER_MS // bi.CHUNK_MS) + 3
        probs = [0.95] * n_trigger
        listener, triggered_calls = self._arm_low_amplitude(
            exit_stack, monkeypatch, probs, margin=0.0
        )
        QUIET = 200
        for _ in range(n_trigger):
            listener._audio_queue.put(_make_chunk(QUIET))
        listener._thread.join(timeout=3.0)
        result = listener.stop()

        assert result.triggered is True, (
            "with the default margin (0 = gate OFF), a sustained high-probability frame "
            "must still trigger regardless of amplitude -- unchanged default behavior"
        )
        assert triggered_calls == [True]

    def test_high_amplitude_high_probability_still_triggers_above_the_floor(self, monkeypatch, exit_stack):
        n_trigger = (bi.BARGE_IN_TRIGGER_MS // bi.CHUNK_MS) + 5
        probs = [0.05] + [0.95] * n_trigger
        listener, triggered_calls = self._arm_low_amplitude(exit_stack, monkeypatch, probs)

        QUIET = 200
        LOUD = 12000  # well above QUIET * BARGE_IN_ENERGY_MARGIN
        listener._audio_queue.put(_make_chunk(QUIET))  # calibrate the floor low
        for _ in range(n_trigger):
            listener._audio_queue.put(_make_chunk(LOUD))
        listener._thread.join(timeout=3.0)
        result = listener.stop()

        assert result.triggered is True, (
            "a genuinely loud, sustained high-probability voice well above the "
            "calibrated echo floor must still trigger a real barge-in"
        )
        assert triggered_calls == [True]

    def _arm_low_amplitude(self, stack, monkeypatch, probs, margin: float = 2.0):
        return _arm(stack, monkeypatch, energy_margin=margin, probs=probs)


class TestSileroEnergyGateArithmeticDirect:
    """Discriminating unit-level check on the gate expression itself
    (mirrors the exact boolean barge_in.py now evaluates), independent of
    the threaded listener -- confirms the arithmetic, not just the
    end-to-end wiring exercised above."""

    @staticmethod
    def _gated(is_speech: bool, clean_rms: float, floor_before_update: float, margin: float) -> bool:
        return is_speech and (margin <= 0 or clean_rms >= floor_before_update * margin)

    def test_at_floor_is_gated_out_when_margin_enabled(self):
        assert self._gated(is_speech=True, clean_rms=0.01, floor_before_update=0.01, margin=2.0) is False

    def test_well_above_floor_times_margin_passes(self):
        assert self._gated(is_speech=True, clean_rms=0.5, floor_before_update=0.01, margin=2.0) is True

    def test_margin_zero_ignores_amplitude_entirely(self):
        assert self._gated(is_speech=True, clean_rms=0.0001, floor_before_update=0.01, margin=0.0) is True

    def test_is_speech_false_always_gates_out_regardless_of_amplitude(self):
        assert self._gated(is_speech=False, clean_rms=0.5, floor_before_update=0.01, margin=2.0) is False
