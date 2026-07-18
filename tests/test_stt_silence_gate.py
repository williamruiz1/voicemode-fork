"""Tests for the pre-STT silence gate (founder-os#11657 Part B).

Near-silent audio is the documented trigger for Whisper silence-hallucinations
("Thank you for watching"). The gate computes a whole-clip normalized RMS on the
int16 recording and, when it is below VOICEMODE_STT_SILENCE_RMS_FLOOR, returns a
`no_speech` result WITHOUT ever calling the STT endpoint — so Whisper never sees
a clip quiet enough to hallucinate.

Calibration reference (measured on the local whisper server, this Mac):
  quiet room / room tone  rms ~0.0008-0.003  -> whisper returns garbage
  quietest real speech    rms ~0.015          -> transcribed correctly
  normal speech           rms ~0.095          -> transcribed correctly
Default floor 0.005 sits between; these tests assert both directions.
"""

import numpy as np
import pytest


def _int16_at_rms(target_rms: float, n: int = 24000, seed: int = 7) -> np.ndarray:
    """A gaussian int16 clip whose normalized ([-1,1]) RMS is ~target_rms."""
    rng = np.random.RandomState(seed)
    x = rng.normal(0.0, target_rms, n)  # normalized-domain samples
    return np.clip(x * 32768.0, -32768, 32767).astype(np.int16)


def _norm_rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float64) / 32768.0) ** 2)))


def test_config_defaults_importable():
    from voice_mode.config import STT_SILENCE_GATE, STT_SILENCE_RMS_FLOOR
    assert isinstance(STT_SILENCE_GATE, bool)
    assert isinstance(STT_SILENCE_RMS_FLOOR, float)
    assert STT_SILENCE_RMS_FLOOR > 0


@pytest.mark.parametrize("target_rms,expect_gated", [
    (0.0008, True),   # quiet room
    (0.003,  True),   # room tone / HVAC
    (0.015,  False),  # quietest real speech — MUST NOT be gated
    (0.095,  False),  # normal speech
])
def test_gate_threshold_matches_calibration(target_rms, expect_gated):
    from voice_mode.config import STT_SILENCE_RMS_FLOOR
    clip = _int16_at_rms(target_rms)
    gated = _norm_rms(clip) < STT_SILENCE_RMS_FLOOR
    assert gated is expect_gated, (
        f"rms~{target_rms}: gated={gated}, expected {expect_gated} "
        f"(floor={STT_SILENCE_RMS_FLOOR})"
    )


def test_sparse_speech_survives_the_gate():
    """A clip that is 95% silence + one brief real-energy utterance must NOT gate
    (energy is squared, so a short loud burst keeps whole-clip RMS well above the
    floor). This is the key false-silence guard."""
    from voice_mode.config import STT_SILENCE_RMS_FLOOR
    n = 24000 * 4
    clip = _int16_at_rms(0.001, n=n)                       # 4s near-silence
    burst = _int16_at_rms(0.09, n=int(24000 * 0.4), seed=9)  # 0.4s of speech-energy
    clip[:len(burst)] = burst
    assert _norm_rms(clip) >= STT_SILENCE_RMS_FLOOR


@pytest.mark.asyncio
async def test_transcribe_short_circuits_silence_without_calling_stt(monkeypatch):
    """End-to-end: transcribe() returns no_speech for a silent clip and never
    invokes simple_stt_failover; a speech clip DOES invoke it."""
    # converse imports audio deps (sounddevice) at module load; skip cleanly in a
    # headless test env that lacks them — the gate math is already covered above.
    pytest.importorskip("sounddevice")
    import voice_mode.tools.converse as conv

    calls = {"n": 0}

    async def fake_stt(audio_file, **kwargs):
        calls["n"] += 1
        return {"text": "should not be reached for silence", "provider": "fake"}

    monkeypatch.setattr("voice_mode.simple_failover.simple_stt_failover", fake_stt)
    monkeypatch.setattr(conv, "STT_SILENCE_GATE", True, raising=False)

    # Silent clip -> gated, STT not called
    silent = _int16_at_rms(0.001)
    result = await conv.speech_to_text(silent)
    assert result.get("error_type") == "no_speech"
    assert result.get("provider") == "silence-gate"
    assert calls["n"] == 0, "STT must NOT be called for near-silent audio"

    # Speech clip -> passes the gate, STT is called
    speech = _int16_at_rms(0.09)
    await conv.speech_to_text(speech)
    assert calls["n"] == 1, "STT must be called for real speech"
