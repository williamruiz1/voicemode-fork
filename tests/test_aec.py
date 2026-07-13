"""Tests for the NLMS software echo canceller (voice_mode/aec.py).

These validate the ALGORITHM against synthetic signals -- a known echo path
convolved with a reference tone, a synthetic "speech" burst overlapping the
echo (the double-talk case a real barge-in trigger window looks like), and
the trivial "no reference = no change" case. They do NOT validate against a
real microphone/speaker acoustic path -- see the natural-voice-mode research
doc's Phase 1 live-trial task for that.
"""

import numpy as np
import pytest

from voice_mode.aec import EchoCanceller


SR = 16000


def _make_reference(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    ref = (
        0.5 * np.sin(2 * np.pi * 220 * t)
        + 0.2 * np.sin(2 * np.pi * 440 * t)
        + 0.05 * rng.standard_normal(n)
    )
    return ref.astype(np.float32)


def _echo_from_reference(far: np.ndarray, delay_samples: int, gain: float = 0.6) -> np.ndarray:
    echo = np.zeros(len(far), dtype=np.float64)
    if delay_samples < len(far):
        echo[delay_samples:] = gain * far[: len(far) - delay_samples]
    return echo


class TestEchoCancellerBasics:
    def test_rejects_invalid_mu(self):
        with pytest.raises(ValueError):
            EchoCanceller(sample_rate=SR, mu=0.0)
        with pytest.raises(ValueError):
            EchoCanceller(sample_rate=SR, mu=1.5)

    def test_filter_length_derived_from_ms(self):
        aec = EchoCanceller(sample_rate=16000, filter_ms=200)
        assert aec.filter_len == 3200
        aec2 = EchoCanceller(sample_rate=8000, filter_ms=100)
        assert aec2.filter_len == 800

    def test_process_preserves_shape_and_dtype(self):
        aec = EchoCanceller(sample_rate=SR, filter_ms=50)
        near = np.zeros(480, dtype=np.float32)
        far = np.zeros(480, dtype=np.float32)
        out = aec.process(near, far)
        assert out.shape == near.shape

    def test_process_handles_empty_input(self):
        aec = EchoCanceller(sample_rate=SR, filter_ms=50)
        out = aec.process(np.array([], dtype=np.float32), np.array([], dtype=np.float32))
        assert len(out) == 0

    def test_process_handles_mismatched_far_length(self):
        """far shorter/longer than near (e.g. TTS just stopped mid-chunk)
        must not raise -- the canceller silence-fills / truncates."""
        aec = EchoCanceller(sample_rate=SR, filter_ms=50)
        near = np.random.default_rng(0).standard_normal(480).astype(np.float32)
        short_far = np.zeros(100, dtype=np.float32)
        long_far = np.zeros(900, dtype=np.float32)
        out_short = aec.process(near, short_far)
        out_long = aec.process(near, long_far)
        assert len(out_short) == 480
        assert len(out_long) == 480

    def test_reset_clears_learned_state(self):
        aec = EchoCanceller(sample_rate=SR, filter_ms=50, mu=0.3)
        n = _make_reference(4800)
        near = _echo_from_reference(n, delay_samples=20)
        for i in range(0, len(n), 480):
            aec.process(near[i:i + 480], n[i:i + 480])
        assert np.any(aec.weights != 0)
        aec.reset()
        assert np.all(aec.weights == 0)
        assert np.all(aec._far_history == 0)


class TestEchoCancellerConvergence:
    """Validate the mechanism actually attenuates a known echo path."""

    def test_silence_reference_passes_near_end_through_unchanged(self):
        """If nothing is playing (far is all zero -- TTS not audible right
        now), the canceller must NOT alter the mic signal at all."""
        rng = np.random.default_rng(1)
        n = 8000
        near = (0.4 * np.sin(2 * np.pi * 300 * np.arange(n) / SR) + 0.01 * rng.standard_normal(n)).astype(np.float32)
        far = np.zeros(n, dtype=np.float32)

        aec = EchoCanceller(sample_rate=SR, filter_ms=50, mu=0.5)
        out = np.zeros(n)
        for i in range(0, n, 480):
            out[i:i + 480] = aec.process(near[i:i + 480], far[i:i + 480])

        assert np.allclose(out, near, atol=1e-6)

    def test_attenuates_known_echo_path(self):
        """A pure echo (delayed, attenuated copy of the reference, no
        independent near-end signal) should converge toward near-silence."""
        n = 16000  # 1s @ 16kHz
        far = _make_reference(n)
        delay_samples = int(0.005 * SR)  # 5ms acoustic delay
        near = _echo_from_reference(far, delay_samples, gain=0.6)
        near = near + 0.01 * np.random.default_rng(2).standard_normal(n)

        aec = EchoCanceller(sample_rate=SR, filter_ms=200, mu=0.5)
        out = np.zeros(n)
        chunk = 480
        for i in range(0, n, chunk):
            out[i:i + chunk] = aec.process(near[i:i + chunk], far[i:i + chunk])

        # Compare the RMS of the LAST second's worth of chunks (after the
        # filter has had time to converge) against the pre-cancellation RMS.
        converged_tail = out[-4000:]
        original_tail = near[-4000:]
        rms_before = np.sqrt(np.mean(original_tail ** 2))
        rms_after = np.sqrt(np.mean(converged_tail ** 2))
        assert rms_after < rms_before * 0.6, (
            f"expected meaningful echo attenuation, got rms_before={rms_before:.4f} "
            f"rms_after={rms_after:.4f}"
        )

    def test_realtime_performance(self):
        """The canceller must run comfortably faster than real time -- it
        runs inside the concurrent barge-in listener alongside a live mic
        stream, so falling behind would make barge-in detection lag."""
        import time

        n = 16000  # 1 second of audio @ 16kHz
        far = _make_reference(n)
        near = _echo_from_reference(far, delay_samples=80, gain=0.5)

        aec = EchoCanceller(sample_rate=SR, filter_ms=200, mu=0.15)
        chunk = 480
        t0 = time.perf_counter()
        for i in range(0, n, chunk):
            aec.process(near[i:i + chunk], far[i:i + chunk])
        elapsed = time.perf_counter() - t0

        assert elapsed < 1.0, f"AEC took {elapsed:.3f}s to process 1s of audio -- not real-time-safe"

    def test_double_talk_does_not_fully_silence_independent_speech(self):
        """During "double talk" (echo AND independent near-end speech present
        at once -- the exact window a barge-in trigger has to detect), the
        default low mu must not cancel the independent speech down to noise
        floor. This is the known NLMS double-talk limitation the config
        comment documents; the assertion pins the CHOSEN default's tradeoff
        so a future change to AEC_STEP_SIZE can't silently regress it back
        toward over-cancellation without a test noticing."""
        n = 16000
        far = _make_reference(n)
        near_echo = _echo_from_reference(far, delay_samples=80, gain=0.6)

        speech = np.zeros(n)
        speech_start = int(0.8 * n)
        t = np.arange(n - speech_start) / SR
        speech[speech_start:] = 0.4 * np.sin(2 * np.pi * 300 * t)

        near = near_echo + speech

        aec = EchoCanceller(sample_rate=SR, filter_ms=200, mu=0.15)
        out = np.zeros(n)
        chunk = 480
        for i in range(0, n, chunk):
            out[i:i + chunk] = aec.process(near[i:i + chunk], far[i:i + chunk])

        speech_region_after = out[speech_start:]
        rms_speech_after = np.sqrt(np.mean(speech_region_after ** 2))
        rms_speech_orig = np.sqrt(np.mean(speech[speech_start:] ** 2))
        survival_ratio = rms_speech_after / rms_speech_orig

        assert survival_ratio > 0.2, (
            f"independent speech was over-cancelled during double-talk: "
            f"survived {survival_ratio*100:.0f}% of original energy"
        )
