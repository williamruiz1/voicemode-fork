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


class TestSpeexEchoCancellerScaleContract:
    """The scale contract SpeexEchoCanceller must honour (founder-os#11658).

    This is the regression guard for a REAL bug caught on hardware, not a
    hypothetical. `BargeInListener` passes NORMALISED FLOAT [-1, 1]
    (`barge_in.py`: `chunk_flat.astype(np.float64) / 32768.0`), matching the
    NLMS `EchoCanceller` this class drops in for. The first cut assumed int16
    PCM and did `np.clip(near, -32768, 32767).astype(np.int16)` — on float
    [-1, 1] the clip is a no-op and the cast truncates EVERY sample to 0, so
    speex was fed silence and returned silence: `rms_clean` was 0.000 across
    all 441 frames of the first real acoustic run.

    That failure is dangerous precisely because it LOOKS like success — an
    all-zero output reads as "infinite cancellation" and as "no false
    positive", while actually meaning the canceller is dead and would swallow
    the real speech barge-in exists to detect. The original synthetic
    convergence test missed it by feeding int16-scale values, where the cast
    happens to be correct. Hence: assert on the float path specifically.
    """

    def _skip_if_unavailable(self):
        try:
            from voice_mode.aec import SPEEX_AVAILABLE
        except Exception:
            pytest.skip("voice_mode.aec unavailable")
        if not SPEEX_AVAILABLE:
            pytest.skip("pyaec/speexdsp not installed")

    def _echo_pair(self, n=SR):
        t = np.arange(n) / SR
        far = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float64)
        near = (0.3 * np.roll(far, 80) + 0.02 * np.random.RandomState(0).randn(n)).astype(np.float64)
        return near, far

    def test_float_input_does_not_collapse_to_silence(self):
        """THE regression: float [-1,1] in must NOT yield an all-zero output."""
        self._skip_if_unavailable()
        from voice_mode.aec import SpeexEchoCanceller
        near, far = self._echo_pair()
        out = SpeexEchoCanceller(sample_rate=SR).process(near, far)
        assert np.abs(out).max() > 0.0, (
            "speex returned all-zeros on float[-1,1] input — the int16 truncation "
            "bug is back; the canceller is dead, not perfect"
        )

    def test_float_in_float_out_same_scale_as_nlms(self):
        """Output must come back in the caller's normalised float domain."""
        self._skip_if_unavailable()
        from voice_mode.aec import SpeexEchoCanceller
        near, far = self._echo_pair()
        out = SpeexEchoCanceller(sample_rate=SR).process(near, far)
        assert out.dtype.kind == "f"
        assert len(out) == len(near)
        # Normalised domain: a [-1,1] input can never produce int16-scale output.
        assert np.abs(out).max() <= 1.5, f"output escaped the [-1,1] domain: {np.abs(out).max()}"

    def test_actually_reduces_echo_on_float_input(self):
        """Beyond 'not zero': it must cancel SOMETHING on the float path."""
        self._skip_if_unavailable()
        from voice_mode.aec import SpeexEchoCanceller
        near, far = self._echo_pair()
        out = SpeexEchoCanceller(sample_rate=SR).process(near, far)
        rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
        # Converged tail only — the filter needs time to adapt.
        assert rms(out[SR // 2:]) < rms(near[SR // 2:]), "no echo reduction on the float path"
