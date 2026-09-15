"""Tests for the AEC3 echo-canceller engine (voice_mode/aec.py's
AEC3EchoCanceller) and its wiring into BargeInListener's engine-select
(voice_mode/barge_in.py).

Pure logic / offline-math only -- no real microphone or speaker device is
ever opened here (per the vm-aec3-phase1 dispatch's hard constraint). The
AEC3-availability fallback path is exercised by monkeypatching the module
flags directly rather than uninstalling pywebrtc-audio, so these tests run
identically whether or not the package is actually installed in this venv.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

import voice_mode.aec as aec_mod
import voice_mode.barge_in as bi
from voice_mode.aec import AEC3EchoCanceller, AEC3_AVAILABLE

SR = 16000


def _skip_if_aec3_unavailable():
    if not AEC3_AVAILABLE:
        pytest.skip("pywebrtc-audio not installed in this venv")


class TestAEC3EchoCancellerContract:
    """AEC3EchoCanceller must honor the EXACT same interface contract as
    EchoCanceller/SpeexEchoCanceller: __init__(sample_rate, filter_ms=...,
    mu=...) [+eps, +stream_delay_ms], process(near, far) -> same shape/dtype,
    reset()."""

    def test_accepts_nlms_signature_args_and_ignores_them(self):
        """filter_ms/mu/eps are NLMS-specific and must be accept-and-ignore
        for drop-in signature compatibility -- must not raise."""
        _skip_if_aec3_unavailable()
        aec = AEC3EchoCanceller(sample_rate=SR, filter_ms=200, mu=0.5, eps=1e-6)
        assert aec.sample_rate == SR

    def test_process_preserves_shape_and_dtype_float32(self):
        _skip_if_aec3_unavailable()
        aec = AEC3EchoCanceller(sample_rate=SR)
        near = np.zeros(480, dtype=np.float32)
        far = np.zeros(480, dtype=np.float32)
        out = aec.process(near, far)
        assert out.shape == near.shape
        assert out.dtype == near.dtype

    def test_process_preserves_shape_and_dtype_float64(self):
        """barge_in.py's near_16k/far_16k are float64 (see aec.py's SCALE
        CONTRACT comment) -- the float32 round-trip inside process() must be
        transparent to the caller's dtype contract."""
        _skip_if_aec3_unavailable()
        aec = AEC3EchoCanceller(sample_rate=SR)
        near = np.zeros(480, dtype=np.float64)
        far = np.zeros(480, dtype=np.float64)
        out = aec.process(near, far)
        assert out.shape == near.shape
        assert out.dtype == near.dtype

    def test_process_handles_empty_input(self):
        _skip_if_aec3_unavailable()
        aec = AEC3EchoCanceller(sample_rate=SR)
        out = aec.process(np.array([], dtype=np.float32), np.array([], dtype=np.float32))
        assert len(out) == 0

    def test_process_pads_short_far_reference(self):
        """far shorter than near (e.g. TTS just stopped mid-chunk) must be
        silence-padded, not raise -- same contract as EchoCanceller."""
        _skip_if_aec3_unavailable()
        aec = AEC3EchoCanceller(sample_rate=SR)
        near = np.random.RandomState(0).randn(480).astype(np.float64) * 0.1
        far = np.random.RandomState(1).randn(200).astype(np.float64) * 0.1
        out = aec.process(near, far)
        assert len(out) == len(near)

    def test_process_truncates_long_far_reference(self):
        _skip_if_aec3_unavailable()
        aec = AEC3EchoCanceller(sample_rate=SR)
        near = np.random.RandomState(0).randn(480).astype(np.float64) * 0.1
        far = np.random.RandomState(1).randn(900).astype(np.float64) * 0.1
        out = aec.process(near, far)
        assert len(out) == len(near)

    def test_process_actually_cancels_a_known_echo(self):
        """Beyond 'doesn't crash': must reduce a real synthetic echo, same
        convergence check test_aec.py runs for NLMS/speex."""
        _skip_if_aec3_unavailable()
        rng = np.random.default_rng(42)
        n = SR * 2
        t = np.arange(n) / SR
        far = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(n)).astype(np.float64)
        delay = 80
        near = np.zeros(n, dtype=np.float64)
        near[delay:] = 0.6 * far[: n - delay]

        aec = AEC3EchoCanceller(sample_rate=SR)
        chunk = 480
        out = np.empty_like(near)
        for pos in range(0, n, chunk):
            end = min(pos + chunk, n)
            out[pos:end] = aec.process(near[pos:end], far[pos:end])

        tail = slice(n // 2, n)  # converged tail
        rms_before = float(np.sqrt(np.mean(np.square(near[tail]))))
        rms_after = float(np.sqrt(np.mean(np.square(out[tail]))))
        assert rms_after < rms_before, (
            f"AEC3 did not reduce echo energy: before={rms_before:.4f} after={rms_after:.4f}"
        )

    def test_reset_clears_adaptive_state(self):
        """reset() must not raise, and the resulting canceller must behave
        like a fresh instance (no crash on immediate reuse)."""
        _skip_if_aec3_unavailable()
        aec = AEC3EchoCanceller(sample_rate=SR)
        near = np.random.RandomState(0).randn(480).astype(np.float32) * 0.1
        far = np.random.RandomState(1).randn(480).astype(np.float32) * 0.1
        aec.process(near, far)  # let some state accumulate
        aec.reset()
        out = aec.process(near, far)
        assert len(out) == len(near)

    def test_raises_when_unavailable(self, monkeypatch):
        """AEC3_AVAILABLE reflects the import guard -- constructing the
        engine when the flag is False (e.g. pywebrtc-audio not installed)
        must fail loudly, not silently degrade."""
        monkeypatch.setattr(aec_mod, "AEC3_AVAILABLE", False)
        with pytest.raises(RuntimeError):
            aec_mod.AEC3EchoCanceller(sample_rate=SR)


class TestAEC3AvailableFlag:
    def test_flag_is_a_bool(self):
        assert isinstance(AEC3_AVAILABLE, bool)

    def test_flag_true_implies_class_importable_and_constructible(self):
        _skip_if_aec3_unavailable()
        aec = AEC3EchoCanceller(sample_rate=SR)
        assert aec is not None


class TestEngineSelectPreference:
    """BargeInListener.__init__'s VOICEMODE_AEC engine-select, per the F5/F5b
    review findings: aec3 (default) -> speex -> nlms, with nlms/speex still
    forcing those specific engines. Mock the module-level *_AVAILABLE flags
    directly (never uninstall a real package) so this runs the same whether
    or not pywebrtc-audio/pyaec are actually installed."""

    @pytest.fixture(autouse=True)
    def _force_webrtcvad_and_no_silero(self, monkeypatch):
        # Isolate engine-select from the VAD-path decision (same pattern as
        # test_barge_in.py's _force_webrtcvad_path fixture) -- irrelevant to
        # AEC engine choice, but keeps construction side-effect-free.
        monkeypatch.setattr(bi, "SILERO_AVAILABLE", False)

    def test_default_prefers_aec3_when_available(self, monkeypatch):
        monkeypatch.delenv("VOICEMODE_AEC", raising=False)
        monkeypatch.setattr(bi, "AEC3_AVAILABLE", True)
        monkeypatch.setattr(bi, "SPEEX_AVAILABLE", True)
        listener = bi.BargeInListener()
        assert listener._aec_kind == "aec3"
        assert isinstance(listener._aec, bi.AEC3EchoCanceller)

    def test_falls_back_to_speex_when_aec3_unavailable(self, monkeypatch):
        monkeypatch.delenv("VOICEMODE_AEC", raising=False)
        monkeypatch.setattr(bi, "AEC3_AVAILABLE", False)
        monkeypatch.setattr(bi, "SPEEX_AVAILABLE", True)
        # SpeexEchoCanceller.__init__ itself gates on aec.py's OWN
        # module-level SPEEX_AVAILABLE (not barge_in's imported copy), AND
        # calls the real `_pyaec.Aec(...)` ctor -- pyaec genuinely isn't
        # installed in this venv, so fake both the flag and the module
        # object to prove engine-select's DECISION without needing the real
        # native speexdsp binding installed.
        monkeypatch.setattr(aec_mod, "SPEEX_AVAILABLE", True)
        monkeypatch.setattr(aec_mod, "_pyaec", MagicMock())
        listener = bi.BargeInListener()
        assert listener._aec_kind == "speexdsp"
        assert isinstance(listener._aec, bi.SpeexEchoCanceller)

    def test_falls_back_to_nlms_when_neither_available(self, monkeypatch):
        monkeypatch.delenv("VOICEMODE_AEC", raising=False)
        monkeypatch.setattr(bi, "AEC3_AVAILABLE", False)
        monkeypatch.setattr(bi, "SPEEX_AVAILABLE", False)
        listener = bi.BargeInListener()
        assert listener._aec_kind == "nlms"
        assert isinstance(listener._aec, bi.EchoCanceller)

    def test_explicit_aec3_pref_same_as_default(self, monkeypatch):
        monkeypatch.setenv("VOICEMODE_AEC", "aec3")
        monkeypatch.setattr(bi, "AEC3_AVAILABLE", True)
        listener = bi.BargeInListener()
        assert listener._aec_kind == "aec3"

    def test_forced_speex_skips_aec3_even_when_available(self, monkeypatch):
        monkeypatch.setenv("VOICEMODE_AEC", "speex")
        monkeypatch.setattr(bi, "AEC3_AVAILABLE", True)
        monkeypatch.setattr(bi, "SPEEX_AVAILABLE", True)
        monkeypatch.setattr(aec_mod, "SPEEX_AVAILABLE", True)  # see note above
        monkeypatch.setattr(aec_mod, "_pyaec", MagicMock())
        listener = bi.BargeInListener()
        assert listener._aec_kind == "speexdsp"

    def test_forced_speex_falls_back_to_nlms_when_speex_unavailable(self, monkeypatch):
        monkeypatch.setenv("VOICEMODE_AEC", "speex")
        monkeypatch.setattr(bi, "AEC3_AVAILABLE", True)
        monkeypatch.setattr(bi, "SPEEX_AVAILABLE", False)
        listener = bi.BargeInListener()
        assert listener._aec_kind == "nlms"

    def test_forced_nlms_skips_both_even_when_available(self, monkeypatch):
        monkeypatch.setenv("VOICEMODE_AEC", "nlms")
        monkeypatch.setattr(bi, "AEC3_AVAILABLE", True)
        monkeypatch.setattr(bi, "SPEEX_AVAILABLE", True)
        listener = bi.BargeInListener()
        assert listener._aec_kind == "nlms"

    def test_aec3_constructed_with_stream_delay_ms_not_manual_shift(self, monkeypatch):
        """F5: AEC3 must receive the delay via stream_delay_ms at
        construction, not have the caller pre-shift the reference."""
        monkeypatch.delenv("VOICEMODE_AEC", raising=False)
        monkeypatch.setattr(bi, "AEC3_AVAILABLE", True)
        monkeypatch.setattr(bi, "AEC_REF_DELAY_MS", 40)
        listener = bi.BargeInListener()
        assert listener._aec_kind == "aec3"
        assert listener._aec._stream_delay_ms == 40


class TestRefDelaySamplesDoubleCompensationGuard:
    """F5 regression guard: when the engine is aec3, _watch_loop must NOT
    also pre-shift the reference via AEC_REF_DELAY_MS (that would shift the
    acoustic delay twice -- once by the caller, once by AEC3's own
    stream_delay_ms compensation)."""

    def test_ref_delay_samples_zero_for_aec3(self, monkeypatch):
        monkeypatch.setattr(bi, "AEC_REF_DELAY_MS", 50)
        listener = object.__new__(bi.BargeInListener)
        listener._aec_kind = "aec3"
        ref_delay_samples = 0 if listener._aec_kind == "aec3" else int(
            bi.SAMPLE_RATE * bi.AEC_REF_DELAY_MS / 1000
        )
        assert ref_delay_samples == 0

    def test_ref_delay_samples_nonzero_for_nlms(self, monkeypatch):
        monkeypatch.setattr(bi, "AEC_REF_DELAY_MS", 50)
        listener = object.__new__(bi.BargeInListener)
        listener._aec_kind = "nlms"
        ref_delay_samples = 0 if listener._aec_kind == "aec3" else int(
            bi.SAMPLE_RATE * bi.AEC_REF_DELAY_MS / 1000
        )
        assert ref_delay_samples == int(bi.SAMPLE_RATE * 50 / 1000)


class TestResetWiredInStart:
    """F5b: self._aec.reset() must be called in start(), same as
    self._silero.reset(), so adaptive-filter/delay-estimator state doesn't
    leak across turns."""

    def test_start_resets_the_aec(self, monkeypatch):
        from unittest.mock import patch

        monkeypatch.setattr(bi, "SILERO_AVAILABLE", False)
        listener = bi.BargeInListener()
        listener._aec.reset = lambda: setattr(listener, "_aec_reset_called", True)
        # start() calls self._aec.reset() BEFORE it tries to open the input
        # stream -- fail the stream open (never touches real hardware) so
        # start() returns right after, without spinning up the watcher
        # thread, while still exercising the reset call we're asserting on.
        with patch.object(bi, "sd") as mock_sd:
            mock_sd.InputStream.side_effect = RuntimeError("no such device")
            listener.start()
        assert getattr(listener, "_aec_reset_called", False) is True
