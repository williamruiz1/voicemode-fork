"""Software acoustic echo cancellation (AEC) for the natural-mode barge-in listener.

Natural mode keeps the mic open WHILE TTS plays (see barge_in.py). On any
device where the mic can hear the speaker at all -- and especially on a single
Bluetooth device serving both directions, like William's AirPods, which are
BOTH the default input and output (system_profiler confirmed this: "WRuiz
AirPod 4" for both) -- the mic will pick up the agent's own TTS audio. Without
cancellation, the barge-in VAD reads that as "he's talking" and the agent
interrupts itself.

This module implements a single-channel Normalized Least-Mean-Squares (NLMS)
adaptive filter against a KNOWN reference (far-end) signal: the exact TTS
samples actually being sent to the output stream, which NonBlockingAudioPlayer
already has (see audio_player.FAR_END_REF). NLMS is the same algorithm family
underlying WebRTC's AEC -- a linear adaptive filter that learns the acoustic
echo path (speaker -> room/BT link -> mic) and subtracts its prediction from
the near-end (mic) signal.

Why NLMS-in-numpy instead of binding the native `webrtc-audio-processing` /
`aec-audio-processing` PyPI packages the research doc named: both ship as
source-only sdists (verified via `pip download` -- no prebuilt wheel for this
platform) requiring a C++ toolchain + the bundled WebRTC APM sources to
compile. That is a much larger, more fragile dependency footprint for Phase 1
than "an incremental patch" -- a broken native build would take down the whole
voicemode server, not just natural mode. NLMS-against-a-known-reference is a
legitimate, standard "software AEC" implementation of the SAME category the
research names (Option A), not a different approach -- it trades off-the-shelf
maturity for zero new native dependencies. EchoCanceller's interface is narrow
enough that a native binding could be swapped in later without touching
barge_in.py's state machine.

HONESTY NOTE (per completeness-honesty-protocol): this has been validated with
a synthetic-echo unit test (see tests/test_aec.py) proving the mechanism
converges and attenuates a known echo path. It has NOT been validated against
a real microphone + real AirPods acoustic path -- that requires a live trial
(natural-voice-mode research doc, Phase 1 task 4) and the three tunables below
(AEC_FILTER_MS, AEC_REF_DELAY_MS, AEC_STEP_SIZE) are exactly the knobs that
trial is expected to retune.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("voicemode.aec")


class EchoCanceller:
    """Single-channel NLMS adaptive echo canceller.

    Operates on float32 samples in [-1, 1] at a caller-chosen sample rate
    (barge_in.py runs this at the VAD's 16kHz working rate -- there is no
    reason to cancel echo at full TTS fidelity when the only consumer of the
    output is a voice-activity detector, not playback or transcription).

    Usage:
        aec = EchoCanceller(sample_rate=16000, filter_ms=200, mu=0.5)
        for near_chunk, far_chunk in stream:
            clean_chunk = aec.process(near_chunk, far_chunk)
            # clean_chunk has echo (predicted from far_chunk) subtracted out
    """

    def __init__(self, sample_rate: int, filter_ms: int = 200, mu: float = 0.5,
                 eps: float = 1e-6):
        """
        Args:
            sample_rate: Sample rate of both near-end and far-end signals (Hz).
            filter_ms: Adaptive filter length in milliseconds -- bounds how
                much acoustic delay (speaker -> mic) the canceller can model.
            mu: NLMS step size, 0 < mu <= 1. Higher converges faster but is
                less stable; 0.5 is a standard NLMS default.
            eps: Small constant to avoid division by zero when the far-end
                reference has near-zero energy (silence).
        """
        if not (0 < mu <= 1):
            raise ValueError(f"mu must be in (0, 1], got {mu}")
        self.sample_rate = sample_rate
        self.filter_len = max(1, int(sample_rate * filter_ms / 1000))
        self.mu = mu
        self.eps = eps
        # Adaptive filter weights (the learned echo-path impulse response).
        self.weights = np.zeros(self.filter_len, dtype=np.float64)
        # Rolling history of far-end (reference) samples, most-recent-last.
        # Needs filter_len history so we can form the reversed tap window for
        # every new near-end sample.
        self._far_history = np.zeros(self.filter_len, dtype=np.float64)

    def reset(self):
        """Clear learned filter state (e.g. after a device/route change)."""
        self.weights.fill(0.0)
        self._far_history.fill(0.0)

    def process(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        """Cancel the predicted echo of `far` out of `near`.

        Args:
            near: Mic-captured samples for this chunk (float32/float64,
                mono, same length convention as `far`).
            far: The reference (far-end) samples actually sent to the output
                stream for the SAME time window as `near` (already
                delay-compensated by the caller via config.AEC_REF_DELAY_MS
                if needed). If shorter than `near` (e.g. TTS isn't playing),
                treat the missing tail as silence.

        Returns:
            Echo-cancelled near-end signal, same shape/dtype as `near`
            (float64 internally, cast back to near's dtype on return).
        """
        near = np.asarray(near, dtype=np.float64).reshape(-1)
        far = np.asarray(far, dtype=np.float64).reshape(-1)

        n = len(near)
        if n == 0:
            return near.astype(near.dtype)

        # Pad/truncate far-end to match near-end length (silence-fill if the
        # reference ran short -- e.g. TTS just stopped mid-chunk).
        if len(far) < n:
            far = np.concatenate([far, np.zeros(n - len(far), dtype=np.float64)])
        elif len(far) > n:
            far = far[:n]

        out = np.empty(n, dtype=np.float64)
        w = self.weights
        hist = self._far_history
        L = self.filter_len
        mu = self.mu
        eps = self.eps

        for i in range(n):
            # Shift the new far-end sample into history (most-recent-last).
            hist = np.concatenate([hist[1:], far[i:i + 1]])
            # Predict the echo as the filter's response to the reference
            # history, reversed so the most recent reference sample aligns
            # with the shortest-delay tap.
            tap_window = hist[::-1]
            predicted_echo = float(np.dot(w, tap_window))
            error = near[i] - predicted_echo
            out[i] = error
            # NLMS weight update: step proportional to error, normalized by
            # reference energy so the update magnitude doesn't blow up when
            # the reference is loud.
            energy = float(np.dot(tap_window, tap_window)) + eps
            w = w + (mu * error / energy) * tap_window

        self.weights = w
        self._far_history = hist
        return out.astype(near.dtype if near.dtype.kind == "f" else np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# founder-os#11658 — speexdsp/pyaec echo canceller (drop-in for EchoCanceller).
#
# The hand-rolled NLMS EchoCanceller above measured only ~3-4 dB of real
# cancellation on William's hardware against ~20 dB of echo (voicemode-fork
# 83a4407). speexdsp is a ~20-year VoIP-grade frequency-domain adaptive echo
# canceller WITH a built-in double-talk-aware preprocessor — directly addressing
# the NLMS limitation (real speech partially swallowed at the interruption
# point). This wraps the `pyaec` speexdsp binding behind the SAME interface as
# EchoCanceller so it drops into the barge-in listener unchanged.
#
# Availability is soft: if pyaec / the native speexdsp lib isn't importable the
# module flag SPEEX_AVAILABLE is False and callers fall back to the NLMS filter.
try:
    import pyaec as _pyaec  # ctypes binding over the bundled speexdsp lib
    SPEEX_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    _pyaec = None
    SPEEX_AVAILABLE = False


class SpeexEchoCanceller:
    """speexdsp echo canceller with the EchoCanceller interface.

    Same constructor + `process(near, far)` + `reset()` contract as the NLMS
    `EchoCanceller`, so it is a literal drop-in. `mu`/`eps` are accepted for
    signature-compatibility and ignored (speex is not an NLMS filter).

    speex operates on fixed-size int16 frames and keeps adaptation state across
    calls, so `process` frames the (possibly longer) input into `frame_size`
    sub-frames; a final partial sub-frame is zero-padded and the output trimmed
    back to the input length, preserving the same-length contract.
    """

    def __init__(self, sample_rate: int, filter_ms: int = 200, mu: float = 0.5,
                 eps: float = 1e-6, frame_size: int = 160):
        if not SPEEX_AVAILABLE:
            raise RuntimeError("pyaec/speexdsp not available")
        self.sample_rate = int(sample_rate)
        self.frame_size = int(frame_size)
        # Echo-tail length in samples (speexdsp models this much speaker->mic delay).
        self.filter_len = max(self.frame_size, int(sample_rate * filter_ms / 1000))
        self._new_aec = lambda: _pyaec.Aec(self.frame_size, self.filter_len, self.sample_rate, True)
        self._aec = self._new_aec()

    def reset(self):
        """Clear learned echo-path state (e.g. after a device/route change)."""
        self._aec = self._new_aec()

    def process(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        near = np.asarray(near)
        far = np.asarray(far)
        n = int(min(len(near), len(far)))
        if n == 0:
            return near.astype(near.dtype)
        out_dtype = near.dtype if near.dtype.kind == "f" else np.float64

        # SCALE CONTRACT (founder-os#11658 — this was a real bug, caught on
        # hardware): the NLMS `EchoCanceller` this class drops in for works in
        # NORMALISED FLOAT [-1, 1], and that is what `BargeInListener` passes
        # (`barge_in.py`: `chunk_flat.astype(np.float64) / 32768.0`). speexdsp
        # needs int16 PCM. The first cut clipped to +/-32768 and cast straight
        # to int16 — on float [-1, 1] the clip is a no-op and the cast TRUNCATES
        # every sample to 0, so speex was handed pure silence and returned pure
        # silence: rms_clean was 0.000 for all 441 frames of the first real
        # acoustic run. That reads as "infinite cancellation" (and as "no false
        # positive") while actually meaning the canceller is DEAD — it would
        # also swallow the real speech barge-in has to detect.
        #
        # The synthetic convergence test missed this because it fed int16-scale
        # values, where the cast is correct. So: scale float input INTO the
        # int16 domain here, and scale the result back on the way out, so the
        # float-in/float-out contract matches NLMS exactly.
        float_in = near.dtype.kind == "f"
        if float_in:
            near_s = np.asarray(near[:n], dtype=np.float64) * 32768.0
            far_s = np.asarray(far[:n], dtype=np.float64) * 32768.0
        else:
            near_s = np.asarray(near[:n], dtype=np.float64)
            far_s = np.asarray(far[:n], dtype=np.float64)
        near_i16 = np.clip(near_s, -32768, 32767).astype(np.int16)
        far_i16 = np.clip(far_s, -32768, 32767).astype(np.int16)
        fs = self.frame_size
        out = np.empty(n, dtype=np.int16)
        pos = 0
        while pos < n:
            end = min(pos + fs, n)
            rec = near_i16[pos:end]
            ref = far_i16[pos:end]
            if len(rec) < fs:  # zero-pad the trailing partial frame
                rec = np.concatenate([rec, np.zeros(fs - len(rec), dtype=np.int16)])
                ref = np.concatenate([ref, np.zeros(fs - len(ref), dtype=np.int16)])
            cleaned = np.asarray(self._aec.cancel_echo(rec.tolist(), ref.tolist()), dtype=np.int16)
            out[pos:end] = cleaned[: end - pos]
            pos = end
        # Back to the caller's domain: float in => normalised float out, so the
        # rms_clean/rms_near ratio the trace computes is dimensionally sane.
        if float_in:
            return (out.astype(np.float64) / 32768.0).astype(out_dtype)
        return out.astype(out_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# AEC3 Phase 1 — WebRTC AEC3 via `pywebrtc-audio` (drop-in for EchoCanceller).
#
# The hand-rolled NLMS filter measures ~3-4 dB of real cancellation on hardware
# (SPEEX_AVAILABLE is False there, so it's the live fallback); AEC3 is WebRTC's
# production echo canceller (Chrome/Meet/Zoom-grade), rated ~25-45 dB. Unlike
# the speexdsp binding above, `pywebrtc-audio` ships PREBUILT macOS arm64
# wheels (v0.2.0) -- no C++ toolchain, no source build, so the "fragile native
# dependency" concern that ruled out webrtc-audio-processing/aec-audio-
# processing (see the module docstring) does not apply to this package.
#
# THE ACTUAL API (discovered by hands-on probe of pywebrtc-audio 0.2.0 -- the
# design doc's assumption of a separate render-then-capture call pair was
# wrong; document the real shape here so nobody re-derives it from scratch):
#   - `pywebrtc_audio.EchoCanceller(sample_rate, num_channels=1,
#      stream_delay_ms=0)` is the AEC3-only processor (there is also a wider
#      `AudioProcessor(echo_cancellation=True, ...)` that bundles NS/AGC/HPF --
#      not used here, we want AEC3 in isolation to compare cleanly against
#      NLMS/speex).
#   - ONE call does the whole job: `.process(near, far) -> np.ndarray` takes
#      BOTH near-end and far-end for the SAME frame and returns the
#      echo-cancelled near-end -- there is no separate `process_render` /
#      `process_capture` pair to sequence.
#   - `near`/`far` MUST be `float32` (or `int16`); `float64` raises
#      `ValueError: audio dtype must be float32 or int16, got format 'd'`.
#      barge_in.py's `near_16k`/`far_16k` are float64 (see the SCALE CONTRACT
#      note on SpeexEchoCanceller above), so `process()` below casts down to
#      float32 for the call and back up on return, same pattern as speex's
#      int16 round-trip.
#   - Frame length is NOT constrained to an exact 10ms multiple -- probed with
#      150, 480 (barge_in's 30ms/16kHz chunk), and other odd lengths, all
#      succeeded; the binding frames internally. No manual sub-framing needed
#      (unlike SpeexEchoCanceller.frame_size above).
#   - `near`/`far` must be the SAME length per call (`ValueError` otherwise) --
#      `process()` below pads/truncates `far` to match `near`, mirroring
#      EchoCanceller's contract.
#   - Delay is a `stream_delay_ms` CONSTRUCTOR ARG (and a live-settable
#      property) on the C++ object itself, NOT something the caller
#      pre-shifts into the reference signal.
#
# DELAY HANDLING (review finding F5 -- do NOT double-compensate): the NLMS/
# speex engines rely on `barge_in.py` pre-shifting the far-end reference by
# `AEC_REF_DELAY_MS` before it ever reaches `.process()` (via
# `audio_player.get_reference_audio(..., delay_samples=ref_delay_samples)`),
# because neither of those engines has its own delay model. AEC3 is
# DIFFERENT: it has a real delay estimator/compensator built on
# `stream_delay_ms`, which expects the RAW, unshifted far-end reference as an
# initial delay ESTIMATE it then refines -- feeding it an ALREADY-shifted
# reference on top of that would shift the acoustic delay twice (once by the
# caller, once by AEC3's own compensation), which is worse than not
# compensating at all. So: `AEC3EchoCanceller` takes `stream_delay_ms` as an
# init-time hint (barge_in.py passes `AEC_REF_DELAY_MS` here, NOT into
# `get_reference_audio`'s `delay_samples`), and barge_in.py's engine-select
# is responsible for passing `delay_samples=0` to `get_reference_audio` only
# when `self._aec_kind == "aec3"` -- see barge_in.py's `_watch_loop`.
#
# Availability is soft, same pattern as SPEEX_AVAILABLE: if pywebrtc-audio (or
# its compiled extension) isn't importable, AEC3_AVAILABLE is False and
# callers fall back to speex, then NLMS.
try:
    import pywebrtc_audio as _pywebrtc_audio
    AEC3_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    _pywebrtc_audio = None
    AEC3_AVAILABLE = False


class AEC3EchoCanceller:
    """WebRTC AEC3 echo canceller with the EchoCanceller interface.

    Same constructor + `process(near, far)` + `reset()` contract as the NLMS
    `EchoCanceller` / `SpeexEchoCanceller`, so it is a literal drop-in.
    `filter_ms`/`mu`/`eps` are accepted for signature-compatibility and
    ignored (AEC3 is not an NLMS filter and has no comparable tunables at
    this binding's surface). `stream_delay_ms` is AEC3-specific -- see the
    DELAY HANDLING note above; barge_in.py is expected to pass
    `AEC_REF_DELAY_MS` here instead of pre-shifting the reference signal.
    """

    def __init__(self, sample_rate: int, filter_ms: int = 200, mu: float = 0.5,
                 eps: float = 1e-6, stream_delay_ms: int = 0):
        if not AEC3_AVAILABLE:
            raise RuntimeError("pywebrtc-audio not available")
        self.sample_rate = int(sample_rate)
        self._stream_delay_ms = int(stream_delay_ms)
        self._new_aec = lambda: _pywebrtc_audio.EchoCanceller(
            sample_rate=self.sample_rate, num_channels=1,
            stream_delay_ms=self._stream_delay_ms,
        )
        self._aec = self._new_aec()

    def reset(self):
        """Clear AEC3's adaptive-filter/delay-estimator state (e.g. after a
        device/route change, or at the start of a new turn -- see F5b: this
        state persists across turns unless explicitly reset)."""
        self._aec = self._new_aec()

    def process(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        near = np.asarray(near).reshape(-1)
        far = np.asarray(far).reshape(-1)
        n = len(near)
        if n == 0:
            return near.astype(near.dtype)
        out_dtype = near.dtype if near.dtype.kind == "f" else np.float64

        # Pad/truncate far-end to match near-end length (silence-fill if the
        # reference ran short), same contract as EchoCanceller/
        # SpeexEchoCanceller above -- pywebrtc-audio raises ValueError on a
        # near/far length mismatch rather than tolerating it.
        if len(far) < n:
            far = np.concatenate([far, np.zeros(n - len(far), dtype=np.float64)])
        elif len(far) > n:
            far = far[:n]

        # SCALE CONTRACT: pywebrtc-audio requires float32 (or int16); the
        # normalised-float [-1, 1] domain barge_in.py passes is float64.
        # Round-trip through float32 -- unlike the speex int16 round-trip,
        # this is NOT a truncation trap: float64->float32 is a precision
        # narrowing, not a domain change, so silence-in silence-out; verified
        # via the offline ERLE harness (tests/aec3_erle_offline.py).
        near32 = np.asarray(near, dtype=np.float32)
        far32 = np.asarray(far, dtype=np.float32)
        clean = self._aec.process(near32, far32)
        return np.asarray(clean, dtype=np.float64).astype(out_dtype)
