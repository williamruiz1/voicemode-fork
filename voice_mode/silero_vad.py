"""Model-based speech/end-of-speech detection (Silero VAD v5, vendored ONNX).

ROOT CAUSE this replaces: webrtcvad (used by both converse.py's turn-taking
record loop and barge_in.py's concurrent listener) returns a BINARY
speech/no-speech decision with no energy awareness of its own. Steady
non-speech energy -- road/engine/room noise, a trailing breath after an
utterance -- reads as "speech" to webrtcvad, so the trailing-silence counter
that ends a turn never accumulates and the mic hangs (see config.py's
VAD_ENERGY_THRESHOLD comment for the band-aid this is meant to obsolete, and
BARGE_IN_ENERGY_MARGIN's comment for the same problem on the barge-in side).

The permanent fix is a MODEL that outputs a continuous P(speech) in [0,1]:
breath/noise score low even when their RMS energy is high, so a plain
threshold on the probability (not the raw energy) separates them from real
speech. This module is that model, wrapped in two pieces:

  - SileroVAD: raw per-frame P(speech), one call per frame, with the model's
    fixed-window (512 samples @16kHz / 256 @8kHz) and per-utterance RNN state
    handled internally so callers can feed arbitrary chunk sizes.
  - Endpointer: the turn-taking STATE MACHINE built on top of SileroVAD --
    "has speech started", "has enough trailing sub-threshold time passed to
    call the turn over". This is the piece converse.py's record loop and
    barge_in.py's listener are each expected to drive their decisions from
    (see the sibling dispatch that wires them in).

Fully offline at runtime: the model is vendored at
voice_mode/resources/models/silero_vad.onnx (MIT-licensed, from
https://github.com/snakers4/silero-vad), and onnxruntime is an OPTIONAL
dependency (`pip install voice-mode[silero]` / the `silero` extra in
pyproject.toml). This module NEVER touches the network at runtime and NEVER
raises at import time just because onnxruntime or the model are missing --
SILERO_AVAILABLE tells callers whether to use it or fall back to webrtcvad.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("voicemode.silero_vad")

# --- Optional onnxruntime import -- see module docstring. A missing package
# is an expected, non-fatal state (webrtcvad-only environments, CI without
# the `silero` extra installed): we degrade SILERO_AVAILABLE to False and
# let every other symbol in this module still import cleanly.
try:
    import onnxruntime as ort
    _ONNXRUNTIME_IMPORTABLE = True
except ImportError:
    ort = None
    _ONNXRUNTIME_IMPORTABLE = False

_MODEL_PATH = Path(__file__).parent / "resources" / "models" / "silero_vad.onnx"

# Silero VAD v5's exported ONNX graph is an LSTM-style recurrent model: it
# takes an (batch, num_samples) audio window, a (2, batch, 128) hidden state,
# and the sample rate, and returns P(speech) plus the updated state. The
# window size the model was trained/calibrated on is fixed per sample rate --
# feeding a different chunk length still RUNS (the ONNX graph itself accepts
# a dynamic axis) but produces an uncalibrated, near-zero-regardless-of-input
# probability, so SileroVAD buffers/reframes internally rather than trusting
# the caller's chunk size.
#
# DISCOVERED DURING THIS BUILD (root-cause, not guessed): a naive
# window-only feed (512/256 raw samples, no context) silently degrades every
# probability toward ~0 -- even genuine macOS `say` speech topped out at
# 0.069 -- which looks like "the model doesn't work" but is actually a
# feeding bug. The official reference wrapper (snakers4/silero-vad
# utils_vad.py OnnxWrapper.__call__) PREPENDS the trailing context_size
# samples (64 @16kHz / 32 @8kHz) of the PREVIOUS window to each new window
# before running the model, and re-derives that context from the tail of the
# just-fed (context+window) tensor afterward -- i.e. the model's real input
# length is window+context (576 @16kHz), not window alone. Confirmed via
# WebFetch of the upstream source and reproduced locally: with context
# prepended, the same `say`-generated speech clip that scored max 0.069
# without it scores >0.9 on real voiced frames (see generate_fixtures.py's
# printed calibration and test_silero_vad.py's real-onnx smoke test).
_WINDOW_SAMPLES = {16000: 512, 8000: 256}
_CONTEXT_SAMPLES = {16000: 64, 8000: 32}
_STATE_SHAPE = (2, 1, 128)


def _load_session() -> "Optional[ort.InferenceSession]":
    """Attempt to construct the shared ONNX session. Never raises -- any
    failure (missing file, corrupt model, onnxruntime internal error) is
    caught and reported via the return value (None) plus a logged warning,
    so SILERO_AVAILABLE below can degrade cleanly instead of crashing the
    whole module at import time."""
    if not _ONNXRUNTIME_IMPORTABLE:
        return None
    if not _MODEL_PATH.exists():
        logger.warning(f"silero_vad: vendored model not found at {_MODEL_PATH}")
        return None
    try:
        return ort.InferenceSession(str(_MODEL_PATH), providers=["CPUExecutionProvider"])
    except Exception as e:
        logger.warning(f"silero_vad: failed to load vendored ONNX model ({e})")
        return None


# Loaded (or attempted) once at import time -- the model is ~2MB and session
# construction is on the order of tens of milliseconds, so doing this once
# and sharing the session across every SileroVAD instance (onnxruntime
# sessions are safe to call concurrently; all per-utterance state is passed
# in/out explicitly as the `state` tensor, never held inside the session) is
# both correct and cheap. This IS the "does the model actually load" half of
# SILERO_AVAILABLE's contract -- a bad/missing file is caught here, not
# deferred to first use.
_SHARED_SESSION = _load_session()

# True iff onnxruntime imports AND the vendored .onnx model loads. Callers
# (the config/converse/barge_in workers) check this BEFORE constructing a
# SileroVAD/Endpointer to decide whether to fall back to webrtcvad.
SILERO_AVAILABLE: bool = _SHARED_SESSION is not None


def _to_float32(frame: np.ndarray) -> np.ndarray:
    """Normalize an arbitrary-dtype 1-D (or flattenable) audio frame to the
    float32 [-1.0, 1.0] range Silero expects. int16 PCM (the format used
    throughout this codebase's mic capture) is scaled by 32768; anything
    already floating-point is assumed to be in-range and passed through."""
    arr = np.asarray(frame)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / 32768.0
    else:
        arr = arr.astype(np.float32)
    return arr.reshape(-1)


class SileroVAD:
    """Per-frame P(speech) via the vendored Silero VAD v5 ONNX model.

    Frozen interface (voicemode-endpointing-bargein W1): sample_rate must be
    8000 or 16000 (the two rates Silero v5 was trained for -- this codebase's
    VAD-facing rate is 16000, see config.VAD_WORK_RATE / barge_in.py). One
    instance = one utterance's worth of RNN state; call reset() at turn
    boundaries (a fresh SileroVAD() already starts zeroed, reset() just lets
    a caller reuse one instance across turns instead of reallocating).
    """

    def __init__(self, sample_rate: int = 16000):
        if sample_rate not in _WINDOW_SAMPLES:
            raise ValueError(
                f"SileroVAD only supports sample_rate in {sorted(_WINDOW_SAMPLES)}, got {sample_rate}"
            )
        self.sample_rate = sample_rate
        self._window_samples = _WINDOW_SAMPLES[sample_rate]
        self._context_samples = _CONTEXT_SAMPLES[sample_rate]
        self._sr_arr = np.array(sample_rate, dtype=np.int64)
        # Construction NEVER raises even when Silero is unavailable -- only
        # prob() does, per the frozen interface's graceful-degradation
        # requirement -- so callers can build the object ahead of a
        # SILERO_AVAILABLE check without special-casing constructor order.
        self._session = _SHARED_SESSION
        self._buffer = np.zeros((0,), dtype=np.float32)
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        # Trailing context_size samples carried from the previous window,
        # prepended to the next -- see the _CONTEXT_SAMPLES comment above.
        # Zeros for the very first window, matching the upstream wrapper.
        self._context = np.zeros(self._context_samples, dtype=np.float32)
        self._last_prob = 0.0

    def reset(self) -> None:
        """Clear RNN state + the internal reframing buffer/context at turn
        boundaries. Silero's hidden state and cross-window context carry
        short-term continuity across calls; leaving them warm across an
        utterance boundary biases the next turn's earliest frames toward
        whatever the previous turn ended on."""
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._buffer = np.zeros((0,), dtype=np.float32)
        self._context = np.zeros(self._context_samples, dtype=np.float32)
        self._last_prob = 0.0

    def prob(self, frame_int16_or_float: np.ndarray) -> float:
        """P(speech) in [0,1] for one frame.

        Handles Silero's fixed window size internally: the incoming frame
        (of whatever length the caller drives -- this codebase's record
        loops use ~30ms / 480-samples@16kHz chunks, not the model's native
        512) is appended to an internal buffer, which is drained in exact
        512-sample (16kHz) / 256-sample (8kHz) windows, running one model
        inference per window and updating the RNN state each time. The
        probability returned is the most recent window's result -- so a
        call that doesn't yet complete a full window returns the previous
        window's probability unchanged (introducing at most one window's
        worth, ~32ms/16ms, of latency versus a hypothetical per-sample
        model).
        """
        if self._session is None:
            raise RuntimeError(
                "SileroVAD.prob() called but Silero VAD is unavailable "
                f"(onnxruntime_importable={_ONNXRUNTIME_IMPORTABLE}, "
                f"model_path={_MODEL_PATH}, model_path_exists={_MODEL_PATH.exists()}). "
                "Check voice_mode.silero_vad.SILERO_AVAILABLE before using SileroVAD "
                "and fall back to webrtcvad when it is False."
            )

        x = _to_float32(frame_int16_or_float)
        self._buffer = np.concatenate([self._buffer, x])

        while len(self._buffer) >= self._window_samples:
            window = self._buffer[: self._window_samples]
            self._buffer = self._buffer[self._window_samples :]
            # Prepend the carried-over context (see _CONTEXT_SAMPLES) -- the
            # model's real input length is context+window, not window alone.
            model_input = np.concatenate([self._context, window])
            inp = model_input.reshape(1, -1).astype(np.float32)
            outputs = self._session.run(
                None, {"input": inp, "state": self._state, "sr": self._sr_arr}
            )
            self._last_prob = float(outputs[0].reshape(-1)[0])
            self._state = outputs[1]
            # Next window's context is the tail of THIS window (post-concat),
            # matching the upstream wrapper's `self._context = x[..., -context_size:]`.
            self._context = model_input[-self._context_samples :]

        return self._last_prob


@dataclass
class EndpointState:
    """Snapshot returned by Endpointer.update() for the frame just fed in."""

    speech_started: bool = False  # speech has been observed at some point this turn
    endpointed: bool = False      # end-of-turn has fired (latches True until reset())
    speech_prob: float = 0.0      # the frame's raw P(speech), for logging/debug


def _frame_duration_ms(frame: np.ndarray, sample_rate: int) -> float:
    return 1000.0 * np.asarray(frame).reshape(-1).shape[0] / sample_rate


class Endpointer:
    """Turn-taking state machine driven by SileroVAD's P(speech) stream.

    This is the piece that fixes the actual bug: webrtcvad's binary decision
    means a noisy/breathy trailing frame counts as "speech" and the silence
    timer restarts, so a turn can hang indefinitely. Endpointer instead
    thresholds the CONTINUOUS probability -- breath and steady noise score
    low even at high energy -- and requires min_endpoint_ms of CONSECUTIVE
    sub-threshold time (after speech has genuinely started) before declaring
    the turn over. min_speech_ms is a symmetric debounce on the other end
    (ignore a single noisy above-threshold blip as "speech started").

    Frozen interface (voicemode-endpointing-bargein W1): thresholds are ctor
    args so the config-layer worker can wire them to env knobs without
    touching this file.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        speech_threshold: float = 0.5,
        min_endpoint_ms: int = 700,
        min_speech_ms: int = 200,
    ):
        self.sample_rate = sample_rate
        self.speech_threshold = speech_threshold
        self.min_endpoint_ms = min_endpoint_ms
        self.min_speech_ms = min_speech_ms

        self._vad = SileroVAD(sample_rate=sample_rate)
        self._speech_started = False
        self._endpointed = False
        # Consecutive time (ms) the prob has been >= threshold, since the
        # last time it dropped below -- gates speech_started via
        # min_speech_ms. Reset to 0 on any sub-threshold frame.
        self._above_ms = 0.0
        # Consecutive time (ms) the prob has been < threshold, since speech
        # last started/resumed -- gates endpointed via min_endpoint_ms. Reset
        # to 0 on any above-threshold frame (a resumed word cancels a
        # trailing-silence run in progress, same as the existing webrtcvad
        # loop's SILENCE_AFTER_SPEECH state in converse.py).
        self._below_ms = 0.0

    def reset(self) -> None:
        """Clear all turn-taking state (and the underlying SileroVAD's RNN
        state/buffer) for a fresh turn."""
        self._vad.reset()
        self._speech_started = False
        self._endpointed = False
        self._above_ms = 0.0
        self._below_ms = 0.0

    def update(self, frame_int16_or_float: np.ndarray) -> EndpointState:
        frame_ms = _frame_duration_ms(frame_int16_or_float, self.sample_rate)
        prob = self._vad.prob(frame_int16_or_float)

        # Once endpointed, latch -- further frames don't un-endpoint. A
        # caller that wants to keep listening past one endpoint (e.g.
        # append-to-turn) calls reset() itself, same contract as the
        # existing converse.py silence-threshold state machine.
        if not self._endpointed:
            if prob >= self.speech_threshold:
                self._above_ms += frame_ms
                self._below_ms = 0.0
                if not self._speech_started and self._above_ms >= self.min_speech_ms:
                    self._speech_started = True
            else:
                self._above_ms = 0.0
                if self._speech_started:
                    self._below_ms += frame_ms
                    if self._below_ms >= self.min_endpoint_ms:
                        self._endpointed = True

        return EndpointState(
            speech_started=self._speech_started,
            endpointed=self._endpointed,
            speech_prob=prob,
        )
