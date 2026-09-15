"""Offline ERLE (Echo Return Loss Enhancement) comparison: NLMS vs AEC3.

This is a SCRIPT, not a pytest test -- it proves AEC3's cancellation
advantage over the hand-rolled NLMS filter WITHOUT opening a real
microphone/speaker (per the vm-aec3-phase1 dispatch's hard constraint: never
open a real audio device from this worktree). It synthesizes:

  1. A known far-end (TTS-like) reference signal -- band-limited noise + a
     couple of voiced-formant-like sinusoids, the same synthetic-signal
     family test_aec.py already uses for the NLMS convergence tests.
  2. A near-end signal = a linear echo of that reference (a fixed delay +
     gain, standing in for the speaker->room/BT-link->mic acoustic path)
     PLUS, on a double-talk segment only, an independent "near speech" burst
     (a different frequency content, uncorrelated with the reference) to
     prove the canceller doesn't just learn to null everything.

For each engine (NLMS `EchoCanceller`, AEC3 `AEC3EchoCanceller`) it streams
the pair through in the SAME chunk size barge_in.py actually uses
(CHUNK_SAMPLES_VAD, 30ms @ 16kHz) and reports ERLE in dB on:
  - the echo-only segment (measures raw cancellation)
  - the double-talk segment (measures cancellation while real "speech" is
    also present -- the harder, more realistic case)

ERLE = 10*log10(rms(near)^2 / rms(clean)^2), computed over the CONVERGED
tail of each segment (the adaptive filters need time to lock on) so an
un-converged startup transient doesn't understate the result.

Run: .venv-aec3/bin/python tests/aec3_erle_offline.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_mode.aec import EchoCanceller, AEC3EchoCanceller, AEC3_AVAILABLE  # noqa: E402

SR = 16000
CHUNK_MS = 30
CHUNK_SAMPLES = int(SR * CHUNK_MS / 1000)  # matches barge_in.py's CHUNK_SAMPLES_VAD

ECHO_DELAY_SAMPLES = 80   # ~5ms acoustic delay
ECHO_GAIN = 0.6
DOUBLE_TALK_GAIN = 0.35   # near-speech relative amplitude during double-talk


def _make_reference(n: int, seed: int = 42) -> np.ndarray:
    """TTS-like far-end: a couple of voiced-formant sinusoids + light noise,
    same shape as test_aec.py's synthetic reference."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    ref = (
        0.5 * np.sin(2 * np.pi * 220 * t)
        + 0.2 * np.sin(2 * np.pi * 440 * t)
        + 0.05 * rng.standard_normal(n)
    )
    return ref.astype(np.float64)


def _make_near_speech(n: int, seed: int = 7) -> np.ndarray:
    """Independent 'his voice' burst -- different fundamental + noise profile
    than the reference, uncorrelated, standing in for a real interruption."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    speech = (
        0.4 * np.sin(2 * np.pi * 180 * t + 0.3)
        + 0.15 * np.sin(2 * np.pi * 900 * t)
        + 0.08 * rng.standard_normal(n)
    )
    return speech.astype(np.float64)


def _echo_from_reference(far: np.ndarray, delay_samples: int, gain: float) -> np.ndarray:
    echo = np.zeros(len(far), dtype=np.float64)
    if delay_samples < len(far):
        echo[delay_samples:] = gain * far[: len(far) - delay_samples]
    return echo


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def _erle_db(before: np.ndarray, after: np.ndarray) -> float:
    rms_before = _rms(before)
    rms_after = _rms(after)
    if rms_after <= 1e-12:
        return float("inf") if rms_before > 1e-12 else 0.0
    return 20.0 * np.log10(max(rms_before, 1e-12) / rms_after)


def _stream_process(aec, near: np.ndarray, far: np.ndarray) -> np.ndarray:
    """Feed near/far through `aec` in barge_in.py's real chunk size, matching
    how BargeInListener._watch_loop actually calls .process() per-frame."""
    out = np.empty_like(near)
    n = len(near)
    pos = 0
    while pos < n:
        end = min(pos + CHUNK_SAMPLES, n)
        out[pos:end] = aec.process(near[pos:end], far[pos:end])
        pos = end
    return out


def run_segment(aec_factory, label: str):
    """Build one echo-only segment followed by one double-talk segment,
    stream them through a fresh `aec`, and return (erle_echo_only, erle_double_talk)."""
    seg_n = SR * 2  # 2s per segment -- enough for the adaptive filters to converge
    far = _make_reference(seg_n * 2)
    far_echo_only, far_double_talk = far[:seg_n], far[seg_n:]

    near_echo_only = _echo_from_reference(far_echo_only, ECHO_DELAY_SAMPLES, ECHO_GAIN)
    near_double_talk = (
        _echo_from_reference(far_double_talk, ECHO_DELAY_SAMPLES, ECHO_GAIN)
        + DOUBLE_TALK_GAIN * _make_near_speech(seg_n)
    )

    near = np.concatenate([near_echo_only, near_double_talk])
    far_full = np.concatenate([far_echo_only, far_double_talk])

    aec = aec_factory()
    clean = _stream_process(aec, near, far_full)
    clean_echo_only, clean_double_talk = clean[:seg_n], clean[seg_n:]

    # Converged tail only (skip the first third of each segment) -- the
    # adaptive filters need time to lock on, same convention test_aec.py uses.
    tail = slice(seg_n // 3, seg_n)
    erle_echo_only = _erle_db(near_echo_only[tail], clean_echo_only[tail])
    erle_double_talk = _erle_db(near_double_talk[tail], clean_double_talk[tail])
    return erle_echo_only, erle_double_talk


def main():
    print(f"AEC3_AVAILABLE = {AEC3_AVAILABLE}")
    if not AEC3_AVAILABLE:
        print("pywebrtc-audio not importable in this venv -- cannot run the AEC3 comparison.")
        sys.exit(1)

    results = {}
    results["NLMS"] = run_segment(
        lambda: EchoCanceller(sample_rate=SR, filter_ms=200, mu=0.5), "NLMS"
    )
    results["AEC3"] = run_segment(
        lambda: AEC3EchoCanceller(sample_rate=SR, stream_delay_ms=0), "AEC3"
    )

    print()
    print(f"{'Engine':<8} {'Echo-only ERLE (dB)':>22} {'Double-talk ERLE (dB)':>24}")
    print("-" * 56)
    for engine, (erle_echo, erle_dt) in results.items():
        print(f"{engine:<8} {erle_echo:>22.1f} {erle_dt:>24.1f}")
    print()

    nlms_echo, nlms_dt = results["NLMS"]
    aec3_echo, aec3_dt = results["AEC3"]
    print(f"AEC3 advantage: +{aec3_echo - nlms_echo:.1f} dB echo-only, "
          f"+{aec3_dt - nlms_dt:.1f} dB double-talk")


if __name__ == "__main__":
    main()
