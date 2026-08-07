#!/usr/bin/env python3
"""Generate the deterministic synthetic-audio fixtures in this directory.

These fixtures are the ground-truth oracle exercising the ONE thing this
whole workstream fixes: webrtcvad's binary speech/no-speech decision treats
steady ENERGY as speech, so noise/breath fools it and a turn's silence timer
never accumulates. A model-based probability (Silero VAD, voice_mode/silero_vad.py)
is supposed to separate "loud" from "speech-shaped" -- these fixtures are
built specifically to have real energy in the wrong places, so a
test that just checks RMS/energy would get the wrong answer.

Run from the repo root:  uv run python tests/fixtures/endpointing/generate_fixtures.py

Requires macOS `say` (see the note below for WHY) to synthesize the speech
segments, and onnxruntime + the vendored model (voice_mode/resources/models/
silero_vad.onnx) to print the calibration stats this script asserts on --
those asserts are a build-time self-check, not something the runtime code
depends on. Output: 3 .wav files, 24kHz int16 mono (voice_mode.config.SAMPLE_RATE),
committed alongside this script -- regenerating them is only needed if the
scenarios themselves change, not for every test run.

WHY macOS `say` for the speech segments, not synthesized tones: Silero VAD is
a real trained neural net, and empirically it is (a) essentially
AMPLITUDE-INVARIANT -- scaling real speech from 1.0x down to 0.02x amplitude
left mean P(speech) unchanged (~0.95) -- and (b) NOT foolable by hand-crafted
harmonic/formant synthesis. Multiple attempts here (pure additive harmonics
with a formant-shaped envelope, a jittered glottal-pulse train through
resonant bandpass "formant" filters -- classic Klatt-style synthesis) all
scored a mean P(speech) under 0.002, indistinguishable from silence, despite
sounding roughly voice-like to a human ear. `say` is LOCAL, OFFLINE macOS
TTS -- used here only at fixture-generation time, never at runtime -- and
produces genuinely speech-shaped audio Silero scores >0.9 on (see the
CALIBRATION block this script prints).

A SECOND, independent bug was found and fixed in voice_mode/silero_vad.py
while building this: even genuine `say` speech, fed as raw 512-sample
windows with no adjustment, scored a max of 0.069 -- looking exactly like
"the model just doesn't work". Root cause (confirmed via WebFetch of the
upstream snakers4/silero-vad OnnxWrapper source, not guessed): the model
expects context_size samples (64 @16kHz) prepended from the PREVIOUS
window, i.e. its real input length is window+context, not window alone.
Fixed in SileroVAD.prob() -- see that file's _CONTEXT_SAMPLES comment.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import signal as sp
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from voice_mode.config import SAMPLE_RATE  # noqa: E402  (24000, see module docstring)
from voice_mode.silero_vad import SileroVAD, SILERO_AVAILABLE  # noqa: E402

FIXTURES_DIR = Path(__file__).parent
VAD_SR = 16000  # Silero's working rate -- SileroVAD resamples internally is
                 # NOT true; callers must resample. This script does the same
                 # resample the Endpointer's real consumers will do, purely
                 # to print honest calibration numbers.


def _say(text: str, voice: str = "Samantha", sr: int = SAMPLE_RATE) -> np.ndarray:
    """Synthesize `text` via macOS `say` at `sr` Hz, mono int16. Raises
    RuntimeError with the offending command on failure (never silently
    returns empty/garbage audio)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        cmd = ["say", "-v", voice, "-o", tmp_path, f"--data-format=LEI16@{sr}", text]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"`say` failed ({result.returncode}): {result.stderr}\ncmd: {cmd}")
        file_sr, data = wavfile.read(tmp_path)
        if file_sr != sr:
            raise RuntimeError(f"say produced {file_sr}Hz, expected {sr}Hz")
        return np.asarray(data, dtype=np.int16).reshape(-1)
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass


def _bandpass_noise(n: int, sr: int, lo: float, hi: float, seed: int) -> np.ndarray:
    """Deterministic (seeded) band-limited noise, normalized to peak 1.0
    float64 -- the caller scales amplitude. Used for breath/babble segments:
    real broadband energy, but NOT speech-shaped, so Silero should score it
    low regardless of how loud it's scaled (see CALIBRATION output)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    b, a = sp.butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    filtered = sp.lfilter(b, a, noise)
    peak = np.max(np.abs(filtered))
    return filtered / peak if peak > 0 else filtered


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    n_out = int(len(x) * sr_out / sr_in)
    return sp.resample(x.astype(np.float64), n_out)


def _to_int16(x_float: np.ndarray) -> np.ndarray:
    return np.clip(x_float * 32768.0, -32768, 32767).astype(np.int16)


def _write(name: str, data_int16: np.ndarray, sr: int = SAMPLE_RATE) -> Path:
    path = FIXTURES_DIR / name
    wavfile.write(path, sr, data_int16)
    return path


def _silero_probs(data_int16: np.ndarray, sr: int) -> np.ndarray:
    """Run the REAL vendored Silero model over a clip (resampled to VAD_SR),
    frame by frame, for the calibration printout / asserts below."""
    x16 = (_resample(data_int16, sr, VAD_SR) / 32768.0).astype(np.float32)
    vad = SileroVAD(sample_rate=VAD_SR)
    win = 512
    probs = []
    for i in range(0, max(len(x16) - win, 0), win):
        probs.append(vad.prob(x16[i : i + win]))
    return np.array(probs)


def _rms(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def build_clean_speech_then_pause() -> None:
    """~2s of real (say-synthesized) speech, then ~1.5s of TRUE silence.
    Baseline/sanity fixture: nothing tricky, both segments should read
    exactly as their names say to any correct detector."""
    speech = _say("Hey, can you hear me okay right now?")
    silence = np.zeros(int(1.5 * SAMPLE_RATE), dtype=np.int16)
    out = np.concatenate([speech, silence])
    path = _write("clean_speech_then_pause.wav", out)
    print(f"[clean_speech_then_pause] speech={len(speech)/SAMPLE_RATE:.2f}s "
          f"silence={len(silence)/SAMPLE_RATE:.2f}s -> {path.name}")

    if not SILERO_AVAILABLE:
        return
    speech_probs = _silero_probs(speech, SAMPLE_RATE)
    silence_probs = _silero_probs(silence, SAMPLE_RATE)
    print(f"  speech  mean_prob={speech_probs.mean():.4f} max={speech_probs.max():.4f} "
          f"frac>0.5={float((speech_probs > 0.5).mean()):.3f}")
    print(f"  silence mean_prob={silence_probs.mean():.4f} max={silence_probs.max():.4f}")
    assert speech_probs.mean() > 0.5, "speech segment should score >0.5 mean P(speech)"
    assert silence_probs.max() < 0.3, "true silence should stay well under threshold"


def build_speech_then_breath_tail() -> None:
    """Real speech, then a breath/plosive-like TAIL with genuine RMS energy
    (a bandpass 100-400Hz noise burst, moderate-to-loud, decaying envelope --
    the frequency band and shape of an exhale/mouth-noise, not a tone) that
    must score LOW Silero probability despite that energy. This is the
    fixture that directly demonstrates the bug being fixed: a plain
    energy-gate (webrtcvad + RMS floor) cannot tell this apart from speech;
    Silero's probability can."""
    speech = _say("Wait, hold on a second.")

    breath_n = int(0.7 * SAMPLE_RATE)
    breath = _bandpass_noise(breath_n, SAMPLE_RATE, 100, 400, seed=3)
    # Decaying envelope (fast attack, slower decay) -- shaped like a real
    # exhale, not a flat noise gate.
    t = np.arange(breath_n) / SAMPLE_RATE
    envelope = np.exp(-t / 0.35)
    envelope[: int(0.03 * SAMPLE_RATE)] *= np.linspace(0, 1, int(0.03 * SAMPLE_RATE))
    breath = breath * envelope
    # Scale to match/exceed the speech segment's own RMS -- "real energy",
    # not a quiet residual.
    speech_rms = _rms(speech.astype(np.float64) / 32768.0)
    breath_peak_rms = _rms(breath)
    breath = breath * (speech_rms * 1.3 / breath_peak_rms)
    breath_int16 = _to_int16(breath)

    trailing_silence = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.int16)
    out = np.concatenate([speech, breath_int16, trailing_silence])
    path = _write("speech_then_breath_tail.wav", out)
    print(f"[speech_then_breath_tail] speech={len(speech)/SAMPLE_RATE:.2f}s "
          f"breath={len(breath_int16)/SAMPLE_RATE:.2f}s -> {path.name}")

    assert _rms(breath) >= speech_rms, "breath tail must carry real energy (>= speech RMS)"
    if not SILERO_AVAILABLE:
        return
    speech_probs = _silero_probs(speech, SAMPLE_RATE)
    breath_probs = _silero_probs(breath_int16, SAMPLE_RATE)
    print(f"  speech mean_prob={speech_probs.mean():.4f}")
    print(f"  breath rms={_rms(breath):.4f} (speech rms={speech_rms:.4f}) "
          f"mean_prob={breath_probs.mean():.4f} max={breath_probs.max():.4f}")
    assert speech_probs.mean() > 0.5, "speech segment should score >0.5 mean P(speech)"
    assert breath_probs.max() < 0.5, "breath tail must stay below the speech threshold despite its energy"


def build_speech_with_background_chatter() -> None:
    """Real (say) foreground speech for its own duration, layered on a
    CONTINUOUS quieter background noise bed that keeps playing after the
    foreground stops. Confirms end-of-speech still gets detected once the
    foreground ends, even though the mic never goes to true silence.

    The background bed is band-limited (150-3000Hz, slow amplitude
    modulation) synthesized noise, NOT a second `say` utterance: Silero was
    empirically found to be amplitude-invariant on genuine speech (scaling
    real speech from 1.0x to 0.02x left mean P(speech) ~unchanged at ~0.95),
    so a quieted-down real voice would NOT reliably read as "not speech" --
    it would defeat the point of this fixture. The noise bed, by contrast,
    stays under 0.5 probability even at levels louder than what's used here
    (see this function's printed calibration)."""
    foreground = _say("I need to grab my keys real quick.")
    tail_s = 1.5
    total_n = len(foreground) + int(tail_s * SAMPLE_RATE)

    bg = _bandpass_noise(total_n, SAMPLE_RATE, 150, 3000, seed=11)
    t = np.arange(total_n) / SAMPLE_RATE
    bg_mod = 0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t + 1.0)
    bg = bg * bg_mod
    bg = bg * 0.25  # quieter than the foreground, but continuous + real energy
    bg_int16 = _to_int16(bg)

    fg_padded = np.concatenate([foreground, np.zeros(total_n - len(foreground), dtype=np.int16)])
    mixed = np.clip(fg_padded.astype(np.int32) + bg_int16.astype(np.int32), -32768, 32767).astype(np.int16)
    path = _write("speech_with_background_chatter.wav", mixed)
    print(f"[speech_with_background_chatter] foreground={len(foreground)/SAMPLE_RATE:.2f}s "
          f"total={total_n/SAMPLE_RATE:.2f}s -> {path.name}")

    if not SILERO_AVAILABLE:
        return
    fg_probs = _silero_probs(fg_padded[: len(foreground)], SAMPLE_RATE)
    bg_only_probs = _silero_probs(bg_int16[len(foreground) :], SAMPLE_RATE)
    print(f"  foreground(+bg) mean_prob={fg_probs.mean():.4f}")
    print(f"  background-only tail rms={_rms(bg_int16[len(foreground):].astype(np.float64)/32768.0):.4f} "
          f"mean_prob={bg_only_probs.mean():.4f} max={bg_only_probs.max():.4f}")
    assert fg_probs.mean() > 0.5, "foreground segment should score >0.5 mean P(speech)"
    assert bg_only_probs.max() < 0.5, "background-only tail must stay below threshold so endpointing still fires"


def main() -> None:
    if not SILERO_AVAILABLE:
        print("SILERO_AVAILABLE is False -- generating fixtures without the calibration "
              "asserts (onnxruntime/model unavailable in this environment). The .wav "
              "files will still be written; re-run in an environment with the `silero` "
              "extra installed to get the calibration printout + build-time asserts.")
    print("=== CALIBRATION (verifies these fixtures actually exercise energy-vs-probability) ===")
    build_clean_speech_then_pause()
    build_speech_then_breath_tail()
    build_speech_with_background_chatter()
    print("=== done ===")


if __name__ == "__main__":
    main()
