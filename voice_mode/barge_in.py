"""Natural-mode barge-in listener (Phase 1).

Natural mode lets William interrupt the agent mid-sentence, the way he can
with the ChatGPT/Claude mobile apps -- unlike "turn mode" (the unchanged
default), where the mic only opens AFTER the agent finishes talking.

The mechanism, concretely:
  1. While TTS is playing (audio_player.is_tts_speaking() is True), open a
     SECOND, concurrent microphone InputStream (the existing turn-taking
     listener in converse.record_audio_with_silence_detection() only ever
     runs sequentially, after playback -- this is a separate stream, not a
     rewrite of that one).
  2. Run each mic frame through a software echo canceller (voice_mode.aec)
     against the KNOWN reference signal actually being sent to the speaker
     (audio_player.get_reference_audio()), so the mic hearing the agent's
     own voice doesn't false-trigger.
  3. Run the post-AEC signal through webrtcvad (the SAME VAD library the
     turn-taking listener already uses).
  4. On BARGE_IN_TRIGGER_MS of sustained post-AEC speech, halt playback
     within one audio buffer (audio_player.trigger_barge_in() -- reuses the
     EXACT mid-buffer stop mechanism already built for the manual Pause
     flag) and hand back the audio captured right at the trigger as
     pre-roll, so his interruption becomes the start of the next turn
     instead of being thrown away and re-prompted.

Entirely inert in turn mode (the default): converse.py only constructs a
BargeInListener when natural_mode_enabled() is True, so turn mode's existing
sequential flow is byte-for-byte unchanged when the flag file is absent.
"""

import logging
import os
import queue
import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import sounddevice as sd

from voice_mode import audio_player
from voice_mode.aec import EchoCanceller
from voice_mode.config import (
    SAMPLE_RATE,
    BARGE_IN_TRIGGER_MS,
    BARGE_IN_VAD_AGGRESSIVENESS,
    AEC_FILTER_MS,
    AEC_REF_DELAY_MS,
    AEC_STEP_SIZE,
    NATURAL_MODE_FLAG_PATH,
)

logger = logging.getLogger("voicemode.barge_in")

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    webrtcvad = None
    VAD_AVAILABLE = False

# webrtcvad only supports 8000 / 16000 / 32000 Hz; the turn-taking listener in
# converse.py already downsamples to 16kHz for the same reason, so the
# barge-in listener's VAD + AEC stage runs at 16kHz too (it only needs to
# drive a speech/no-speech decision, not preserve full TTS fidelity).
VAD_WORK_RATE = 16000
CHUNK_MS = 30  # webrtcvad frame size must be 10, 20, or 30ms
CHUNK_SAMPLES_MIC = int(SAMPLE_RATE * CHUNK_MS / 1000)
CHUNK_SAMPLES_VAD = int(VAD_WORK_RATE * CHUNK_MS / 1000)
# Bound how much raw mic audio we retain as "pre-roll" so a trigger can hand
# back what he actually said leading INTO the interruption, not just the
# instant it crossed the trigger threshold. ~600ms is comfortably more than
# BARGE_IN_TRIGGER_MS's default (300ms), so the pre-roll always covers the
# whole speech run that caused the trigger.
_MAX_PRE_ROLL_CHUNKS = 20


def natural_mode_enabled() -> bool:
    """True while natural mode's flag file exists. Absence (the default) is
    turn mode -- mirrors the existing focus-hold / pause-flag pattern already
    used elsewhere in this codebase (a plain os.path.exists check, no daemon
    watcher needed since mode is a per-turn/session setting, not something
    toggled mid-utterance)."""
    try:
        return os.path.exists(NATURAL_MODE_FLAG_PATH)
    except Exception:
        return False


@dataclass
class BargeInResult:
    """What happened while the listener was armed."""
    triggered: bool
    pre_roll: Optional[np.ndarray] = None  # raw mic audio at SAMPLE_RATE, int16
    error: Optional[str] = None


class BargeInListener:
    """Concurrent mic listener armed for the duration of one TTS playback."""

    def __init__(self, vad_aggressiveness: Optional[int] = None):
        self._vad_aggressiveness = (
            vad_aggressiveness if vad_aggressiveness is not None else BARGE_IN_VAD_AGGRESSIVENESS
        )
        self._stream: Optional[sd.InputStream] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._audio_queue: "queue.Queue" = queue.Queue()
        self._result_lock = threading.Lock()
        self._triggered = False
        self._pre_roll_chunks: List[np.ndarray] = []
        self._error: Optional[str] = None
        self._aec = EchoCanceller(sample_rate=VAD_WORK_RATE, filter_ms=AEC_FILTER_MS, mu=AEC_STEP_SIZE)
        self._vad = webrtcvad.Vad(self._vad_aggressiveness) if VAD_AVAILABLE else None

    def start(self):
        """Open the concurrent input stream and start the watcher thread.

        No-op (logs a warning, leaves triggered=False) if webrtcvad isn't
        available or the input stream can't open -- natural mode degrades to
        "TTS just plays like turn mode this utterance" rather than guessing
        at speech onset without VAD.
        """
        if not VAD_AVAILABLE:
            logger.warning("barge-in: webrtcvad unavailable — natural-mode listener not started this turn")
            self._error = "webrtcvad unavailable"
            return

        audio_player.reset_barge_in_event()
        self._stop_event.clear()

        def _callback(indata, frames, time_info, status):
            if status:
                logger.debug(f"barge-in input stream status: {status}")
            try:
                self._audio_queue.put_nowait(indata.copy())
            except Exception:
                pass

        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype=np.int16,
                callback=_callback,
                blocksize=CHUNK_SAMPLES_MIC,
            )
            self._stream.start()
        except Exception as e:
            logger.warning(f"barge-in: could not open concurrent input stream ({e}) — natural mode inactive this turn")
            self._error = str(e)
            self._stream = None
            return

        self._thread = threading.Thread(target=self._watch_loop, daemon=True, name="natural-mode-barge-in")
        self._thread.start()

    def _watch_loop(self):
        from scipy import signal as scipy_signal

        speech_run_ms = 0
        ref_delay_samples = int(SAMPLE_RATE * AEC_REF_DELAY_MS / 1000)

        while not self._stop_event.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            except Exception:
                continue

            chunk_flat = chunk.flatten()

            # Bounded pre-roll of raw mic audio at native rate -- handed back
            # on trigger so converse.py can seed the next turn's recording
            # with what he actually said, not just silence-from-here-on.
            self._pre_roll_chunks.append(chunk_flat.copy())
            if len(self._pre_roll_chunks) > _MAX_PRE_ROLL_CHUNKS:
                self._pre_roll_chunks.pop(0)

            if not audio_player.is_tts_speaking():
                # Nothing playing right now — nothing to barge in on. Reset
                # the speech-run counter (a later burst should start clean)
                # but keep listening; playback may resume any moment.
                speech_run_ms = 0
                continue

            # Downsample mic capture to the VAD/AEC working rate.
            near_16k = scipy_signal.resample(
                chunk_flat.astype(np.float64) / 32768.0,
                int(len(chunk_flat) * VAD_WORK_RATE / SAMPLE_RATE),
            )
            near_16k = _fit_length(near_16k, CHUNK_SAMPLES_VAD)

            # Pull the matching window of the KNOWN reference (TTS) signal,
            # already delay-compensated per config.AEC_REF_DELAY_MS.
            far_native = audio_player.get_reference_audio(len(chunk_flat), delay_samples=ref_delay_samples)
            far_16k = scipy_signal.resample(
                far_native.astype(np.float64),
                int(len(chunk_flat) * VAD_WORK_RATE / SAMPLE_RATE),
            )
            far_16k = _fit_length(far_16k, CHUNK_SAMPLES_VAD)

            clean = self._aec.process(near_16k, far_16k)

            clean_int16 = np.clip(clean * 32768.0, -32768, 32767).astype(np.int16)
            frame_bytes = clean_int16.tobytes()

            try:
                is_speech = self._vad.is_speech(frame_bytes, VAD_WORK_RATE)
            except Exception as e:
                logger.debug(f"barge-in VAD error: {e}")
                is_speech = False

            speech_run_ms = speech_run_ms + CHUNK_MS if is_speech else 0

            if speech_run_ms >= BARGE_IN_TRIGGER_MS:
                logger.info(
                    f"🗣️ Barge-in detected ({speech_run_ms}ms sustained post-AEC speech during playback) — interrupting"
                )
                with self._result_lock:
                    self._triggered = True
                audio_player.trigger_barge_in()
                self._stop_event.set()
                break

    def stop(self) -> BargeInResult:
        """Stop the listener (idempotent) and report what happened."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        with self._result_lock:
            triggered = self._triggered

        pre_roll = None
        if triggered and self._pre_roll_chunks:
            pre_roll = np.concatenate(self._pre_roll_chunks)

        return BargeInResult(triggered=triggered, pre_roll=pre_roll, error=self._error)


def _fit_length(arr: np.ndarray, n: int) -> np.ndarray:
    """Pad with zeros or truncate `arr` to exactly `n` samples."""
    if len(arr) == n:
        return arr
    if len(arr) > n:
        return arr[:n]
    return np.concatenate([arr, np.zeros(n - len(arr), dtype=arr.dtype)])
