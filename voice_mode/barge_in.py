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

import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
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
    BARGE_IN_ENERGY_MARGIN,
    NATURAL_MODE_FLAG_PATH,
)
from voice_mode.utils.event_logger import (
    log_barge_in_armed,
    log_barge_in_unavailable,
    log_barge_in_triggered,
    log_barge_in_disarmed,
)

logger = logging.getLogger("voicemode.barge_in")

# --- Decision-trace logging (per-frame evidence for live-trial post-mortems) -
# The 2026-07-13 live trial left ZERO evidence of what actually happened:
# barge_in.py's logger.* calls only reach stderr via logging.basicConfig
# (config.setup_logging only adds a FileHandler when VOICEMODE_DEBUG=true, and
# it wasn't that day), and no BARGE_IN_* event types existed in the event
# logger. So whether the listener even armed, whether it ever saw plausible
# speech energy, and whether trigger_barge_in() was ever called was simply
# unknown -- not "the mechanism failed", but "there is no record either way".
#
# This trace is opt-in (VOICEMODE_BARGE_IN_TRACE=1) because per-frame logging
# at CHUNK_MS=30 is ~33 writes/sec -- too dense to leave on by default -- but
# it is exactly what the next live trial (or the acoustic test harness) needs:
# per-frame RMS of the near-end (raw mic), far-end (known TTS reference), and
# post-AEC clean signal, plus the VAD's is_speech decision and the running
# speech-run counter. If natural mode fails again, this file answers "was
# there ANY detected speech energy, was the AEC actually attenuating the
# echo, and how close did it get to the trigger threshold" instead of a
# second round of guessing.
_TRACE_ENABLED = os.getenv("VOICEMODE_BARGE_IN_TRACE", "").lower() in ("1", "true", "yes")
_TRACE_DIR = Path(os.path.expanduser("~/.voicemode/logs/barge_in"))


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))

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
        # Evidence trail (see module docstring re: 2026-07-13) -- populated in
        # start()/stop() regardless of whether the trace file is enabled, so
        # BARGE_IN_ARMED/DISARMED events always carry accurate frame counts.
        self._armed_at: Optional[float] = None
        self._frames_processed: int = 0
        self._trace_fh = None

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
            log_barge_in_unavailable("webrtcvad unavailable")
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
            log_barge_in_unavailable(f"stream open failed: {e}")
            return

        if _TRACE_ENABLED:
            try:
                _TRACE_DIR.mkdir(parents=True, exist_ok=True)
                trace_path = _TRACE_DIR / f"trace_{time.strftime('%Y-%m-%d')}.jsonl"
                self._trace_fh = open(trace_path, "a")
            except Exception as e:
                logger.debug(f"barge-in: could not open trace file ({e}) — continuing without it")
                self._trace_fh = None

        self._armed_at = time.monotonic()
        self._frames_processed = 0
        log_barge_in_armed(self._vad_aggressiveness)

        self._thread = threading.Thread(target=self._watch_loop, daemon=True, name="natural-mode-barge-in")
        self._thread.start()

    def _watch_loop(self):
        from scipy import signal as scipy_signal

        speech_run_ms = 0
        ref_delay_samples = int(SAMPLE_RATE * AEC_REF_DELAY_MS / 1000)
        # Adaptive echo-floor gate state (see config.BARGE_IN_ENERGY_MARGIN
        # docstring for why this exists). echo_floor is an ASYMMETRIC
        # minimum-statistics tracker of rms_clean -- the same family of
        # technique used for noise-floor estimation in real-time speech
        # processing (fast down / slow up). It updates on EVERY frame
        # (unlike an earlier version of this gate that only updated while
        # speech_run_ms==0 -- that version silently never calibrated when
        # TTS was loud from frame 1, which is the common case, making the
        # gate an inert no-op for the entire turn). Moving down fast lets it
        # track the true ambient echo level quickly; moving up slow means a
        # genuine interruption's higher energy can't drag the floor up to
        # swallow itself mid-run.
        echo_floor = 0.0
        echo_floor_initialized = False
        ENERGY_FLOOR_ALPHA_DOWN = 0.2
        ENERGY_FLOOR_ALPHA_UP = 0.02

        while not self._stop_event.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            except Exception:
                continue

            chunk_flat = chunk.flatten()
            self._frames_processed += 1

            # Bounded pre-roll of raw mic audio at native rate -- handed back
            # on trigger so converse.py can seed the next turn's recording
            # with what he actually said, not just silence-from-here-on.
            self._pre_roll_chunks.append(chunk_flat.copy())
            if len(self._pre_roll_chunks) > _MAX_PRE_ROLL_CHUNKS:
                self._pre_roll_chunks.pop(0)

            tts_speaking = audio_player.is_tts_speaking()

            if not tts_speaking:
                # Nothing playing right now — nothing to barge in on. Reset
                # the speech-run counter (a later burst should start clean)
                # but keep listening; playback may resume any moment. Still
                # traced (rms_near only) so a live trial can confirm the mic
                # callback is actually firing even between utterances.
                if self._trace_fh is not None:
                    self._trace_write(
                        rms_near=_rms(chunk_flat.astype(np.float64) / 32768.0),
                        rms_far=0.0, rms_clean=0.0, is_speech=False,
                        speech_run_ms=0, tts_speaking=False,
                    )
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

            clean_rms = _rms(clean)

            # Always update the asymmetric echo-floor tracker first (see the
            # ENERGY_FLOOR_ALPHA_* comment above), THEN decide the gate off
            # its pre-update value -- floor tracking and gating must not be
            # entangled, or the floor never calibrates when the very first
            # frame is already loud (a real, previously-shipped bug: gating
            # the floor update on speech_run_ms==0 meant a continuously-loud
            # TTS onset from frame 1 never let speech_run_ms return to 0,
            # so the floor update condition never fired -- the gate silently
            # never engaged for that entire turn).
            if BARGE_IN_ENERGY_MARGIN > 0:
                floor_before_update = echo_floor if echo_floor_initialized else clean_rms
                if not echo_floor_initialized:
                    echo_floor = clean_rms
                    echo_floor_initialized = True
                elif clean_rms < echo_floor:
                    echo_floor = (1 - ENERGY_FLOOR_ALPHA_DOWN) * echo_floor + ENERGY_FLOOR_ALPHA_DOWN * clean_rms
                else:
                    echo_floor = (1 - ENERGY_FLOOR_ALPHA_UP) * echo_floor + ENERGY_FLOOR_ALPHA_UP * clean_rms

                gated_speech = is_speech and clean_rms >= floor_before_update * BARGE_IN_ENERGY_MARGIN
            else:
                gated_speech = is_speech

            speech_run_ms = speech_run_ms + CHUNK_MS if gated_speech else 0

            if self._trace_fh is not None:
                self._trace_write(
                    rms_near=_rms(near_16k), rms_far=_rms(far_16k), rms_clean=clean_rms,
                    is_speech=is_speech, speech_run_ms=speech_run_ms, tts_speaking=True,
                    echo_floor=echo_floor if BARGE_IN_ENERGY_MARGIN > 0 else None,
                )

            if speech_run_ms >= BARGE_IN_TRIGGER_MS:
                elapsed = time.monotonic() - self._armed_at if self._armed_at else 0.0
                logger.info(
                    f"🗣️ Barge-in detected ({speech_run_ms}ms sustained post-AEC speech during playback) — interrupting"
                )
                log_barge_in_triggered(speech_run_ms, elapsed)
                with self._result_lock:
                    self._triggered = True
                audio_player.trigger_barge_in()
                self._stop_event.set()
                break

    def _trace_write(self, *, rms_near: float, rms_far: float, rms_clean: float,
                      is_speech: bool, speech_run_ms: int, tts_speaking: bool,
                      echo_floor: Optional[float] = None):
        """Append one per-frame decision-trace record (VOICEMODE_BARGE_IN_TRACE=1
        only). Never lets a trace-write failure break the barge-in loop."""
        try:
            record = {
                "t": round(time.time(), 3),
                "rms_near": round(rms_near, 5),
                "rms_far": round(rms_far, 5),
                "rms_clean": round(rms_clean, 5),
                "is_speech": is_speech,
                "speech_run_ms": speech_run_ms,
                "tts_speaking": tts_speaking,
                "vad_aggressiveness": self._vad_aggressiveness,
            }
            if echo_floor is not None:
                record["echo_floor"] = round(echo_floor, 5)
            self._trace_fh.write(json.dumps(record) + "\n")
        except Exception:
            pass

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

        if self._armed_at is not None:
            log_barge_in_disarmed(triggered, self._frames_processed, time.monotonic() - self._armed_at)
        if self._trace_fh is not None:
            try:
                self._trace_fh.close()
            except Exception:
                pass
            self._trace_fh = None

        return BargeInResult(triggered=triggered, pre_roll=pre_roll, error=self._error)


def _fit_length(arr: np.ndarray, n: int) -> np.ndarray:
    """Pad with zeros or truncate `arr` to exactly `n` samples."""
    if len(arr) == n:
        return arr
    if len(arr) > n:
        return arr[:n]
    return np.concatenate([arr, np.zeros(n - len(arr), dtype=arr.dtype)])
