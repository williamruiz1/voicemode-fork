"""Non-blocking audio player using callback-based playback.

This module provides a queue-based audio playback system that allows multiple
concurrent audio streams without blocking or interference.
"""

import logging
import os
import queue
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger("voicemode.audio_player")

# --- Convomode live pause (interrupt mid-sentence) ---------------------------
# The menu-bar widget (or `convomode-floor.py`) can pause live speech the instant
# William taps Pause — e.g. an incoming phone call — by creating the flag file
# below; removing it (Proceed) lets the next utterance play normally.
#
# The realtime audio callback must NOT do file I/O (it runs on the audio thread),
# so a single daemon watcher polls the flag (~50ms) and reflects it into an
# in-memory Event; the callback only reads that Event. When set, the current
# stream is stopped immediately, cutting speech mid-word.
PAUSE_FLAG_PATH = os.path.expanduser("~/.voicemode/pause.flag")
_pause_event = threading.Event()


def convomode_paused() -> bool:
    """True while convomode is paused (the pause flag file exists)."""
    return _pause_event.is_set()


def _pause_flag_watcher():
    while True:
        try:
            if os.path.exists(PAUSE_FLAG_PATH):
                _pause_event.set()
            else:
                _pause_event.clear()
        except Exception:
            _pause_event.clear()
        time.sleep(0.05)


# Daemon so it never blocks interpreter exit; started once at import.
threading.Thread(target=_pause_flag_watcher, daemon=True, name="convomode-pause-watcher").start()


# --- Convomode "speaking" signal (drives the menu-bar live-audio indicator) ---
# While TTS is ACTUALLY playing, this flag file exists so VibeDispatcher's waveform
# bars animate ONLY when an agent is really talking (William 2026-06-08: "I only want
# it to move when it's talking" — before, the bars looped continuously whenever a
# session held the floor). Ref-counted so overlapping playbacks don't clear it early;
# the flag is removed only when the LAST active playback finishes.
SPEAKING_FLAG_PATH = os.path.expanduser("~/.voicemode/speaking.flag")
_speaking_lock = threading.Lock()
_speaking_count = 0


def _speaking_inc():
    global _speaking_count
    with _speaking_lock:
        _speaking_count += 1
        if _speaking_count == 1:
            try:
                os.makedirs(os.path.dirname(SPEAKING_FLAG_PATH), exist_ok=True)
                open(SPEAKING_FLAG_PATH, "w").close()
            except Exception:
                pass


def _speaking_dec():
    global _speaking_count
    with _speaking_lock:
        if _speaking_count > 0:
            _speaking_count -= 1
        if _speaking_count == 0:
            try:
                os.remove(SPEAKING_FLAG_PATH)
            except FileNotFoundError:
                pass
            except Exception:
                pass


def is_tts_speaking() -> bool:
    """True while TTS is audibly playing (in-process ref count, not the flag
    file -- no I/O race, and this is the same signal the barge-in listener
    needs to decide "is there anything to barge in on right now")."""
    return _speaking_count > 0


# --- Natural-mode barge-in (Phase 1) -----------------------------------------
# Natural mode (voice_mode/barge_in.py) runs a CONCURRENT mic listener while
# TTS plays. When it detects sustained post-AEC speech during playback, it
# calls trigger_barge_in() below -- a SEPARATE event from the manual pause
# flag above (pause = "William tapped Pause for an incoming call, no new turn
# implied"; barge-in = "he's talking, that IS the next turn, and the caller
# needs to know to seed the recording with what was just captured"), but it
# reuses the EXACT SAME instant mid-buffer stop mechanism -- the callback
# below checks both events, so no new stop-playback code path was needed.
_barge_in_event = threading.Event()


def trigger_barge_in():
    """Called by the natural-mode listener the instant it detects a genuine
    barge-in. Halts the current TTS stream within one audio buffer."""
    _barge_in_event.set()


def barge_in_triggered() -> bool:
    return _barge_in_event.is_set()


def reset_barge_in_event():
    """Clear the barge-in flag before starting a new listen cycle, so a
    trigger from a prior turn can't stale-fire the next one."""
    _barge_in_event.clear()


# --- Far-end reference ring buffer (the AEC's known-echo-source signal) ------
# Acoustic echo cancellation needs the EXACT reference signal being sent to
# the speaker so it can predict what echoes back into the mic. Rather than
# re-deriving that from scratch, every audio buffer NonBlockingAudioPlayer
# actually writes to the output stream is copied into this fixed-size
# circular buffer, keyed by a monotonically increasing sample counter so a
# reader can ask for "the reference audio as of N samples ago" (accounting
# for the acoustic/Bluetooth round-trip delay -- see config.AEC_REF_DELAY_MS).
#
# NOTE: if multiple players overlap (e.g. DJ background music playing
# alongside TTS), this buffer reflects whichever callback wrote most
# recently -- natural mode + concurrent DJ audio is an acknowledged Phase 1
# gap, not silently handled.
_REF_BUFFER_SECONDS = 5
_ref_buffer_lock = threading.Lock()
_ref_buffer: Optional[np.ndarray] = None
_ref_buffer_capacity = 0
_ref_write_pos = 0  # monotonically increasing total samples written


def _ref_buffer_init(sample_rate: int):
    global _ref_buffer, _ref_buffer_capacity
    with _ref_buffer_lock:
        capacity = int(sample_rate * _REF_BUFFER_SECONDS)
        if _ref_buffer is None or _ref_buffer_capacity != capacity:
            _ref_buffer = np.zeros(capacity, dtype=np.float32)
            _ref_buffer_capacity = capacity


def _ref_buffer_write(samples: np.ndarray):
    """Append played samples (mono, float32) to the ring buffer."""
    global _ref_write_pos
    if _ref_buffer is None or _ref_buffer_capacity == 0:
        return
    mono = samples if samples.ndim == 1 else samples[:, 0]
    n = len(mono)
    if n == 0:
        return
    with _ref_buffer_lock:
        cap = _ref_buffer_capacity
        start_idx = _ref_write_pos % cap
        end_idx = start_idx + n
        if end_idx <= cap:
            _ref_buffer[start_idx:end_idx] = mono
        else:
            first_part = cap - start_idx
            _ref_buffer[start_idx:] = mono[:first_part]
            _ref_buffer[: n - first_part] = mono[first_part:]
        _ref_write_pos += n


def get_reference_audio(n_samples: int, delay_samples: int = 0) -> np.ndarray:
    """Return the `n_samples` of TTS reference audio ending `delay_samples`
    samples ago (float32, mono). Missing history (buffer not warmed up yet,
    or asking further back than what's been written) is zero-filled --
    silence is the correct reference when nothing has played yet.
    """
    with _ref_buffer_lock:
        if _ref_buffer is None or _ref_buffer_capacity == 0:
            return np.zeros(n_samples, dtype=np.float32)
        cap = _ref_buffer_capacity
        end_total = _ref_write_pos - delay_samples
        start_total = end_total - n_samples

        out = np.zeros(n_samples, dtype=np.float32)
        if end_total <= 0:
            return out  # nothing written yet at all, within the delay window

        # Clip the readable range to what's actually been written.
        read_start = max(start_total, 0)
        read_end = min(end_total, _ref_write_pos)
        if read_end <= read_start:
            return out
        # Also never read further back than the buffer's capacity holds.
        oldest_available = max(read_start, _ref_write_pos - cap)
        read_start = max(read_start, oldest_available)
        if read_end <= read_start:
            return out

        n_read = read_end - read_start
        out_offset = read_start - start_total  # where in `out` this slice lands
        start_idx = read_start % cap
        end_idx = start_idx + n_read
        if end_idx <= cap:
            out[out_offset:out_offset + n_read] = _ref_buffer[start_idx:end_idx]
        else:
            first_part = cap - start_idx
            out[out_offset:out_offset + first_part] = _ref_buffer[start_idx:]
            out[out_offset + first_part:out_offset + n_read] = _ref_buffer[: n_read - first_part]
        return out


class NonBlockingAudioPlayer:
    """Non-blocking audio player using callback-based playback.

    This player uses a queue-based callback system to play audio without blocking
    the calling thread. It allows multiple instances to play audio concurrently
    by leveraging the system's audio mixing capabilities (Core Audio on macOS,
    PulseAudio/ALSA on Linux).

    Example:
        player = NonBlockingAudioPlayer()
        player.play(audio_samples, sample_rate=24000)
        player.wait()  # Wait for playback to complete
    """

    def __init__(self, buffer_size: int = 2048):
        """Initialize the audio player.

        Args:
            buffer_size: Size of audio buffer chunks for callback (default: 2048)
        """
        self.buffer_size = buffer_size
        self.audio_queue: Optional[queue.Queue] = None
        self.stream: Optional[sd.OutputStream] = None
        self.playback_complete = threading.Event()
        self.playback_error: Optional[Exception] = None

    def _audio_callback(self, outdata, frames, time_info, status):
        """Callback function called by sounddevice for each audio buffer.

        Args:
            outdata: Output buffer to fill with audio data
            frames: Number of frames requested
            time_info: Timing information
            status: Status flags
        """
        if status:
            logger.warning(f"Audio callback status: {status}")

        # Convomode pause — cut speech instantly, mid-sentence, the moment the
        # pause flag is set (e.g. William taps Pause for an incoming call).
        # Natural-mode barge-in reuses this SAME instant mid-buffer stop path —
        # see trigger_barge_in() above — via a separate event (a barge-in
        # implies a new turn is starting; a manual pause does not).
        if _pause_event.is_set() or _barge_in_event.is_set():
            outdata[:] = 0
            self.playback_complete.set()
            raise sd.CallbackStop()

        try:
            # Get audio chunk from queue
            chunk = self.audio_queue.get_nowait()

            # Handle end-of-stream marker
            if chunk is None:
                outdata[:] = 0
                self.playback_complete.set()
                raise sd.CallbackStop()

            # Fill output buffer
            chunk_len = len(chunk)
            if chunk_len < frames:
                # Partial chunk - pad with zeros
                if chunk.ndim == 1:
                    # Mono audio - reshape for sounddevice
                    outdata[:chunk_len, 0] = chunk
                    outdata[chunk_len:, 0] = 0
                else:
                    # Multi-channel audio
                    outdata[:chunk_len] = chunk
                    outdata[chunk_len:] = 0
                # Mark playback complete after this chunk
                self.playback_complete.set()
                raise sd.CallbackStop()
            else:
                if chunk.ndim == 1:
                    # Mono audio - reshape for sounddevice
                    outdata[:, 0] = chunk[:frames]
                else:
                    # Multi-channel audio
                    outdata[:] = chunk[:frames]

            # Mirror what was ACTUALLY sent to the speaker into the far-end
            # reference ring buffer -- this is the exact known-echo-source
            # signal the natural-mode AEC needs (see get_reference_audio()
            # above). Cheap (a numpy copy of one buffer) and unconditional --
            # harmless when natural mode / barge-in isn't running, since
            # nothing reads the buffer in that case.
            try:
                _ref_buffer_write(outdata[:, 0] if outdata.ndim > 1 else outdata)
            except Exception:
                pass  # never let reference-buffer bookkeeping break playback

        except queue.Empty:
            # No data available - output silence
            outdata[:] = 0
            logger.debug("Audio queue empty - outputting silence")

    def play(self, samples: np.ndarray, sample_rate: int, blocking: bool = False):
        """Play audio samples using non-blocking callback system.

        Args:
            samples: Audio samples to play (numpy array)
            sample_rate: Sample rate in Hz
            blocking: If True, wait for playback to complete before returning

        Raises:
            Exception: If playback error occurs
        """
        # Reset state
        self.playback_complete.clear()
        self.playback_error = None

        # Ensure samples are float32
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        # Determine number of channels
        if samples.ndim == 1:
            channels = 1
        else:
            channels = samples.shape[1]

        # Create queue and fill with audio chunks
        self.audio_queue = queue.Queue()

        # Split samples into chunks
        for i in range(0, len(samples), self.buffer_size):
            chunk = samples[i:i + self.buffer_size]
            self.audio_queue.put(chunk)

        # Add end-of-stream marker
        self.audio_queue.put(None)

        # Size the far-end reference ring buffer to this stream's sample rate
        # (a no-op after the first call at a given rate).
        _ref_buffer_init(sample_rate)

        # Create and start output stream
        try:
            self.stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=channels,
                callback=self._audio_callback,
                blocksize=self.buffer_size,
                dtype=np.float32
            )
            self.stream.start()
            _speaking_inc()   # TTS is now audibly playing → raise the "speaking" flag

            if blocking:
                try:
                    self.wait()
                finally:
                    _speaking_dec()
            else:
                # Non-blocking: clear the flag when THIS playback finishes. The
                # callback always sets playback_complete (end-of-stream, error, or
                # pause), so this never strands the flag.
                def _clear_speaking_when_done(ev=self.playback_complete):
                    ev.wait()
                    _speaking_dec()
                threading.Thread(target=_clear_speaking_when_done, daemon=True,
                                 name="convomode-speaking-clear").start()

        except Exception as e:
            self.playback_error = e
            logger.error(f"Error starting audio playback: {e}")
            raise

    def wait(self, timeout: Optional[float] = None):
        """Wait for playback to complete.

        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)

        Raises:
            Exception: If playback error occurred
        """
        # Wait for playback to complete
        if not self.playback_complete.wait(timeout=timeout):
            logger.warning("Playback wait timed out")

        # Stop and close stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        # Raise any error that occurred during playback
        if self.playback_error:
            raise self.playback_error

    def stop(self):
        """Stop playback immediately."""
        self.playback_complete.set()
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        # Clear queue
        if self.audio_queue:
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
