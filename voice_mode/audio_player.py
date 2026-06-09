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
        if _pause_event.is_set():
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
