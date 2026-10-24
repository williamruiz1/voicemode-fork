"""AirPods stem press = Send while a listen is open (macOS 14+).

While voicemode records, the AirPods are in call mode: a single stem press is
"Mute & Unmute", which macOS delivers ONLY to an app that registered
``AVAudioApplication.setInputMuteStateChangeHandler`` and is itself capturing
input (Apple docs + WWDC23 10233). voicemode is that capturing process, so it
registers the handler for the duration of each listen and treats every mute
toggle as Send: it touches ``~/.voicemode/turn-end.signal`` (the same file the
VibeDispatcher widget's Send button writes) and returns True WITHOUT muting
capture.

Details:
  * The system calls the handler once when the input stream opens, with the
    app's current state (an "initial sync", not a press). A real press always
    CHANGES the state, so a call whose value equals the last known state is
    ignored -- no timing window needed.
  * Never fires while convomode is paused (``~/.voicemode/pause.flag``): Pause
    mutes the mic itself, and a phantom Send during a pause skips the turn (B25).
  * The handler runs on a system thread with no run loop requirement (verified
    with PortAudio input + a sleeping main thread), so registering from the
    recorder's executor thread is fine. It only touches a file -- never blocks.
  * Gated by VOICEMODE_AIRPODS_PRESS_SEND (default on). Fails open: without
    PyObjC's AVFAudio bindings (``pip install 'voice-mode[airpods]'``), on
    non-macOS, or on any error, the listen proceeds exactly as before.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from typing import Iterator, Optional

logger = logging.getLogger("voicemode")

ENV_VAR = "VOICEMODE_AIRPODS_PRESS_SEND"


def enabled() -> bool:
    value = os.getenv(ENV_VAR, "").strip().lower()
    return value in ("true", "1", "yes", "on") if value else True


def _turn_end_path() -> str:
    return os.path.expanduser("~/.voicemode/turn-end.signal")


def _pause_flag_path() -> str:
    return os.path.expanduser("~/.voicemode/pause.flag")


def _load_app():
    """The shared AVAudioApplication, or None when unavailable (fail-open)."""
    try:
        import AVFAudio  # pyobjc-framework-AVFoundation, macOS 14+
        cls = getattr(AVFAudio, "AVAudioApplication", None)
        if cls is None:
            return None
        return cls.sharedInstance()
    except Exception as e:  # ImportError, non-macOS, old macOS
        logger.debug(f"AirPods press-send unavailable: {e}")
        return None


class PressSender:
    """One listen's registration. ``handle`` is the AVAudioApplication callback."""

    def __init__(self, app, initial_muted: bool):
        self._app = app
        self._lock = threading.Lock()
        self._last = bool(initial_muted)
        self._closing = False
        self.presses = 0

    def handle(self, muted) -> bool:
        muted = bool(muted)
        with self._lock:
            changed = muted != self._last
            self._last = muted
            if self._closing or not changed:
                return True  # initial sync / our own reset: not a press
            self.presses += 1
        if os.path.exists(_pause_flag_path()):
            logger.info("AirPods press ignored: convomode is paused")
            return True
        try:
            path = _turn_end_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a"):
                os.utime(path, None)
            logger.info("✓ AirPods press -> Send (turn-end.signal)")
        except Exception as e:
            logger.warning(f"AirPods press: could not write turn-end.signal: {e}")
        return True  # never mute capture

    def close(self) -> None:
        with self._lock:
            self._closing = True
            was_muted = self._last
        try:
            if was_muted:
                # Leave the app unmuted for the next listen / other apps' indicators.
                self._app.setInputMuted_error_(False, None)
        except Exception as e:
            logger.debug(f"AirPods press: reset mute state failed (ignored): {e}")
        try:
            self._app.setInputMuteStateChangeHandler_error_(None, None)
        except Exception as e:
            logger.debug(f"AirPods press: unregister failed (ignored): {e}")


def register() -> Optional[PressSender]:
    """Register the press handler; None if disabled or unavailable."""
    if not enabled():
        return None
    app = _load_app()
    if app is None:
        return None
    try:
        initial = bool(app.isInputMuted())
    except Exception:
        initial = False
    sender = PressSender(app, initial)
    try:
        ok = app.setInputMuteStateChangeHandler_error_(sender.handle, None)
        if isinstance(ok, tuple):
            ok = ok[0]
        if not ok:
            logger.debug("AirPods press: handler registration returned False")
            return None
    except Exception as e:
        logger.debug(f"AirPods press: handler registration failed: {e}")
        return None
    return sender


@contextlib.contextmanager
def listening() -> Iterator[Optional[PressSender]]:
    """Hold the press handler for the duration of one listen. Never raises."""
    sender = None
    try:
        sender = register()
    except Exception as e:
        logger.debug(f"AirPods press: register crashed (ignored): {e}")
    try:
        yield sender
    finally:
        if sender is not None:
            try:
                sender.close()
            except Exception:
                pass
