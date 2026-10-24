"""AirPods stem press = Send while a listen is open (macOS 14+).

While voicemode records, the AirPods are in call mode: a single stem press is
"Mute & Unmute", which macOS delivers ONLY to an app that is capturing input,
registered ``AVAudioApplication.setInputMuteStateChangeHandler``, AND opted in
by observing ``AVAudioApplicationInputMuteStateChangeNotification`` (Apple docs
+ WWDC23 10233). voicemode is that capturing process, so it opts in once per
process, registers the handler for the duration of each listen and treats every
mute toggle as Send: it writes ``~/.voicemode/turn-end.signal`` with the source
tag "airpods" (the same file the VibeDispatcher widget's Send button writes)
and returns True WITHOUT muting capture.

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
  * Opt-in (B33): without the notification observer audiomxd rejects the press
    ("audio app is not allowed to input mute (isOptedIn: 0, hasHandler: 1)")
    and the AirPods play the reject chime. Adding the observer sets the app
    property ``PrefersBluetoothAccessoryMutingMacOS``; it stays set for the
    process's life (removing the observer does not clear it), so the observer
    is added once and kept.
  * Every registration, handler call and unregister is appended to
    ``~/.voicemode/logs/airpods.log`` (size-capped) for diagnosis.
  * Fallback (Darwin notification): audioaccessoryd also posts
    ``com.apple.private.restrict-post.audioaccessoryd.MuteState`` (older macOS:
    ``com.apple.audioaccessoryd.MuteState``) on every stem press in call mode.
    It needs no bundle and no mic permission. While a listen is open a small
    thread watches both names via libSystem ``notify_register_check`` /
    ``notify_check`` (every 50 ms) and treats a post as Send, source
    "airpods". A press seen by both paths within 300 ms counts once. Gated by
    VOICEMODE_AIRPODS_DARWIN_NOTIFY (default on); fail-open.
  * Gated by VOICEMODE_AIRPODS_PRESS_SEND (default on). Fails open: without
    PyObjC's AVFAudio bindings (``pip install 'voice-mode[airpods]'``), on
    non-macOS, or on any error, the listen proceeds exactly as before.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from typing import Iterator, Optional

logger = logging.getLogger("voicemode")

ENV_VAR = "VOICEMODE_AIRPODS_PRESS_SEND"
DARWIN_ENV_VAR = "VOICEMODE_AIRPODS_DARWIN_NOTIFY"
DARWIN_NOTIFY_NAMES = (
    "com.apple.private.restrict-post.audioaccessoryd.MuteState",
    "com.apple.audioaccessoryd.MuteState",
)
DARWIN_POLL_SECONDS = 0.05
DEDUPE_SECONDS = 0.3
SEND_SOURCE = "airpods"
DEBUG_LOG_MAX_BYTES = 512 * 1024  # airpods.log rotates to airpods.log.1 past this

_debug_lock = threading.Lock()
_opt_in_lock = threading.Lock()
_opt_in_observer = None  # kept for the process's life once opted in
_send_lock = threading.Lock()
_last_send = (0.0, "")  # (monotonic time, origin) of the last emitted Send


def enabled() -> bool:
    value = os.getenv(ENV_VAR, "").strip().lower()
    return value in ("true", "1", "yes", "on") if value else True


def darwin_enabled() -> bool:
    value = os.getenv(DARWIN_ENV_VAR, "").strip().lower()
    return value in ("true", "1", "yes", "on") if value else True


def _turn_end_path() -> str:
    return os.path.expanduser("~/.voicemode/turn-end.signal")


def _pause_flag_path() -> str:
    return os.path.expanduser("~/.voicemode/pause.flag")


def _debug_log_path() -> str:
    return os.path.expanduser("~/.voicemode/logs/airpods.log")


def debug_log(message: str) -> None:
    """Append one line to ~/.voicemode/logs/airpods.log. Never raises."""
    try:
        t = threading.current_thread()
        now = time.time()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)) + f".{int(now * 1000) % 1000:03d}"
        line = f"{stamp} pid={os.getpid()} thread={t.name}({t.ident}) main={t is threading.main_thread()} {message}\n"
        path = _debug_log_path()
        with _debug_lock:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                if os.path.getsize(path) > DEBUG_LOG_MAX_BYTES:
                    os.replace(path, path + ".1")
            except OSError:
                pass
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass


def _write_send_signal() -> None:
    """turn-end.signal with this module's source tag, written atomically so the
    listen never reads a half-written file."""
    path = _turn_end_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(SEND_SOURCE + "\n")
    os.replace(tmp, path)


def _emit_send(origin: str, detail: str) -> bool:
    """Treat one press as Send: honour pause, collapse the same press seen by
    two paths (different origins within DEDUPE_SECONDS), write turn-end.signal.
    Returns True if a signal was written. Never raises."""
    global _last_send
    if os.path.exists(_pause_flag_path()):
        debug_log(f"{origin} {detail} -> ignored (paused)")
        logger.info("AirPods press ignored: convomode is paused")
        return False
    now = time.monotonic()
    with _send_lock:
        last_t, last_origin = _last_send
        if last_origin and last_origin != origin and now - last_t < DEDUPE_SECONDS:
            debug_log(f"{origin} {detail} -> ignored (duplicate of {last_origin})")
            return False
        _last_send = (now, origin)
    try:
        _write_send_signal()
        debug_log(f"{origin} {detail} -> SEND (turn-end.signal)")
        logger.info("✓ AirPods press -> Send (turn-end.signal)")
        return True
    except Exception as e:
        debug_log(f"{origin} {detail} -> write failed: {e!r}")
        logger.warning(f"AirPods press: could not write turn-end.signal: {e}")
        return False


def _ensure_opted_in() -> bool:
    """Opt this process in to the AirPods mute gesture by observing
    AVAudioApplicationInputMuteStateChangeNotification (once; kept). Without
    it audiomxd rejects every press (isOptedIn: 0). Never raises."""
    global _opt_in_observer
    with _opt_in_lock:
        if _opt_in_observer is not None:
            return True
        try:
            import AVFAudio
            import Foundation

            name = AVFAudio.AVAudioApplicationInputMuteStateChangeNotification

            def _on_notification(note):
                try:
                    info = note.userInfo() or {}
                    debug_log(f"notification mute_state={info.get('AVAudioApplicationMuteStateKey')!r}")
                except Exception:
                    pass

            _opt_in_observer = Foundation.NSNotificationCenter.defaultCenter().addObserverForName_object_queue_usingBlock_(
                name, None, None, _on_notification)
            debug_log("opt-in: observing AVAudioApplicationInputMuteStateChangeNotification")
            return True
        except Exception as e:
            debug_log(f"opt-in failed: {e!r}")
            logger.debug(f"AirPods press: opt-in failed: {e}")
            return False


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
            last = self._last
            changed = muted != last
            self._last = muted
            closing = self._closing
            if not closing and changed:
                self.presses += 1
        if closing or not changed:
            debug_log(f"handler muted={muted} last={last} closing={closing} -> ignored "
                      f"({'reset' if closing else 'initial sync / no change'})")
            return True  # initial sync / our own reset: not a press
        _emit_send("avaudioapplication", f"handler muted={muted} last={last}")
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
            debug_log(f"unregistered (presses={self.presses}, reset_mute={was_muted})")
        except Exception as e:
            debug_log(f"unregister failed: {e!r}")
            logger.debug(f"AirPods press: unregister failed (ignored): {e}")


def register() -> Optional[PressSender]:
    """Register the press handler; None if disabled or unavailable."""
    if not enabled():
        debug_log(f"register skipped: {ENV_VAR} is off")
        return None
    app = _load_app()
    if app is None:
        debug_log("register skipped: AVAudioApplication unavailable")
        return None
    opted_in = _ensure_opted_in()
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
            debug_log(f"register failed: handler registration returned False (opted_in={opted_in})")
            logger.debug("AirPods press: handler registration returned False")
            return None
    except Exception as e:
        debug_log(f"register failed: {e!r}")
        logger.debug(f"AirPods press: handler registration failed: {e}")
        return None
    debug_log(f"registered ok=True initial_muted={initial} opted_in={opted_in}")
    return sender


class DarwinNotifyWatcher:
    """Watches the audioaccessoryd MuteState Darwin notifications for one
    listen. ``notify_check`` reports a post since the previous check, so a
    polling thread needs no run loop or ctypes block. Names that fail to
    register are skipped; if none register the watcher is not started."""

    def __init__(self, lib, names=DARWIN_NOTIFY_NAMES, poll=DARWIN_POLL_SECONDS):
        self._lib = lib
        self._poll = poll
        self._stop = threading.Event()
        self._tokens = []  # (name, token)
        self._thread = None
        self.posts = 0
        import ctypes
        self._ctypes = ctypes
        for name in names:
            token = ctypes.c_int(-1)
            status = lib.notify_register_check(name.encode(), ctypes.byref(token))
            if status != 0:
                debug_log(f"darwin-notify: register {name} failed status={status}")
                continue
            self._prime(token.value)  # first check always reports True; swallow it
            self._tokens.append((name, token.value))
            state = ctypes.c_uint64(0)
            lib.notify_get_state(token.value, ctypes.byref(state))
            debug_log(f"darwin-notify: registered {name} state={state.value}")

    def _check(self, token) -> bool:
        flag = self._ctypes.c_int(0)
        if self._lib.notify_check(token, self._ctypes.byref(flag)) != 0:
            return False
        return bool(flag.value)

    def _prime(self, token) -> None:
        self._check(token)

    @property
    def active(self) -> bool:
        return bool(self._tokens)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="airpods-darwin-notify", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._poll):
            for name, token in self._tokens:
                try:
                    if self._check(token):
                        self.posts += 1
                        _emit_send("darwin-notify", f"post {name}")
                except Exception as e:
                    debug_log(f"darwin-notify: poll error {e!r}")

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        for _, token in self._tokens:
            try:
                self._lib.notify_cancel(token)
            except Exception:
                pass
        debug_log(f"darwin-notify: unsubscribed (posts={self.posts})")
        self._tokens = []


def _load_libsystem():
    try:
        import ctypes
        import sys
        if sys.platform != "darwin":
            return None
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        lib.notify_register_check.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        lib.notify_check.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        lib.notify_get_state.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
        lib.notify_cancel.argtypes = [ctypes.c_int]
        return lib
    except Exception as e:
        debug_log(f"darwin-notify: libSystem unavailable: {e!r}")
        return None


def register_darwin_notify() -> Optional[DarwinNotifyWatcher]:
    """Start the Darwin-notification watcher; None if disabled/unavailable."""
    if not darwin_enabled():
        debug_log(f"darwin-notify skipped: {DARWIN_ENV_VAR} is off")
        return None
    lib = _load_libsystem()
    if lib is None:
        return None
    watcher = DarwinNotifyWatcher(lib)
    if not watcher.active:
        return None
    watcher.start()
    return watcher


@contextlib.contextmanager
def listening() -> Iterator[Optional[PressSender]]:
    """Hold the press handler for the duration of one listen. Never raises."""
    sender = None
    try:
        sender = register()
    except Exception as e:
        logger.debug(f"AirPods press: register crashed (ignored): {e}")
    watcher = None
    try:
        watcher = register_darwin_notify()
    except Exception as e:
        logger.debug(f"AirPods press: darwin-notify crashed (ignored): {e}")
    try:
        yield sender
    finally:
        if watcher is not None:
            try:
                watcher.close()
            except Exception:
                pass
        if sender is not None:
            try:
                sender.close()
            except Exception:
                pass
