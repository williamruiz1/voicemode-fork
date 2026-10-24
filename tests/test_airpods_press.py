"""AirPods stem press = Send (voice_mode/airpods_press.py).

AVFAudio is faked: a FakeApp records the registered handler, and a test
"presses" by invoking it the way macOS does (with the new mute state).
"""

import contextlib
import os
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault("webrtcvad", MagicMock())

from voice_mode import airpods_press as ap

_real_ensure_opted_in = ap._ensure_opted_in


class FakeApp:
    def __init__(self, muted=False, register_ok=True, raise_on_register=False):
        self.muted = muted
        self.handler = None
        self.register_ok = register_ok
        self.raise_on_register = raise_on_register
        self.register_calls = []
        self.set_muted_calls = []

    def isInputMuted(self):
        return self.muted

    def setInputMuteStateChangeHandler_error_(self, handler, err):
        if self.raise_on_register:
            raise RuntimeError("boom")
        self.register_calls.append(handler)
        self.handler = handler
        return self.register_ok

    def setInputMuted_error_(self, muted, err):
        self.set_muted_calls.append(muted)
        self.muted = muted
        if self.handler:
            self.handler(muted)  # macOS routes setInputMuted through the handler
        return True

    # helper: what the system does on a stem press
    def press(self):
        self.muted = not self.muted
        return self.handler(self.muted)


@pytest.fixture(autouse=True)
def isolate_darwin_notify(monkeypatch):
    """Existing tests exercise the AVAudioApplication path only; the real
    Darwin watcher is off unless a test turns it on. Reset the dedupe state."""
    monkeypatch.setenv(ap.DARWIN_ENV_VAR, "0")
    monkeypatch.setattr(ap, "_last_send", (0.0, ""))


@pytest.fixture(autouse=True)
def opt_in_calls(monkeypatch):
    """Never touch the real NSNotificationCenter; record opt-in attempts."""
    calls = []
    monkeypatch.setattr(ap, "_ensure_opted_in", lambda: calls.append(1) or True)
    return calls


@pytest.fixture
def home(monkeypatch):
    # conftest's isolate_home_directory already redirects expanduser("~").
    from pathlib import Path
    monkeypatch.delenv(ap.ENV_VAR, raising=False)
    h = Path(os.path.expanduser("~"))
    (h / ".voicemode").mkdir(parents=True, exist_ok=True)
    return h


@pytest.fixture
def fake_app(monkeypatch):
    app = FakeApp()
    monkeypatch.setattr(ap, "_load_app", lambda: app)
    return app


def _signal(home):
    return home / ".voicemode" / "turn-end.signal"


def test_press_during_listen_touches_turn_end_and_does_not_mute(home, fake_app):
    with ap.listening() as sender:
        assert sender is not None
        assert fake_app.press() is True  # returns success, capture untouched
        assert _signal(home).exists()
        assert sender.presses == 1


def test_initial_sync_is_not_a_press(home, fake_app):
    with ap.listening() as sender:
        fake_app.handler(False)  # stream open: system reports current (unmuted) state
        assert not _signal(home).exists()
        assert sender.presses == 0


def test_every_toggle_is_a_press(home, fake_app):
    with ap.listening() as sender:
        fake_app.press()
        _signal(home).unlink()  # listen consumed the Send
        fake_app.press()
        assert _signal(home).exists()
        assert sender.presses == 2


def test_press_ignored_while_paused(home, fake_app):
    (home / ".voicemode" / "pause.flag").touch()
    with ap.listening():
        fake_app.press()
        assert not _signal(home).exists()


def test_unregisters_and_resets_mute_after_listen(home, fake_app):
    with ap.listening() as sender:
        fake_app.press()  # app state now "muted"
    assert fake_app.register_calls[-1] is None  # handler removed
    assert fake_app.set_muted_calls == [False]
    assert fake_app.muted is False
    # the reset itself (routed through the handler while closing) is not a press
    assert sender.presses == 1


def test_no_reset_when_left_unmuted(home, fake_app):
    with ap.listening():
        fake_app.press()
        fake_app.press()
    assert fake_app.set_muted_calls == []
    assert fake_app.register_calls[-1] is None


def test_disabled_by_env(home, fake_app, monkeypatch):
    monkeypatch.setenv(ap.ENV_VAR, "false")
    with ap.listening() as sender:
        assert sender is None
    assert fake_app.register_calls == []


def test_default_on(monkeypatch):
    monkeypatch.delenv(ap.ENV_VAR, raising=False)
    assert ap.enabled() is True


def test_fail_open_without_avfaudio(home, monkeypatch):
    monkeypatch.setitem(sys.modules, "AVFAudio", None)  # import raises ImportError
    assert ap._load_app() is None
    with ap.listening() as sender:
        assert sender is None


def test_fail_open_when_registration_raises(home, monkeypatch):
    app = FakeApp(raise_on_register=True)
    monkeypatch.setattr(ap, "_load_app", lambda: app)
    with ap.listening() as sender:
        assert sender is None


def test_fail_open_when_registration_returns_false(home, monkeypatch):
    app = FakeApp(register_ok=False)
    monkeypatch.setattr(ap, "_load_app", lambda: app)
    with ap.listening() as sender:
        assert sender is None


def test_load_app_uses_shared_instance(monkeypatch):
    shared = object()
    mod = types.SimpleNamespace(AVAudioApplication=types.SimpleNamespace(sharedInstance=lambda: shared))
    monkeypatch.setitem(sys.modules, "AVFAudio", mod)
    assert ap._load_app() is shared


def test_every_listen_holds_the_handler(home, monkeypatch):
    """record_audio_with_silence_detection wraps the recorder in listening()."""
    import voice_mode.tools.converse as C

    app = FakeApp()
    monkeypatch.setattr(ap, "_load_app", lambda: app)
    seen = {}

    def fake_record(*args, listen_stats=None, **kwargs):
        seen["registered"] = app.handler is not None
        app.press()  # William presses mid-listen
        return (np.zeros(10, dtype=np.int16), True)

    monkeypatch.setattr(C, "_record_audio_with_silence_detection", fake_record)
    C.record_audio_with_silence_detection(5.0)
    assert seen["registered"] is True
    assert _signal(home).exists()
    assert app.register_calls[-1] is None  # released when the listen ended


def test_listen_still_runs_when_unavailable(home, monkeypatch):
    import voice_mode.tools.converse as C

    monkeypatch.setattr(ap, "_load_app", lambda: None)
    monkeypatch.setattr(C, "_record_audio_with_silence_detection",
                        lambda *a, listen_stats=None, **k: (np.zeros(3, dtype=np.int16), False))
    audio, speech = C.record_audio_with_silence_detection(5.0)
    assert len(audio) == 3 and speech is False


# ---- B32 source tag / B33 opt-in + debug log ----

def _debug_log(home):
    return home / ".voicemode" / "logs" / "airpods.log"


def test_press_writes_airpods_source_tag(home, fake_app):
    with ap.listening():
        fake_app.press()
    assert _signal(home).read_text() == "airpods\n"
    assert not list((home / ".voicemode").glob("*.tmp"))


def test_press_source_is_read_back_by_converse(home, fake_app):
    import voice_mode.tools.converse as C

    with ap.listening():
        fake_app.press()
    assert C._consume_send() == "airpods"
    assert C._take_send_source() == "airpods"


def test_register_opts_in_before_handler(home, fake_app, opt_in_calls):
    with ap.listening():
        assert opt_in_calls == [1]


def test_debug_log_records_register_calls_and_unregister(home, fake_app):
    with ap.listening():
        fake_app.handler(False)  # initial sync
        fake_app.press()
    text = _debug_log(home).read_text()
    assert "registered ok=True initial_muted=False opted_in=True" in text
    assert "handler muted=False last=False closing=False -> ignored (initial sync / no change)" in text
    assert "handler muted=True last=False -> SEND (turn-end.signal)" in text
    assert "-> ignored (reset)" in text  # close() resets mute via the handler
    assert "unregistered (presses=1, reset_mute=True)" in text
    assert "thread=" in text and "pid=" in text


def test_debug_log_records_paused_and_skips(home, fake_app, monkeypatch):
    (home / ".voicemode" / "pause.flag").touch()
    with ap.listening():
        fake_app.press()
    monkeypatch.setenv(ap.ENV_VAR, "off")
    with ap.listening():
        pass
    text = _debug_log(home).read_text()
    assert "-> ignored (paused)" in text
    assert "register skipped: VOICEMODE_AIRPODS_PRESS_SEND is off" in text


def test_debug_log_is_size_capped(home, monkeypatch):
    monkeypatch.setattr(ap, "DEBUG_LOG_MAX_BYTES", 200)
    for i in range(20):
        ap.debug_log(f"line {i:02d} " + "x" * 40)
    log = _debug_log(home)
    assert log.stat().st_size <= 200 + 200
    assert (log.parent / "airpods.log.1").exists()
    assert "line 19" in log.read_text()


def test_debug_log_never_raises(monkeypatch):
    monkeypatch.setattr(ap, "_debug_log_path", lambda: "/nonexistent-root-dir/x/airpods.log")
    ap.debug_log("hello")  # no exception


def test_ensure_opted_in_adds_observer_once(home, monkeypatch):
    monkeypatch.setattr(ap, "_opt_in_observer", None)
    added = []

    class FakeCenter:
        def addObserverForName_object_queue_usingBlock_(self, name, obj, queue, block):
            added.append(name)
            block(types.SimpleNamespace(userInfo=lambda: {"AVAudioApplicationMuteStateKey": 1}))
            return "observer"

    center = FakeCenter()
    monkeypatch.setitem(sys.modules, "AVFAudio", types.SimpleNamespace(
        AVAudioApplicationInputMuteStateChangeNotification="AVAudioApplicationInputMuteStateChangeNotification"))
    monkeypatch.setitem(sys.modules, "Foundation", types.SimpleNamespace(
        NSNotificationCenter=types.SimpleNamespace(defaultCenter=lambda: center)))
    assert _real_ensure_opted_in() is True
    assert _real_ensure_opted_in() is True
    assert added == ["AVAudioApplicationInputMuteStateChangeNotification"]
    text = _debug_log(home).read_text()
    assert "opt-in: observing" in text and "notification mute_state=1" in text


def test_opt_in_failure_is_fail_open(home, fake_app, monkeypatch):
    monkeypatch.setattr(ap, "_ensure_opted_in", lambda: False)
    with ap.listening() as sender:
        assert sender is not None  # handler still registered
        fake_app.press()
    assert _signal(home).exists()
    assert "opted_in=False" in _debug_log(home).read_text()


# ---- Darwin notification fallback ------------------------------------------

class FakeLib:
    """Stands in for libSystem: notify_check reports the posts queued per token."""

    def __init__(self, register_status=0):
        self.register_status = register_status
        self.pending = {}  # token -> count of unseen posts
        self.names = {}
        self.cancelled = []
        self._next = 10
        self.first_check = set()

    def notify_register_check(self, name, tok_ref):
        if self.register_status:
            return self.register_status
        tok = self._next
        self._next += 1
        tok_ref._obj.value = tok
        self.names[name.decode()] = tok
        self.pending[tok] = 0
        return 0

    def notify_check(self, tok, flag_ref):
        if tok not in self.first_check:  # real notify_check reports True once at first
            self.first_check.add(tok)
            flag_ref._obj.value = 1
            return 0
        flag_ref._obj.value = 1 if self.pending[tok] else 0
        self.pending[tok] = 0
        return 0

    def notify_get_state(self, tok, st_ref):
        st_ref._obj.value = 1
        return 0

    def notify_cancel(self, tok):
        self.cancelled.append(tok)
        return 0

    def post(self, name):
        self.pending[self.names[name]] += 1


def _wait(cond, timeout=2.0):
    import time
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


@pytest.fixture
def fake_lib(home, monkeypatch):
    lib = FakeLib()
    monkeypatch.setenv(ap.DARWIN_ENV_VAR, "1")
    monkeypatch.setattr(ap, "_load_libsystem", lambda: lib)
    monkeypatch.setattr(ap, "_load_app", lambda: None)  # darwin path alone
    return lib


def test_darwin_post_writes_airpods_source(home, fake_lib):
    with ap.listening():
        assert not _signal(home).exists()  # priming check is not a press
        fake_lib.post(ap.DARWIN_NOTIFY_NAMES[0])
        assert _wait(lambda: _signal(home).exists())
    assert _signal(home).read_text().strip() == "airpods"


def test_darwin_both_names_are_watched(home, fake_lib):
    with ap.listening():
        assert set(fake_lib.names) == set(ap.DARWIN_NOTIFY_NAMES)
        fake_lib.post(ap.DARWIN_NOTIFY_NAMES[1])
        assert _wait(lambda: _signal(home).exists())


def test_darwin_ignored_while_paused(home, fake_lib):
    (home / ".voicemode" / "pause.flag").touch()
    with ap.listening():
        fake_lib.post(ap.DARWIN_NOTIFY_NAMES[0])
        import time; time.sleep(0.3)
        assert not _signal(home).exists()


def test_darwin_unsubscribes_when_listen_ends(home, fake_lib):
    with ap.listening():
        pass
    assert sorted(fake_lib.cancelled) == sorted(fake_lib.names.values())


def test_darwin_post_after_listen_is_ignored(home, fake_lib):
    with ap.listening():
        pass
    fake_lib.post(ap.DARWIN_NOTIFY_NAMES[0])
    import time; time.sleep(0.2)
    assert not _signal(home).exists()


def test_darwin_disabled_by_env(home, fake_lib, monkeypatch):
    monkeypatch.setenv(ap.DARWIN_ENV_VAR, "0")
    with ap.listening():
        assert fake_lib.names == {}


def test_darwin_default_on(monkeypatch):
    monkeypatch.delenv(ap.DARWIN_ENV_VAR, raising=False)
    assert ap.darwin_enabled() is True


def test_darwin_fail_open_when_register_fails(home, monkeypatch):
    monkeypatch.setenv(ap.DARWIN_ENV_VAR, "1")
    monkeypatch.setattr(ap, "_load_libsystem", lambda: FakeLib(register_status=1))
    monkeypatch.setattr(ap, "_load_app", lambda: None)
    with ap.listening() as sender:
        assert sender is None  # listen proceeds


def test_darwin_fail_open_without_libsystem(home, monkeypatch):
    monkeypatch.setenv(ap.DARWIN_ENV_VAR, "1")
    monkeypatch.setattr(ap, "_load_libsystem", lambda: None)
    monkeypatch.setattr(ap, "_load_app", lambda: None)
    with ap.listening():
        pass


def test_press_seen_by_both_paths_counts_once(home, fake_lib, fake_app, monkeypatch):
    monkeypatch.setattr(ap, "_load_app", lambda: fake_app)
    with ap.listening():
        fake_app.press()
        assert _signal(home).exists()
        _signal(home).unlink()  # listen consumed the Send
        fake_lib.post(ap.DARWIN_NOTIFY_NAMES[0])
        import time; time.sleep(0.3)
        assert not _signal(home).exists()  # duplicate collapsed


def test_darwin_post_is_logged_with_source(home, fake_lib):
    with ap.listening():
        fake_lib.post(ap.DARWIN_NOTIFY_NAMES[0])
        assert _wait(lambda: _signal(home).exists())
    log = (home / ".voicemode" / "logs" / "airpods.log").read_text()
    assert "darwin-notify" in log and "SEND" in log


def test_live_libsystem_subscription_smoke():
    """Real libSystem: registration succeeds and state is readable (no press)."""
    import sys
    if sys.platform != "darwin":
        pytest.skip("macOS only")
    lib = ap._load_libsystem()
    w = ap.DarwinNotifyWatcher(lib)
    try:
        assert w.active
    finally:
        w.close()
