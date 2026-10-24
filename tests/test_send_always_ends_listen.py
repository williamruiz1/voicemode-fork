"""B17: Send (turn-end.signal) must end a listen within ~0.5s in EVERY state.

Incident 2026-10-23 22:00 (held listen, AirPods 4): the input stream stopped
delivering callbacks and Send never ended the listen. The loop was alive
(polling an empty queue every 0.1s) but Send was gated on
``recording_duration >= MIN_RECORDING_DURATION`` and recording_duration only
advances when a chunk ARRIVES -- a stream that stalls before 0.5s of audio
defers Send forever (until the 600s hold ceiling).

Each test drives the real loop with a fake sounddevice and runs it on a
daemon thread with a hard join timeout, so a regression fails fast instead of
hanging the suite.
"""

import sys
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules.setdefault("webrtcvad", MagicMock())

import voice_mode.tools.converse as C
from voice_mode.tools.converse import record_audio_with_silence_detection
from voice_mode.config import SAMPLE_RATE, VAD_CHUNK_DURATION_MS


_REAL_RECORD_AUDIO = C.record_audio  # the sim fixture replaces C.record_audio
CHUNK_SAMPLES = int(SAMPLE_RATE * VAD_CHUNK_DURATION_MS / 1000)
SEND_DEADLINE_S = 1.0  # requirement is ~0.5s; slack for a loaded CI box


class _PortAudioError(Exception):
    pass


def _chunk(loud=True):
    if loud:
        return np.random.randint(-8000, 8000, size=(CHUNK_SAMPLES, 1), dtype=np.int16)
    return np.zeros((CHUNK_SAMPLES, 1), dtype=np.int16)


class _StallingStream:
    """Delivers `deliver` chunks, then goes silent forever with NO status and
    NO error -- the incident's stalled CoreAudio stream (no IOProc running)."""

    deliver = 0
    status_after = None  # str: deliver this status (device error) after `deliver` chunks
    opened = 0

    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]
        self._stop = threading.Event()
        _StallingStream.opened += 1

    def __enter__(self):
        def pump():
            for _ in range(_StallingStream.deliver):
                if self._stop.is_set():
                    return
                self.callback(_chunk(), CHUNK_SAMPLES, None, None)
                time.sleep(0.0005)
            if _StallingStream.status_after:
                self.callback(_chunk(), CHUNK_SAMPLES, None, _StallingStream.status_after)
        threading.Thread(target=pump, daemon=True).start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        return False


class _VadAlwaysSpeech:
    def __init__(self, *a):
        pass

    def is_speech(self, chunk_bytes, rate):
        return True


@pytest.fixture
def sim(monkeypatch, isolate_home_directory):
    home = isolate_home_directory / ".voicemode"
    home.mkdir(parents=True, exist_ok=True)
    fake_sd = MagicMock()
    fake_sd.InputStream = _StallingStream
    fake_sd.Stream = _StallingStream
    fake_sd.PortAudioError = _PortAudioError
    monkeypatch.setattr(C, "sd", fake_sd)
    monkeypatch.setattr(C, "_bluetooth_input_active", lambda: False)
    monkeypatch.setattr(C, "VAD_AVAILABLE", True)
    monkeypatch.setattr(C, "DISABLE_SILENCE_DETECTION", False)
    monkeypatch.setattr(C, "ENDPOINTING_ENABLED", False)
    monkeypatch.setattr(C, "VAD_ENERGY_THRESHOLD", 0)
    monkeypatch.setattr(C, "STEP_AWAY_ENV", False)
    monkeypatch.setattr(C, "STEP_AWAY_FLAG_PATH", str(home / "nonexistent-step-away"))
    monkeypatch.setattr(C, "APPEND_WINDOW_MS", 0)
    monkeypatch.setattr(C, "APPEND_WINDOW_FLAG_PATH", str(home / "nonexistent-append"))
    monkeypatch.setattr(C, "record_audio", MagicMock(side_effect=AssertionError("fixed-length record_audio used")))
    fake_webrtc = MagicMock()
    fake_webrtc.Vad = _VadAlwaysSpeech
    monkeypatch.setattr(C, "webrtcvad", fake_webrtc)
    # Bounded so a regression's leaked thread still ends on its own.
    monkeypatch.setenv("VOICEMODE_HOLD_MAX_SECONDS", "8")
    monkeypatch.delenv("VOICEMODE_STREAM_STALL_SECONDS", raising=False)
    _StallingStream.deliver = 0
    _StallingStream.status_after = None
    _StallingStream.opened = 0
    return home


def _listen_async(**kwargs):
    """Start a listen on a daemon thread. Returns (thread, box); box gets
    'result', 'stats', 'ended_at' (or 'error')."""
    box = {"stats": {}}

    def run():
        try:
            box["result"] = record_audio_with_silence_detection(listen_stats=box["stats"], **kwargs)
        except BaseException as e:  # surfaced by the test
            box["error"] = e
        box["ended_at"] = time.monotonic()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, box


def _send_and_wait(home, thread, box, after=0.4, timeout=SEND_DEADLINE_S):
    """Let the listen run `after` seconds, press Send, and require the listen
    to end within `timeout`. Returns seconds from Send to return."""
    time.sleep(after)
    assert thread.is_alive(), f"listen ended before Send: {box}"
    sent = time.monotonic()
    (home / "turn-end.signal").touch()
    thread.join(timeout)
    assert not thread.is_alive(), (
        f"Send did not end the listen within {timeout}s "
        f"(turn-end.signal still on disk: {(home / 'turn-end.signal').exists()})")
    assert "error" not in box, box.get("error")
    return box["ended_at"] - sent


class TestStalledStream:
    """The incident path (H2): stream stops delivering, no error."""

    def test_held_stalled_from_start_send_ends_listen(self, sim):
        (sim / "hold-mode").touch()
        _StallingStream.deliver = 0
        t, box = _listen_async(max_duration=30.0)
        _send_and_wait(sim, t, box)
        assert box["stats"]["end_reason"] == "send"
        assert not (sim / "turn-end.signal").exists()

    def test_held_stalled_below_floor_send_ends_listen(self, sim):
        # 5 chunks = 0.15s of audio < MIN_RECORDING_DURATION (0.5s): the
        # incident's exact gate. (Control: test_held_stalled_above_floor.)
        (sim / "hold-mode").touch()
        _StallingStream.deliver = 5
        t, box = _listen_async(max_duration=30.0)
        _send_and_wait(sim, t, box)
        assert box["stats"]["end_reason"] == "send"
        audio, speech = box["result"]
        assert len(audio) == 5 * CHUNK_SAMPLES  # captured audio is kept
        assert speech is True

    def test_held_stalled_above_floor_send_ends_listen(self, sim):
        # Control: >= 0.5s delivered before the stall -- Send always worked here.
        (sim / "hold-mode").touch()
        _StallingStream.deliver = 20
        t, box = _listen_async(max_duration=30.0)
        _send_and_wait(sim, t, box)
        assert box["stats"]["end_reason"] == "send"

    def test_not_held_stalled_send_ends_listen(self, sim):
        _StallingStream.deliver = 0
        t, box = _listen_async(max_duration=30.0)
        _send_and_wait(sim, t, box)
        assert box["stats"]["end_reason"] == "send"

    def test_stalled_stream_ends_itself_without_send(self, sim, monkeypatch):
        # A dead stream must not hold the mic (and the floor) until the 600s
        # hold ceiling: it ends with end_reason=stream_stalled.
        monkeypatch.setenv("VOICEMODE_STREAM_STALL_SECONDS", "0.6")
        (sim / "hold-mode").touch()
        _StallingStream.deliver = 10
        t, box = _listen_async(max_duration=30.0)
        t.join(3.0)
        assert not t.is_alive(), "stalled stream never ended the listen"
        assert box["stats"]["end_reason"] == "stream_stalled"
        assert len(box["result"][0]) == 10 * CHUNK_SAMPLES

    def test_not_held_stalled_stream_ends_itself(self, sim, monkeypatch):
        monkeypatch.setenv("VOICEMODE_STREAM_STALL_SECONDS", "0.6")
        _StallingStream.deliver = 0
        t, box = _listen_async(max_duration=30.0)
        t.join(3.0)
        assert not t.is_alive(), "non-held stalled stream never ended (no wall-clock backstop)"
        assert box["stats"]["end_reason"] == "stream_stalled"


class TestDeviceErrorRecovery:
    """H1: stream open keeps failing with a device error -> recovery loop."""

    def _failing_open(self, monkeypatch, counter):
        def _open(**kwargs):
            counter.append(time.monotonic())
            raise _PortAudioError("Error opening InputStream: Device unavailable [PaErrorCode -9985]")
        monkeypatch.setattr(C.sd, "InputStream", _open)

    def test_device_gone_recovery_is_capped(self, sim, monkeypatch):
        opens = []
        self._failing_open(monkeypatch, opens)
        (sim / "hold-mode").touch()
        t, box = _listen_async(max_duration=30.0)
        t.join(5.0)
        assert not t.is_alive(), f"recovery loop did not give up ({len(opens)} opens)"
        assert "error" not in box, box.get("error")
        assert len(opens) == C.DEVICE_RECOVERY_MAX_ATTEMPTS
        assert box["stats"]["end_reason"] == "device_error"
        audio, speech = box["result"]
        assert len(audio) == 0 and speech is False

    def test_send_during_recovery_ends_listen(self, sim, monkeypatch):
        opens = []
        self._failing_open(monkeypatch, opens)
        monkeypatch.setattr(C, "DEVICE_RECOVERY_MAX_ATTEMPTS", 1000)
        monkeypatch.setattr(C, "DEVICE_RECOVERY_DELAY_SECONDS", 0.5)
        (sim / "hold-mode").touch()
        t, box = _listen_async(max_duration=30.0)
        _send_and_wait(sim, t, box, after=0.3)
        assert box["stats"]["end_reason"] == "send"
        # Recovery must not have wiped the Send before honoring it.
        assert not (sim / "turn-end.signal").exists()

    def test_device_error_mid_held_listen_ends_with_audio(self, sim):
        # Stream delivers audio, then reports a device error mid held listen:
        # the listen ends promptly on its own (no Send needed) and keeps the
        # audio captured before the error.
        _StallingStream.deliver = 10
        _StallingStream.status_after = "device unavailable"
        (sim / "hold-mode").touch()
        t, box = _listen_async(max_duration=30.0)
        t.join(SEND_DEADLINE_S)
        assert not t.is_alive(), "device error left the held listen running"
        assert box["stats"]["end_reason"] == "device_error"
        assert len(box["result"][0]) == 10 * CHUNK_SAMPLES

    def test_close_error_after_send_does_not_reopen(self, sim, monkeypatch):
        # Send ends the listen, then closing the stream raises a device error:
        # the listen is delivered, not restarted by device recovery.
        _StallingStream.deliver = 10
        opens = []

        class _DiesOnClose(_StallingStream):
            def __exit__(self, *a):
                super().__exit__(*a)
                raise _PortAudioError("Unanticipated host error [PaErrorCode -9999]")

        def _open(**kwargs):
            opens.append(1)
            return _DiesOnClose(**kwargs)

        monkeypatch.setattr(C.sd, "InputStream", _open)
        (sim / "hold-mode").touch()
        t, box = _listen_async(max_duration=30.0)
        _send_and_wait(sim, t, box)
        assert len(opens) == 1
        assert box["stats"]["end_reason"] == "send"
        audio, speech = box["result"]
        assert len(audio) == 10 * CHUNK_SAMPLES and speech is True

    def test_reopen_after_device_error_keeps_audio_and_send(self, sim, monkeypatch):
        # First open fails, second succeeds and stalls; Send pressed during the
        # retry is honored (not deleted by the retry's stale-signal cleanup).
        calls = []

        def _open(**kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise _PortAudioError("Device unavailable")
            return _StallingStream(**kwargs)

        monkeypatch.setattr(C.sd, "InputStream", _open)
        (sim / "hold-mode").touch()
        t, box = _listen_async(max_duration=30.0)
        _send_and_wait(sim, t, box, after=0.8)
        assert len(calls) == 2
        assert box["stats"]["end_reason"] == "send"


class TestBlockingCallbacks:
    """H3: a callback invoked from the loop blocks (step-away check-in TTS)."""

    def test_slow_checkin_does_not_block_send(self, sim, monkeypatch):
        monkeypatch.setattr(C, "STEP_AWAY_CHECKIN_SECONDS", 0.0)
        monkeypatch.setattr(C, "STEP_AWAY_GRACE_SECONDS", 30.0)

        class _SilentVad:
            def __init__(self, *a):
                pass

            def is_speech(self, *a):
                return False

        monkeypatch.setattr(C.webrtcvad, "Vad", _SilentVad)
        # A live stream (real-time-ish), so only the check-in can block Send.
        monkeypatch.setattr(_StallingStream, "deliver", 100000)
        checkin_started = threading.Event()

        def slow_checkin():
            checkin_started.set()
            time.sleep(5.0)  # "Still there?" TTS waiting up to 20s on the loop

        t, box = _listen_async(
            max_duration=30.0,
            pause_check=lambda: True,
            checkin_callback=slow_checkin,
            step_away_enabled_override=True,
        )
        assert checkin_started.wait(2.0)
        _send_and_wait(sim, t, box, after=0.2)
        assert box["stats"]["end_reason"] == "send"


class TestHeldNoVadPath:
    """_record_held_until_send (disable_silence_detection + held) had the same
    audio-time floor on Send."""

    def test_no_vad_held_stalled_send_ends_listen(self, sim):
        (sim / "hold-mode").touch()
        _StallingStream.deliver = 3
        t, box = _listen_async(max_duration=30.0, disable_silence_detection=True)
        _send_and_wait(sim, t, box)
        assert box["stats"]["end_reason"] == "send"
        assert len(box["result"][0]) == 3 * CHUNK_SAMPLES

    def test_no_vad_held_stalled_ends_itself(self, sim, monkeypatch):
        monkeypatch.setenv("VOICEMODE_STREAM_STALL_SECONDS", "0.6")
        (sim / "hold-mode").touch()
        _StallingStream.deliver = 0
        t, box = _listen_async(max_duration=30.0, disable_silence_detection=True)
        t.join(3.0)
        assert not t.is_alive()
        assert box["stats"]["end_reason"] == "stream_stalled"


class TestFixedLengthFallback:
    """record_audio(): the fallback a failed VAD stream drops into."""

    def test_record_audio_honors_send_when_asked(self, sim, monkeypatch):
        monkeypatch.setattr(C, "record_audio", _REAL_RECORD_AUDIO)
        fake_sd = C.sd
        fake_sd.rec = MagicMock(side_effect=lambda n, **k: np.zeros((n, 1), dtype=np.int16))
        fake_sd.wait = MagicMock(side_effect=AssertionError("blocking sd.wait() used"))
        box = {}

        def run():
            box["audio"] = C.record_audio(30.0, honor_send=True)
            box["ended_at"] = time.monotonic()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        time.sleep(0.3)
        sent = time.monotonic()
        (sim / "turn-end.signal").touch()
        t.join(SEND_DEADLINE_S)
        assert not t.is_alive(), "record_audio ignored Send"
        assert box["ended_at"] - sent < SEND_DEADLINE_S
        fake_sd.stop.assert_called()
        assert 0 < len(box["audio"]) < 30 * SAMPLE_RATE  # truncated to what was recorded

    def test_record_audio_device_retry_is_capped(self, sim, monkeypatch):
        monkeypatch.setattr(C, "record_audio", _REAL_RECORD_AUDIO)
        monkeypatch.setattr(C, "DEVICE_RECOVERY_DELAY_SECONDS", 0.01)
        calls = []

        def rec(n, **k):
            calls.append(1)
            raise _PortAudioError("Device unavailable")

        C.sd.rec = rec
        audio = C.record_audio(1.0)
        assert len(audio) == 0
        assert len(calls) == C.DEVICE_RECOVERY_MAX_ATTEMPTS
