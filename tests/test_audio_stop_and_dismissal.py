"""A stuck audio stop can't freeze the service, a dismissal stops a call already running,
and each leaves a line in the activity log."""

import asyncio
import json
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from voice_mode import activity, audio_player, audio_stop, client_identity, pause_resume
from voice_mode.conch import Conch
from voice_mode.daemon import LoopWatchdog


class StuckStream:
    """stop() blocks like the CoreAudio deadlock did, until the test releases it."""

    def __init__(self):
        self.release = threading.Event()
        self.closed = False

    def stop(self):
        self.release.wait(30)

    abort = stop

    def close(self):
        self.closed = True


@pytest.fixture
def activity_log(tmp_path, monkeypatch):
    path = tmp_path / "activity.jsonl"
    monkeypatch.setenv("VOICEMODE_ACTIVITY_LOG", str(path))

    def events():
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return events


@pytest.fixture(autouse=True)
def no_abandoned_streams():
    audio_stop._abandoned.clear()
    yield
    audio_stop._abandoned.clear()


@pytest.fixture
def clean_conch():
    if Conch.LOCK_FILE.exists():
        Conch.LOCK_FILE.unlink()
    yield
    if Conch.LOCK_FILE.exists():
        Conch.LOCK_FILE.unlink()


class TestStuckStop:
    @pytest.mark.asyncio
    async def test_event_loop_keeps_running_while_a_stop_is_stuck(self, activity_log):
        stream = StuckStream()
        ticks = 0

        async def tick():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.01)
        ticker = asyncio.ensure_future(tick())
        finished = await audio_stop.stop_stream_async(stream, limit=0.3)
        ticker.cancel()
        stream.release.set()

        assert finished is False
        assert ticks > 10, "the event loop stopped running during the stop"
        assert audio_stop.stuck() is True
        assert [e["event"] for e in activity_log()] == ["audio_stop_stuck"]

    def test_abandoned_stream_is_never_touched_again(self, activity_log):
        stream = StuckStream()
        assert audio_stop.stop_stream(stream, limit=0.1) is False
        assert audio_stop.stop_stream(stream, stop=False) is False
        stream.release.set()
        time.sleep(0.05)
        assert audio_stop.stuck() is True

    def test_healthy_stream_stops_and_closes(self):
        stream = MagicMock()
        assert audio_stop.stop_stream(stream) is True
        stream.stop.assert_called_once()
        stream.close.assert_called_once()
        assert audio_stop.stuck() is False

    def test_abort_failure_falls_back_to_stop(self):
        stream = MagicMock()
        stream.abort.side_effect = RuntimeError("no")
        assert audio_stop.stop_stream(stream, abort=True) is True
        stream.stop.assert_called_once()


class TestPlayerWaitsForCallbackStop:
    def test_no_stop_while_the_callback_is_still_stopping_the_stream(self):
        """The Oct 31 deadlock: stop() ran the moment the callback set playback_complete."""
        order = []
        player = audio_player.NonBlockingAudioPlayer()
        stream = MagicMock()
        stream.stop.side_effect = lambda: order.append("stop")
        player.stream = stream
        player.playback_complete.set()

        def callback_finishes_later():
            time.sleep(0.2)
            order.append("finished")
            player._stream_finished.set()
        threading.Thread(target=callback_finishes_later).start()
        player.wait()
        assert order == ["finished", "stop"]

    def test_dead_device_still_stops(self, monkeypatch):
        monkeypatch.setattr(audio_player, "STREAM_FINISH_WAIT_SECONDS", 0.05)
        player = audio_player.NonBlockingAudioPlayer()
        stream = MagicMock()
        player.stream = stream
        player.playback_complete.set()
        player.wait()
        stream.stop.assert_called_once()

    def test_callback_ends_stream_when_a_stop_is_requested(self):
        class Stop(Exception):
            pass
        player = audio_player.NonBlockingAudioPlayer()
        player._end_requested = True
        with patch.object(audio_player.sd, "CallbackStop", Stop), pytest.raises(Stop):
            player._audio_callback(np.zeros((8, 1), dtype=np.float32), 8, None, None)


class TestDismissed:
    def test_marker_for_this_session(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pause_resume, "DISMISS_DIR", str(tmp_path))
        token = client_identity.bind(client_identity.ClientIdentity("pid:4242", 4242))
        try:
            with patch.object(pause_resume, "_ancestor_pids", return_value={4242}):
                assert pause_resume.dismissed_from_voice() is False
                (tmp_path / "4242").write_text("x")
                assert pause_resume.dismissed_from_voice() is True
        finally:
            client_identity.unbind(token)

    def test_another_tabs_marker_does_not_count(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pause_resume, "DISMISS_DIR", str(tmp_path))
        (tmp_path / "1111").write_text("x")
        with patch.object(pause_resume, "_ancestor_pids", return_value={4242}):
            assert pause_resume.dismissed_from_voice() is False

    def test_shared_service_with_no_known_caller_is_not_dismissed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pause_resume, "DISMISS_DIR", str(tmp_path))
        monkeypatch.setattr(client_identity, "SHARED_MODE", True)
        (tmp_path / "4242").write_text("x")
        assert pause_resume.dismissed_from_voice() is False


class TestDismissedDuringCall:
    """The 6:32 AM report: dismissed while waiting for the mic, then spoke anyway."""

    @pytest.mark.asyncio
    async def test_dismissal_during_the_wait_ends_the_call_silently(self, clean_conch, activity_log):
        from voice_mode.tools import converse as mod

        answers = iter([False, False, False])
        tts = AsyncMock()
        with patch.object(Conch, "try_acquire", return_value=False), \
             patch.object(Conch, "get_holder", return_value={"pid": 99999, "agent": "other"}), \
             patch.object(Conch, "request_yield"), \
             patch.object(mod, "CONCH_CHECK_INTERVAL", 0.01), \
             patch.object(mod, "text_to_speech_with_failover", tts), \
             patch.object(mod, "dismissed_from_voice", side_effect=lambda: next(answers, True)):
            result = await getattr(mod.converse, "fn", mod.converse)(
                message="hello", wait_for_response=False, wait_for_conch=True)

        assert "DISMISSED" in result
        tts.assert_not_called()
        refused = [e for e in activity_log() if e["event"] == "call_refused_dismissed"]
        assert refused and refused[0]["at"] == "waiting_for_mic"

    @pytest.mark.asyncio
    async def test_dismissal_landing_as_the_mic_frees_ends_the_call_silently(self, clean_conch, activity_log):
        from voice_mode.tools import converse as mod

        tts = AsyncMock()
        answers = iter([False, True])
        with patch.object(mod, "text_to_speech_with_failover", tts), \
             patch.object(mod, "dismissed_from_voice", side_effect=lambda: next(answers, True)):
            result = await getattr(mod.converse, "fn", mod.converse)(message="hello", wait_for_response=False)

        assert "DISMISSED" in result
        tts.assert_not_called()
        assert not Conch.LOCK_FILE.exists(), "the mic must be free again"
        events = [e["event"] for e in activity_log()]
        assert "call_refused_dismissed" in events and "mic_taken" not in events and "call_end" in events

    @pytest.mark.asyncio
    async def test_not_dismissed_speaks_as_before(self, clean_conch, activity_log):
        from voice_mode.tools import converse as mod

        tts = AsyncMock(return_value=(True, {}, {"provider": "test", "voice": "x"}))
        with patch.object(mod, "text_to_speech_with_failover", tts), \
             patch.object(mod, "dismissed_from_voice", return_value=False):
            result = await getattr(mod.converse, "fn", mod.converse)(message="hello", wait_for_response=False)

        assert "DISMISSED" not in result
        tts.assert_called_once()
        events = [e["event"] for e in activity_log()]
        assert events.index("call_start") < events.index("mic_taken") < events.index("speak_start") < events.index("call_end")


class TestStuckAudioRefusesCalls:
    @pytest.mark.asyncio
    async def test_a_new_call_gets_an_answer_instead_of_hanging(self, clean_conch, activity_log):
        from voice_mode.tools import converse as mod

        audio_stop._abandoned.append(object())
        tts = AsyncMock()
        with patch.object(mod, "text_to_speech_with_failover", tts), \
             patch.object(mod, "dismissed_from_voice", return_value=False):
            result = await getattr(mod.converse, "fn", mod.converse)(message="hello", wait_for_response=True)
        assert "AUDIO STUCK" in result
        tts.assert_not_called()
        assert [e for e in activity_log() if e["event"] == "call_refused_audio_stuck"][0]["at"] == "start"

    @pytest.mark.asyncio
    async def test_a_stop_that_sticks_during_speech_keeps_the_mic_closed(self, clean_conch, activity_log):
        from voice_mode.tools import converse as mod

        async def speak_then_stick(*a, **k):
            audio_stop._abandoned.append(object())
            return True, {}, {"provider": "test", "voice": "x"}

        with patch.object(mod, "text_to_speech_with_failover", speak_then_stick), \
             patch.object(mod, "dismissed_from_voice", return_value=False), \
             patch.object(mod, "play_audio_feedback", AsyncMock()) as chime, \
             patch.object(mod, "record_audio_with_silence_detection") as record:
            result = await getattr(mod.converse, "fn", mod.converse)(message="hello", wait_for_response=True)
        assert "AUDIO STUCK" in result
        chime.assert_not_called()
        record.assert_not_called()


class TestRefreshSkipped:
    @pytest.mark.asyncio
    async def test_no_device_refresh_while_another_call_is_using_audio(self, clean_conch, activity_log):
        """Pa_Terminate closes every stream in the process, another tab's mic included."""
        import sounddevice as sd
        from voice_mode.tools import converse as mod

        with patch.object(sd, "_terminate") as term, patch.object(sd, "_initialize"), \
             patch.object(Conch, "try_acquire", return_value=False), \
             patch.object(Conch, "get_holder", return_value={"pid": 99999, "agent": "other"}), \
             patch.object(mod, "dismissed_from_voice", return_value=False):
            async with mod.audio_operation_lock:
                await getattr(mod.converse, "fn", mod.converse)(message="hello", wait_for_response=False)
            term.assert_not_called()
            await getattr(mod.converse, "fn", mod.converse)(message="hello", wait_for_response=False)
            term.assert_called_once()


class TestLoopWatchdog:
    def make(self, activity_log):
        now = [0.0]
        frozen, exits = [], []
        dog = LoopWatchdog(alert_after=20, exit_after=60, clock=lambda: now[0],
                           on_frozen=frozen.append, on_exit=lambda: exits.append(True))
        return dog, now, frozen, exits

    def test_alerts_once_then_restarts(self, activity_log):
        dog, now, frozen, exits = self.make(activity_log)
        now[0] = 19
        dog.check()
        assert not frozen
        now[0] = 21
        dog.check()
        now[0] = 40
        dog.check()
        assert len(frozen) == 1 and not exits
        now[0] = 61
        dog.check()
        assert exits
        assert [e["event"] for e in activity_log()] == ["service_frozen", "service_restart"]

    def test_recovery_is_recorded_and_can_alert_again(self, activity_log):
        dog, now, frozen, exits = self.make(activity_log)
        now[0] = 25
        dog.check()
        dog._last_beat = now[0]
        dog.check()
        now[0] = 50
        dog.check()
        assert len(frozen) == 2 and not exits
        assert [e["event"] for e in activity_log()] == ["service_frozen", "service_recovered", "service_frozen"]

    @pytest.mark.asyncio
    async def test_a_real_blocked_loop_is_noticed_from_the_thread(self, activity_log):
        frozen = []
        dog = LoopWatchdog(alert_after=0.2, exit_after=100, on_frozen=frozen.append)
        beat = asyncio.ensure_future(dog.beat(every=0.02))
        await asyncio.sleep(0.05)
        dog.start_thread(poll=0.02)
        time.sleep(0.5)  # blocks this event loop on purpose
        await asyncio.sleep(0.1)
        beat.cancel()
        dog.stop()
        assert len(frozen) == 1
        assert [e["event"] for e in activity_log()] == ["service_frozen", "service_recovered"]


class TestStuckAudioRestart:
    @pytest.mark.asyncio
    async def test_waits_for_running_requests_then_exits(self, activity_log):
        import uvicorn
        from voice_mode.daemon import ActiveRequests, DrainingServer

        now = [0.0]
        active = ActiveRequests(MagicMock())
        server = DrainingServer(uvicorn.Config(active), active, max_wait=55.0, clock=lambda: now[0])
        audio_stop._abandoned.append(object())
        active.count = 1
        with patch("voice_mode.daemon.os._exit") as exit_now:
            await server.on_tick(0)
            exit_now.assert_not_called()
            active.count = 0
            await server.on_tick(1)
            exit_now.assert_called_once_with(0)
        assert activity_log()[-1]["reason"] == "audio_stuck"

    @pytest.mark.asyncio
    async def test_a_call_stuck_on_it_cannot_hold_the_restart_forever(self, activity_log):
        import uvicorn
        from voice_mode.daemon import ActiveRequests, DrainingServer

        now = [0.0]
        active = ActiveRequests(MagicMock())
        server = DrainingServer(uvicorn.Config(active), active, max_wait=55.0, clock=lambda: now[0])
        audio_stop._abandoned.append(object())
        active.count = 1
        with patch("voice_mode.daemon.os._exit") as exit_now:
            await server.on_tick(0)
            now[0] = 56
            await server.on_tick(1)
            exit_now.assert_called_once_with(0)


class TestActivityLog:
    def test_rolls_at_the_limit_and_never_raises(self, tmp_path, monkeypatch):
        path = tmp_path / "activity.jsonl"
        monkeypatch.setenv("VOICEMODE_ACTIVITY_LOG", str(path))
        monkeypatch.setattr(activity, "MAX_BYTES", 10)
        activity.log("one")
        activity.log("two")
        assert json.loads((tmp_path / "activity.jsonl.1").read_text())["event"] == "one"
        assert json.loads(path.read_text())["event"] == "two"
        monkeypatch.setenv("VOICEMODE_ACTIVITY_LOG", "/proc/nope/nope/activity.jsonl")
        activity.log("three")
