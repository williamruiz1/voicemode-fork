"""Tests for the conch yieldable-listen mechanism (vibedispatcher#132).

Covers the acceptance criteria of the dispatch:
- R1: a listening (idle) holder yields on another agent's request — the
  wanted-file signal fires (fresh, live requester) and the listen-loop yield
  decision ends the idle listen.
- R2: TTS is never preempted mid-utterance — the waiter's preempt path grants
  grace while the speaking flag is present.
- R3: the hard-timeout path preempts-and-acquires instead of failing.
- R4: existing expiry + dead-holder reap behavior is untouched (see
  test_conch.py, which must keep passing).
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

import pytest

from voice_mode.conch import Conch


@pytest.fixture
def clean_conch():
    """Ensure no conch/wanted files exist before/after tests."""
    for f in (Conch.LOCK_FILE, Conch.WANTED_FILE):
        if f.exists():
            f.unlink()
    yield
    for f in (Conch.LOCK_FILE, Conch.WANTED_FILE):
        if f.exists():
            f.unlink()


def _write_wanted(pid: int, requested: datetime, agent: str = "other") -> None:
    Conch.WANTED_FILE.parent.mkdir(parents=True, exist_ok=True)
    Conch.WANTED_FILE.write_text(json.dumps({
        "pid": pid,
        "agent": agent,
        "requested": requested.isoformat(),
    }))


def _spawn_live_process() -> subprocess.Popen:
    """A real, alive, other-pid process to stand in as the requester."""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])


class TestWantedSignal:
    """R1 — the request/yield signal channel."""

    def test_not_wanted_when_no_file(self, clean_conch):
        assert Conch.is_wanted() is False

    def test_request_yield_writes_wanted_file(self, clean_conch):
        Conch.request_yield("agent_b")
        data = json.loads(Conch.WANTED_FILE.read_text())
        assert data["pid"] == os.getpid()
        assert data["agent"] == "agent_b"
        assert "requested" in data

    def test_own_request_does_not_count_as_wanted(self, clean_conch):
        """A process never yields to its own request."""
        Conch.request_yield("me")
        assert Conch.is_wanted() is False

    def test_fresh_request_from_live_process_is_wanted(self, clean_conch):
        proc = _spawn_live_process()
        try:
            _write_wanted(proc.pid, datetime.now())
            assert Conch.is_wanted() is True
        finally:
            proc.kill()
            proc.wait()

    def test_stale_request_is_not_wanted(self, clean_conch):
        proc = _spawn_live_process()
        try:
            _write_wanted(proc.pid, datetime.now() - timedelta(seconds=600))
            assert Conch.is_wanted() is False
        finally:
            proc.kill()
            proc.wait()

    def test_dead_requester_is_not_wanted_and_reaped(self, clean_conch):
        _write_wanted(999999999, datetime.now())
        assert Conch.is_wanted() is False
        assert not Conch.WANTED_FILE.exists()  # stale request reaped

    def test_malformed_wanted_file_fails_open(self, clean_conch):
        Conch.WANTED_FILE.parent.mkdir(parents=True, exist_ok=True)
        Conch.WANTED_FILE.write_text("{not json")
        assert Conch.is_wanted() is False

    def test_clear_yield_request_only_clears_own(self, clean_conch):
        proc = _spawn_live_process()
        try:
            _write_wanted(proc.pid, datetime.now())
            Conch.clear_yield_request()  # not ours — must stay
            assert Conch.WANTED_FILE.exists()
        finally:
            proc.kill()
            proc.wait()

        Conch.request_yield("me")
        Conch.clear_yield_request()  # ours — removed
        assert not Conch.WANTED_FILE.exists()


class TestIdleListenYieldDecision:
    """R1 — the listen-loop yield decision in record_audio_with_silence_detection.

    The loop yields ONLY while idle: `yield_check is not None and not
    speech_detected and yield_check()`. Exercised at the decision level
    (real audio capture needs hardware; the live two-session soak is the
    VD GM's R1/R3 verification step).
    """

    @staticmethod
    def _decision(yield_check, speech_detected):
        # Mirrors the guard wired into the recording loop in converse.py.
        if yield_check is not None and not speech_detected:
            return bool(yield_check())
        return False

    def test_idle_listen_yields_on_wanted(self, clean_conch):
        proc = _spawn_live_process()
        try:
            _write_wanted(proc.pid, datetime.now())
            assert self._decision(Conch.is_wanted, speech_detected=False) is True
        finally:
            proc.kill()
            proc.wait()

    def test_active_speech_never_yields(self, clean_conch):
        proc = _spawn_live_process()
        try:
            _write_wanted(proc.pid, datetime.now())
            assert self._decision(Conch.is_wanted, speech_detected=True) is False
        finally:
            proc.kill()
            proc.wait()

    def test_no_yield_check_never_yields(self, clean_conch):
        # skip_conch / yield-disabled path: no callable wired in.
        assert self._decision(None, speech_detected=False) is False


class TestPreemptAcquire:
    """R2 + R3 — the waiter's hard-timeout preempt path."""

    def test_preempt_takes_lock_from_foreign_holder(self, clean_conch):
        """R3: a fresh lock held by another (alive) process is preempted."""
        proc = _spawn_live_process()
        try:
            Conch.LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
            Conch.LOCK_FILE.write_text(json.dumps({
                "pid": proc.pid,
                "agent": "stuck_holder",
                "acquired": datetime.now().isoformat(),
                "expires": None,
            }))
            conch = Conch(agent_name="preempter")
            # Normal try_acquire may or may not fail (no flock held by the fake
            # holder) — the preempt path must succeed regardless.
            assert conch.preempt_acquire("preempter") is True
            data = json.loads(Conch.LOCK_FILE.read_text())
            assert data["pid"] == os.getpid()
            conch.release()
        finally:
            proc.kill()
            proc.wait()

    def test_preempt_with_no_lock_still_acquires(self, clean_conch):
        conch = Conch(agent_name="preempter")
        assert conch.preempt_acquire() is True
        conch.release()

    @pytest.mark.asyncio
    async def test_preempt_never_fires_while_speaking_flag_present(self, clean_conch, tmp_path, monkeypatch):
        """R2: TTS is never preempted mid-utterance.

        Runs the REAL waiter grace helper (_preempt_conch_after_tts_grace):
        while the speaking flag exists, preempt_acquire must not fire; it may
        only fire after the flag clears (TTS finished) or the grace expires.
        """
        import asyncio
        from unittest.mock import patch as mock_patch
        import voice_mode.tools.converse as conv

        monkeypatch.setattr(conv, "CONCH_PREEMPT_TTS_GRACE", 5.0)
        monkeypatch.setattr(conv, "CONCH_CHECK_INTERVAL", 0.05)

        flag = tmp_path / "speaking.flag"
        flag.write_text("")

        conch = Conch(agent_name="waiter")
        preempted_while_flag = []
        real_preempt = conch.preempt_acquire

        def spy_preempt(agent_name=None):
            preempted_while_flag.append(flag.exists())
            return real_preempt(agent_name)

        monkeypatch.setattr(conch, "preempt_acquire", spy_preempt)

        # Holder never releases voluntarily during the grace window.
        with mock_patch.object(Conch, "try_acquire", return_value=False):
            async def finish_tts():
                await asyncio.sleep(0.3)
                flag.unlink()  # TTS finishes mid-grace

            task = asyncio.create_task(finish_tts())
            await conv._preempt_conch_after_tts_grace(conch, str(flag))
            await task

        assert preempted_while_flag == [False], (
            "preempt_acquire must fire exactly once, and only AFTER the "
            "speaking flag cleared"
        )

    @pytest.mark.asyncio
    async def test_preempt_fires_immediately_when_not_speaking(self, clean_conch, tmp_path):
        """R3: at the hard timeout with no TTS in progress, the waiter
        preempts-and-acquires instead of failing."""
        import voice_mode.tools.converse as conv

        proc = _spawn_live_process()
        try:
            Conch.LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
            Conch.LOCK_FILE.write_text(json.dumps({
                "pid": proc.pid,
                "agent": "stuck_holder",
                "acquired": datetime.now().isoformat(),
                "expires": None,
            }))
            conch = Conch(agent_name="waiter")
            acquired = await conv._preempt_conch_after_tts_grace(
                conch, str(tmp_path / "absent-speaking.flag")
            )
            assert acquired is True
            assert json.loads(Conch.LOCK_FILE.read_text())["pid"] == os.getpid()
            conch.release()
        finally:
            proc.kill()
            proc.wait()
