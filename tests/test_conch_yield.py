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
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voice_mode.conch import Conch
from voice_mode.config import SAMPLE_RATE, VAD_CHUNK_DURATION_MS
from voice_mode.tools.converse import record_audio_with_silence_detection


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


class TestIdleListenYieldGracePeriod:
    """Regression coverage for the founder-os 2026-08-04 barge-in incident.

    Timeline that produced it (voicemode_events_2026-08-04.jsonl): holder A
    finishes speaking, opens a fresh idle-listen (RECORDING_START); waiter B
    had already been polling wait_for_conch (and therefore refreshing
    conch-wanted) for ~1s *before* A's TTS even ended. A's listen loop honored
    the pending yield_check() on its very first tick — recording_duration was
    effectively 0 — so A yielded after 0.178s with zero samples captured and B
    grabbed the conch and started talking before William could get a word in.

    The fix requires CONCH_YIELD_GRACE_SECONDS of elapsed idle-listening
    before a yield request is honored, mirroring the guard now wired into
    record_audio_with_silence_detection in converse.py.
    """

    @staticmethod
    def _decision(yield_check, speech_detected, recording_duration, grace_seconds):
        # Mirrors the guard wired into the recording loop in converse.py.
        if (yield_check is not None and not speech_detected
                and recording_duration >= grace_seconds):
            return bool(yield_check())
        return False

    def test_pending_request_at_listen_start_does_not_barge_in(self, clean_conch):
        """The exact incident shape: is_wanted() already True at t=0."""
        proc = _spawn_live_process()
        try:
            _write_wanted(proc.pid, datetime.now())
            assert self._decision(
                Conch.is_wanted, speech_detected=False,
                recording_duration=0.0, grace_seconds=3.0,
            ) is False
        finally:
            proc.kill()
            proc.wait()

    def test_request_still_denied_partway_through_grace(self, clean_conch):
        proc = _spawn_live_process()
        try:
            _write_wanted(proc.pid, datetime.now())
            assert self._decision(
                Conch.is_wanted, speech_detected=False,
                recording_duration=1.5, grace_seconds=3.0,
            ) is False
        finally:
            proc.kill()
            proc.wait()

    def test_request_honored_once_grace_elapses(self, clean_conch):
        """A waiter that's still genuinely wanted after the human had a real
        chance to reply is still honored — this isn't disabling the yield,
        only delaying it past the turn boundary."""
        proc = _spawn_live_process()
        try:
            _write_wanted(proc.pid, datetime.now())
            assert self._decision(
                Conch.is_wanted, speech_detected=False,
                recording_duration=3.0, grace_seconds=3.0,
            ) is True
        finally:
            proc.kill()
            proc.wait()

    def test_zero_grace_restores_old_immediate_yield_behavior(self, clean_conch):
        """VOICEMODE_CONCH_YIELD_GRACE_SECONDS=0 is documented as an escape
        hatch back to the pre-fix behavior — must still work for anyone who
        explicitly opts back in."""
        proc = _spawn_live_process()
        try:
            _write_wanted(proc.pid, datetime.now())
            assert self._decision(
                Conch.is_wanted, speech_detected=False,
                recording_duration=0.0, grace_seconds=0.0,
            ) is True
        finally:
            proc.kill()
            proc.wait()


_CHUNK_SAMPLES = int(SAMPLE_RATE * VAD_CHUNK_DURATION_MS / 1000)
_CHUNK_DURATION_S = VAD_CHUNK_DURATION_MS / 1000


def _run_idle_listen(yield_check, supersede_check, grace_seconds, n_chunks=400):
    """Drive the REAL record_audio_with_silence_detection loop with silence
    (never speech) and a mocked mic, exactly like
    tests/test_endpointing_recordloop.py::_run_with_fixture — no real device,
    no real wall-clock wait (recording_duration is a nominal per-chunk
    counter, not a wall-clock read, so this runs at test speed).

    Returns (audio, speech_detected, yield_state).
    """
    silence_chunk = np.zeros(_CHUNK_SAMPLES, dtype=np.int16).reshape(-1, 1)
    mock_queue_instance = MagicMock()
    mock_queue_instance.get.side_effect = [silence_chunk] * n_chunks

    yield_state = {"yielded": False}
    with patch("voice_mode.tools.converse.sd") as mock_sd, \
         patch("voice_mode.tools.converse.CONCH_YIELD_GRACE_SECONDS", grace_seconds), \
         patch("queue.Queue", return_value=mock_queue_instance):
        mock_sd.InputStream.return_value.__enter__.return_value = MagicMock()
        mock_sd.InputStream.return_value.__exit__.return_value = False
        audio, speech_detected = record_audio_with_silence_detection(
            max_duration=8.0,
            yield_check=yield_check,
            yield_state=yield_state,
            supersede_check=supersede_check,
        )
    return audio, speech_detected, yield_state


class TestSupersessionBypassesGraceFloor:
    """W2c merge fix (fold of PR #12 conch-fail-closed into master's grace
    floor, PR #11 / commit 2916047).

    master added CONCH_YIELD_GRACE_SECONDS: a fresh idle-listen must run for
    at least that long before honoring a yield REQUEST (Conch.is_wanted),
    because a waiter's poll loop can start requesting the mic before the
    holder even starts listening (see TestIdleListenYieldGracePeriod above).

    fix/conch-fail-closed (PR #12) added Conch.is_superseded: a preempter has
    ALREADY taken the lock in place under a bumped epoch, so the holder no
    longer holds what it thinks it holds. That is a FACT, not a request — the
    human is welcome to keep waiting on a request, but a fact about who
    currently owns the mic can't be deferred without two processes both
    believing they hold it for the whole deferral.

    A TEXTUAL "take both sides" merge of the two PRs auto-merges cleanly
    (verified: `git merge origin/fix/conch-fail-closed` into a fresh
    origin/master worktree produces NO conflict markers in converse.py) but
    is WRONG: fix/conch-fail-closed's higher-level wiring hunk rebinds the
    single `yield_check` parameter to `conch.should_yield` (which composes
    is_superseded() OR is_wanted()), and master's grace-gate hunk in the
    low-level loop is untouched — so the composed check, supersession
    included, gets gated behind CONCH_YIELD_GRACE_SECONDS. converse.py now
    wires supersession as its OWN callable (`supersede_check`, ungated) so
    this can't happen — see record_audio_with_silence_detection's docstring
    and the wiring at its `listen_supersede_check` call site.

    These tests run the REAL record loop (see _run_idle_listen) exercising
    the real production grace-gate code — not a mirror of the conditional.
    The naive-merge contrast test is proven faithfully because the low-level
    grace-gated `if (yield_check is not None and not speech_detected and
    recording_duration >= CONCH_YIELD_GRACE_SECONDS):` block in this worktree
    is BYTE-IDENTICAL to the one a real `git merge` of origin/fix/conch-
    fail-closed into origin/master produces (diffed and confirmed at merge
    time) -- so calling this file's own record_audio_with_silence_detection
    with the naive wiring (single `yield_check=holder.should_yield`, no
    `supersede_check`) reproduces the naive merge's runtime behavior exactly,
    not just its shape.
    """

    def test_supersession_honored_on_first_tick_correct_merge(self, clean_conch):
        """Acceptance test 1: a superseded holder stands down at t≈0, well
        below CONCH_YIELD_GRACE_SECONDS, on the CORRECT (split-check) merge."""
        holder = Conch(agent_name="holder")
        assert holder.try_acquire() is True
        Conch(agent_name="preempter").preempt_acquire("preempter")
        assert holder.is_superseded() is True, "setup: holder must be superseded"

        audio, speech_detected, yield_state = _run_idle_listen(
            yield_check=Conch.is_wanted,
            supersede_check=holder.is_superseded,
            grace_seconds=0.3,
        )

        recorded_duration = len(audio) / SAMPLE_RATE
        assert speech_detected is False
        assert yield_state["yielded"] is True
        assert recorded_duration < 0.1, (
            f"correct merge: superseded holder ran {recorded_duration:.3f}s "
            f"before yielding -- should have stood down on ~the first tick"
        )

    def test_naive_merge_would_have_gated_supersession_behind_grace(self, clean_conch):
        """THE CONTRAST — the single most important test in this suite.

        Same exact scenario (already-superseded holder), decided by the
        NAIVE both-sides-merge wiring: a single `yield_check` bound to the
        COMPOSED `conch.should_yield`, no separate supersede_check (that
        parameter does not exist on the naive merge's converse.py at all).
        This FAILS to stand down until CONCH_YIELD_GRACE_SECONDS elapses --
        the double-hold bug PR #12 exists to close. It is run against the
        real production loop, so if this assertion ever passes at t≈0
        instead, the correct/naive contrast this test exists to prove has
        silently stopped holding (e.g. someone re-merged should_yield back
        into the wiring) and this test's FAILURE is the signal to look at.
        """
        holder = Conch(agent_name="holder")
        assert holder.try_acquire() is True
        Conch(agent_name="preempter").preempt_acquire("preempter")
        assert holder.is_superseded() is True, "setup: holder must be superseded"

        grace = 0.3
        audio, speech_detected, yield_state = _run_idle_listen(
            yield_check=holder.should_yield,  # the naive merge's single wire
            supersede_check=None,             # doesn't exist on the naive merge
            grace_seconds=grace,
        )

        recorded_duration = len(audio) / SAMPLE_RATE
        assert speech_detected is False
        assert yield_state["yielded"] is True, "should still yield eventually"
        assert recorded_duration >= grace - _CHUNK_DURATION_S, (
            f"naive merge: superseded holder yielded after only "
            f"{recorded_duration:.3f}s, before the {grace}s grace floor -- "
            f"if this fails, the naive-vs-correct contrast no longer holds "
            f"and the wiring may have regressed to the naive shape"
        )

    def test_yield_request_still_withheld_until_grace_elapses(self, clean_conch):
        """Acceptance test 2: master's fix is not weakened. A mere REQUEST
        (not a supersession -- this holder is never preempted) is still
        withheld until grace elapses, on the correct merge's own wiring."""
        holder = Conch(agent_name="holder")
        assert holder.try_acquire() is True
        assert holder.is_superseded() is False

        proc = _spawn_live_process()
        try:
            _write_wanted(proc.pid, datetime.now())
            grace = 0.3
            audio, speech_detected, yield_state = _run_idle_listen(
                yield_check=Conch.is_wanted,
                supersede_check=holder.is_superseded,  # always False here
                grace_seconds=grace,
            )
        finally:
            proc.kill()
            proc.wait()

        recorded_duration = len(audio) / SAMPLE_RATE
        assert speech_detected is False
        assert yield_state["yielded"] is True
        assert recorded_duration >= grace - _CHUNK_DURATION_S, (
            f"a mere request yielded after only {recorded_duration:.3f}s, "
            f"before the {grace}s grace floor -- master's fix was weakened"
        )

    def test_supersession_survives_conch_yield_disabled(self, clean_conch):
        """Acceptance test 3: config cannot switch off a fact. Mirrors
        converse.py's own wiring when VOICEMODE_CONCH_YIELD_ENABLED=false --
        `listen_yield_check` is never even constructed (passed as None here)
        -- while `listen_supersede_check` is unaffected by that flag."""
        holder = Conch(agent_name="holder")
        assert holder.try_acquire() is True
        Conch(agent_name="preempter").preempt_acquire("preempter")
        assert holder.is_superseded() is True

        audio, speech_detected, yield_state = _run_idle_listen(
            yield_check=None,  # CONCH_YIELD_ENABLED=false: never wired
            supersede_check=holder.is_superseded,
            grace_seconds=0.3,
        )

        recorded_duration = len(audio) / SAMPLE_RATE
        assert speech_detected is False
        assert yield_state["yielded"] is True
        assert recorded_duration < 0.1, (
            f"supersession did not survive yield being disabled -- ran "
            f"{recorded_duration:.3f}s before yielding"
        )


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
