"""Tests for the Conch lock file mechanism."""

import json
import logging
import multiprocessing
import os
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from voice_mode.conch import Conch, ConchUnavailable


@pytest.fixture
def clean_conch():
    """Ensure no conch file exists before/after tests."""
    conch_file = Conch.LOCK_FILE
    if conch_file.exists():
        conch_file.unlink()
    yield
    if conch_file.exists():
        conch_file.unlink()


class TestConch:
    """Tests for Conch class."""

    def test_is_active_returns_false_when_no_lock_file(self, clean_conch):
        """is_active() returns False when lock file doesn't exist."""
        assert Conch.is_active() is False

    def test_acquire_creates_lock_file(self, clean_conch):
        """acquire() creates the lock file with correct content."""
        conch = Conch(agent_name="test_agent")
        conch.acquire()

        assert Conch.LOCK_FILE.exists()

        data = json.loads(Conch.LOCK_FILE.read_text())
        assert data["pid"] == os.getpid()
        assert data["agent"] == "test_agent"
        assert "acquired" in data
        assert data["expires"] is None

        conch.release()

    def test_release_removes_lock_file(self, clean_conch):
        """release() removes the lock file."""
        conch = Conch()
        conch.acquire()
        assert Conch.LOCK_FILE.exists()

        conch.release()
        assert not Conch.LOCK_FILE.exists()

    def test_is_active_returns_true_when_lock_held(self, clean_conch):
        """is_active() returns True when lock file exists and PID is alive."""
        conch = Conch()
        conch.acquire()

        assert Conch.is_active() is True

        conch.release()

    def test_is_active_returns_false_for_stale_lock(self, clean_conch):
        """is_active() returns False when PID in lock file is dead."""
        # Create a lock file with a non-existent PID
        Conch.LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pid": 999999999,  # Very unlikely to be a valid PID
            "agent": "dead_agent",
            "acquired": "2026-01-01T00:00:00",
            "expires": None
        }
        Conch.LOCK_FILE.write_text(json.dumps(data))

        assert Conch.is_active() is False

    def test_context_manager_acquires_and_releases(self, clean_conch):
        """Context manager properly acquires and releases lock."""
        assert Conch.is_active() is False

        with Conch(agent_name="context_test"):
            assert Conch.is_active() is True

        assert Conch.is_active() is False

    def test_context_manager_releases_on_exception(self, clean_conch):
        """Context manager releases lock even if exception occurs."""
        assert Conch.is_active() is False

        try:
            with Conch(agent_name="exception_test"):
                assert Conch.is_active() is True
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert Conch.is_active() is False

    def test_get_holder_returns_lock_info(self, clean_conch):
        """get_holder() returns lock holder information."""
        assert Conch.get_holder() is None

        with Conch(agent_name="holder_test"):
            holder = Conch.get_holder()
            assert holder is not None
            assert holder["agent"] == "holder_test"
            assert holder["pid"] == os.getpid()

        assert Conch.get_holder() is None

    def test_acquire_with_override_agent_name(self, clean_conch):
        """acquire() can override agent name set in constructor."""
        conch = Conch(agent_name="original")
        conch.acquire(agent_name="override")

        data = json.loads(Conch.LOCK_FILE.read_text())
        assert data["agent"] == "override"

        conch.release()

    def test_release_handles_missing_file_gracefully(self, clean_conch):
        """release() doesn't error if lock file doesn't exist."""
        conch = Conch()
        # This should not raise
        conch.release()

    def test_is_active_handles_invalid_json(self, clean_conch):
        """is_active() returns False for invalid JSON in lock file."""
        Conch.LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        Conch.LOCK_FILE.write_text("not valid json {{{")

        assert Conch.is_active() is False


class TestConchConfig:
    """Tests for conch configuration options."""

    def test_conch_enabled_default(self):
        """CONCH_ENABLED defaults to True."""
        from voice_mode.config import CONCH_ENABLED
        # Default should be True (unless env var overrides)
        assert isinstance(CONCH_ENABLED, bool)

    def test_conch_timeout_default(self):
        """CONCH_TIMEOUT defaults to 60 seconds."""
        from voice_mode.config import CONCH_TIMEOUT
        assert isinstance(CONCH_TIMEOUT, float)
        assert CONCH_TIMEOUT == 60.0

    def test_conch_check_interval_default(self):
        """CONCH_CHECK_INTERVAL defaults to 0.5 seconds."""
        from voice_mode.config import CONCH_CHECK_INTERVAL
        assert isinstance(CONCH_CHECK_INTERVAL, float)
        assert CONCH_CHECK_INTERVAL == 0.5

    def test_conch_enabled_env_var(self):
        """CONCH_ENABLED can be set via environment variable."""
        import os
        import importlib
        import voice_mode.config

        # Test with false
        os.environ["VOICEMODE_CONCH_ENABLED"] = "false"
        importlib.reload(voice_mode.config)
        assert voice_mode.config.CONCH_ENABLED is False

        # Test with true
        os.environ["VOICEMODE_CONCH_ENABLED"] = "true"
        importlib.reload(voice_mode.config)
        assert voice_mode.config.CONCH_ENABLED is True

        # Clean up
        del os.environ["VOICEMODE_CONCH_ENABLED"]
        importlib.reload(voice_mode.config)

    def test_conch_timeout_env_var(self):
        """CONCH_TIMEOUT can be set via environment variable."""
        import os
        import importlib
        import voice_mode.config

        os.environ["VOICEMODE_CONCH_TIMEOUT"] = "120"
        importlib.reload(voice_mode.config)
        assert voice_mode.config.CONCH_TIMEOUT == 120.0

        # Clean up
        del os.environ["VOICEMODE_CONCH_TIMEOUT"]
        importlib.reload(voice_mode.config)


def _try_acquire_worker(name: str, queue: multiprocessing.Queue, hold_time: float = 0.5):
    """Worker function for multiprocessing tests.

    Args:
        name: Agent name for the conch
        queue: Queue to report results back
        hold_time: How long to hold the lock if acquired
    """
    conch = Conch(agent_name=name)
    acquired = conch.try_acquire()
    queue.put((name, acquired))
    if acquired:
        time.sleep(hold_time)
        conch.release()


def _acquire_release_then_signal(name: str, queue: multiprocessing.Queue, barrier):
    """Acquire, release, then signal completion via barrier."""
    conch = Conch(agent_name=name)
    acquired = conch.try_acquire()
    queue.put((name, "first_try", acquired))
    if acquired:
        time.sleep(0.1)
        conch.release()
    barrier.wait()


def _wait_then_acquire(name: str, queue: multiprocessing.Queue, barrier):
    """Wait for barrier then try to acquire."""
    barrier.wait()
    time.sleep(0.05)  # Small delay to ensure release is complete
    conch = Conch(agent_name=name)
    acquired = conch.try_acquire()
    queue.put((name, "second_try", acquired))
    if acquired:
        conch.release()


class TestConchAtomicLocking:
    """Tests for atomic fcntl-based locking."""

    @pytest.fixture(autouse=True)
    def clean_conch_file(self):
        """Ensure no conch file exists before/after tests."""
        conch_file = Conch.LOCK_FILE
        if conch_file.exists():
            conch_file.unlink()
        yield
        if conch_file.exists():
            conch_file.unlink()

    def test_try_acquire_succeeds_when_not_held(self):
        """try_acquire() returns True when lock is not held."""
        conch = Conch(agent_name="test_agent")
        assert conch.try_acquire() is True
        conch.release()

    def test_try_acquire_fails_when_held(self):
        """try_acquire() returns False when lock is held by another."""
        conch1 = Conch(agent_name="first")
        conch2 = Conch(agent_name="second")

        assert conch1.try_acquire() is True
        assert conch2.try_acquire() is False

        conch1.release()

    def test_try_acquire_returns_true_if_already_holding(self):
        """try_acquire() returns True if we already hold the lock."""
        conch = Conch(agent_name="test")
        assert conch.try_acquire() is True
        # Second call should also return True
        assert conch.try_acquire() is True
        conch.release()

    def test_release_allows_next(self):
        """After release, another process can acquire."""
        conch1 = Conch(agent_name="first")
        conch2 = Conch(agent_name="second")

        assert conch1.try_acquire() is True
        assert conch2.try_acquire() is False

        conch1.release()

        assert conch2.try_acquire() is True
        conch2.release()

    def test_held_seconds_tracking(self):
        """release() returns correct held duration."""
        conch = Conch(agent_name="timing_test")
        conch.try_acquire()
        time.sleep(0.1)
        held = conch.release()
        # Allow some timing slack
        assert 0.09 < held < 0.3, f"Expected held time ~0.1s, got {held}s"

    def test_held_seconds_zero_when_not_acquired(self):
        """release() returns 0.0 when lock was never acquired."""
        conch = Conch()
        held = conch.release()
        assert held == 0.0

    def test_atomic_acquisition_multiprocess(self):
        """Only one process can acquire at a time (multiprocessing test)."""
        results = multiprocessing.Queue()

        # Start two processes simultaneously
        p1 = multiprocessing.Process(target=_try_acquire_worker, args=("agent1", results))
        p2 = multiprocessing.Process(target=_try_acquire_worker, args=("agent2", results))

        p1.start()
        p2.start()
        p1.join(timeout=5)
        p2.join(timeout=5)

        # Collect results
        acquisitions = []
        while not results.empty():
            acquisitions.append(results.get())

        # Exactly one should have acquired
        acquired_count = sum(1 for _, acq in acquisitions if acq)
        assert acquired_count == 1, f"Expected 1 acquisition, got {acquired_count}: {acquisitions}"

    def test_sequential_acquisition_after_release_multiprocess(self):
        """After first process releases, second can acquire (multiprocessing)."""
        results = multiprocessing.Queue()
        barrier = multiprocessing.Barrier(2)

        p1 = multiprocessing.Process(
            target=_acquire_release_then_signal,
            args=("first", results, barrier)
        )
        p2 = multiprocessing.Process(
            target=_wait_then_acquire,
            args=("second", results, barrier)
        )

        p1.start()
        p2.start()
        p1.join(timeout=5)
        p2.join(timeout=5)

        # Collect results
        acquisitions = {}
        while not results.empty():
            name, phase, acquired = results.get()
            acquisitions[(name, phase)] = acquired

        # First should acquire on first try
        assert acquisitions.get(("first", "first_try")) is True
        # Second should acquire after first releases
        assert acquisitions.get(("second", "second_try")) is True

    def test_lock_file_contains_correct_data(self):
        """Lock file contains PID, agent, and timestamp after try_acquire."""
        conch = Conch(agent_name="data_test")
        conch.try_acquire()

        assert Conch.LOCK_FILE.exists()
        data = json.loads(Conch.LOCK_FILE.read_text())

        assert data["pid"] == os.getpid()
        assert data["agent"] == "data_test"
        assert "acquired" in data
        assert data["expires"] is None

        conch.release()

    def test_try_acquire_with_override_agent_name(self):
        """try_acquire() can override agent name set in constructor."""
        conch = Conch(agent_name="original")
        conch.try_acquire(agent_name="override")

        data = json.loads(Conch.LOCK_FILE.read_text())
        assert data["agent"] == "override"

        conch.release()

    def test_try_acquire_clears_dead_holder_lock(self):
        """try_acquire() unlinks a lock owned by a dead PID and acquires.

        Uses the fork-and-reap pattern: spawn a child, wait for it to exit,
        then write a lock file with the reaped (now dead) PID. Avoids the
        flaky "PID 999999" pattern -- high-PID systems may have it in use.
        """
        # Fork a child that exits immediately, then reap it so the PID is
        # genuinely dead.
        pid = os.fork()
        if pid == 0:
            # Child -- exit immediately
            os._exit(0)
        # Parent -- reap the child
        os.waitpid(pid, 0)

        # Sanity check: signal 0 against the reaped PID should now raise.
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)

        # Write a lock file with the dead PID and a fresh timestamp
        # (so timestamp-based expiry would NOT fire).
        Conch.LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        data = {
            "pid": pid,
            "agent": "dead_agent",
            "acquired": datetime.now().isoformat(),
            "expires": None,
        }
        Conch.LOCK_FILE.write_text(json.dumps(data))

        # A fresh Conch should clear the dead-holder lock and acquire.
        new_conch = Conch(agent_name="new_agent")
        assert new_conch.try_acquire() is True

        # The lock should now be ours.
        new_data = json.loads(Conch.LOCK_FILE.read_text())
        assert new_data["pid"] == os.getpid()
        assert new_data["agent"] == "new_agent"

        new_conch.release()

    def test_try_acquire_clears_dead_holder_lock_with_expiry_disabled(self):
        """Dead-PID clearance works even when CONCH_LOCK_EXPIRY <= 0.

        Operators may opt out of timestamp-based expiry, but a dead holder
        is unambiguously stale and must still be cleared.
        """
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        os.waitpid(pid, 0)

        Conch.LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        data = {
            "pid": pid,
            "agent": "dead_agent",
            "acquired": datetime.now().isoformat(),
            "expires": None,
        }
        Conch.LOCK_FILE.write_text(json.dumps(data))

        # Patch the deferred lock-expiry getter to simulate disabled expiry.
        with patch("voice_mode.conch._get_lock_expiry", return_value=0):
            new_conch = Conch(agent_name="new_agent")
            assert new_conch.try_acquire() is True

        new_conch.release()

    def test_try_acquire_respects_live_holder_lock(self):
        """try_acquire() returns False when a live PID holds a fresh lock."""
        # Write a lock file with our own (live) PID and fresh timestamp.
        # We DON'T use Conch.acquire() because that doesn't take an flock --
        # we need a flock-protected lock to genuinely block try_acquire.
        holder = Conch(agent_name="holder")
        assert holder.try_acquire() is True

        # Sanity: lock file has our live PID.
        data = json.loads(Conch.LOCK_FILE.read_text())
        assert data["pid"] == os.getpid()

        # A fresh Conch should NOT acquire.
        contender = Conch(agent_name="contender")
        assert contender.try_acquire() is False

        # Lock file is unchanged (still belongs to holder).
        assert Conch.LOCK_FILE.exists()

        holder.release()

    def test_try_acquire_clears_stale_timestamp_with_live_pid(self):
        """Live PID + expired timestamp: takeover still succeeds.

        The outcome is unchanged, but the mechanism is now an IN-PLACE
        truncate-and-rewrite under the flock rather than an unlink, so no
        inode swap occurs. See TestConchFailClosed for the inode assertions.
        """
        # Write a lock file with our live PID but an ancient timestamp.
        Conch.LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pid": os.getpid(),
            "agent": "stuck_agent",
            "acquired": "2000-01-01T00:00:00",  # Way past any expiry
            "expires": None,
        }
        Conch.LOCK_FILE.write_text(json.dumps(data))

        # A fresh Conch should clear the stale-timestamp lock and acquire.
        new_conch = Conch(agent_name="new_agent")
        assert new_conch.try_acquire() is True

        new_data = json.loads(Conch.LOCK_FILE.read_text())
        assert new_data["agent"] == "new_agent"

        new_conch.release()

    def test_check_and_clear_handles_permission_error(self):
        """PermissionError from os.kill is treated as 'alive' -- lock preserved.

        If os.kill raises PermissionError, the process exists but is owned
        by another user. We must NOT clear the lock in that case.
        """
        # Write a lock file with a fresh timestamp and some PID.
        Conch.LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        data = {
            "pid": 12345,
            "agent": "other_user_agent",
            "acquired": datetime.now().isoformat(),
            "expires": None,
        }
        Conch.LOCK_FILE.write_text(json.dumps(data))

        # Mock os.kill to raise PermissionError.
        with patch("voice_mode.conch.os.kill", side_effect=PermissionError):
            conch = Conch(agent_name="probe")
            conch._check_and_clear_stale_lock()

        # Lock file must still exist -- treated as alive.
        assert Conch.LOCK_FILE.exists()
        preserved = json.loads(Conch.LOCK_FILE.read_text())
        assert preserved["pid"] == 12345
        assert preserved["agent"] == "other_user_agent"

    def test_release_without_acquire_does_not_delete_lock_file(self):
        """release() on non-holder must NOT delete the lock file.

        Regression test: Previously, release() would unconditionally delete
        ~/.voicemode/conch even when the caller never acquired the lock. This
        destroyed the flock held by the actual owner (on a different inode),
        allowing multiple agents to speak simultaneously.
        """
        # Agent A acquires the conch
        holder = Conch(agent_name="holder")
        assert holder.try_acquire() is True
        assert Conch.LOCK_FILE.exists()

        # Agent B fails to acquire
        blocked = Conch(agent_name="blocked")
        assert blocked.try_acquire() is False

        # Agent B calls release() — this should NOT delete the lock file
        blocked.release()

        # Lock file should still exist (belongs to Agent A)
        assert Conch.LOCK_FILE.exists(), (
            "release() on non-holder deleted the lock file, "
            "breaking flock coordination for the actual holder"
        )

        # Agent A should still be holding the lock
        assert Conch.is_active()

        # Clean up
        holder.release()

    def test_non_holder_release_preserves_flock_coordination(self):
        """After non-holder release(), a third agent cannot acquire.

        This tests the full failure scenario: if release() deletes the file,
        a third caller creates a new file (new inode) and gets its own flock,
        resulting in two agents holding 'exclusive' locks simultaneously.
        """
        # Agent A acquires
        agent_a = Conch(agent_name="agent_a")
        assert agent_a.try_acquire() is True

        # Agent B fails and releases (should be a no-op for the file)
        agent_b = Conch(agent_name="agent_b")
        assert agent_b.try_acquire() is False
        agent_b.release()

        # Agent C should NOT be able to acquire (Agent A still holds it)
        agent_c = Conch(agent_name="agent_c")
        assert agent_c.try_acquire() is False, (
            "Agent C acquired the conch while Agent A still holds it! "
            "This means release() destroyed the lock file and broke coordination."
        )

        # Clean up
        agent_a.release()


class TestConchFailClosed:
    """Fail-closed invariants: an enforcement verdict never comes from a
    swallowed exception, and stale clearance never unlinks a live holder.

    Both bug classes below let two processes hold a real, simultaneous
    "exclusive" lock -- the mechanism behind unchecked double-speak.
    """

    # ---- Constraint 1: an unreadable lock file means HELD, not free ----

    def test_unreadable_lock_refuses_acquire_when_no_flock_held(self, caplog):
        """A corrupt lock file must NOT read as "microphone free".

        The dangerous shape: a lock record whose holder holds NO flock, so the
        flock in try_acquire() cannot protect it -- a record left by a legacy
        writer, or by the flock-less arm of preempt_acquire(). Before the fix,
        the corrupt-read exception was swallowed into "no conversation active"
        and the contender acquired while a live holder was recorded.
        """
        Conch.LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        # A live, flock-less holder record, then corrupted in place.
        Conch.LOCK_FILE.write_text(json.dumps({
            "pid": os.getpid(), "agent": "holder",
            "acquired": datetime.now().isoformat(), "expires": None,
        }))
        Conch.LOCK_FILE.write_text('{"pid": 1234, "agen')

        contender = Conch(agent_name="contender")
        with caplog.at_level(logging.WARNING, logger="voicemode.conch"):
            acquired = contender.try_acquire()

        assert acquired is False, (
            "try_acquire() granted the conch off an unreadable lock file -- "
            "a swallowed parse error became a 'microphone is free' verdict"
        )
        assert any(
            "unreadable" in record.getMessage().lower()
            for record in caplog.records
        ), (
            "the unreadable-lock condition was silently absorbed; it must be "
            f"logged. Records seen: {[r.getMessage() for r in caplog.records]}"
        )

    def test_unreadable_lock_is_logged_even_when_flock_also_refuses(self, caplog):
        """The flock refusal must not mask the unreadable-lock condition.

        When the holder DID take an flock, the contender was already refused
        -- but silently, with no record that the lock file had been corrupted.
        A refusal whose reason is invisible cannot be operated on.
        """
        holder = Conch(agent_name="holder")
        assert holder.try_acquire() is True
        Conch.LOCK_FILE.write_text("not json at all")

        contender = Conch(agent_name="contender")
        with caplog.at_level(logging.WARNING, logger="voicemode.conch"):
            assert contender.try_acquire() is False
        assert any(
            "unreadable" in record.getMessage().lower()
            for record in caplog.records
        ), "unreadable lock file was never logged"

        holder.release()

    def test_absent_lock_file_still_reads_as_free(self, clean_conch):
        """Fail-closed must not over-reach: NO lock file is genuinely free.

        Guards against collapsing "file absent" into "file unreadable" --
        that would deadlock every first acquisition.
        """
        assert not Conch.LOCK_FILE.exists()
        assert Conch.read_lock_state()[0] == Conch.STATE_FREE
        assert Conch.is_held() is False

        conch = Conch(agent_name="first")
        assert conch.try_acquire() is True
        conch.release()

    def test_read_lock_state_reports_unreadable_for_corrupt_file(self):
        """The authoritative accessor names the condition instead of hiding it."""
        Conch.LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        Conch.LOCK_FILE.write_text("{broken")

        state, data = Conch.read_lock_state()
        assert state == Conch.STATE_UNREADABLE
        assert data is None
        assert Conch.is_held() is True, (
            "an unreadable lock must resolve to HELD for enforcement"
        )

        # The non-authoritative status reader keeps its documented contract.
        assert Conch.is_active() is False

    # ---- Constraint 2: stale clearance must not unlink a live holder ----

    def test_stale_timestamp_with_live_holder_refused_and_inode_stable(self):
        """A live holder past the stale threshold must NOT lose its inode.

        Before the fix, _check_and_clear_stale_lock() unlinked on timestamp
        expiry even with the holder alive. The holder kept a valid flock on
        the orphaned inode while the contender took a valid flock on a brand
        new one: two live processes, two real locks, both entitled to speak.

        The inode number is the discriminator -- if it changed, the swap
        happened regardless of what the return value says.
        """
        holder = Conch(agent_name="holder")
        assert holder.try_acquire() is True
        inode_before = os.stat(Conch.LOCK_FILE).st_ino

        # Backdate the holder's record IN PLACE: same inode, flock intact.
        # (write_text truncates and rewrites; flock is advisory and survives.)
        Conch.LOCK_FILE.write_text(json.dumps({
            "pid": os.getpid(),          # holder is alive -- it is us
            "agent": "holder",
            "acquired": "2000-01-01T00:00:00",  # way past any expiry
            "expires": None,
        }))
        assert os.stat(Conch.LOCK_FILE).st_ino == inode_before, (
            "test setup changed the inode; the assertion below would be void"
        )

        contender = Conch(agent_name="contender")
        acquired = contender.try_acquire()
        inode_after = os.stat(Conch.LOCK_FILE).st_ino

        assert inode_after == inode_before, (
            f"stale clearance unlinked a LIVE holder's lock: inode "
            f"{inode_before} -> {inode_after}. The holder still flocks the "
            f"old inode; the contender now flocks a new one -- two live "
            f"holders, both 'exclusive'."
        )
        assert acquired is False, (
            "contender acquired the conch while a live holder still flocked it"
        )

        record = json.loads(Conch.LOCK_FILE.read_text())
        assert record["agent"] == "holder", (
            "the live holder's record was overwritten by a refused contender"
        )

        holder.release()

    def test_stale_timestamp_takeover_happens_in_place_not_by_unlink(self):
        """A stale record with NO live flock is taken over on the same inode.

        This is the legitimate takeover case (holder gone or never flocked).
        It must still succeed -- and must do so by truncate-and-rewrite under
        the held descriptor, not by creating a new inode.
        """
        Conch.LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        Conch.LOCK_FILE.write_text(json.dumps({
            "pid": os.getpid(),                  # alive, but holds no flock
            "agent": "stuck_agent",
            "acquired": "2000-01-01T00:00:00",
            "expires": None,
        }))
        inode_before = os.stat(Conch.LOCK_FILE).st_ino

        new_conch = Conch(agent_name="new_agent")
        assert new_conch.try_acquire() is True
        assert os.stat(Conch.LOCK_FILE).st_ino == inode_before, (
            "takeover created a new inode instead of rewriting in place"
        )
        assert json.loads(Conch.LOCK_FILE.read_text())["agent"] == "new_agent"

        new_conch.release()

    def test_dead_holder_unlink_path_still_works(self):
        """Unlink stays permitted against a CONFIRMED-dead PID.

        Regression guard: the constraint-2 fix must not disable the
        dead-holder recovery path, which is the only reason the unlink exists.
        """
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        os.waitpid(pid, 0)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)

        Conch.LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        Conch.LOCK_FILE.write_text(json.dumps({
            "pid": pid,
            "agent": "dead_agent",
            "acquired": datetime.now().isoformat(),
            "expires": None,
        }))

        state, data = Conch.read_lock_state()
        assert state == Conch.STATE_FREE, "a confirmed-dead holder is free"
        assert data is not None and data["agent"] == "dead_agent"

        new_conch = Conch(agent_name="new_agent")
        assert new_conch.try_acquire() is True
        assert json.loads(Conch.LOCK_FILE.read_text())["agent"] == "new_agent"
        new_conch.release()

    def test_preempt_acquire_recovers_from_permanently_corrupt_lock(self):
        """The documented escape hatch out of a fail-closed corrupt lock.

        Because an unreadable lock now refuses acquisition forever, the
        deliberate preempt path must still be able to clear it -- otherwise
        one bad write wedges the microphone permanently.
        """
        Conch.LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        Conch.LOCK_FILE.write_text("}{ garbage")

        blocked = Conch(agent_name="blocked")
        assert blocked.try_acquire() is False

        rescuer = Conch(agent_name="rescuer")
        assert rescuer.preempt_acquire("rescuer") is True
        assert json.loads(Conch.LOCK_FILE.read_text())["agent"] == "rescuer"
        rescuer.release()

class TestConchSupersession:
    """Preemption must REVOKE the conch, never ORPHAN the old holder.

    `flock` cannot be revoked from outside its holder, so preempt_acquire()
    used to unlink the lock file instead -- leaving the superseded holder
    flocking a deleted inode where it could never learn it had lost. Two live
    processes, both entitled to speak. The monotonic epoch replaces that: the
    record is rewritten in place, and the old holder can SEE it changed.
    """

    def test_preempt_does_not_unlink_and_bumps_the_epoch(self):
        """The inode must survive preemption, and the epoch must advance."""
        holder = Conch(agent_name="holder")
        assert holder.try_acquire() is True
        inode_before = os.stat(Conch.LOCK_FILE).st_ino
        first_epoch = json.loads(Conch.LOCK_FILE.read_text())[Conch.EPOCH_KEY]
        assert first_epoch == 1

        preempter = Conch(agent_name="preempter")
        assert preempter.preempt_acquire("preempter") is True

        inode_after = os.stat(Conch.LOCK_FILE).st_ino
        assert inode_after == inode_before, (
            f"preempt_acquire() unlinked the lock: inode {inode_before} -> "
            f"{inode_after}. The superseded holder keeps its flock on the old "
            f"inode and can never learn it lost -- two live holders."
        )
        record = json.loads(Conch.LOCK_FILE.read_text())
        assert record["agent"] == "preempter"
        assert record["pid"] == os.getpid()
        assert record[Conch.EPOCH_KEY] == first_epoch + 1, (
            "preemption must advance the epoch, or supersession is undetectable"
        )

    def test_superseded_holder_detects_it_and_stands_down(self):
        """The holder-side half: the old holder must SEE that it lost."""
        holder = Conch(agent_name="holder")
        assert holder.try_acquire() is True
        assert holder.is_superseded() is False, "holder is not superseded yet"
        assert holder.should_yield() is False

        preempter = Conch(agent_name="preempter")
        assert preempter.preempt_acquire("preempter") is True

        assert holder.is_superseded() is True, (
            "the superseded holder cannot tell it lost the conch -- it will "
            "keep speaking alongside the preempter"
        )
        assert holder.should_yield() is True, (
            "should_yield() must report supersession so the listen loop ends"
        )

    def test_should_yield_is_a_valid_zero_arg_callable_for_the_listen_loop(self):
        """Contract check for converse.py's `listen_yield_check`.

        The record loop polls a bound zero-arg callable and breaks on True.
        """
        holder = Conch(agent_name="holder")
        assert holder.try_acquire() is True
        check = holder.should_yield  # exactly what converse.py passes
        assert check() is False

        Conch(agent_name="preempter").preempt_acquire("preempter")
        assert check() is True

    def test_converse_wires_should_yield_into_the_listen_loop(self):
        """Wiring guard: the holder-side check must stay connected.

        Both halves of this fix are useless alone -- an epoch nobody reads
        changes nothing. This pins the one expression in converse.py that
        connects them, so a future refactor cannot silently unwire it.
        """
        import voice_mode.tools.converse as conv
        source = Path(conv.__file__).read_text()
        assert "listen_yield_check" in source
        assert "conch.should_yield" in source, (
            "converse.py no longer passes conch.should_yield as the listen "
            "yield check -- a superseded holder will not stand down"
        )

    def test_superseded_holder_release_keeps_the_new_holders_record(self):
        """A superseded holder's release() must not delete the new record.

        Same damage class as a non-holder release: the preempter is left with
        no lock file while believing it holds the conch.
        """
        holder = Conch(agent_name="holder")
        assert holder.try_acquire() is True
        preempter = Conch(agent_name="preempter")
        assert preempter.preempt_acquire("preempter") is True

        holder.release()  # stands down

        assert Conch.LOCK_FILE.exists(), (
            "the superseded holder deleted the preempter's lock file"
        )
        assert json.loads(Conch.LOCK_FILE.read_text())["agent"] == "preempter"

    def test_unsuperseded_holder_release_still_removes_the_lock(self):
        """Guard the other direction: a real holder must still clean up."""
        holder = Conch(agent_name="holder")
        assert holder.try_acquire() is True
        holder.release()
        assert not Conch.LOCK_FILE.exists()

    def test_supersession_is_honoured_even_when_yield_is_disabled(self):
        """is_superseded is a FACT; is_wanted is a REQUEST.

        VOICEMODE_CONCH_YIELD_ENABLED=false opts out of polite hand-over. It
        must not opt out of standing down from a conch we no longer hold.
        """
        holder = Conch(agent_name="holder")
        assert holder.try_acquire() is True

        with patch("voice_mode.conch._get_yield_enabled", return_value=False):
            # A mere request is ignored when yielding is off...
            Conch.WANTED_FILE.parent.mkdir(parents=True, exist_ok=True)
            Conch.WANTED_FILE.write_text(json.dumps({
                "pid": 1, "agent": "asker",
                "requested": datetime.now().isoformat(),
            }))
            assert holder.should_yield() is False

            # ...but an actual supersession is not.
            Conch(agent_name="preempter").preempt_acquire("preempter")
            assert holder.should_yield() is True

    def test_preempt_recovers_a_corrupt_lock_without_unlinking(self):
        """The safe recovery path out of a fail-closed corrupt lock.

        try_acquire() now refuses an unreadable lock forever, so preemption is
        the escape hatch -- and it must be the SAFE one: same inode, epoch
        restarted at 1 because the old record's epoch is unreadable.
        """
        Conch.LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        Conch.LOCK_FILE.write_text("}{ garbage")
        inode_before = os.stat(Conch.LOCK_FILE).st_ino

        blocked = Conch(agent_name="blocked")
        assert blocked.try_acquire() is False

        rescuer = Conch(agent_name="rescuer")
        assert rescuer.preempt_acquire("rescuer") is True
        assert os.stat(Conch.LOCK_FILE).st_ino == inode_before, (
            "corrupt-lock recovery unlinked instead of rewriting in place"
        )
        record = json.loads(Conch.LOCK_FILE.read_text())
        assert record["agent"] == "rescuer"
        assert record[Conch.EPOCH_KEY] == 1
        rescuer.release()

    def test_epoch_is_monotonic_across_successive_in_place_takeovers(self):
        """Each takeover of the same inode advances the generation counter."""
        import fcntl as _fcntl
        seen = []
        for agent in ("a", "b", "c"):
            c = Conch(agent_name=agent)
            assert c.preempt_acquire(agent) is True
            seen.append(json.loads(Conch.LOCK_FILE.read_text())[Conch.EPOCH_KEY])
            # Drop our flock but LEAVE the record, so the next takeover reads it.
            c._acquired = False
            if c._fd is not None:
                _fcntl.flock(c._fd, _fcntl.LOCK_UN)
                os.close(c._fd)
                c._fd = None
        assert seen == [1, 2, 3], f"epoch not monotonic: {seen}"

    def test_is_superseded_is_false_for_a_process_that_never_held_it(self):
        """No claim, nothing to lose -- must not report supersession."""
        never = Conch(agent_name="never")
        assert never.is_superseded() is False
        assert never.should_yield() is False


class TestConchAcquireIsAtomic:
    """acquire() / `with Conch(...)` must not stomp the current holder."""

    def test_acquire_returns_false_when_another_process_holds_it(self):
        """acquire() used to be a bare write_text() that always returned True."""
        holder = Conch(agent_name="holder")
        assert holder.try_acquire() is True

        contender = Conch(agent_name="contender")
        assert contender.acquire() is False, (
            "acquire() overwrote a live holder's lock and reported success"
        )
        assert json.loads(Conch.LOCK_FILE.read_text())["agent"] == "holder"
        holder.release()

    def test_context_manager_refuses_to_enter_without_the_lock(self):
        """`with Conch(...)` must raise rather than run the body unlocked."""
        holder = Conch(agent_name="holder")
        assert holder.try_acquire() is True

        body_ran = False
        with pytest.raises(ConchUnavailable):
            with Conch(agent_name="intruder"):
                body_ran = True

        assert body_ran is False, (
            "the context-manager body ran while another agent held the conch"
        )
        assert json.loads(Conch.LOCK_FILE.read_text())["agent"] == "holder"
        holder.release()

    def test_acquire_takes_a_real_flock(self):
        """After acquire(), a second atomic attempt must be refused."""
        holder = Conch(agent_name="holder")
        assert holder.acquire() is True
        assert Conch(agent_name="other").try_acquire() is False
        holder.release()
