"""Conch - Simple lock file for voice conversation coordination.

The Conch provides a lock file mechanism to indicate when a voice conversation
is active. This allows other processes (like sound effect hooks) to check
whether to suppress their audio output.

Lock file location: ~/.voicemode/conch

Usage:
    # As context manager (recommended)
    with Conch(agent_name="cora"):
        # ... voice conversation logic ...

    # Manual acquire/release
    conch = Conch()
    conch.acquire(agent_name="cora")
    try:
        # ... voice conversation logic ...
    finally:
        conch.release()

    # Check if converse is active (for external scripts)
    # NON-AUTHORITATIVE: answers False on any read error.
    if Conch.is_active():
        print("Someone is in a voice conversation")

    # Authoritative, FAIL-CLOSED check -- use this when the answer GATES an
    # action. An unreadable lock file reads as held, never as free.
    if Conch.is_held():
        print("Do not take the microphone")
"""

import fcntl
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("voicemode.conch")


def _log_conch_event(name: str, payload: dict) -> None:
    """Best-effort observability event.

    Safe no-op if the event logger is unset or the import fails (avoids
    circular-import / startup-order issues). Never raises.
    """
    try:
        from voice_mode.utils.event_logger import get_event_logger
        event_logger = get_event_logger()
        if event_logger:
            event_logger.log_event(name, payload)
    except Exception:
        pass


# Import config for lock expiry - deferred to avoid circular import
def _get_lock_expiry() -> float:
    """Get lock expiry from config, with fallback."""
    try:
        from voice_mode.config import CONCH_LOCK_EXPIRY
        return CONCH_LOCK_EXPIRY
    except ImportError:
        return 120.0  # Default 2 minutes


def _get_yield_enabled() -> bool:
    """Get the yieldable-listen switch from config, with fallback."""
    try:
        from voice_mode.config import CONCH_YIELD_ENABLED
        return CONCH_YIELD_ENABLED
    except ImportError:
        return True


def _get_wanted_fresh() -> float:
    """Get the conch-wanted freshness window from config, with fallback."""
    try:
        from voice_mode.config import CONCH_WANTED_FRESH
        return CONCH_WANTED_FRESH
    except ImportError:
        return 5.0


class ConchUnavailable(RuntimeError):
    """Raised when a `with Conch(...)` block cannot take the conch.

    The context manager must never enter without holding the lock: a body that
    believes it owns the microphone while another agent actually does is the
    failure this whole module exists to prevent.
    """


class Conch:
    """Simple lock file for voice conversation coordination.

    Creates a lock file at ~/.voicemode/conch when a voice conversation
    is active. The lock file contains:
    - pid: Process ID of the lock holder (for stale lock detection)
    - agent: Name of the agent holding the lock
    - acquired: ISO timestamp when lock was acquired
    - expires: Optional expiry time (reserved for future use)
    - epoch: monotonic generation counter for THIS lock file. Bumped by every
      takeover of the same inode, so a holder can detect that it was
      superseded in place (see is_superseded). Records written before this
      field existed read as epoch 0.
    """

    LOCK_FILE = Path.home() / ".voicemode" / "conch"

    # Authoritative lock states returned by read_lock_state(). Anything other
    # than STATE_FREE means "treat the microphone as taken."
    STATE_FREE = "free"
    STATE_HELD = "held"
    STATE_UNREADABLE = "unreadable"

    # Record key holding the monotonic generation counter (see is_superseded).
    EPOCH_KEY = "epoch"

    def __init__(self, agent_name: Optional[str] = None):
        """Initialize Conch with optional agent name.

        Args:
            agent_name: Name of the agent (e.g., "cora"). Used for debugging/logging.
        """
        self.agent_name = agent_name
        self._acquired = False
        self._fd = None  # File descriptor for flock
        self._acquire_time = None  # Track when acquired
        self._epoch = None  # Generation we wrote; None until acquired

    def acquire(self, agent_name: Optional[str] = None) -> bool:
        """Acquire the conch. Thin alias for the atomic try_acquire().

        This used to be a bare write_text() that took no lock and always
        returned True -- so it silently overwrote whoever held the microphone,
        and `with Conch(...)` (which calls it) stomped the current holder while
        reporting success. It now routes through the same flock-protected path
        as every other acquisition and returns a REAL boolean.

        Args:
            agent_name: Override the agent name set in __init__

        Returns:
            True if the lock was acquired, False if another process holds it
        """
        return self.try_acquire(agent_name)

    def try_acquire(self, agent_name: Optional[str] = None) -> bool:
        """Atomically try to acquire the conch.

        Uses fcntl.flock() for true atomic locking across processes.
        Also handles stale locks: if a lock is older than CONCH_LOCK_EXPIRY
        seconds, it will be forcibly released and re-acquired.

        Args:
            agent_name: Name of the agent acquiring the lock

        Returns:
            True if lock acquired, False if already held by another process
        """
        if self._acquired:
            return True  # Already holding it

        agent = agent_name or self.agent_name or "unknown"
        self.LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)

        # First check (FAIL CLOSED): a lock file that EXISTS but cannot be
        # read or parsed is not evidence that the microphone is free. Refuse,
        # and say so -- never let a swallowed exception hand out a second
        # "exclusive" lock. An ABSENT lock file is a different thing and is
        # still genuinely free.
        #
        # Recovery from a permanently-corrupt lock file is the deliberate
        # preempt_acquire() path (the waiter's hard timeout), not a silent
        # grab here.
        state, _ = self.read_lock_state()
        if state == self.STATE_UNREADABLE:
            logger.warning(
                "Conch lock file %s exists but is unreadable/unparseable; "
                "refusing to acquire (fail closed). Use the preempt path to "
                "clear it.", self.LOCK_FILE
            )
            _log_conch_event("CONCH_UNREADABLE_LOCK", {
                "pid": os.getpid(),
                "agent": agent,
                "lock_file": str(self.LOCK_FILE),
            })
            return False

        # Second check: is there a stale lock we can forcibly clear?
        self._check_and_clear_stale_lock()

        try:
            # Open file for read/write, create if doesn't exist
            self._fd = os.open(str(self.LOCK_FILE), os.O_CREAT | os.O_RDWR, 0o644)

            # Try to get exclusive lock (non-blocking)
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Got lock - bump the generation counter and write our info. The
            # epoch is read from the record we are about to replace (we hold
            # the flock, so nobody can be mid-write), which makes it monotonic
            # per inode and lets a superseded holder notice (see
            # is_superseded).
            prev_epoch = self._read_epoch_from_fd(self._fd)
            self._acquire_time = datetime.now()
            self._epoch = prev_epoch + 1
            data = {
                "pid": os.getpid(),
                "agent": agent,
                "acquired": self._acquire_time.isoformat(),
                "expires": None,
                self.EPOCH_KEY: self._epoch,
            }

            os.ftruncate(self._fd, 0)
            os.lseek(self._fd, 0, os.SEEK_SET)
            os.write(self._fd, json.dumps(data, indent=2).encode())
            os.fsync(self._fd)  # Ensure data is written

            self._acquired = True
            return True

        except (BlockingIOError, OSError) as e:
            # Lock held by another process, or other OS error
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                self._fd = None
            self._epoch = None
            return False

    @staticmethod
    def _read_epoch_from_fd(fd: int) -> int:
        """Read the monotonic epoch out of the record currently in `fd`.

        Returns 0 when the file is empty, unparseable, or was written by a
        version predating the epoch field -- so the next writer starts at 1.
        """
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            raw = os.read(fd, 65536)
        except OSError:
            return 0
        if not raw:
            return 0
        try:
            data = json.loads(raw.decode())
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return 0
        if not isinstance(data, dict):
            return 0
        epoch = data.get(Conch.EPOCH_KEY)
        if isinstance(epoch, int) and not isinstance(epoch, bool) and epoch >= 0:
            return epoch
        return 0

    def _check_and_clear_stale_lock(self) -> None:
        """Check for and clear stale locks.

        Two paths:
        1. Dead-holder fast-fail: if the recorded PID no longer exists,
           unlink the lock immediately. This runs even when timestamp-based
           expiry is disabled (CONCH_LOCK_EXPIRY <= 0) -- a dead holder is
           unambiguously stale.
        2. Timestamp-based expiry: if the lock is older than
           CONCH_LOCK_EXPIRY seconds, note it and let try_acquire() take the
           lock over IN PLACE. This handles the case where the holder is
           alive but stuck -- WITHOUT unlinking (see below).

        Unlink is permitted ONLY against a confirmed-dead PID (path 1).
        Unlinking a lock whose holder may still be alive creates a NEW INODE:
        the holder keeps a valid flock on the orphaned inode while the next
        caller takes a valid flock on the fresh one, so two live processes
        each hold a real "exclusive" lock and both are entitled to speak.
        That inode swap was the mechanism behind double-speak on long turns.
        """
        if not self.LOCK_FILE.exists():
            return

        try:
            data = json.loads(self.LOCK_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return

        # Fast-fail on dead holder -- no need to wait for timestamp expiry.
        pid = data.get("pid")
        if pid is not None:
            try:
                os.kill(pid, 0)
                # Process is alive -- fall through to timestamp check.
            except ProcessLookupError:
                # Holder is dead -- clear the lock immediately.
                stale_agent = data.get("agent", "unknown")
                try:
                    self.LOCK_FILE.unlink()
                except OSError:
                    pass
                # Best-effort observability event. Safe no-op if logger unset
                # or import fails (avoids circular-import / startup-order issues).
                try:
                    from voice_mode.utils.event_logger import get_event_logger
                    event_logger = get_event_logger()
                    if event_logger:
                        event_logger.log_event("CONCH_DEAD_HOLDER_CLEARED", {
                            "stale_pid": pid,
                            "stale_agent": stale_agent,
                        })
                except Exception:
                    pass
                return
            except PermissionError:
                # Process exists but we can't signal it -- treat as alive.
                pass
            except (TypeError, OSError):
                # PID isn't a valid int or other OS error -- skip dead-PID path,
                # fall through to timestamp check.
                pass

        # Timestamp-based staleness.
        #
        # We deliberately do NOT unlink here. Control only reaches this point
        # when the recorded holder was NOT confirmed dead above -- it is alive,
        # unsignalable, or unknown. Unlinking in that state swaps the inode out
        # from under a possibly-live flock holder and produces two
        # simultaneously-valid locks (see this method's docstring).
        #
        # Instead, takeover happens IN PLACE: try_acquire() opens this SAME
        # inode and attempts flock(LOCK_EX | LOCK_NB). If the holder is truly
        # gone -- or never took an flock -- we win the lock and rewrite the
        # holder record under the held descriptor, same inode. If a live holder
        # still holds the flock we are refused, which is the correct answer.
        # Deliberately breaking a live holder's flock remains possible, but only
        # through the explicit preempt_acquire() path.
        lock_expiry = _get_lock_expiry()
        if lock_expiry <= 0:
            return  # Stale lock detection disabled

        acquired_str = data.get("acquired")
        if not acquired_str:
            return

        try:
            acquired_time = datetime.fromisoformat(acquired_str)
        except ValueError:
            return

        age_seconds = (datetime.now() - acquired_time).total_seconds()
        if age_seconds > lock_expiry:
            logger.debug(
                "Conch lock held by pid %s (agent %s) is %.1fs old (expiry "
                "%.1fs); attempting in-place takeover, not unlinking.",
                pid, data.get("agent", "unknown"), age_seconds, lock_expiry,
            )
            _log_conch_event("CONCH_STALE_TIMESTAMP_LIVE_HOLDER", {
                "pid": os.getpid(),
                "stale_pid": pid,
                "stale_agent": data.get("agent", "unknown"),
                "age_seconds": age_seconds,
                "lock_expiry": lock_expiry,
            })

    def release(self) -> float:
        """Release the lock and return seconds held.

        Only removes the lock file if this instance actually acquired the lock.
        Removing it when not acquired would destroy the lock held by another
        process (they'd be flocking different inodes after re-creation).

        Returns:
            Seconds the lock was held, or 0.0 if not acquired
        """
        held_seconds = 0.0

        if self._acquire_time:
            held_seconds = (datetime.now() - self._acquire_time).total_seconds()

        # Are we STILL the recorded holder? A preempter may have superseded us
        # IN PLACE -- same inode, new epoch. Unlinking then would delete the
        # NEW holder's record, which is the same class of damage as a
        # non-holder release. Decide before clearing our own state.
        superseded = self._acquired and self.is_superseded()
        if superseded:
            logger.info(
                "Conch release: we were superseded (epoch %s no longer current)"
                " -- dropping our flock but leaving the new holder's record.",
                self._epoch,
            )
            _log_conch_event("CONCH_RELEASE_AFTER_SUPERSEDED", {
                "pid": os.getpid(),
                "agent": self.agent_name or "unknown",
                "our_epoch": self._epoch,
            })

        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

        # Only remove the lock file if we actually acquired the lock AND are
        # still the recorded holder. If we didn't acquire it, or were
        # superseded, the file belongs to another process.
        if self._acquired and not superseded and self.LOCK_FILE.exists():
            try:
                self.LOCK_FILE.unlink()
            except OSError:
                pass

        self._acquired = False
        self._acquire_time = None
        self._epoch = None

        return held_seconds

    @classmethod
    def read_lock_state(cls) -> Tuple[str, Optional[dict]]:
        """Authoritative, FAIL-CLOSED read of the lock file.

        This is the accessor that enforcement decisions must use. Unlike
        is_active() / get_holder() -- non-authoritative status readers that
        answer False/None on any error -- this method never lets a swallowed
        exception become a "the microphone is free" verdict.

        Returns (state, data):
          STATE_FREE       The lock file does not exist (data None), or its
                           recorded holder PID is CONFIRMED dead (data is the
                           parsed record, so callers can log who it was).
          STATE_HELD       A readable record whose holder is alive, or alive
                           but unsignalable. data is the record.
          STATE_UNREADABLE The file EXISTS but could not be statted, read or
                           parsed, or carries no usable holder PID. Resolves
                           to held for enforcement purposes: an unreadable
                           lock is never evidence that nobody is speaking.

        Note the deliberate distinction: an ABSENT lock file is FREE; a
        PRESENT-but-unreadable one is not. Collapsing those two cases is the
        fail-open bug this method exists to prevent.

        Note also the deliberate divergence from is_active(): a holder whose
        timestamp has expired but whose process is still ALIVE reads HELD
        here. is_active() calls that inactive; for an enforcement verdict,
        "alive but slow" is still a live microphone holder.
        """
        try:
            if not cls.LOCK_FILE.exists():
                return cls.STATE_FREE, None
        except OSError:
            # Cannot even stat the path -- refuse to call it free.
            return cls.STATE_UNREADABLE, None

        try:
            raw = cls.LOCK_FILE.read_text()
        except (OSError, ValueError):
            return cls.STATE_UNREADABLE, None

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return cls.STATE_UNREADABLE, None

        if not isinstance(data, dict):
            return cls.STATE_UNREADABLE, None

        pid = data.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            # No usable holder PID: liveness is undeterminable, so the lock
            # cannot be SHOWN to be free. (pid <= 0 is also rejected because
            # os.kill would address a process group, not a process.)
            return cls.STATE_UNREADABLE, None

        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            # Confirmed dead holder -- genuinely free.
            return cls.STATE_FREE, data
        except PermissionError:
            # Process exists but is owned by another user -- alive.
            return cls.STATE_HELD, data
        except OSError:
            # Liveness undeterminable -- fail closed.
            return cls.STATE_UNREADABLE, data

        return cls.STATE_HELD, data

    @classmethod
    def is_held(cls) -> bool:
        """Authoritative, fail-closed answer to "is the microphone taken?"

        True unless the lock file is genuinely absent or its holder is
        confirmed dead. Use this -- not is_active() -- anywhere the answer
        gates an action rather than merely reporting status.
        """
        return cls.read_lock_state()[0] != cls.STATE_FREE

    def is_superseded(self) -> bool:
        """True if this holder has ALREADY lost the conch to a preempter.

        `flock` cannot be revoked from outside the holder, so preemption works
        by rewriting the holder record IN PLACE under a new epoch (see
        preempt_acquire). That makes supersession something the superseded
        holder can DETECT -- which is the whole point. Before the epoch existed,
        preemption unlinked the lock file instead, leaving the old holder
        flocking an orphaned inode where it could never learn it had lost:
        two live processes, both believing they held an exclusive lock.

        Fail-closed -- every uncertain case reads as superseded, because a
        holder that cannot demonstrate it still owns the conch must stand down:
          - the record is unreadable      -> superseded
          - the record is gone            -> superseded
          - the record names another pid  -> superseded
          - the epoch moved on            -> superseded

        Returns False when we never acquired the lock (nothing to lose).
        """
        if not self._acquired or self._epoch is None:
            return False

        state, data = self.read_lock_state()
        if state == self.STATE_UNREADABLE:
            return True
        if not isinstance(data, dict):
            # No record at all -- we are not the holder any more.
            return True
        if data.get("pid") != os.getpid():
            return True
        if data.get(self.EPOCH_KEY, 0) != self._epoch:
            return True
        return False

    def should_yield(self) -> bool:
        """Holder-side stand-down check, polled while a held listen is IDLE.

        COMPOSES the two reasons a holder ends a listen early -- it does not
        replace either:

        1. is_superseded() -- MANDATORY. We have already lost the conch; this
           is a fact, not a request, so it is honoured regardless of the
           yieldable-listen config switch.
        2. is_wanted() -- COOPERATIVE. Another agent is ASKING for the mic
           (transient, freshness-gated). Honoured only when yieldable listen is
           enabled, so VOICEMODE_CONCH_YIELD_ENABLED=false still opts out of
           polite hand-over exactly as before.
        """
        if self.is_superseded():
            logger.info(
                "Conch superseded in place (our epoch %s is no longer current)"
                " -- standing down from this listen.", self._epoch,
            )
            return True
        return _get_yield_enabled() and self.is_wanted()

    @classmethod
    def is_active(cls) -> bool:
        """Check if a voice conversation is currently active.

        NON-AUTHORITATIVE status reader: answers False on any read error and
        treats a stale timestamp as inactive. Do NOT gate an enforcement
        decision on this -- use is_held() / read_lock_state(), which fail
        closed instead.

        A conversation is considered active if:
        1. The lock file exists
        2. The PID in the file corresponds to a running process
        3. The lock is not stale (acquired within CONCH_LOCK_EXPIRY seconds)

        Returns:
            True if converse is active, False otherwise
        """
        if not cls.LOCK_FILE.exists():
            return False

        try:
            data = json.loads(cls.LOCK_FILE.read_text())
            pid = data.get("pid")

            if pid is None:
                return False

            # Check if process is alive (signal 0 doesn't actually send a signal)
            os.kill(pid, 0)

            # Check if lock is stale based on timestamp
            lock_expiry = _get_lock_expiry()
            if lock_expiry > 0:
                acquired_str = data.get("acquired")
                if acquired_str:
                    acquired_time = datetime.fromisoformat(acquired_str)
                    age_seconds = (datetime.now() - acquired_time).total_seconds()
                    if age_seconds > lock_expiry:
                        # Lock is stale - consider it inactive
                        return False

            return True
        except (json.JSONDecodeError, ProcessLookupError, PermissionError, OSError, ValueError):
            # JSON invalid, process dead, no permission to signal, or invalid timestamp
            return False

    @classmethod
    def get_holder(cls) -> Optional[dict]:
        """Get information about the current lock holder.

        NON-AUTHORITATIVE status reader (for display and logging): returns
        None on any read error, so "None" must never be read as "nobody holds
        the conch." Use read_lock_state() when the answer gates an action.

        Returns:
            Dict with lock info if active, None otherwise
        """
        if not cls.is_active():
            return None

        try:
            return json.loads(cls.LOCK_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    # ---- Yield-request channel (transient + preemptible audio focus) ----
    #
    # A waiter that wants the mic writes/refreshes ~/.voicemode/conch-wanted.
    # A holder that is merely LISTENING (idle) polls is_wanted() and ends its
    # listen early to hand the mic over; a holder that is SPEAKING ignores it
    # (TTS is never preempted mid-utterance). The request goes stale after
    # CONCH_WANTED_FRESH seconds (waiters refresh every poll), so a crashed
    # waiter can't force yields forever.

    WANTED_FILE = Path.home() / ".voicemode" / "conch-wanted"

    @classmethod
    def request_yield(cls, agent_name: Optional[str] = None) -> None:
        """Write/refresh the conch-wanted request as this process.

        Best-effort: any failure is swallowed (the waiter still has the
        timeout/preempt path as its backstop).
        """
        try:
            cls.WANTED_FILE.parent.mkdir(parents=True, exist_ok=True)
            cls.WANTED_FILE.write_text(json.dumps({
                "pid": os.getpid(),
                "agent": agent_name or "unknown",
                "requested": datetime.now().isoformat(),
            }, indent=2))
        except OSError:
            pass

    @classmethod
    def clear_yield_request(cls) -> None:
        """Remove this process's conch-wanted request (no-op if it isn't ours).

        Only the requester clears its own request — clearing another waiter's
        fresh request would silently cancel their preemption.
        """
        try:
            data = json.loads(cls.WANTED_FILE.read_text())
            if data.get("pid") == os.getpid():
                cls.WANTED_FILE.unlink()
        except (json.JSONDecodeError, OSError):
            pass

    @classmethod
    def is_wanted(cls) -> bool:
        """True if another live process is currently requesting the conch.

        A request counts only when ALL of:
        - the wanted file exists and parses,
        - the requester is not this process,
        - the requester PID is still alive,
        - the request timestamp is within CONCH_WANTED_FRESH seconds.

        Fails open to False on any error (never yields on a bad read).
        """
        try:
            data = json.loads(cls.WANTED_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return False

        pid = data.get("pid")
        if not isinstance(pid, int) or pid == os.getpid():
            return False

        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            # Requester is dead — reap its stale request.
            try:
                cls.WANTED_FILE.unlink()
            except OSError:
                pass
            return False
        except (PermissionError, OSError):
            pass  # Exists but not signalable — treat as alive.

        requested_str = data.get("requested")
        if not requested_str:
            return False
        try:
            requested_time = datetime.fromisoformat(requested_str)
        except ValueError:
            return False

        age = (datetime.now() - requested_time).total_seconds()
        return age <= _get_wanted_fresh()

    def preempt_acquire(self, agent_name: Optional[str] = None) -> bool:
        """Supersede the current holder IN PLACE -- never unlink.

        Used by the waiter's hard-timeout path (the old CONCH_TIMEOUT failure
        becomes preempt-and-acquire). Callers are responsible for the
        never-mid-utterance grace (checking the speaking flag) BEFORE
        preempting.

        REVOKE, DO NOT ORPHAN. This used to unlink the lock file, and its own
        docstring named the consequence: "the stuck holder keeps its flock on
        the old inode, but new acquisitions get a fresh file." That is two live
        processes each holding a real exclusive lock, and the superseded one
        never finds out -- the double-speak mechanism. Instead we bump the
        monotonic epoch and rewrite the holder record on the SAME inode. The
        superseded holder sees the epoch change via is_superseded() /
        should_yield() during its hold and stands down; its release() then
        leaves our record alone.

        `flock` cannot be revoked from outside its holder, so a preempter may
        end up owning the record without owning the flock. We make ONE
        non-blocking attempt to take it (a holder standing down drops it
        quickly). Failing that we hold by epoch alone: after a preemption the
        EPOCH is what entitles us to speak. If a late flock winner then takes
        over, it bumps the epoch in turn -- so the loser of that race can
        detect it too. The race becomes detectable rather than silent.

        Returns True once we are the recorded holder.
        """
        if self._acquired:
            return True

        agent = agent_name or self.agent_name or "unknown"
        self.LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Nothing to preempt -- this is an ordinary acquisition.
        if not self.LOCK_FILE.exists():
            return self.try_acquire(agent)

        old_holder = None
        try:
            parsed = json.loads(self.LOCK_FILE.read_text())
            if isinstance(parsed, dict):
                old_holder = parsed
        except (json.JSONDecodeError, OSError, ValueError):
            # Corrupt record. Since try_acquire() now fails closed on an
            # unreadable lock, this path is also its recovery route -- so it
            # must work without being able to parse what it replaces.
            logger.warning(
                "Conch preempt: existing lock record at %s is unreadable; "
                "superseding it.", self.LOCK_FILE,
            )

        prev_epoch = 0
        if old_holder is not None:
            raw_epoch = old_holder.get(self.EPOCH_KEY)
            if isinstance(raw_epoch, int) and not isinstance(raw_epoch, bool) and raw_epoch >= 0:
                prev_epoch = raw_epoch

        # Open the EXISTING inode. Never O_TRUNC here and never unlink -- the
        # inode identity is what lets the superseded holder see our epoch.
        try:
            fd = os.open(str(self.LOCK_FILE), os.O_CREAT | os.O_RDWR, 0o644)
        except OSError as e:
            logger.warning("Conch preempt could not open %s: %s", self.LOCK_FILE, e)
            return False

        have_flock = False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            have_flock = True
            # We hold the flock, so the on-disk record cannot be mid-write:
            # prefer the epoch read under the lock.
            prev_epoch = max(prev_epoch, self._read_epoch_from_fd(fd))
        except (BlockingIOError, OSError):
            # The superseded holder still flocks this inode. It stands down on
            # its next should_yield() poll; we proceed by epoch.
            pass

        self._acquire_time = datetime.now()
        self._epoch = prev_epoch + 1
        data = {
            "pid": os.getpid(),
            "agent": agent,
            "acquired": self._acquire_time.isoformat(),
            "expires": None,
            self.EPOCH_KEY: self._epoch,
        }

        try:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, json.dumps(data, indent=2).encode())
            os.fsync(fd)
        except OSError as e:
            logger.warning("Conch preempt could not write the holder record: %s", e)
            try:
                os.close(fd)
            except OSError:
                pass
            self._epoch = None
            self._acquire_time = None
            return False

        if have_flock:
            self._fd = fd
        else:
            # Do not keep a descriptor carrying no lock -- release() would try
            # to unflock it. The epoch is our claim in this case.
            try:
                os.close(fd)
            except OSError:
                pass
            self._fd = None

        self._acquired = True
        logger.info(
            "Conch preempted in place: epoch %s -> %s, superseded pid %s "
            "(agent %s), flock_held=%s",
            prev_epoch, self._epoch,
            (old_holder or {}).get("pid"), (old_holder or {}).get("agent"),
            have_flock,
        )
        _log_conch_event("CONCH_PREEMPT", {
            "pid": os.getpid(),
            "agent": agent,
            "preempted_pid": (old_holder or {}).get("pid"),
            "preempted_agent": (old_holder or {}).get("agent"),
            "prev_epoch": prev_epoch,
            "epoch": self._epoch,
            "flock_held": have_flock,
            "in_place": True,
        })
        return True

    def __enter__(self):
        """Context manager entry - acquire the lock, or refuse to enter.

        Raises ConchUnavailable rather than entering the body without holding
        the conch. Previously acquire() always returned True and this ignored
        it, so the body ran believing it owned a microphone another agent held.
        """
        if not self.acquire():
            holder = self.get_holder() or {}
            raise ConchUnavailable(
                "Could not acquire the conch"
                + (f"; held by {holder.get('agent', 'unknown')} "
                   f"(pid {holder.get('pid')})" if holder else "")
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release the lock."""
        self.release()
        return False  # Don't suppress exceptions
