"""Pause = cut speech now, finish the rest after Proceed.

When convomode is paused (``~/.voicemode/pause.flag``) mid-message, the TTS
playback paths stop within ~one audio chunk and mark their metrics
``paused=True`` (see ``voice_mode/streaming.py``). ``converse`` then calls
:func:`resume_after_pause`, which blocks (never listening) until the flag is
removed, then replays the not-yet-heard remainder via a caller-supplied
``replay`` coroutine, before the normal listen starts.
"""

import asyncio
import logging
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from . import audio_player
from . import client_identity

logger = logging.getLogger("voicemode.pause_resume")

# Comfortably under Claude Code's 30-minute (1800s) stdio MCP idle timeout, so
# the "[PAUSED — timed out ...]" result returns before the host gives up.
DEFAULT_PAUSE_MAX_SECONDS = 1500.0
# A message can be paused again during the replay; bound the loop.
MAX_REPLAYS = 5


def pause_max_seconds() -> float:
    """VOICEMODE_PAUSE_MAX_SECONDS, read at call time (default 1500)."""
    try:
        return float(os.environ.get("VOICEMODE_PAUSE_MAX_SECONDS", DEFAULT_PAUSE_MAX_SECONDS))
    except ValueError:
        return DEFAULT_PAUSE_MAX_SECONDS


FLOOR_QUEUE_PATH = os.path.expanduser("~/.voicemode/floor-queue.json")


def _ancestor_pids(pid: Optional[int] = None, limit: int = 12) -> set:
    """This process plus its parent chain (the MCP server is a child of the
    Claude session whose pid the floor queue keys on)."""
    pid = pid or os.getpid()
    chain = set()
    for _ in range(limit):
        if pid <= 1 or pid in chain:
            break
        chain.add(pid)
        try:
            out = subprocess.run(["/bin/ps", "-o", "ppid=", "-p", str(pid)],
                                 capture_output=True, text=True, timeout=2).stdout.strip()
            pid = int(out)
        except Exception:
            break
    return chain


def floor_holder_pid() -> Optional[int]:
    """pid of the current floor holder from floor-queue.json, or None if the
    floor is free / the file is missing or unreadable (fail-open)."""
    try:
        with open(FLOOR_QUEUE_PATH) as fh:
            holder = json.load(fh).get("floor_holder")
        pid = int((holder or {}).get("pid") or 0)
        return pid or None
    except Exception:
        return None


def we_hold_floor() -> Optional[bool]:
    """True/False when a holder is recorded (is it this session?); None if the
    floor has no holder or the state is unreadable.

    Under the shared HTTP daemon this process is no session's child, so the
    session is the bound MCP client (client_identity); with no client bound
    there the answer is unknown (None), never a guess from the daemon's pid."""
    holder = floor_holder_pid()
    if holder is None:
        return None
    client_pid = client_identity.current_client_pid()
    if client_pid is None and client_identity.SHARED_MODE:
        return None
    return holder in _ancestor_pids(client_pid)


def floor_handed_off(held_at_start: bool) -> bool:
    """True only if this session held the floor when it started speaking and a
    DIFFERENT holder is recorded now (a jump/advance displaced it)."""
    return bool(held_at_start) and we_hold_floor() is False


@dataclass
class PauseOutcome:
    timed_out: bool = False
    replayed: bool = False
    skipped_by_send: bool = False
    handed_off: bool = False
    undelivered_text: str = ""
    paused_for_seconds: float = 0.0
    final_metrics: Optional[dict] = field(default=None)


async def wait_for_resume(max_seconds: float, poll: float = 0.1) -> bool:
    """Block until the pause flag clears. False if `max_seconds` elapsed first."""
    deadline = time.monotonic() + max_seconds  # wall clock, not summed sleeps
    while audio_player.convomode_paused():
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(poll)
    return True


def _consume_send(turn_end_signal_path: Optional[str]) -> bool:
    """True (and consume it) if Send was pressed during the pause."""
    if not turn_end_signal_path:
        return False
    try:
        if os.path.exists(turn_end_signal_path):
            os.remove(turn_end_signal_path)
            return True
    except OSError:
        pass
    return False


async def resume_after_pause(
    message: str,
    tts_metrics: Optional[dict],
    replay: Callable[[str], Awaitable[Optional[dict]]],
    turn_end_signal_path: Optional[str] = None,
    max_seconds: Optional[float] = None,
    held_floor_at_start: bool = False,
) -> Optional[PauseOutcome]:
    """Handle a playback cut by pause. Returns None if playback wasn't paused.

    Send (turn-end.signal) pressed during the pause means "skip the rest": it is
    consumed here and the remainder is not replayed (never silently dropped and
    never left to be deleted as stale by the listen that follows).
    """
    if not tts_metrics or not tts_metrics.get("paused"):
        return None

    limit = pause_max_seconds() if max_seconds is None else max_seconds
    outcome = PauseOutcome(final_metrics=tts_metrics)
    current = tts_metrics
    started = time.monotonic()

    for _ in range(MAX_REPLAYS):
        remaining_budget = max(0.0, limit - (time.monotonic() - started))
        if not await wait_for_resume(remaining_budget):
            outcome.timed_out = True
            break
        if _consume_send(turn_end_signal_path):
            outcome.skipped_by_send = True
            break
        remaining = current.get("undelivered_text")
        if remaining is None:
            remaining = message  # unknown split: replay all rather than lose any
        # A floor jump/advance pulses pause.flag to cut the displaced holder; the
        # flag clearing then is NOT a Proceed. Don't talk over the new holder.
        if floor_handed_off(held_floor_at_start):
            outcome.handed_off = True
            outcome.undelivered_text = remaining
            break
        if not remaining.strip():
            break
        logger.info("Pause cleared; replaying the not-yet-heard remainder")
        new_metrics = await replay(remaining)
        outcome.replayed = True
        if not new_metrics:
            break
        outcome.final_metrics = new_metrics
        if not new_metrics.get("paused"):
            break
        current = new_metrics
    outcome.paused_for_seconds = time.monotonic() - started
    return outcome
