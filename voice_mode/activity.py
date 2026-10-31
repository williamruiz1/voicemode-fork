"""One timeline of what voice did: ~/.voicemode/activity.jsonl.

Every piece that can change what the user hears appends here: this server, the
convomode floor script, and the VibeDispatcher widget and helper. The other logs
each hold one piece; this file answers "what happened, in what order".

Line shape: {"ts": "<local ISO time>", "source": "voice", "event": "<name>", ...}.
Each line is one O_APPEND write, so writers in different processes don't interleave.
The file rolls to activity.jsonl.1 at MAX_BYTES. A write never blocks or fails a call.
Read it with `voice-activity.py` (claude-user-config, bin/founder-os).
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Optional

logger = logging.getLogger("voicemode.activity")

MAX_BYTES = 5 * 1024 * 1024
ACTIVE_DIR = os.path.expanduser("~/.voicemode/convomode-active")


def _path() -> str:
    return os.environ.get("VOICEMODE_ACTIVITY_LOG") or os.path.expanduser("~/.voicemode/activity.jsonl")


def _tab_name(client_pid: Optional[int]) -> Optional[str]:
    if not client_pid:
        return None
    try:
        with open(os.path.join(ACTIVE_DIR, str(client_pid))) as f:
            return f.read().strip() or None
    except OSError:
        return None


def log(event: str, **fields) -> None:
    """Append one event. The calling tab is added when a tool call is in progress."""
    try:
        from .live_transcript import resolve_client_pid

        client_pid = fields.pop("client_pid", None) or resolve_client_pid()
        entry = {
            "ts": datetime.datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "source": "voice",
            "event": event,
            "server_pid": os.getpid(),
        }
        if client_pid:
            entry["client_pid"] = client_pid
            tab = _tab_name(client_pid)
            if tab:
                entry["tab"] = tab
        entry.update(fields)
        path = _path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            if os.path.getsize(path) > MAX_BYTES:
                os.replace(path, path + ".1")
        except OSError:
            pass
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, (json.dumps(entry, ensure_ascii=False) + "\n").encode())
        finally:
            os.close(fd)
    except Exception as e:
        logger.debug(f"activity log write failed: {e}")
