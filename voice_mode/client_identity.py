"""Per-client identity for a shared (HTTP) voicemode server.

Under stdio every Claude Code session spawns its own voicemode process, so
"who am I" was always ``os.getpid()`` (the conch) or "my parent chain" (the
convomode floor). One shared daemon serving every session breaks both: all
sessions are the same process. This module gives each MCP request its own
identity instead.

Identity is an *owner token*, ``"pid:<N>"``:

- stdio (no client bound): N is this process's pid -- exactly what the conch
  compared before, so single-process behaviour is unchanged.
- shared HTTP daemon: N is the pid of the client process on the other end of
  the TCP connection (a Claude Code session), found from the kernel's socket
  table. That is the same pid the convomode floor queue keys on, and it needs
  no client configuration. A client may instead send ``X-Voicemode-Client-Pid``.

The token namespace is shared with legacy records (which only carry ``pid``),
so a daemon and old per-session stdio servers coordinate correctly while both
exist during migration.
"""

from __future__ import annotations

import contextvars
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("voicemode.client_identity")

CLIENT_PID_HEADER = "x-voicemode-client-pid"


@dataclass(frozen=True)
class ClientIdentity:
    owner: str
    pid: Optional[int]


_current: contextvars.ContextVar[Optional[ClientIdentity]] = contextvars.ContextVar(
    "voicemode_client_identity", default=None
)

# True once this process serves several clients (set by `voicemode serve`).
# Identity-dependent checks that run without a bound client then answer
# "unknown" instead of guessing from this process's own pid.
SHARED_MODE = False


def owner_for_pid(pid: int) -> str:
    return f"pid:{int(pid)}"


def bind(identity: Optional[ClientIdentity]) -> contextvars.Token:
    return _current.set(identity)


def unbind(token: contextvars.Token) -> None:
    _current.reset(token)


def current() -> Optional[ClientIdentity]:
    return _current.get()


def current_owner() -> str:
    ident = _current.get()
    return ident.owner if ident else owner_for_pid(os.getpid())


def current_client_pid() -> Optional[int]:
    """The bound client's pid, or None when no client is bound."""
    ident = _current.get()
    return ident.pid if ident else None


def record_owner(data: dict) -> Optional[str]:
    """Owner token of a conch/wanted record (legacy records carry only pid)."""
    owner = data.get("owner")
    if isinstance(owner, str) and owner:
        return owner
    pid = data.get("pid")
    if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0:
        return owner_for_pid(pid)
    return None


# ---- Resolving the client behind a loopback TCP connection ----

_PEER_CACHE_TTL = 30.0
_peer_cache: dict[int, tuple[int, float]] = {}


def _lsof_peer_pid(client_port: int, server_port: int) -> Optional[int]:
    """pid owning the client end (127.0.0.1:client_port -> :server_port)."""
    try:
        out = subprocess.run(
            ["/usr/sbin/lsof", "-nP", f"-iTCP@127.0.0.1:{client_port}",
             "-sTCP:ESTABLISHED", "-Fpn"],
            capture_output=True, text=True, timeout=3,
        ).stdout
    except Exception as e:
        logger.debug(f"lsof peer lookup failed: {e}")
        return None
    me = os.getpid()
    pid = None
    want = f"127.0.0.1:{client_port}->127.0.0.1:{server_port}"
    for line in out.splitlines():
        if line.startswith("p"):
            try:
                pid = int(line[1:])
            except ValueError:
                pid = None
        elif line.startswith("n") and pid and pid != me and line[1:].startswith(want):
            return pid
    return None


def peer_pid(client_port: int, server_port: int) -> Optional[int]:
    now = time.monotonic()
    hit = _peer_cache.get(client_port)
    if hit and now - hit[1] < _PEER_CACHE_TTL:
        return hit[0]
    pid = _lsof_peer_pid(client_port, server_port)
    if pid:
        _peer_cache[client_port] = (pid, now)
    return pid


def identify_request(request) -> Optional[ClientIdentity]:
    """Identity for a Starlette request, or None when it can't be determined."""
    header = request.headers.get(CLIENT_PID_HEADER, "").strip()
    if header.isdigit() and int(header) > 1:
        return ClientIdentity(owner_for_pid(int(header)), int(header))
    client, server = request.scope.get("client"), request.scope.get("server")
    if client and server and client[0] in ("127.0.0.1", "::1", "localhost"):
        pid = peer_pid(int(client[1]), int(server[1]))
        if pid:
            return ClientIdentity(owner_for_pid(pid), pid)
    return None
