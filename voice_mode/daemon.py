"""Shared-daemon support for `voicemode serve`.

One voicemode process serves every Claude Code session over streamable HTTP
instead of one stdio server per session. This module holds the pieces that
mode needs beyond plain `serve`:

- ClientIdentityMiddleware: binds each tool call to the client session that
  made it (see client_identity), so the conch and the convomode floor tell
  sessions apart even though they share one process.
- IdleExit: the daemon exits after a quiet period with no tool call running.
  launchd socket activation starts it again on the next connection.
- launchd_sockets: adopt the listening socket launchd passes a socket-activated
  job (launch_activate_socket), so the port keeps accepting while the daemon
  is not running.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import socket
import time
from typing import Callable, Optional

from fastmcp.server.dependencies import get_http_request
from fastmcp.server.middleware import Middleware, MiddlewareContext

from . import client_identity

logger = logging.getLogger("voicemode.daemon")

DEFAULT_IDLE_EXIT_SECONDS = 2700.0


class IdleExit:
    """Tracks tool calls; reports idle once none ran for `seconds`."""

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic):
        self.seconds = seconds
        self._clock = clock
        self.in_flight = 0
        self.last_activity = clock()

    def begin(self) -> None:
        self.in_flight += 1
        self.last_activity = self._clock()

    def end(self) -> None:
        self.in_flight = max(0, self.in_flight - 1)
        self.last_activity = self._clock()

    def idle_for(self) -> float:
        return 0.0 if self.in_flight else self._clock() - self.last_activity

    def is_idle(self) -> bool:
        return self.seconds > 0 and self.in_flight == 0 and self.idle_for() >= self.seconds

    async def watch(self, on_idle: Callable[[], None], poll: Optional[float] = None) -> None:
        poll = poll if poll is not None else min(60.0, max(0.5, self.seconds / 10))
        while True:
            await asyncio.sleep(poll)
            if self.is_idle():
                logger.info(f"No tool call for {self.idle_for():.0f}s; idle exit")
                on_idle()
                return


class ClientIdentityMiddleware(Middleware):
    """Bind each tool call to its client session and record activity."""

    def __init__(self, idle: Optional[IdleExit] = None, poll: float = 1.0):
        self.idle = idle
        self.poll = poll

    async def _run_until_client_gone(self, identity, call_next, context):
        """Run the call, cancelling it if its client process dies mid-call.

        A tab killed mid-converse would otherwise leave its call running in
        this long-lived daemon, holding the conch for the whole listen. The
        client's pid is watched (not the socket via request.is_disconnected(),
        which would race the MCP transport's own receive loop). Clients whose
        pid is unknown (anon identity) cannot be watched; the conch's
        timestamp expiry is their backstop.
        """
        if identity is None or identity.pid is None:
            return await call_next(context)
        task = asyncio.ensure_future(call_next(context))
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=self.poll)
                if done:
                    return task.result()
                if client_identity.owner_is_dead(identity.owner):
                    logger.warning(f"Client {identity.owner} died mid-call; cancelling its tool call")
                    task.cancel()
                    try:
                        await task
                    except BaseException:
                        pass
                    raise asyncio.CancelledError()
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
            raise

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        identity = None
        try:
            identity = client_identity.identify_request(get_http_request())
        except RuntimeError:
            pass  # not an HTTP request (e.g. in-process client)
        if identity is None:
            logger.warning("Tool call from an unidentified client; conch/floor "
                           "checks treat its identity as unknown")
            identity = client_identity.anon_identity()
        token = client_identity.bind(identity)
        if self.idle:
            self.idle.begin()
        try:
            return await self._run_until_client_gone(identity, call_next, context)
        finally:
            if self.idle:
                self.idle.end()
            client_identity.unbind(token)


def launchd_sockets(name: str) -> list[socket.socket]:
    """Listening sockets launchd holds for this job under Sockets/<name>."""
    libc = ctypes.CDLL(None)
    fn = libc.launch_activate_socket
    fn.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                   ctypes.POINTER(ctypes.c_size_t)]
    fn.restype = ctypes.c_int
    fds = ctypes.POINTER(ctypes.c_int)()
    count = ctypes.c_size_t(0)
    rc = fn(name.encode(), ctypes.byref(fds), ctypes.byref(count))
    if rc != 0:
        raise OSError(rc, f"launch_activate_socket({name!r}) failed")
    try:
        return [socket.socket(fileno=fds[i]) for i in range(count.value)]
    finally:
        libc.free.argtypes = [ctypes.c_void_p]
        libc.free(ctypes.cast(fds, ctypes.c_void_p))
