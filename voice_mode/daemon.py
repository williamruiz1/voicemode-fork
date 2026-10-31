"""Shared-daemon support for `voicemode serve`.

One voicemode process serves every Claude Code session over streamable HTTP
instead of one stdio server per session. This module holds the pieces that
mode needs beyond plain `serve`:

- ClientIdentityMiddleware: binds each tool call to the client session that
  made it (see client_identity), so the conch and the convomode floor tell
  sessions apart even though they share one process.
- IdleExit: the daemon exits after a quiet period with no tool call running.
  launchd socket activation starts it again on the next connection.
- ActiveRequests, DrainingServer: a SIGTERM (launchctl kickstart -k, bootout)
  waits for running requests, up to launchd's limit.
- LoopWatchdog: notices the event loop not answering, records where it is stuck,
  alerts, and restarts the service if it never comes back.
- launchd_sockets: adopt the listening socket launchd passes a socket-activated
  job (launch_activate_socket), so the port keeps accepting while the daemon
  is not running.
"""

from __future__ import annotations

import asyncio
import ctypes
import faulthandler
import logging
import os
import signal
import socket
import subprocess
import threading
import time
from types import FrameType
from typing import Callable, Optional

import uvicorn
from fastmcp.server.dependencies import get_http_request
from fastmcp.server.middleware import Middleware, MiddlewareContext

from . import activity, audio_stop, client_identity

logger = logging.getLogger("voicemode.daemon")

DEFAULT_IDLE_EXIT_SECONDS = 2700.0

# Longest a SIGTERM waits for running requests. launchd caps a job's ExitTimeOut
# at 60s and then sends SIGKILL, so this stays under it (the job sets
# ExitTimeOut 60 in launchagents/com.founder-os.voicemode-daemon.plist). A longer
# listen can still be cut by a raw SIGTERM, so tabs restart this service with
# voicemode-daemon-restart.sh, which waits for the call to end first.
DRAIN_MAX_SECONDS = 55.0

# How long the event loop can go without answering before LoopWatchdog acts.
FROZEN_ALERT_SECONDS = 20.0
FROZEN_EXIT_SECONDS = 60.0


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


class ActiveRequests:
    """ASGI wrapper that counts HTTP requests still being answered.

    A tool call's result is written after the tool itself returns, so this is
    what tells a shutdown that the client has its answer.
    """

    def __init__(self, app):
        self.app = app
        self.count = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        self.count += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.count -= 1


class DrainingServer(uvicorn.Server):
    """uvicorn server whose SIGTERM waits until no request is being answered.

    uvicorn's own shutdown cancels a running converse, which loses whatever the
    user is saying. New calls are still served while waiting; the first moment
    none is running (between turns of a conversation) the server shuts down as
    usual, and launchd starts it again on the next connection.
    """

    def __init__(self, config: uvicorn.Config, active: ActiveRequests,
                 max_wait: float = DRAIN_MAX_SECONDS, clock: Callable[[], float] = time.monotonic):
        super().__init__(config)
        self.active = active
        self.max_wait = max_wait
        self._clock = clock
        self._drain_since: Optional[float] = None
        self._stuck_since: Optional[float] = None

    def handle_exit(self, sig: int, frame: Optional[FrameType]) -> None:
        if sig == signal.SIGTERM and self.active.count:
            if self._drain_since is None:
                self._drain_since = self._clock()
            return
        super().handle_exit(sig, frame)

    def _exit_for_stuck_audio(self) -> bool:
        """After an abandoned stream no new stream opens in this process (seen on a real
        deadlock), so it needs replacing. converse refuses calls now, so the running ones
        end quickly with that answer; max_wait covers one that can't."""
        if not audio_stop.stuck():
            return False
        if self._stuck_since is None:
            self._stuck_since = self._clock()
        return not self.active.count or self._clock() - self._stuck_since >= self.max_wait

    def _drain_finished(self) -> bool:
        if self._drain_since is None or self.should_exit:
            return False
        return not self.active.count or self._clock() - self._drain_since >= self.max_wait

    async def on_tick(self, counter: int) -> bool:
        if self._exit_for_stuck_audio():
            # Skips interpreter teardown: Pa_Terminate would block on the stuck stream.
            logger.error(f"Restarting: an audio stream is stuck ({self.active.count} requests still running)")
            activity.log("service_restart", reason="audio_stuck", running_requests=self.active.count)
            logging.shutdown()
            os._exit(0)
        if self._drain_finished():
            logger.info(f"Stopping after waiting {self._clock() - self._drain_since:.0f}s "
                        f"for running requests ({self.active.count} still running)")
            super().handle_exit(signal.SIGTERM, None)
        return await super().on_tick(counter)


class LoopWatchdog:
    """Notices a stuck event loop from outside it.

    Every tab on the shared service stops getting answers when the loop is stuck, and
    nothing inside the loop can say so (Oct 31: 10 minutes down, nothing logged). A
    thread watches a heartbeat the loop writes. A silent loop gets its stacks written
    to the log and the user alerted; one that stays silent can't finish any call, so
    the process exits and launchd starts a new one on the next connection.
    """

    def __init__(self, alert_after: float = FROZEN_ALERT_SECONDS, exit_after: float = FROZEN_EXIT_SECONDS,
                 clock: Callable[[], float] = time.monotonic,
                 on_frozen: Optional[Callable[[float], None]] = None,
                 on_exit: Optional[Callable[[], None]] = None):
        self.alert_after = alert_after
        self.exit_after = exit_after
        self._clock = clock
        self._last_beat = clock()
        self._frozen = False
        self._on_frozen = on_frozen or alert_frozen
        self._on_exit = on_exit or (lambda: os._exit(1))
        self._stopped = threading.Event()

    async def beat(self, every: float = 1.0) -> None:
        while True:
            self._last_beat = self._clock()
            await asyncio.sleep(every)

    def check(self) -> None:
        silent = self._clock() - self._last_beat
        if silent >= self.exit_after:
            logger.critical(f"Event loop silent for {silent:.0f}s; restarting")
            activity.log("service_restart", reason="frozen", silent_seconds=round(silent))
            logging.shutdown()
            self._on_exit()
        elif silent >= self.alert_after and not self._frozen:
            self._frozen = True
            logger.critical(f"Event loop silent for {silent:.0f}s; every tab on this service is waiting")
            activity.log("service_frozen", silent_seconds=round(silent))
            self._on_frozen(silent)
        elif silent < self.alert_after and self._frozen:
            self._frozen = False
            logger.warning("Event loop answering again")
            activity.log("service_recovered")

    def start_thread(self, poll: float = 1.0) -> threading.Thread:
        def run():
            while not self._stopped.wait(poll):
                try:
                    self.check()
                except Exception as e:
                    logger.warning(f"loop watchdog check failed: {e}")
        thread = threading.Thread(target=run, daemon=True, name="voicemode-loop-watchdog")
        thread.start()
        return thread

    def stop(self) -> None:
        self._stopped.set()


def alert_frozen(silent: float) -> None:
    """Write where the loop is stuck and buzz the user's phone. Never raises."""
    try:
        for handler in logging.getLogger().handlers:
            stream = getattr(handler, "stream", None)
            if stream is not None and hasattr(stream, "fileno"):
                faulthandler.dump_traceback(file=stream)
                break
    except Exception as e:
        logger.warning(f"could not write stacks: {e}")
    try:
        subprocess.Popen(
            [os.path.expanduser("~/.local/bin/founder-os/foscc-push-notify.sh"), "--audience", "phone",
             "--sender", "voice-service", "--title", "Voice stopped answering",
             "--ask", f"Nothing to do yet: it restarts itself after {FROZEN_EXIT_SECONDS:.0f} seconds. "
                      "Tell a tab if voice is still dead after that.",
             "--why-now", "Every tab on the shared voice service is stuck until this clears.",
             "--dedupe-key", "voice-service-frozen"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError as e:
        logger.warning(f"could not send freeze alert: {e}")


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
