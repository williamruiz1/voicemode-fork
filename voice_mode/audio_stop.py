"""Stop audio streams without letting a stuck stop take the process with it.

A PortAudio stop can block forever inside CoreAudio (Oct 31: two threads stopping
one stream deadlocked, and because that stop ran on the event loop it froze the
shared service for every tab). Stops here run on their own thread with a time
limit. A stream that misses the limit is abandoned: it is never touched again, a
reference is kept so its destructor can't touch it either, and `stuck()` turns
true so the shared service restarts itself at the first quiet moment
(DrainingServer) and nothing calls Pa_Terminate, which would block on it too.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Optional

from . import activity

logger = logging.getLogger("voicemode.audio_stop")

# A normal stop returns in well under a second. Long enough never to abandon a
# healthy stream on a slow Bluetooth device.
STOP_LIMIT_SECONDS = 10.0

_abandoned: list = []


def stuck() -> bool:
    """True once this process has abandoned a stream; only a restart clears it."""
    return bool(_abandoned)


def _begin(stream, stop: bool, abort: bool, close: bool) -> threading.Event:
    done = threading.Event()

    def _run():
        try:
            if stop:
                try:
                    stream.abort() if abort else stream.stop()
                except Exception as e:
                    logger.debug(f"stream {'abort' if abort else 'stop'} failed: {e}")
                    stream.stop()
            if close:
                stream.close()
        except Exception as e:
            logger.warning(f"stopping audio stream failed: {e}")
        finally:
            done.set()

    threading.Thread(target=_run, daemon=True, name="voicemode-audio-stop").start()
    return done


def _give_up(stream, limit: float) -> None:
    _abandoned.append(stream)
    logger.error(f"Audio stream did not stop within {limit:.0f}s; abandoning it")
    activity.log("audio_stop_stuck", limit_seconds=limit)


def stop_stream(stream, *, stop: bool = True, abort: bool = False, close: bool = True,
                limit: Optional[float] = None) -> bool:
    """Stop and close `stream`. True if it finished within the limit. For threads, not the event loop."""
    if stream in _abandoned:
        return False
    limit = STOP_LIMIT_SECONDS if limit is None else limit
    if _begin(stream, stop, abort, close).wait(limit):
        return True
    _give_up(stream, limit)
    return False


async def stop_stream_async(stream, *, stop: bool = True, abort: bool = False, close: bool = True,
                            limit: Optional[float] = None) -> bool:
    """stop_stream for the event loop: other calls keep running while this waits."""
    if stream in _abandoned:
        return False
    limit = STOP_LIMIT_SECONDS if limit is None else limit
    done = _begin(stream, stop, abort, close)
    started = time.monotonic()
    while not done.is_set():
        if time.monotonic() - started >= limit:
            _give_up(stream, limit)
            return False
        await asyncio.sleep(0.01)
    return True


class guarded:
    """`with guarded(stream):` is `with stream:`, with the stop on exit under the time limit."""

    def __init__(self, stream):
        self.stream = stream

    def __enter__(self):
        return self.stream.__enter__()

    def __exit__(self, *exc):
        errors = []
        done = threading.Event()

        def _run():
            try:
                self.stream.__exit__(*exc)
            except Exception as e:
                errors.append(e)
            finally:
                done.set()

        threading.Thread(target=_run, daemon=True, name="voicemode-audio-stop").start()
        if not done.wait(STOP_LIMIT_SECONDS):
            _give_up(self.stream, STOP_LIMIT_SECONDS)
        elif errors:
            raise errors[0]
        return False
