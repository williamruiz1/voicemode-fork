"""Live transcript feed and incremental STT for converse turns.

Two independent pieces, both fail-open (a failure here must never break or slow
a converse call):

1. ``append_event`` -- appends JSON lines to ``<BASE_DIR>/live-transcript.jsonl``
   so a viewer can show each party's words as they happen. Line schema::

       {"ts": <unix float>, "kind": "agent" | "user_partial" | "user_final",
        "turn_id": "<converse session id>", "server_pid": <int>,
        "client_pid": <int | null>, "text": "<text>"}

   ``agent`` is written when TTS starts, ``user_partial`` carries the cumulative
   transcript so far (each replaces the previous one for that turn_id), and
   ``user_final`` the final transcript (empty text = no speech).

2. ``IncrementalTranscriber`` -- while the mic is open, cut finished audio at
   natural pauses and transcribe it on a worker thread, so after Send only the
   untranscribed tail remains. Any failure makes ``collect`` return None and the
   caller falls back to one whole-recording transcription (B27).
"""

import json
import logging
import os
import subprocess
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("voicemode")

FEED_FILENAME = "live-transcript.jsonl"
FEED_MAX_BYTES = 1024 * 1024

_feed_lock = threading.Lock()


def feed_path() -> Path:
    """Feed file location; resolved per call so tests can redirect BASE_DIR."""
    from voice_mode import config
    return Path(config.BASE_DIR) / FEED_FILENAME


# ---- client pid ----

_claude_pid_cache: dict = {}


def _ps_ppid_comm(pid: int) -> Optional[Tuple[int, str]]:
    try:
        out = subprocess.run(
            ["ps", "-o", "ppid=,comm=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
    except Exception as e:
        logger.debug(f"live transcript: ps lookup failed: {e}")
        return None
    parts = out.split(None, 1)
    if len(parts) != 2 or not parts[0].isdigit():
        return None
    return int(parts[0]), parts[1]


def _find_claude_ancestor(start_pid: int, max_hops: int = 8) -> Optional[int]:
    """Walk ppid from start_pid to the nearest process whose comm contains 'claude'."""
    pid = start_pid
    for _ in range(max_hops):
        if pid <= 1:
            return None
        row = _ps_ppid_comm(pid)
        if row is None:
            return None
        ppid, comm = row
        if "claude" in os.path.basename(comm).lower():
            return pid
        pid = ppid
    return None


def resolve_client_pid() -> Optional[int]:
    """Claude Code pid that owns this call, or None when unknown.

    Daemon mode: the pid client_identity already resolved for the conch owner.
    stdio mode: the nearest 'claude' ancestor of this server process.
    """
    try:
        from voice_mode import client_identity
        bound = client_identity.current_client_pid()
        if bound:
            return bound
        if client_identity.SHARED_MODE:
            return None
        me = os.getpid()
        if me not in _claude_pid_cache:
            _claude_pid_cache[me] = _find_claude_ancestor(os.getppid())
        return _claude_pid_cache[me]
    except Exception as e:
        logger.debug(f"live transcript: client pid resolution failed: {e}")
        return None


# ---- feed writer ----

def _needs_truncate(path: Path, now: float) -> bool:
    try:
        st = path.stat()
    except FileNotFoundError:
        return False
    return st.st_size > FEED_MAX_BYTES or date.fromtimestamp(st.st_mtime) != date.fromtimestamp(now)


def append_event(kind: str, turn_id: str, text: str, client_pid: Optional[int] = None) -> bool:
    """Append one feed line. Returns False (never raises) if it could not be written."""
    try:
        now = time.time()
        line = json.dumps({
            "ts": now,
            "kind": kind,
            "turn_id": turn_id,
            "server_pid": os.getpid(),
            "client_pid": client_pid if client_pid is not None else resolve_client_pid(),
            "text": text,
        }, ensure_ascii=False) + "\n"
        path = feed_path()
        with _feed_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            if _needs_truncate(path, now):
                flags |= os.O_TRUNC
            fd = os.open(path, flags, 0o600)
            try:
                os.write(fd, line.encode("utf-8"))  # one write: O_APPEND keeps the line whole
            finally:
                os.close(fd)
        return True
    except Exception as e:
        logger.debug(f"live transcript: feed write failed (ignored): {e}")
        return False


# ---- incremental STT ----

def incremental_stt_enabled() -> bool:
    return os.getenv("VOICEMODE_INCREMENTAL_STT", "true").strip().lower() in ("1", "true", "yes", "on")


def rolling_seconds() -> float:
    """How often the words so far are re-transcribed for display. 0 turns it off."""
    return _env_float("VOICEMODE_INCREMENTAL_STT_ROLLING_S", 2.0)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


class IncrementalTranscriber:
    """Cuts a growing recording into segments and transcribes each in the background.

    Feed it the same chunks the recorder appends. A segment is cut when at least
    ``min_segment_s`` of untranscribed audio ends in a pause (``pause_s`` of
    low-RMS frames), or, failing that, at the quietest frame once it reaches
    ``max_segment_s``. Segments do not overlap, so joining is plain text concat.

    ``transcribe`` is a blocking callable ``(int16 ndarray) -> str`` (return ""
    for no speech, raise on failure); it runs on a single worker thread so
    segment order is preserved and the recorder loop never blocks.

    With ``rolling_s`` set, what has been said since the last cut is also
    transcribed every ``rolling_s`` seconds and published as provisional text, so
    the words show while the user talks, not only when a segment ends. That text is
    for display only: the turn's final text still comes from the segments. Segments
    go first (Whisper handles one request at a time), and a rolling failure only
    ends the rolling updates. With ``wait_for_speech``, nothing is published until
    ``mark_speech()``: Whisper invents words from a quiet room.
    """

    FRAME_S = 0.05
    ROLLING_MIN_NEW_S = 0.8     # don't re-transcribe for less new audio than this

    def __init__(self, transcribe: Callable[[np.ndarray], str], sample_rate: int,
                 on_partial: Optional[Callable[[str], None]] = None,
                 min_segment_s: Optional[float] = None, max_segment_s: Optional[float] = None,
                 pause_s: float = 0.4, silence_rms: float = 0.01,
                 rolling_s: float = 0.0, wait_for_speech: bool = False):
        self._transcribe = transcribe
        self._sr = sample_rate
        self._on_partial = on_partial
        # >= 4096 samples so a segment always holds its whole fingerprint window.
        self._min_seg = max(4096, int((min_segment_s if min_segment_s is not None
                                       else _env_float("VOICEMODE_INCREMENTAL_STT_MIN_SEGMENT_S", 6.0)) * sample_rate))
        self._max_seg = int((max_segment_s if max_segment_s is not None
                             else _env_float("VOICEMODE_INCREMENTAL_STT_MAX_SEGMENT_S", 15.0)) * sample_rate)
        self._pause = int(pause_s * sample_rate)
        self._silence_rms = silence_rms
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="incremental-stt")
        self._lock = threading.Lock()
        self._rolling_s = rolling_s
        self._rolling_stop = threading.Event()
        self._rolling_thread: Optional[threading.Thread] = None
        self._speech_seen = not wait_for_speech
        self._reset()

    def _reset(self) -> None:
        # Generation: a segment still running from before a reset must not write into the new state.
        self._gen = getattr(self, "_gen", 0) + 1
        self._buf: List[np.ndarray] = []
        self._buf_len = 0
        self._consumed = 0          # samples already handed to the worker
        self._fingerprint = 0       # crc of the last samples before the cut point
        self._futures: list = []
        self._texts: dict = {}
        self._failed = False
        self._tail_text = ""        # provisional text of the audio not yet cut
        self._provisional: dict = {}  # provisional text of a cut segment still being transcribed
        self._rolled_at = 0         # total samples fed when the tail was last transcribed

    # -- recorder-facing --

    def begin(self, pre_roll: Optional[np.ndarray]) -> None:
        """Start (or restart, after device recovery) a listen seeded with ``pre_roll``.

        Segments already cut stay valid when ``pre_roll`` still starts with the
        same audio; otherwise everything is dropped (whole-file fallback).
        """
        try:
            if pre_roll is not None and len(pre_roll):
                self.mark_speech()  # a barge-in's audio, or a listen resuming after device recovery
            self._start_rolling()
            if pre_roll is not None and len(pre_roll) >= self._consumed and self._consumed > 0 \
                    and self._crc(pre_roll[:self._consumed]) == self._fingerprint:
                self._buf, self._buf_len = [], 0
                self.feed(pre_roll[self._consumed:])
                return
            self._futures_cancel()
            self._reset()
            if pre_roll is not None and len(pre_roll):
                self.feed(pre_roll)
        except Exception as e:
            logger.debug(f"incremental STT begin failed (falling back): {e}")
            self._failed = True

    def feed(self, chunk: np.ndarray) -> None:
        """Add recorded audio; never raises, never blocks on STT."""
        try:
            if self._failed or chunk is None or len(chunk) == 0:
                return
            self._buf.append(np.asarray(chunk).reshape(-1))
            self._buf_len += len(self._buf[-1])
            if self._buf_len >= self._min_seg:
                self._maybe_cut()
        except Exception as e:
            logger.debug(f"incremental STT feed failed (falling back): {e}")
            self._failed = True

    def mark_speech(self) -> None:
        """The recorder heard speech: provisional text can show from here on."""
        self._speech_seen = True

    def _start_rolling(self) -> None:
        if self._rolling_s <= 0 or self._on_partial is None or self._rolling_thread is not None:
            return
        self._rolling_thread = threading.Thread(target=self._rolling_loop, daemon=True,
                                                name="incremental-stt-rolling")
        self._rolling_thread.start()

    def _rolling_loop(self) -> None:
        while not self._rolling_stop.wait(self._rolling_s):
            try:
                self._roll_once()
            except Exception as e:
                # Stay off the failover path too: it could reach a paid provider every 2 seconds.
                logger.debug(f"incremental STT rolling update failed, stopping them: {e}")
                return

    def _roll_once(self) -> None:
        if not self._speech_seen or self._failed:
            return
        with self._lock:
            if any(i not in self._texts for i in range(len(self._futures))):
                return  # a segment is waiting for Whisper; it goes first
        gen, base, parts = self._gen, self._consumed, list(self._buf)
        if not parts:
            return
        tail = np.concatenate(parts)
        total = base + len(tail)
        if len(tail) < self.ROLLING_MIN_NEW_S * self._sr or total - self._rolled_at < self.ROLLING_MIN_NEW_S * self._sr:
            return
        if self._rms(tail[-(total - self._rolled_at):]) < self._silence_rms:
            return  # nothing new was said
        self._rolled_at = total
        text = (self._transcribe(tail) or "").strip()
        with self._lock:
            # Cut or restarted meanwhile: this audio now belongs to a segment.
            if gen != self._gen or base != self._consumed or self._rolling_stop.is_set():
                return
            self._tail_text = text
            partial = self._display_text()
        self._publish(partial)

    def _display_text(self) -> str:
        """Finished segments, then provisional text for the rest. Caller holds the lock."""
        pieces = [self._texts.get(i) or self._provisional.get(i, "") for i in range(len(self._futures))]
        pieces.append(self._tail_text)
        return " ".join(p for p in pieces if p)

    def _publish(self, partial: str) -> None:
        if partial and self._on_partial and self._speech_seen:
            try:
                self._on_partial(partial)
            except Exception as e:
                logger.debug(f"incremental STT partial publish failed (ignored): {e}")

    def _rms(self, a: np.ndarray) -> float:
        if len(a) == 0:
            return 0.0
        return float(np.sqrt(np.mean((a.astype(np.float64) / 32768.0) ** 2)))

    @staticmethod
    def _crc(a: np.ndarray) -> int:
        return zlib.crc32(np.ascontiguousarray(a[-4096:]).astype(np.int16).tobytes())

    def _maybe_cut(self) -> None:
        audio = np.concatenate(self._buf)
        frame = int(self.FRAME_S * self._sr)
        cut = None
        # Pause: the newest `pause` samples are quiet and the segment is long enough.
        if len(audio) >= self._min_seg and self._rms(audio[-self._pause:]) < self._silence_rms:
            cut = len(audio)
        elif len(audio) >= self._max_seg:
            # Forced: quietest frame in the last third, so we never split a word if avoidable.
            lo = max(self._min_seg, len(audio) - len(audio) // 3)
            n = (len(audio) - lo) // frame
            if n > 0:
                rms = [self._rms(audio[lo + i * frame: lo + (i + 1) * frame]) for i in range(n)]
                cut = lo + (int(np.argmin(rms)) + 1) * frame
            else:
                cut = len(audio)
        if cut is None:
            return
        segment, rest = audio[:cut], audio[cut:]
        self._buf, self._buf_len = ([rest], len(rest)) if len(rest) else ([], 0)
        self._consumed += len(segment)
        self._fingerprint = self._crc(segment)  # same as crc of the recording's prefix ending here
        idx = len(self._futures)
        with self._lock:
            self._provisional[idx], self._tail_text = self._tail_text, ""
            self._futures.append(self._pool.submit(self._run_segment, self._gen, idx, segment))

    def _run_segment(self, gen: int, idx: int, segment: np.ndarray) -> None:
        try:
            text = (self._transcribe(segment) or "").strip()
        except Exception as e:
            logger.warning(f"incremental STT segment {idx} failed: {e}")
            with self._lock:
                if gen == self._gen:
                    self._failed = True
            return
        with self._lock:
            if gen != self._gen:
                return  # stale: the recording was restarted while this ran
            self._texts[idx] = text
            partial = self._display_text()
        self._publish(partial)

    def _futures_cancel(self) -> None:
        for f in self._futures:
            f.cancel()

    # -- converse-facing --

    @property
    def segments_started(self) -> int:
        return len(self._futures)

    def collect(self, full_audio: np.ndarray) -> Optional[Tuple[str, np.ndarray]]:
        """Wait for in-flight segments; return ``(prefix_text, tail_audio)``.

        None means "do not trust the incremental result" (a segment failed, or the
        recorded audio no longer matches what was cut): transcribe the whole
        recording instead. Blocking -- call from an executor.
        """
        self._rolling_stop.set()
        try:
            for f in self._futures:
                f.result()
            with self._lock:
                if self._failed or len(self._texts) != len(self._futures):
                    return None
                texts = [self._texts[i] for i in range(len(self._futures))]
            if self._consumed > len(full_audio) or \
                    (self._consumed and self._crc(full_audio[:self._consumed]) != self._fingerprint):
                logger.warning("incremental STT: recording does not match cut segments - transcribing whole recording")
                return None
            return " ".join(t for t in texts if t), full_audio[self._consumed:]
        except Exception as e:
            logger.warning(f"incremental STT collect failed - transcribing whole recording: {e}")
            return None
        finally:
            self._pool.shutdown(wait=False)

    def close(self) -> None:
        """Release the worker without collecting (turn abandoned)."""
        self._rolling_stop.set()
        self._futures_cancel()
        self._pool.shutdown(wait=False)
