"""Live (per-call) voice settings.

A tiny JSON file, ``~/.voicemode/live-settings.json``, is re-read on every
converse() call so settings can change on the very next voice turn with no
restart and no re-import (e.g. from a menu bar app). Missing file, bad JSON,
or an out-of-range value all fall back to the configured value; nothing here
ever raises.

Schema (all keys optional)::

    {
      "silence_threshold_ms": 1000,   # int,    100..10000
      "vad_aggressiveness": 2,        # int,    0..3
      "listen_duration_max": 120,     # number, >0..3600 seconds
      "listen_duration_min": 2.0,     # number, 0..3600 seconds
      "append_window_ms": 0,          # int,    0..10000
      "endpointing_min_endpoint_ms": 500  # int,  100..5000 (Silero end-of-turn silence)
    }
"""

import json
import os
from typing import Any, Dict, Optional

LIVE_SETTINGS_ENV = "VOICEMODE_LIVE_SETTINGS_PATH"
_DEFAULT_PATH = "~/.voicemode/live-settings.json"


def live_settings_path() -> str:
    # Resolved on every call so HOME / env changes are honored (and tests can use a temp HOME).
    return os.path.expanduser(os.getenv(LIVE_SETTINGS_ENV, _DEFAULT_PATH))


def _is_num(v: Any) -> bool:
    return (
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and v == v
        and v not in (float("inf"), float("-inf"))
    )


def _int_in(lo: int, hi: int):
    def check(v: Any) -> Optional[int]:
        if not _is_num(v) or int(v) != v:
            return None
        return int(v) if lo <= v <= hi else None
    return check


def _num_in(lo: float, hi: float, lo_exclusive: bool = False):
    def check(v: Any) -> Optional[float]:
        if not _is_num(v):
            return None
        if v < lo or v > hi or (lo_exclusive and v == lo):
            return None
        return float(v)
    return check


_VALIDATORS = {
    "silence_threshold_ms": _int_in(100, 10000),
    "vad_aggressiveness": _int_in(0, 3),
    "listen_duration_max": _num_in(0, 3600, lo_exclusive=True),
    "listen_duration_min": _num_in(0, 3600),
    "append_window_ms": _int_in(0, 10000),
    "endpointing_min_endpoint_ms": _int_in(100, 5000),
}


def load_live_settings() -> Dict[str, Any]:
    """Return only the valid keys currently in the live settings file ({} on any problem)."""
    try:
        with open(live_settings_path()) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        out: Dict[str, Any] = {}
        for key, check in _VALIDATORS.items():
            if key in data:
                val = check(data[key])
                if val is not None:
                    out[key] = val
        return out
    except Exception:
        return {}


def get_live(name: str, default: Any) -> Any:
    """Live value for ``name`` if present and valid, else ``default``."""
    try:
        return load_live_settings().get(name, default)
    except Exception:
        return default
