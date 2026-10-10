"""The shell hook receiver must read the conch lock FAIL-CLOSED.

voicemode-hook-receiver.sh used to answer "no conversation active" whenever the
lock file was missing OR unparseable, so a corrupt record let sound effects
play over a live conversation. Python side already fails closed
(Conch.read_lock_state); this pins the shell reader to the same contract.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parent.parent / "voice_mode" / "data" / "hooks" / "voicemode-hook-receiver.sh"
PAYLOAD = '{"tool_name":"Task","hook_event_name":"PreToolUse"}'
SKIP_MSG = "skipping sound playback"


def _run(home: Path) -> str:
    env = {"HOME": str(home), "PATH": os.environ["PATH"], "VOICEMODE_HOOK_DEBUG": "1"}
    out = subprocess.run(["bash", str(HOOK)], input=PAYLOAD, env=env,
                         capture_output=True, text=True, timeout=30)
    return out.stdout + out.stderr


@pytest.fixture
def home(tmp_path):
    (tmp_path / ".voicemode").mkdir()
    return tmp_path


def _lock(home, text):
    (home / ".voicemode" / "conch").write_text(text)


@pytest.mark.parametrize("body", ["{corrupt", "", "[]", '{"agent": "x"}', '{"pid": "abc"}', '{"pid": 0}'])
def test_unreadable_or_pidless_lock_reads_as_active(home, body):
    _lock(home, body)
    assert SKIP_MSG in _run(home)


def test_live_holder_reads_as_active(home):
    _lock(home, json.dumps({"pid": os.getpid()}))
    assert SKIP_MSG in _run(home)


def test_absent_lock_is_not_active(home):
    assert SKIP_MSG not in _run(home)


def test_dead_holder_is_not_active(home):
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    _lock(home, json.dumps({"pid": p.pid}))
    assert SKIP_MSG not in _run(home)
