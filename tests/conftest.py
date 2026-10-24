"""Shared test fixtures and configuration for VoiceMode tests."""

import os
import sys
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import pytest_asyncio

# Point HOME at a throwaway directory for the whole session, BEFORE any
# voice_mode module is imported. Many modules resolve ~/.voicemode paths at
# import time (config.BASE_DIR/LOGS_DIR/AUDIO_DIR, conversation_logger,
# audio_player.SPEAKING_FLAG_PATH, ...), so the per-test isolate_home_directory
# patch below came too late for them: a full run appended to the real
# exchange and barge-in logs, repointed the real audio/latest*.wav symlinks
# and deleted the live speaking.flag. Child processes (multiprocessing
# spawn) inherit it too. The per-test fixture still gives each test its own home.
_REAL_HOME = os.environ.get("HOME")
_SESSION_HOME = tempfile.mkdtemp(prefix="voicemode-test-home-")
os.environ["HOME"] = _SESSION_HOME


def pytest_sessionfinish(session, exitstatus):
    import shutil
    shutil.rmtree(_SESSION_HOME, ignore_errors=True)
    if _REAL_HOME is not None:
        os.environ["HOME"] = _REAL_HOME


# Add voice_mode to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))


# Commands that should never run in tests - these affect system services
BLOCKED_COMMANDS = {
    "launchctl",
    "systemctl",
    "brew",
    # Audio players: a test must never play sound through the real speakers.
    "afplay",
    "ffplay",
    "say",
    "aplay",
    "paplay",
}


# sounddevice entry points that open a REAL audio device -- input, output or
# both. `query_devices` and `default` are deliberately NOT listed -- reading
# device metadata is harmless and plenty of code does it. `Stream` is
# full-duplex: with a Bluetooth default input, record_audio() and the VAD loop
# capture via sd.Stream (F1). Output matters too: unmocked chimes/TTS played
# through the user's headphones mid-conversation.
BLOCKED_AUDIO_CAPTURE = (
    "InputStream", "OutputStream", "Stream",
    "RawInputStream", "RawOutputStream", "RawStream",
    "rec", "play", "playrec",
)

# The real sounddevice module, captured while conftest loads -- before test
# modules are collected. Some test modules replace sys.modules["sounddevice"]
# with a MagicMock at import time (e.g. test_min_duration_integration.py), so
# a later `import sounddevice` inside the fixture returns that mock while
# voice_mode.tools.converse still holds the real module. Guarding only the mock
# left the real device open to every test in a full run.
try:
    import sounddevice as _REAL_SOUNDDEVICE
except Exception:
    _REAL_SOUNDDEVICE = None


def _safe_subprocess_run(original_run):
    """Wrapper that blocks dangerous system commands during tests."""
    def wrapper(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        if isinstance(cmd, str):
            cmd_parts = cmd.split()
        else:
            cmd_parts = list(cmd) if cmd else []

        if cmd_parts and cmd_parts[0] in BLOCKED_COMMANDS:
            # Return a mock result instead of running the command
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            return mock_result

        return original_run(*args, **kwargs)
    return wrapper


def _safe_subprocess_popen(original_popen):
    """Wrapper that blocks dangerous system commands during tests.

    Returns a callable class (not a function) so that type subscripting like
    Popen[bytes] works. The mcp library uses this in type annotations and
    Python 3.10 raises TypeError on subscripted functions.
    """
    class SafePopen:
        """Callable wrapper for subprocess.Popen that blocks dangerous commands."""

        def __class_getitem__(cls, params):
            """Support type subscripting (e.g., Popen[bytes])."""
            return original_popen[params]

        def __new__(cls, *args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])
            if isinstance(cmd, str):
                cmd_parts = cmd.split()
            else:
                cmd_parts = list(cmd) if cmd else []

            if cmd_parts and cmd_parts[0] in BLOCKED_COMMANDS:
                # Return a mock process instead of running the command
                mock_proc = MagicMock()
                mock_proc.pid = 99999
                mock_proc.poll.return_value = 0
                mock_proc.returncode = 0
                mock_proc.communicate.return_value = (b"", b"")
                mock_proc.stdout = MagicMock()
                mock_proc.stderr = MagicMock()
                return mock_proc

            return original_popen(*args, **kwargs)

    return SafePopen


@pytest.fixture(autouse=True)
def block_dangerous_commands(monkeypatch):
    """
    Automatically block dangerous system commands in all tests.

    This prevents tests from accidentally running launchctl, systemctl,
    or other system commands that could affect running services.
    Tests that need to verify these commands are called should use
    explicit mocking with patch().
    """
    original_run = subprocess.run
    original_popen = subprocess.Popen

    monkeypatch.setattr("subprocess.run", _safe_subprocess_run(original_run))
    monkeypatch.setattr("subprocess.Popen", _safe_subprocess_popen(original_popen))


@pytest.fixture(autouse=True)
def block_real_microphone(monkeypatch, request):
    """Make it structurally impossible for a test to open the real microphone.

    A test whose recording mock targets the wrong function silently falls
    through to `sounddevice` and HOLDS THE MACHINE'S INPUT DEVICE for the full
    listen duration -- measured at ~90s for one test on this branch. On a
    machine running live voice sessions that is a real-world side effect, not a
    flaky test, so it is blocked structurally rather than by remembering to
    mock the right name.

    A test that patches `sd.InputStream` / `sd.rec` itself still wins: its
    patch is applied after this fixture's. This only bites when nothing else
    intercepted the call.

    Opt out with @pytest.mark.real_microphone if a test genuinely needs a
    device.
    """
    # Record the fixture's own decision so a test can assert on it without
    # poking sounddevice -- other tests replace that module with a MagicMock,
    # which makes any attribute-shape assertion order-dependent.
    request.node.mic_guard_armed = False

    if request.node.get_closest_marker("real_microphone"):
        yield
        return

    targets = [m for m in (_REAL_SOUNDDEVICE, sys.modules.get("sounddevice")) if m is not None]
    if not targets:
        # No audio backend available -- nothing to guard.
        yield
        return

    def _refuse(*args, **kwargs):
        raise RuntimeError(
            "This test tried to open the REAL microphone or speaker via "
            "sounddevice. Mock the function the code actually calls -- for "
            "recording that is record_audio_with_silence_detection, not just "
            "record_audio -- or mark the test @pytest.mark.real_microphone if "
            "it truly needs a device. See tests/conftest.py:block_real_microphone."
        )

    for sd in {id(m): m for m in targets}.values():
        for name in BLOCKED_AUDIO_CAPTURE:
            if hasattr(sd, name):
                monkeypatch.setattr(sd, name, _refuse)
                request.node.mic_guard_armed = True
        # Catch-all: every sounddevice stream class (and play/rec/playrec,
        # which build one internally) opens PortAudio in _StreamBase.__init__,
        # so this also covers a name held from before the patch above.
        base = getattr(sd, "_StreamBase", None)
        if isinstance(base, type):
            monkeypatch.setattr(base, "__init__", _refuse)

    # pydub's playback fallback goes through simpleaudio when it is installed.
    try:
        import simpleaudio
        monkeypatch.setattr(simpleaudio, "play_buffer", _refuse)
    except Exception:
        pass

    yield


@pytest.fixture(autouse=True)
def block_signals_to_foreign_processes(monkeypatch):
    """Refuse to signal any process this test run did not start.

    Service code stops a server by finding whatever listens on its port and
    terminating it. Under the fake home no plist exists, so that fallback ran
    against the REAL machine: tests/test_whisper_model_cli.py::
    test_set_active_model terminated the live whisper-server on every full run
    (launchd then restarted it mid-conversation). Signal 0 (liveness probes,
    used by the conch) is still allowed.
    """
    import psutil

    me = os.getpid()

    def _ours(pid):
        try:
            proc = psutil.Process(pid)
            while proc is not None:
                if proc.pid == me:
                    return True
                proc = proc.parent()
        except Exception:
            pass
        return False

    def _refuse(pid, what):
        raise PermissionError(
            f"This test tried to {what} pid {pid}, a process it did not start "
            "(a real service on this machine). Mock the process lookup. See "
            "tests/conftest.py:block_signals_to_foreign_processes."
        )

    original_kill = os.kill

    def _kill(pid, sig):
        if sig != 0 and pid > 0 and not _ours(pid):
            _refuse(pid, f"send signal {sig} to")
        return original_kill(pid, sig)

    monkeypatch.setattr(os, "kill", _kill)

    for name in ("terminate", "kill", "send_signal", "suspend", "resume"):
        original = getattr(psutil.Process, name)

        def _make(original, name):
            def _guarded(self, *args, **kwargs):
                if not _ours(self.pid):
                    _refuse(self.pid, name)
                return original(self, *args, **kwargs)
            return _guarded

        monkeypatch.setattr(psutil.Process, name, _make(original, name))

    yield


@pytest.fixture(autouse=True)
def isolate_home_directory(tmp_path, monkeypatch):
    """
    Redirect Path.home() and os.path.expanduser() to a temporary directory.

    This prevents tests from writing plist files to ~/Library/LaunchAgents/
    or systemd service files to ~/.config/systemd/user/. The isolation is
    automatic for all tests.

    Without this fixture, running pytest would install service files to
    the real home directory, causing:
    - Apple notifications about services being configured
    - Services potentially starting automatically
    - Developer's local service configuration being affected
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()

    # Create expected subdirectories that tests may write to
    (fake_home / "Library" / "LaunchAgents").mkdir(parents=True, exist_ok=True)
    (fake_home / ".config" / "systemd" / "user").mkdir(parents=True, exist_ok=True)
    (fake_home / ".voicemode" / "logs").mkdir(parents=True, exist_ok=True)
    (fake_home / ".voicemode" / "services").mkdir(parents=True, exist_ok=True)
    (fake_home / ".voicemode" / "config").mkdir(parents=True, exist_ok=True)

    # Mock Path.home() to return the fake home directory
    monkeypatch.setattr("pathlib.Path.home", lambda: fake_home)

    # Mock os.path.expanduser() to handle ~ expansion
    original_expanduser = os.path.expanduser

    def mock_expanduser(path):
        if path.startswith("~"):
            return str(fake_home) + path[1:]
        return original_expanduser(path)

    monkeypatch.setattr("os.path.expanduser", mock_expanduser)

    # Conch.LOCK_FILE / Conch.WANTED_FILE are class attributes computed ONCE
    # at module-import time (`Path.home() / ".voicemode" / "conch"`), which
    # happens at collection time -- before this per-test monkeypatch of
    # Path.home()/expanduser takes effect. So without patching them directly,
    # test_conch_yield.py's `clean_conch` fixture unlinks and rewrites the
    # REAL ~/.voicemode/conch + conch-wanted lock files on every run -- files
    # that a live voice-mode session elsewhere on the machine may be actively
    # holding to signal "don't let another agent barge in on my mic." A test
    # run deleting that lock mid-conversation would let a concurrent session
    # think the mic is free. Patch the class attributes directly so tests
    # only ever touch the fake home's conch files.
    try:
        from voice_mode.conch import Conch
        monkeypatch.setattr(Conch, "LOCK_FILE", fake_home / ".voicemode" / "conch")
        monkeypatch.setattr(Conch, "WANTED_FILE", fake_home / ".voicemode" / "conch-wanted")
        # Keep preempt tests fast: don't wait a real second for a holder that
        # (in single-process tests) is never going to stand down.
        monkeypatch.setattr(Conch, "PREEMPT_FLOCK_WAIT", 0.05)
    except ImportError:
        pass

    yield fake_home


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_mcp():
    """Create a mock FastMCP instance for testing tools."""
    from fastmcp import FastMCP
    mcp = MagicMock(spec=FastMCP)
    mcp.tool = MagicMock()
    
    # Make the decorator work properly
    def tool_decorator(**kwargs):
        def decorator(func):
            func._mcp_tool_config = kwargs
            return func
        return decorator
    
    mcp.tool.side_effect = tool_decorator
    return mcp


@pytest.fixture
def mock_subprocess(monkeypatch):
    """Mock subprocess calls."""
    import subprocess
    mock = MagicMock()
    mock.Popen = MagicMock()
    mock.run = MagicMock()
    mock.DEVNULL = subprocess.DEVNULL
    mock.PIPE = subprocess.PIPE
    monkeypatch.setattr("subprocess.Popen", mock.Popen)
    monkeypatch.setattr("subprocess.run", mock.run)
    return mock
