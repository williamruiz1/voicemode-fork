"""Conversation tools for interactive voice interactions."""

import asyncio
import functools
import json
import logging
import os
import threading
import time
import traceback
from typing import Callable, Optional, Literal, Tuple, Dict, Union
from pathlib import Path
from datetime import datetime

import numpy as np
import sounddevice as sd
from scipy import signal  # module level: a cold import inside the listen loop stalled Send >1s
from scipy.io.wavfile import write
from pydub import AudioSegment
from openai import AsyncOpenAI
import httpx

# Optional webrtcvad for silence detection
try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError as e:
    webrtcvad = None
    VAD_AVAILABLE = False

# Silero VAD-based endpointing (voice_mode/silero_vad.py) -- an opt-in
# replacement for the webrtcvad decision above (config.ENDPOINTING_ENABLED).
# Frozen interface; SILERO_AVAILABLE degrades gracefully to False (no import
# error) when onnxruntime or the vendored model aren't present.
from voice_mode.silero_vad import SILERO_AVAILABLE, Endpointer

from voice_mode.server import mcp
from voice_mode.conch import Conch
from voice_mode.conversation_logger import get_conversation_logger
from voice_mode.config import (
    audio_operation_lock,
    SAMPLE_RATE,
    CHANNELS,
    DEBUG,
    DEBUG_DIR,
    VAD_DEBUG,
    SAVE_AUDIO,
    AUDIO_DIR,
    OPENAI_API_KEY,
    PREFER_LOCAL,
    AUDIO_FEEDBACK_ENABLED,
    service_processes,
    HTTP_CLIENT_CONFIG,
    save_transcription,
    SAVE_TRANSCRIPTIONS,
    DISABLE_SILENCE_DETECTION,
    VAD_AGGRESSIVENESS,
    SILENCE_THRESHOLD_MS,
    MIN_RECORDING_DURATION,
    VAD_ENERGY_THRESHOLD,
    SKIP_TTS,
    TTS_SPEED,
    VAD_CHUNK_DURATION_MS,
    INITIAL_SILENCE_GRACE_PERIOD,
    ENDPOINTING_ENABLED,
    ENDPOINTING_MIN_ENDPOINT_MS,
    ENDPOINTING_SPEECH_THRESHOLD,
    ENDPOINTING_MIN_SPEECH_MS,
    DEFAULT_LISTEN_DURATION,
    TTS_VOICES,
    TTS_MODELS,
    REPEAT_PHRASES,
    WAIT_PHRASES,
    WAIT_DURATION,
    METRICS_LEVEL,
    STT_AUDIO_FORMAT,
    STT_SAVE_FORMAT,
    MP3_BITRATE,
    CONCH_ENABLED,
    CONCH_TIMEOUT,
    CONCH_CHECK_INTERVAL,
    CONCH_YIELD_ENABLED,
    CONCH_YIELD_GRACE_SECONDS,
    CONCH_PREEMPT_TTS_GRACE,
    AUTO_FOCUS_PANE,
    STT_MODEL,
    STEP_AWAY_ENV,
    STEP_AWAY_FLAG_PATH,
    STEP_AWAY_GRACE_SECONDS,
    STEP_AWAY_CHECKIN_SECONDS,
    APPEND_WINDOW_MS,
    APPEND_WINDOW_FLAG_PATH,
    STEP_AWAY_PHRASES,
    STEP_AWAY_RESUME_PHRASES,
)
import voice_mode.config
from voice_mode.provider_discovery import provider_registry
from voice_mode.core import (
    get_openai_clients,
    text_to_speech,
    cleanup as cleanup_clients,
    save_debug_file,
    get_debug_filename,
    get_audio_path,
    play_chime_start,
    play_chime_end,
    play_system_audio
)
from voice_mode.audio_player import NonBlockingAudioPlayer, SPEAKING_FLAG_PATH, convomode_paused, PAUSE_FLAG_PATH
from voice_mode import audio_player
from voice_mode import airpods_press
from voice_mode.pause_resume import resume_after_pause, pause_max_seconds, we_hold_floor
from voice_mode import barge_in
from voice_mode.statistics_tracking import track_voice_interaction
from voice_mode.utils import (
    get_event_logger,
    log_recording_start,
    log_recording_end,
    log_stt_start,
    log_stt_complete,
    log_tool_request_start,
    log_tool_request_end,
    update_latest_symlinks
)
from voice_mode.pronounce import get_manager as get_pronounce_manager, is_enabled as pronounce_enabled

logger = logging.getLogger("voicemode")

# Log silence detection config at module load time
logger.info(f"Module loaded with DISABLE_SILENCE_DETECTION={DISABLE_SILENCE_DETECTION}")


def is_tmux() -> bool:
    """Check if the current process is running inside a tmux session."""
    return bool(os.environ.get("TMUX"))


def _is_focus_held() -> bool:
    """Check if another tool recently took visual focus (the 'visual conch').

    Returns True if ~/.voicemode/focus-hold exists and was modified within
    the hold period, meaning auto-focus should be suppressed to let the
    user view what was shown (e.g. a file opened by show-me).

    The hold duration is read from the file contents (written by show-me's
    --hold flag), falling back to VOICEMODE_FOCUS_HOLD_SECONDS env var,
    then 30 seconds.
    """
    hold_file = os.path.expanduser("~/.voicemode/focus-hold")
    default_hold = float(os.environ.get("VOICEMODE_FOCUS_HOLD_SECONDS", "30"))
    try:
        age = time.time() - os.path.getmtime(hold_file)
        # Read hold duration from file (written by show-me --hold)
        try:
            with open(hold_file) as f:
                hold_seconds = float(f.read().strip())
        except (ValueError, OSError):
            hold_seconds = default_hold
        return age < hold_seconds
    except (OSError, ValueError):
        return False


def focus_tmux_pane() -> None:
    """Make the agent's tmux window visible, and optionally switch a client.

    Steps:
    1. Check focus-hold sentinel — skip if another tool recently took focus
    2. select-window: make the agent's window current (without changing active pane)
    3. Check if any client is already showing this session — if so, stop
    4. If no client is showing the session, switch the focused client to it

    Deliberately does NOT call select-pane — this avoids stealing focus from
    whichever pane the user is currently working in.  The window becomes
    visible so the user can see the agent is speaking, but their cursor stays
    where it was.

    Silent no-op if not in tmux, TMUX_PANE is unset, or tmux is not found.
    """
    import subprocess

    tmux_pane = os.environ.get("TMUX_PANE", "")
    if not tmux_pane:
        return

    # Respect the visual conch — another tool recently took focus
    if _is_focus_held():
        return

    try:
        # Select the window containing our pane (without changing active pane).
        # This makes the window visible but doesn't steal focus from whichever
        # pane the user is currently looking at.
        subprocess.run(["tmux", "select-window", "-t", tmux_pane], capture_output=True)

        # Find which session owns this pane
        r = subprocess.run(
            ["tmux", "display-message", "-t", tmux_pane, "-p", "#{session_name}"],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            return
        session_name = r.stdout.strip()

        # Check if any client is already attached to this session
        r = subprocess.run(
            ["tmux", "list-clients", "-t", session_name, "-F", "#{client_tty}"],
            capture_output=True, text=True
        )
        if r.returncode == 0 and r.stdout.strip():
            # Session already visible in a terminal — don't steal focus
            return

        # No client is showing our session — switch the focused client to it
        r = subprocess.run(
            ["tmux", "list-clients", "-F", "#{client_tty} #{client_flags}"],
            capture_output=True, text=True
        )
        for line in r.stdout.strip().split("\n"):
            parts = line.split(" ", 1)
            if len(parts) == 2 and "focused" in parts[1]:
                client_tty = parts[0]
                subprocess.run(
                    ["tmux", "switch-client", "-c", client_tty, "-t", session_name],
                    capture_output=True
                )
                break
    except FileNotFoundError:
        pass  # tmux binary not installed


# DJ Ducking Configuration
DJ_SOCKET_PATH = "/tmp/voicemode-mpv.sock"
DJ_VOLUME_DUCK_AMOUNT = int(os.environ.get("VOICEMODE_DJ_DUCK_AMOUNT", "20"))  # Volume reduction during TTS


def _dj_command(cmd: str) -> Optional[str]:
    """Send a command to mpv-dj via IPC socket.

    Args:
        cmd: JSON command to send (e.g., '{ "command": ["get_property", "volume"] }')

    Returns:
        Response string from mpv, or None if DJ not running
    """
    import subprocess
    import json

    if not os.path.exists(DJ_SOCKET_PATH):
        return None

    try:
        result = subprocess.run(
            ["socat", "-", DJ_SOCKET_PATH],
            input=cmd + "\n",
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_dj_volume() -> Optional[float]:
    """Get current DJ volume level.

    Returns:
        Current volume (0-100) or None if DJ not running
    """
    import json
    response = _dj_command('{ "command": ["get_property", "volume"] }')
    if response:
        try:
            data = json.loads(response)
            if "data" in data:
                return float(data["data"])
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
    return None


def set_dj_volume(volume: float) -> bool:
    """Set DJ volume level.

    Args:
        volume: Volume level (0-100)

    Returns:
        True if successful, False otherwise
    """
    import json
    volume = max(0, min(100, volume))  # Clamp to valid range
    response = _dj_command(f'{{ "command": ["set_property", "volume", {volume}] }}')
    if response:
        try:
            data = json.loads(response)
            return data.get("error") == "success"
        except json.JSONDecodeError:
            pass
    return False


class DJDucker:
    """Context manager for ducking DJ volume during TTS playback."""

    def __init__(self, duck_amount: int = None):
        self.duck_amount = duck_amount if duck_amount is not None else DJ_VOLUME_DUCK_AMOUNT
        self.original_volume: Optional[float] = None
        self.ducked = False

    def __enter__(self):
        self.original_volume = get_dj_volume()
        if self.original_volume is not None:
            ducked_volume = max(0, self.original_volume - self.duck_amount)
            if set_dj_volume(ducked_volume):
                self.ducked = True
                logger.debug(f"DJ ducked: {self.original_volume:.0f}% -> {ducked_volume:.0f}%")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ducked and self.original_volume is not None:
            if set_dj_volume(self.original_volume):
                logger.debug(f"DJ restored: {self.original_volume:.0f}%")
        return False  # Don't suppress exceptions


def should_repeat(text: str) -> bool:
    """
    Check if the transcribed text ends with a repeat phrase.

    Args:
        text: The transcribed text to check

    Returns:
        True if text ends with a repeat phrase, False otherwise
    """
    if not text:
        return False

    # Normalize text for comparison (lowercase, strip whitespace and punctuation)
    import string
    normalized_text = text.lower().strip().rstrip(string.punctuation).strip()

    # Check if any repeat phrase appears at the end
    for phrase in REPEAT_PHRASES:
        if normalized_text.endswith(phrase.lower().strip()):
            logger.info(f"Repeat phrase detected: '{phrase}' in '{text}'")
            return True

    return False


def should_wait(text: str) -> bool:
    """
    Check if the transcribed text ends with a wait phrase.

    Args:
        text: The transcribed text to check

    Returns:
        True if text ends with a wait phrase, False otherwise
    """
    if not text:
        return False

    # Normalize text for comparison (lowercase, strip whitespace and punctuation)
    import string
    normalized_text = text.lower().strip().rstrip(string.punctuation).strip()

    # Check if any wait phrase appears at the end
    for phrase in WAIT_PHRASES:
        if normalized_text.endswith(phrase.lower().strip()):
            logger.info(f"Wait phrase detected: '{phrase}' in '{text}'")
            return True

    return False


# Track last session end time for measuring AI thinking time
last_session_end_time = None

# Initialize OpenAI clients - now using provider registry for endpoint discovery
openai_clients = get_openai_clients(OPENAI_API_KEY or "dummy-key-for-local", None, None)

# Provider-specific clients are now created dynamically by the provider registry


async def startup_initialization():
    """Initialize services on startup based on configuration"""
    if voice_mode.config._startup_initialized:
        return
    
    voice_mode.config._startup_initialized = True
    logger.info("Running startup initialization...")
    
    # Initialize provider registry
    logger.info("Initializing provider registry...")
    await provider_registry.initialize()
    
    # Check if we should auto-start Kokoro
    auto_start_kokoro = os.getenv("VOICE_MODE_AUTO_START_KOKORO", "").lower() in ("true", "1", "yes", "on")
    if auto_start_kokoro:
        try:
            # Check if Kokoro is already running
            async with httpx.AsyncClient(timeout=3.0) as client:
                base_url = 'http://127.0.0.1:8880'  # Kokoro default
                health_url = f"{base_url}/health"
                response = await client.get(health_url)
                
                if response.status_code == 200:
                    logger.info("Kokoro TTS is already running externally")
                else:
                    raise Exception("Not running")
        except:
            # Kokoro is not running, start it
            logger.info("Auto-starting Kokoro TTS service...")
            try:
                # Import here to avoid circular dependency
                import subprocess
                if "kokoro" not in service_processes:
                    process = subprocess.Popen(
                        ["uvx", "kokoro-fastapi"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env={**os.environ}
                    )
                    service_processes["kokoro"] = process
                    
                    # Wait a moment for it to start
                    await asyncio.sleep(2.0)
                    
                    # Verify it started
                    if process.poll() is None:
                        logger.info(f"✓ Kokoro TTS started successfully (PID: {process.pid})")
                    else:
                        logger.error("Failed to start Kokoro TTS")
            except Exception as e:
                logger.error(f"Error auto-starting Kokoro: {e}")
    
    # Log initial status
    logger.info("Service initialization complete")


async def get_tts_config(provider: Optional[str] = None, voice: Optional[str] = None, model: Optional[str] = None, instructions: Optional[str] = None):
    """Get TTS configuration - simplified to use direct config"""
    from voice_mode.provider_discovery import detect_provider_type
    from voice_mode.voice_profiles import is_clone_voice, get_profile

    # Check if this is a clone voice — override provider/model/base_url
    if voice and is_clone_voice(voice):
        profile = get_profile(voice)
        logger.info(f"Voice '{voice}' is a clone profile: {profile.description}")
        return {
            'base_url': profile.base_url,
            'model': profile.model,
            'voice': voice,
            'instructions': None,
            'provider_type': 'clone'
        }

    # Validate instructions usage
    if instructions and model != "gpt-4o-mini-tts":
        logger.warning(f"Instructions parameter is only supported with gpt-4o-mini-tts model, ignoring for model: {model}")
        instructions = None

    # Map provider names to base URLs
    provider_urls = {
        'openai': 'https://api.openai.com/v1',
        'kokoro': 'http://127.0.0.1:8880/v1'
    }

    # Convert provider name to URL if it's a known provider
    base_url = None
    if provider:
        base_url = provider_urls.get(provider, provider)

    # Use first available endpoint from config
    if not base_url:
        base_url = TTS_BASE_URLS[0] if TTS_BASE_URLS else 'https://api.openai.com/v1'

    provider_type = detect_provider_type(base_url)

    # Return simplified configuration
    return {
        'base_url': base_url,
        'model': model or TTS_MODELS[0] if TTS_MODELS else 'tts-1',
        'voice': voice or TTS_VOICES[0] if TTS_VOICES else 'alloy',
        'instructions': instructions,
        'provider_type': provider_type
    }


async def get_stt_config(provider: Optional[str] = None):
    """Get STT configuration - simplified to use direct config"""
    from voice_mode.provider_discovery import detect_provider_type
    from voice_mode.config import STT_BASE_URLS

    # Map provider names to base URLs
    provider_urls = {
        'whisper-local': 'http://127.0.0.1:2022/v1',
        'openai-whisper': 'https://api.openai.com/v1'
    }

    # Convert provider name to URL if it's a known provider
    base_url = None
    if provider:
        base_url = provider_urls.get(provider, provider)

    # Use first available endpoint from config
    if not base_url:
        base_url = STT_BASE_URLS[0] if STT_BASE_URLS else 'https://api.openai.com/v1'

    provider_type = detect_provider_type(base_url)

    # Return simplified configuration
    return {
        'base_url': base_url,
        'model': STT_MODEL,
        'provider': 'whisper-local' if '127.0.0.1' in base_url or 'localhost' in base_url else 'openai-whisper',
        'provider_type': provider_type
    }



def resolve_ref_text(ref_text: Optional[str]) -> Optional[str]:
    """Resolve a ``ref_text`` argument that may be a file path OR literal text.

    Auto-detect: if the value names an existing file, its contents (stripped)
    are used as the transcript; otherwise the value is treated as the literal
    transcript text. Returns ``None`` when no override was supplied, so callers
    can distinguish "use the profile/sidecar transcript" from an explicit
    override.

    Note: path detection is local-only. For a remote TTS host the *audio*
    path must exist on that host, but ``ref_text`` is sent as a string, so
    reading the transcript file locally here is always correct.
    """
    if ref_text is None:
        return None
    candidate = os.path.expanduser(ref_text)
    try:
        if os.path.isfile(candidate):
            with open(candidate, "r") as fh:
                return fh.read().strip()
    except OSError:
        # Fall through and treat the value as literal text.
        pass
    return ref_text


async def text_to_speech_with_failover(
    message: str,
    voice: Optional[str] = None,
    model: Optional[str] = None,
    instructions: Optional[str] = None,
    audio_format: Optional[str] = None,
    initial_provider: Optional[str] = None,
    speed: Optional[float] = None,
    ref_text: Optional[str] = None
) -> Tuple[bool, Optional[dict], Optional[dict]]:
    """
    Text to speech with automatic failover to next available endpoint.
    
    Returns:
        Tuple of (success, tts_metrics, tts_config)
    """
    # Apply pronunciation rules if enabled
    if pronounce_enabled():
        pronounce_mgr = get_pronounce_manager()
        message = pronounce_mgr.process_tts(message)

    # Always use simple failover (the only mode now)
    from voice_mode.simple_failover import simple_tts_failover
    return await simple_tts_failover(
        text=message,
        voice=voice or TTS_VOICES[0],
        model=model or TTS_MODELS[0],
        instructions=instructions,
        audio_format=audio_format,
        debug=DEBUG,
        debug_dir=DEBUG_DIR if DEBUG else None,
        save_audio=SAVE_AUDIO,
        audio_dir=AUDIO_DIR if SAVE_AUDIO else None,
        speed=speed,
        ref_text=ref_text
    )


def prepare_audio_for_stt(audio_data: np.ndarray, output_format: str = "mp3") -> bytes:
    """
    Prepare audio data for STT upload with optional compression.

    Converts raw audio to the specified format, optionally compressing and
    downsampling to 16kHz (Whisper's native rate) for optimal bandwidth.

    Args:
        audio_data: Raw audio data as numpy array (16-bit PCM)
        output_format: Target format ('mp3', 'wav', 'flac', etc.)

    Returns:
        Compressed audio data as bytes
    """
    import io

    # Create AudioSegment from raw data
    # Audio is recorded at SAMPLE_RATE (24kHz), 16-bit mono
    audio = AudioSegment(
        audio_data.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=2,  # 16-bit = 2 bytes
        channels=CHANNELS
    )

    # Calculate original size for logging
    original_size = len(audio_data) * 2  # 16-bit = 2 bytes per sample

    # Downsample to 16kHz (Whisper's native rate) for better compression
    # This also reduces size by ~33% even before compression
    whisper_sample_rate = 16000
    if SAMPLE_RATE != whisper_sample_rate:
        audio = audio.set_frame_rate(whisper_sample_rate)

    # Export to target format
    buffer = io.BytesIO()

    if output_format == "mp3":
        # Use configured bitrate for MP3 (default 32k for speech)
        audio.export(buffer, format="mp3", bitrate=MP3_BITRATE)
    elif output_format == "wav":
        # WAV is uncompressed but we still benefit from downsampling
        audio.export(buffer, format="wav")
    elif output_format == "flac":
        # FLAC is lossless compression
        audio.export(buffer, format="flac")
    else:
        # Default to MP3 for unknown formats
        logger.warning(f"Unknown STT format '{output_format}', falling back to MP3")
        audio.export(buffer, format="mp3", bitrate=MP3_BITRATE)

    compressed_data = buffer.getvalue()
    compressed_size = len(compressed_data)

    # Log compression ratio
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    logger.info(f"STT audio prepared: {original_size/1024:.1f}KB -> {compressed_size/1024:.1f}KB "
                f"({output_format}, {compression_ratio:.1f}x compression)")

    return compressed_data


async def speech_to_text(
    audio_data: np.ndarray,
    save_audio: bool = False,
    audio_dir: Optional[Path] = None,
    transport: str = "local"
) -> Optional[Dict]:
    """
    Convert audio to text with automatic failover.

    Handles audio file preparation (saving permanently or using temp file) and
    delegates to simple_stt_failover for the actual transcription attempts.

    For remote endpoints: Audio is compressed (MP3 at 32kbps) and downsampled
    to 16kHz to reduce bandwidth usage when uploading.

    For local endpoints: Audio is sent as WAV to skip compression overhead,
    since network bandwidth isn't a bottleneck for localhost/LAN connections.

    Original full-quality WAV is saved separately when save_audio is enabled.

    Args:
        audio_data: Raw audio data as numpy array
        save_audio: Whether to save the audio file permanently
        audio_dir: Directory to save audio files (if save_audio is True)
        transport: Transport method (for logging context)

    Returns:
        Dict with transcription result or error information:
        - Success: {"text": "...", "provider": "...", "endpoint": "..."}
        - No speech: {"error_type": "no_speech", "provider": "..."}
        - All failed: {"error_type": "connection_failed", "attempted_endpoints": [...]}
    """
    import tempfile
    import io
    from voice_mode.conversation_logger import get_conversation_logger
    from voice_mode.core import save_debug_file, get_debug_filename
    from voice_mode.simple_failover import simple_stt_failover
    from voice_mode.config import STT_BASE_URLS, STT_COMPRESS, STT_SILENCE_GATE, STT_SILENCE_RMS_FLOOR
    from voice_mode.provider_discovery import is_local_provider

    # Pre-STT silence gate (founder-os#11657 Part B) — skip near-silent audio so
    # Whisper never sees a clip quiet enough to hallucinate a closing phrase
    # ("Thank you for watching"). audio_data is int16 PCM; normalize to [-1, 1]
    # for a scale-independent whole-clip RMS. Inert/reversible: gated on
    # STT_SILENCE_GATE (default on) + a floor calibrated well below real speech.
    if STT_SILENCE_GATE and STT_SILENCE_RMS_FLOOR > 0 and audio_data is not None and len(audio_data) > 0:
        rms = float(np.sqrt(np.mean((audio_data.astype(np.float64) / 32768.0) ** 2)))
        if rms < STT_SILENCE_RMS_FLOOR:
            logger.info(
                f"STT: silence gate tripped — clip RMS {rms:.5f} < floor "
                f"{STT_SILENCE_RMS_FLOOR:.5f}; skipping STT call (no_speech) to "
                f"avoid Whisper silence-hallucination"
            )
            return {
                "error_type": "no_speech",
                "provider": "silence-gate",
                "metrics": {
                    "file_size_bytes": len(audio_data) * 2,
                    "request_time_ms": 0.0,
                    "is_local": True,
                    "rms": rms,
                    "rms_floor": STT_SILENCE_RMS_FLOOR,
                },
            }

    # Determine compression based on STT_COMPRESS mode
    # Options: auto (default), always, never
    primary_endpoint = STT_BASE_URLS[0] if STT_BASE_URLS else 'https://api.openai.com/v1'
    is_local = is_local_provider(primary_endpoint)

    if STT_COMPRESS == "never":
        # Never compress - always use WAV
        stt_format = "wav"
        logger.info(f"STT: Compression disabled (mode=never), using WAV")
    elif STT_COMPRESS == "always":
        # Always compress regardless of endpoint type
        stt_format = STT_AUDIO_FORMAT if STT_AUDIO_FORMAT != "pcm" else "mp3"
        logger.info(f"STT: Compression forced (mode=always), using {stt_format}")
    else:
        # Auto mode (default): compress for remote, skip for local
        if is_local:
            # Local endpoint: use WAV to skip compression overhead (~200-800ms saved)
            stt_format = "wav"
            logger.info(f"STT: Local endpoint detected ({primary_endpoint}), skipping compression")
        else:
            # Remote endpoint: compress to reduce bandwidth (~90% smaller)
            stt_format = STT_AUDIO_FORMAT if STT_AUDIO_FORMAT != "pcm" else "mp3"
            logger.info(f"STT: Remote endpoint ({primary_endpoint}), using {stt_format} compression")

    # Prepare audio for upload (compressed for remote, WAV for local)
    compressed_audio = prepare_audio_for_stt(audio_data, stt_format)

    # Determine file extension based on format
    file_extension = stt_format if stt_format in ["mp3", "wav", "flac", "m4a", "ogg"] else "mp3"

    # Determine if we should save the file permanently or use a temp file
    if save_audio and audio_dir:
        # Save files for debugging/analysis
        conversation_logger = get_conversation_logger()
        conversation_id = conversation_logger.conversation_id

        # Create year/month directory structure
        now = datetime.now()
        year_dir = audio_dir / str(now.year)
        month_dir = year_dir / f"{now.month:02d}"
        month_dir.mkdir(parents=True, exist_ok=True)

        # Save recording in configured format (default: wav for full quality)
        save_filename = get_debug_filename("stt", STT_SAVE_FORMAT, conversation_id)
        save_file_path = month_dir / save_filename

        if STT_SAVE_FORMAT == "wav":
            # Save as uncompressed WAV for full quality archival
            write(str(save_file_path), SAMPLE_RATE, audio_data)
        else:
            # Save in configured compressed format
            saved_audio = prepare_audio_for_stt(audio_data, STT_SAVE_FORMAT)
            with open(save_file_path, 'wb') as f:
                f.write(saved_audio)

        logger.info(f"STT audio saved to: {save_file_path} (format: {STT_SAVE_FORMAT})")

        # Update latest symlinks for quick access to most recent STT audio
        update_latest_symlinks(save_file_path, "stt")

        # Use compressed audio for upload (temporary file)
        # Windows fix: close temp file before reopening (Issue #135)
        tmp_file = tempfile.NamedTemporaryFile(suffix=f'.{file_extension}', delete=False)
        tmp_path = tmp_file.name
        try:
            tmp_file.write(compressed_audio)
            tmp_file.flush()
            tmp_file.close()  # Close before reopening on Windows

            with open(tmp_path, 'rb') as audio_file:
                result = await simple_stt_failover(
                    audio_file=audio_file,
                )
        finally:
            # Clean up temp file (we keep the WAV)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    else:
        # Use temporary file that will be deleted
        # Windows fix: close temp file before reopening (Issue #135)
        tmp_file = tempfile.NamedTemporaryFile(suffix=f'.{file_extension}', delete=False)
        tmp_path = tmp_file.name
        try:
            tmp_file.write(compressed_audio)
            tmp_file.flush()
            tmp_file.close()  # Close before reopening on Windows

            with open(tmp_path, 'rb') as audio_file:
                result = await simple_stt_failover(
                    audio_file=audio_file,
                )
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return result


async def play_audio_feedback(
    text: str,
    openai_clients: dict,
    enabled: Optional[bool] = None,
    style: str = "whisper",
    feedback_type: Optional[str] = None,
    voice: str = "nova",
    model: str = "gpt-4o-mini-tts",
    chime_leading_silence: Optional[float] = None,
    chime_trailing_silence: Optional[float] = None
) -> None:
    """Play an audio feedback chime

    Args:
        text: Which chime to play (either "listening" or "finished")
        openai_clients: OpenAI client instances (kept for compatibility, not used)
        enabled: Override global audio feedback setting
        style: Kept for compatibility, not used
        feedback_type: Kept for compatibility, not used
        voice: Kept for compatibility, not used
        model: Kept for compatibility, not used
        chime_leading_silence: Optional override for pre-chime silence duration
        chime_trailing_silence: Optional override for post-chime silence duration
    """
    # Use parameter override if provided, otherwise use global setting
    if enabled is False:
        return
    
    # If enabled is None, use global setting
    if enabled is None:
        enabled = AUDIO_FEEDBACK_ENABLED
    
    # Skip if disabled
    if not enabled:
        return
    
    try:
        # Play appropriate chime with optional delay overrides
        if text == "listening":
            await play_chime_start(
                leading_silence=chime_leading_silence,
                trailing_silence=chime_trailing_silence
            )
        elif text == "finished":
            await play_chime_end(
                leading_silence=chime_leading_silence,
                trailing_silence=chime_trailing_silence
            )
    except Exception as e:
        logger.debug(f"Audio feedback failed: {e}")
        # Don't interrupt the main flow if feedback fails


def _bluetooth_input_active() -> bool:
    """True when the default input device is a Bluetooth headset (AirPods etc.).

    On macOS an input-only open does not reliably force a Bluetooth link into
    HFP (mic-capable) mode — the link stays in A2DP playback and the mic reads
    pure digital silence (airpods-mic-pinned-playback RCA 2026-06-27:
    input-only RMS 0.000 DEAD vs full-duplex RMS 0.099 LIVE). Bluetooth inputs
    therefore get a FULL-DUPLEX open (silent output + live input) for the
    listen window so the OS negotiates and holds HFP. Wired/built-in inputs
    keep the plain input-only path. Predicate mirrors core.py's device check.
    """
    try:
        device_name = str(sd.query_devices(kind='input')['name']).lower()
    except Exception as e:
        logger.debug(f"Bluetooth input probe failed (assuming non-BT): {e}")
        return False
    return 'airpod' in device_name or 'bluetooth' in device_name or 'bt' in device_name


def _duplex_device_pair():
    """(input, output) device pair for a full-duplex open on a BT headset.

    Prefer the output device whose name matches the default input (the same
    headset — the duplex open must ride the SAME Bluetooth link to pin HFP);
    fall back to the system default output (None) when no name match exists.
    """
    in_dev = sd.default.device[0]
    try:
        in_name = sd.query_devices(kind='input')['name']
        for idx, dev in enumerate(sd.query_devices()):
            if dev['max_output_channels'] > 0 and dev['name'] == in_name:
                return (in_dev, idx)
    except Exception as e:
        logger.debug(f"Duplex output resolution failed (using default output): {e}")
    return (in_dev, None)


def _record_audio_duplex(samples_to_record: int, honor_send: bool = False) -> np.ndarray:
    """Fixed-length capture over a full-duplex stream whose output is silence.

    Used instead of sd.rec() when the input is a Bluetooth headset — see
    _bluetooth_input_active. Returns an (n, 1) int16 array like sd.rec would.
    honor_send: stop early (keeping what was captured) on turn-end.signal.
    """
    import time as _time
    captured = []
    remaining = [samples_to_record]

    def _callback(indata, outdata, frames, time, status):
        if status:
            logger.warning(f"Duplex stream status: {status}")
        outdata.fill(0)
        if remaining[0] > 0:
            captured.append(indata.copy())
            remaining[0] -= frames

    device = _duplex_device_pair()
    logger.info(f"🎧 Bluetooth input — full-duplex capture (F1), device pair {device}")
    deadline = _time.monotonic() + samples_to_record / SAMPLE_RATE + 2.0
    with sd.Stream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.int16,
                   device=device, callback=_callback):
        while remaining[0] > 0 and _time.monotonic() < deadline:
            if honor_send and _send_pending():
                logger.info("✓ Manual turn-end signal received — stopping fixed-length recording")
                _consume_send()
                break
            sd.sleep(50)
    if not captured:
        return np.zeros((0, 1), dtype=np.int16)
    data = np.concatenate([c.reshape(-1) for c in captured])
    return data[:samples_to_record].reshape(-1, 1)


def record_audio(duration: float, honor_send: bool = False, _attempt: int = 0) -> np.ndarray:
    """Record audio from microphone.

    honor_send: end early on turn-end.signal, returning what was captured --
    set by every listen path that falls back here, so Send still ends the
    listen. Device-error retries are capped at DEVICE_RECOVERY_MAX_ATTEMPTS.
    """
    logger.info(f"🎤 Recording audio for {duration}s...")
    if DEBUG:
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            logger.debug(f"Default input device: {default_input} - {devices[default_input]['name'] if default_input is not None else 'None'}")
            logger.debug(f"Recording config - Sample rate: {SAMPLE_RATE}Hz, Channels: {CHANNELS}, dtype: int16")
        except Exception as dev_e:
            logger.error(f"Error querying audio devices: {dev_e}")
    
    # Save current stdio state
    import sys
    original_stdin = sys.stdin
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    try:
        samples_to_record = int(duration * SAMPLE_RATE)
        logger.debug(f"Recording {samples_to_record} samples...")
        
        if _bluetooth_input_active():
            # F1 (founder-os#13215): BT headsets need a full-duplex open to
            # hold HFP for the listen window; input-only reads dead silence.
            recording = _record_audio_duplex(samples_to_record, honor_send=honor_send)
        else:
            recording = sd.rec(
                samples_to_record,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=np.int16
            )
            if honor_send:
                # Poll instead of sd.wait(), which blocks until the full
                # duration has been captured and never sees Send.
                started = time.monotonic()
                while True:
                    elapsed = time.monotonic() - started
                    if _send_pending():
                        logger.info("✓ Manual turn-end signal received — stopping fixed-length recording")
                        _consume_send()
                        recording = recording[:min(samples_to_record, int(elapsed * SAMPLE_RATE))]
                        break
                    if elapsed >= duration + 0.25:
                        break
                    time.sleep(SEND_POLL_SECONDS)
                sd.stop()
            else:
                sd.wait()
        
        flattened = recording.flatten()
        logger.info(f"✓ Recorded {len(flattened)} samples")
        
        if DEBUG:
            logger.debug(f"Recording stats - Min: {flattened.min()}, Max: {flattened.max()}, Mean: {flattened.mean():.2f}")
            # Check if recording contains actual audio (not silence)
            rms = np.sqrt(np.mean(flattened.astype(float) ** 2))
            logger.debug(f"RMS level: {rms:.2f} ({'likely silence' if rms < 100 else 'audio detected'})")
        
        return flattened
        
    except Exception as e:
        logger.error(f"Recording failed: {e}")
        logger.error(f"Audio config when error occurred - Sample rate: {SAMPLE_RATE}, Channels: {CHANNELS}")
        
        # Check if this is a device error that might be recoverable
        error_str = str(e).lower()
        recoverable = any(err in error_str for err in ['device unavailable', 'device disconnected',
                                                        'invalid device', 'unanticipated host error',
                                                        'portaudio error'])
        if recoverable and _attempt + 1 >= DEVICE_RECOVERY_MAX_ATTEMPTS:
            logger.error(f"Audio device still failing after {DEVICE_RECOVERY_MAX_ATTEMPTS} attempts - giving up")
        elif recoverable:
            logger.info("Audio device error detected - attempting to reinitialize audio system")
            
            # Try to reinitialize sounddevice
            try:
                # Get current default device info before reinit
                try:
                    old_device = sd.query_devices(kind='input')
                    old_device_name = old_device.get('name', 'Unknown')
                except:
                    old_device_name = 'Previous device'
                
                sd._terminate()
                sd._initialize()
                
                # Get new default device info
                try:
                    new_device = sd.query_devices(kind='input')
                    new_device_name = new_device.get('name', 'Unknown')
                    logger.info(f"Audio system reinitialized - switched from '{old_device_name}' to '{new_device_name}'")
                except:
                    logger.info("Audio system reinitialized - retrying with new default device")
                
                # Wait a moment for the system to stabilize (Send ends it)
                if honor_send and _sleep_unless_send(DEVICE_RECOVERY_DELAY_SECONDS):
                    logger.info("✓ Manual turn-end signal received during device recovery")
                    _consume_send()
                    return np.array([], dtype=np.int16)
                if not honor_send:
                    time.sleep(DEVICE_RECOVERY_DELAY_SECONDS)

                # Try recording again with the new device (recursive call)
                logger.info("Retrying recording with new audio device...")
                return record_audio(duration, honor_send=honor_send, _attempt=_attempt + 1)
                
            except Exception as reinit_error:
                logger.error(f"Failed to reinitialize audio: {reinit_error}")
                # Fall through to normal error handling
        
        # Import here to avoid circular imports
        from voice_mode.utils.audio_diagnostics import get_audio_error_help
        
        # Get helpful error message
        help_message = get_audio_error_help(e)
        logger.error(f"\n{help_message}")
        
        # Try to get more info about audio devices
        try:
            devices = sd.query_devices()
            logger.error(f"Available input devices:")
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    logger.error(f"  {i}: {device['name']} (inputs: {device['max_input_channels']})")
        except Exception as dev_e:
            logger.error(f"Cannot query audio devices: {dev_e}")
        
        return np.array([])
    finally:
        # Restore stdio if it was changed
        if sys.stdin != original_stdin:
            sys.stdin = original_stdin
        if sys.stdout != original_stdout:
            sys.stdout = original_stdout
        if sys.stderr != original_stderr:
            sys.stderr = original_stderr


async def _preempt_conch_after_tts_grace(conch: Conch, speaking_flag_path: str = None) -> bool:
    """Preempt-and-acquire the conch at the waiter's hard timeout — never mid-utterance.

    While the current holder is actually playing TTS (the speaking flag is
    present), grant up to CONCH_PREEMPT_TTS_GRACE extra seconds — the holder
    normally releases (or its idle listen yields) in that window. Only after
    the flag clears or the grace is exhausted does the force-clear happen.

    Returns True if the conch ended up acquired.
    """
    flag_path = speaking_flag_path or SPEAKING_FLAG_PATH
    grace = 0.0
    while (not conch.try_acquire()
           and os.path.exists(flag_path)
           and grace < CONCH_PREEMPT_TTS_GRACE):
        await asyncio.sleep(CONCH_CHECK_INTERVAL)
        grace += CONCH_CHECK_INTERVAL
    if not conch._acquired:
        conch.preempt_acquire("converse")
    return conch._acquired


# ==================== PAUSE / STEP-AWAY / APPEND HELPERS (founder-os#11655) ==========
# Small PURE helpers so the record loop's new behavior is unit-testable without a
# microphone. All resolve to inert values by default (step-away off, append 0).

def step_away_enabled() -> bool:
    """Runtime check: step-away is ON if the env flag is set OR the flag file
    exists (mirrors the natural-mode.flag discipline — toggle live, no respawn).
    Default (no env, no file) = False = fully inert."""
    try:
        if STEP_AWAY_ENV:
            return True
        return os.path.exists(STEP_AWAY_FLAG_PATH)
    except Exception:
        return False


def append_window_ms() -> int:
    """Runtime resolve of the append-to-turn grace (ms). A live knob file
    (~/.voicemode/append-window-ms with an integer body) overrides the env so it
    can be tuned without a respawn. Default 0 = OFF = byte-for-byte prior loop."""
    try:
        if os.path.exists(APPEND_WINDOW_FLAG_PATH):
            with open(APPEND_WINDOW_FLAG_PATH) as f:
                return max(0, int((f.read() or "0").strip() or "0"))
    except Exception:
        pass
    return max(0, APPEND_WINDOW_MS)


# ==================== HOLD-OPEN PUSH-TO-TALK ==========
# Three signal files under ~/.voicemode, resolved at CALL time (expanduser) so a
# hotkey can flip them live and tests can point HOME at a temp dir:
#   turn-end.signal  SEND -- ends the listen now (pre-existing, honored always).
#   hold-open.signal per-turn HOLD -- once seen during a listen, silence never
#                    ends it; removed by the listen on every exit.
#   hold-mode        persistent HOLD MODE -- every listen is held until SEND;
#                    never removed here.
# While held, the ceiling is VOICEMODE_HOLD_MAX_SECONDS (default 600) instead of
# listen_duration_max. Absence of both hold files = byte-for-byte prior behavior.

HOLD_OPEN_STALE_SECONDS = 15 * 60
DEFAULT_HOLD_MAX_SECONDS = 600.0

# Send must end a listen within ~0.5s in every recorder state (B17). A stream
# that stops delivering callbacks never advances the audio clock, so Send's
# floor and the stall watchdog are measured on the wall clock; device-error
# recovery is capped and its waits poll Send.
DEVICE_RECOVERY_MAX_ATTEMPTS = 3      # stream opens per listen before giving up
DEVICE_RECOVERY_DELAY_SECONDS = 0.5   # settle time after reinitializing PortAudio
DEFAULT_STREAM_STALL_SECONDS = 5.0
SEND_POLL_SECONDS = 0.05


def _turn_end_signal_path() -> str:
    return os.path.expanduser("~/.voicemode/turn-end.signal")


def _send_pending() -> bool:
    try:
        return os.path.exists(_turn_end_signal_path())
    except Exception:
        return False


# Which control pressed Send (B32): writers put a one-line tag in the signal
# file ("widget", "hotkey", "airpods", "script"); an empty file (older writers,
# a bare `touch`) reads as "unknown". The listen that consumes it reports the
# tag as RECORDING_END's ``send_source``.
SEND_SOURCE_UNKNOWN = "unknown"
_SEND_SOURCE_MAX_LEN = 32
_send_source_lock = threading.Lock()
_last_send_source: Optional[str] = None


def _read_send_source(path: str) -> str:
    """The signal file's first line as a short [a-z0-9._-] tag, else "unknown"."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            first = f.read(256).splitlines()
    except Exception:
        return SEND_SOURCE_UNKNOWN
    tag = first[0].strip().lower() if first else ""
    tag = "".join(c for c in tag if c.isalnum() or c in "._-")[:_SEND_SOURCE_MAX_LEN]
    return tag or SEND_SOURCE_UNKNOWN


def _consume_send() -> str:
    """Delete turn-end.signal, returning (and remembering for this listen's
    RECORDING_END) which control wrote it."""
    global _last_send_source
    path = _turn_end_signal_path()
    source = _read_send_source(path)
    try:
        os.remove(path)
    except Exception:
        pass
    with _send_source_lock:
        _last_send_source = source
    logger.info(f"✓ Send source: {source}")
    return source


def _take_send_source() -> Optional[str]:
    """Return and clear the source of the last consumed Send (None if none)."""
    global _last_send_source
    with _send_source_lock:
        source, _last_send_source = _last_send_source, None
    return source


def _clear_stale_send() -> None:
    """Listen entry: a Send left from an earlier turn must not pre-end this one.
    Its source is discarded, not reported -- it ended nothing."""
    try:
        os.remove(_turn_end_signal_path())
    except Exception:
        pass


def _run_checkin(callback: Callable[[], None]) -> None:
    try:
        callback()
    except Exception:
        pass


def _sleep_unless_send(seconds: float) -> bool:
    """Sleep up to `seconds`, polling Send. True = Send arrived (not consumed)."""
    deadline = time.monotonic() + seconds
    while True:
        if _send_pending():
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(SEND_POLL_SECONDS, remaining))


class _StreamStalled(Exception):
    """A held listen's input stream went quiet. Raised inside the listen loop so
    the device-recovery path reopens the stream (audio kept, Send preserved)."""


def stream_stall_seconds() -> float:
    """How long a listen tolerates an input stream that delivers nothing before
    ending with end_reason=stream_stalled. VOICEMODE_STREAM_STALL_SECONDS;
    missing, unparseable, or non-positive -> DEFAULT_STREAM_STALL_SECONDS."""
    try:
        value = float(os.getenv("VOICEMODE_STREAM_STALL_SECONDS", str(DEFAULT_STREAM_STALL_SECONDS)))
    except (TypeError, ValueError):
        return DEFAULT_STREAM_STALL_SECONDS
    return value if value > 0 else DEFAULT_STREAM_STALL_SECONDS


def _hold_open_signal_path() -> str:
    return os.path.expanduser("~/.voicemode/hold-open.signal")


def _hold_mode_path() -> str:
    return os.path.expanduser("~/.voicemode/hold-mode")


def hold_active() -> bool:
    """True while hold mode is on or the per-turn hold signal is present.
    Fail-safe: any error reads as not held (the normal VAD stop applies)."""
    try:
        return os.path.exists(_hold_mode_path()) or os.path.exists(_hold_open_signal_path())
    except Exception:
        return False


def hold_max_seconds() -> float:
    """Ceiling for a held listen, read from VOICEMODE_HOLD_MAX_SECONDS at call
    time. Missing, unparseable, or non-positive -> DEFAULT_HOLD_MAX_SECONDS."""
    try:
        value = float(os.getenv("VOICEMODE_HOLD_MAX_SECONDS", str(DEFAULT_HOLD_MAX_SECONDS)))
    except (TypeError, ValueError):
        return DEFAULT_HOLD_MAX_SECONDS
    return value if value > 0 else DEFAULT_HOLD_MAX_SECONDS


def _clear_stale_hold_open() -> None:
    """On listen entry, drop a hold-open.signal left over from a crashed turn
    (mtime older than HOLD_OPEN_STALE_SECONDS). A fresh one is kept: the user
    may press hold during TTS playback, before the listen starts."""
    path = _hold_open_signal_path()
    try:
        if time.time() - os.path.getmtime(path) > HOLD_OPEN_STALE_SECONDS:
            os.remove(path)
            logger.info("Removed stale hold-open signal")
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.debug(f"stale hold-open check failed (ignored): {e}")


def _clear_hold_open() -> None:
    """The per-turn hold ends with the turn. Never touches hold-mode."""
    try:
        os.remove(_hold_open_signal_path())
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.debug(f"hold-open cleanup failed (ignored): {e}")


def _listening_state_path() -> str:
    return os.path.expanduser("~/.voicemode/listening.json")


def _write_listening_state(started_at: float, held: bool) -> None:
    """Atomically publish ~/.voicemode/listening.json while the mic is open, so
    a hotkey/indicator can tell a listen is live and whether it is held."""
    path = _listening_state_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({"started_at": started_at, "held": bool(held), "pid": os.getpid()}, f)
        os.replace(tmp, path)
    except Exception as e:
        logger.debug(f"listening state write failed (ignored): {e}")


def _clear_listening_state() -> None:
    try:
        os.remove(_listening_state_path())
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.debug(f"listening state cleanup failed (ignored): {e}")


def _drain_delivered_audio(audio_queue, chunks: list) -> int:
    """On Send, keep every chunk the mic already delivered but the loop has
    not processed yet (a slow consumer would otherwise discard the backlog,
    i.e. the tail of what was said). Returns the number of chunks drained."""
    import queue as _queue
    drained = 0
    while True:
        try:
            chunk = audio_queue.get_nowait()
        except _queue.Empty:
            return drained
        if chunk is None:  # device-error sentinel
            continue
        chunks.append(chunk.flatten())
        drained += 1


def recording_end_event_data(duration: float, samples: int, listen_stats: Optional[dict]) -> Tuple[dict, bool]:
    """Build the RECORDING_END event payload and decide INPUT_STARVED.

    audio_seconds = samples / SAMPLE_RATE; wall_seconds = how long the mic was
    open. Starved = the mic was open for >= 1s but delivered less than half that
    much audio -- the signature of an input stream that stalled or a consumer
    that fell behind (chunks_delivered vs chunks_processed tells which)."""
    stats = listen_stats or {}
    wall = float(stats.get("wall_seconds", duration) or 0.0)
    audio_seconds = samples / SAMPLE_RATE
    data = {
        "duration": duration,
        "samples": samples,
        "audio_seconds": round(audio_seconds, 3),
        "wall_seconds": round(wall, 3),
        "end_reason": stats.get("end_reason", "other"),
        "held": bool(stats.get("held", False)),
    }
    if "send_source" in stats or data["end_reason"] == "send":
        data["send_source"] = stats.get("send_source") or SEND_SOURCE_UNKNOWN
    for key in ("chunks_delivered", "chunks_processed", "chunks_drained"):
        if key in stats:
            data[key] = stats[key]
    starved = wall >= 1.0 and audio_seconds < 0.5 * wall
    return data, starved


def _record_held_until_send(fallback_max_duration: float, listen_stats: Optional[dict] = None) -> Tuple[np.ndarray, bool]:
    """Held listen without VAD (disable_silence_detection, or webrtcvad missing):
    stream the mic until turn-end.signal (after MIN_RECORDING_DURATION) or the
    hold ceiling. Plain record_audio() never checks signals, so Send could not
    end it. Returns (audio, True) like the fixed-duration path, or (empty, False)
    when nothing was captured."""
    import queue

    stats = listen_stats if listen_stats is not None else {}
    stats.update(held=True, end_reason="hold_ceiling", chunks_delivered=0, chunks_processed=0)
    ceiling = hold_max_seconds()
    logger.info(f"🎤 Held listen (no VAD): recording until Send (max {ceiling:.0f}s)...")
    chunk_samples = int(SAMPLE_RATE * VAD_CHUNK_DURATION_MS / 1000)
    audio_queue = queue.Queue()

    def _callback(indata, frames, time_info, status):
        stats["chunks_delivered"] += 1
        audio_queue.put(indata.copy())

    def _duplex_callback(indata, outdata, frames, time_info, status):
        outdata.fill(0)
        _callback(indata, frames, time_info, status)

    turn_end = _turn_end_signal_path()
    _clear_stale_send()

    chunks = []
    recorded = 0.0
    loop_done = False
    started = time.monotonic()
    stall_s = stream_stall_seconds()
    attempt = 0
    try:
        while True:
            if _bluetooth_input_active():
                stream_ctx = sd.Stream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.int16,
                                       device=_duplex_device_pair(), callback=_duplex_callback,
                                       blocksize=chunk_samples)
            else:
                stream_ctx = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.int16,
                                            callback=_callback, blocksize=chunk_samples)
            stalled = False
            with stream_ctx:
                last_audio_at = time.monotonic()
                # Wall-clock backstop: a stream that stops delivering never
                # advances `recorded`, and a held listen must still end.
                while recorded < ceiling and time.monotonic() - started < ceiling + 1.0:
                    now = time.monotonic()
                    # Send's floor is wall time: a stalled stream never advances
                    # `recorded`, which deferred Send forever (B17).
                    if os.path.exists(turn_end) and max(recorded, now - started) >= MIN_RECORDING_DURATION:
                        logger.info("✓ Manual turn-end signal received — stopping held recording")
                        _consume_send()
                        stats["end_reason"] = "send"
                        stats["chunks_drained"] = _drain_delivered_audio(audio_queue, chunks)
                        break
                    if now - last_audio_at >= stall_s:
                        stalled = True
                        break
                    try:
                        chunk = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    last_audio_at = time.monotonic()
                    flat = chunk.flatten()
                    chunks.append(flat)
                    stats["chunks_processed"] += 1
                    recorded += len(flat) / SAMPLE_RATE
            if stalled and attempt + 1 < DEVICE_RECOVERY_MAX_ATTEMPTS:
                # A stall on a held turn is a recoverable device error (AirPods
                # can deliver audio intermittently): reopen with the captured
                # audio and Send state intact; end only once the cap is hit.
                attempt += 1
                logger.warning(f"Input stream delivered nothing for {stall_s:.1f}s - reopening held listen "
                               f"(attempt {attempt + 1}/{DEVICE_RECOVERY_MAX_ATTEMPTS})")
                if _sleep_unless_send(DEVICE_RECOVERY_DELAY_SECONDS):
                    logger.info("✓ Manual turn-end signal received during stream reopen — stopping held recording")
                    _consume_send()
                    stats["end_reason"] = "send"
                    break
                continue
            if stalled:
                logger.warning(f"Input stream delivered nothing for {stall_s:.1f}s - ending held listen")
                stats["end_reason"] = "stream_stalled"
            break
        loop_done = True
    except Exception as e:
        if not loop_done:  # a close failure after the loop keeps its end_reason
            stats["end_reason"] = "device_error"
        if chunks:
            logger.error(f"Held recording failed: {e} - returning the {recorded:.1f}s captured")
            return (np.concatenate(chunks), True)
        logger.error(f"Held recording failed: {e} - falling back to fixed duration recording")
        return (record_audio(fallback_max_duration, honor_send=True), True)

    if not chunks:
        return (np.array([], dtype=np.int16), False)
    logger.info(f"✓ Held recording ended after {recorded:.1f}s")
    return (np.concatenate(chunks), True)


def _normalize_phrase(text: Optional[str]) -> str:
    """Lowercase, strip punctuation/extra spaces for whole-utterance matching."""
    if not text:
        return ""
    import re
    return re.sub(r"[^a-z0-9\s']", " ", text.lower()).strip()


def is_step_away_phrase(text: Optional[str]) -> bool:
    """True if the WHOLE utterance is a step-away phrase ("hold on", "one sec").
    Whole-utterance (near-exact) so a passing mention mid-sentence never fires.
    Fail-safe: empty/None → False."""
    norm = _normalize_phrase(text)
    if not norm:
        return False
    return any(norm == _normalize_phrase(p) for p in STEP_AWAY_PHRASES)


def is_resume_phrase(text: Optional[str]) -> bool:
    """True if the WHOLE utterance is a resume phrase ("I'm back", "resume").
    Fail-safe: empty/None → False."""
    norm = _normalize_phrase(text)
    if not norm:
        return False
    return any(norm == _normalize_phrase(p) for p in STEP_AWAY_RESUME_PHRASES)


def _format_truncation_note(tts_metrics: dict) -> Optional[str]:
    """Render the W3c barge-in truncation-context block for the converse
    tool result -- the one place the calling agent actually reads (see
    voice_mode/streaming.py / voice_mode/truncation.py for how these
    numbers were derived). Returns None when `tts_metrics` doesn't describe
    a truncated turn, so callers can unconditionally prepend the result.

    William's rule (see the convomode skill for the full statement):
    everything reported here as "delivered" counts as heard, and is
    probably what he's responding to; the "undelivered" remainder is
    neither spoken nor assumed heard — the agent decides whether to fold
    it into the next reply or bring it back later.
    """
    if not tts_metrics or not tts_metrics.get('truncated'):
        return None

    delivered = tts_metrics.get('delivered_text') or ""
    undelivered = tts_metrics.get('undelivered_text') or ""
    fraction = tts_metrics.get('delivered_fraction')
    confidence = tts_metrics.get('truncation_confidence') or "unknown"
    basis = tts_metrics.get('truncation_basis') or ""

    pct = f"{fraction * 100:.0f}%" if isinstance(fraction, (int, float)) else "an unknown fraction"

    lines = [
        f"[TTS INTERRUPTED — William talked over you. Estimated ~{pct} of your reply "
        f"was delivered before playback cut off (confidence: {confidence} — a "
        f"sentence-boundary estimate, not a word-level position; see basis below).",
        f"  Delivered (counts as heard — he's probably responding to this): "
        f"\"{delivered}\"" if delivered else "  Delivered: (nothing — cut before any full sentence was heard)",
        f"  NOT delivered (do not assume he heard this): \"{undelivered}\"" if undelivered
        else "  Not delivered: (nothing — the estimate says the whole message got out)",
        f"  Basis: {basis}" if basis else None,
        "  Your call: fold the undelivered remainder into your next reply, or bring it "
        "back later — whichever fits what the interruption was about.]",
    ]
    return "\n".join(line for line in lines if line)


class StepAwayTracker:
    """Pure state machine for graceful step-away during an IDLE listen (Gaps 1 & 2).

    Inert unless `enabled` AND a pause is observed while still idle. No I/O, no
    audio — the record loop feeds it (paused, elapsed, speech_detected) each idle
    iteration and it returns whether a ONE-TIME check-in should be spoken now. It
    also computes the extended listen deadline (effective_max) so a paused idle
    listen waits `grace_seconds` instead of timing out at the normal max.
    """

    def __init__(self, enabled: bool, base_max_duration: float,
                 grace_seconds: float, checkin_seconds: float):
        self.enabled = bool(enabled)
        self.base_max = float(base_max_duration)
        self.grace = float(grace_seconds)
        self.checkin = float(checkin_seconds)
        self.active = False           # currently in a paused idle wait
        self.start_elapsed = None     # recording elapsed (s) when the pause began
        self.checkin_done = False     # the ONE check-in already spoken
        self.stepped_away = False     # a step-away happened at all this listen
        self.resumed = False          # pause cleared while still idle (he's back)

    def effective_max(self) -> float:
        """The current listen deadline. Extends to grace only while actively
        paused-and-idle; otherwise the unchanged base max (inert)."""
        if self.enabled and self.active and self.start_elapsed is not None:
            return max(self.base_max, self.start_elapsed + self.grace)
        return self.base_max

    def observe(self, paused: bool, elapsed: float, speech_detected: bool) -> bool:
        """Feed one idle-iteration observation. Returns True EXACTLY ONCE when the
        spoken check-in is due. Once speech is detected this turn, step-away no
        longer applies (an in-progress utterance always completes)."""
        if not self.enabled or speech_detected:
            return False
        if paused:
            if not self.active:
                self.active = True
                self.start_elapsed = elapsed
                self.stepped_away = True
            if (not self.checkin_done) and self.start_elapsed is not None \
                    and (elapsed - self.start_elapsed) >= self.checkin:
                self.checkin_done = True
                return True
        else:
            if self.active:
                # pause cleared while still idle → he's back / took the floor
                self.resumed = True
                self.active = False
        return False


def record_audio_with_silence_detection(*args, listen_stats: Optional[dict] = None, **kwargs) -> Tuple[np.ndarray, bool]:
    """Hold-open bookkeeping around _record_audio_with_silence_detection (which
    has the full signature and docs): clear a stale hold-open.signal on entry,
    publish ~/.voicemode/listening.json while the mic is open, and on every exit
    remove both and fill ``listen_stats`` (end_reason, held, samples,
    audio_seconds, wall_seconds, chunk counters) for the RECORDING_END event."""
    stats = listen_stats if listen_stats is not None else {}
    _clear_stale_hold_open()
    started = time.monotonic()
    stats["held"] = hold_active()
    stats["started_at"] = time.time()
    _write_listening_state(stats["started_at"], stats["held"])
    _take_send_source()  # a Send consumed outside any listen is not this one's
    result = None
    try:
        # AirPods stem press = Send while the mic is open (no-op if unavailable).
        with airpods_press.listening():
            result = _record_audio_with_silence_detection(*args, listen_stats=stats, **kwargs)
        return result
    finally:
        samples = len(result[0]) if result is not None else 0
        stats.setdefault("end_reason", "other")
        stats["samples"] = samples
        stats["audio_seconds"] = samples / SAMPLE_RATE
        stats["wall_seconds"] = time.monotonic() - started
        _send_source = _take_send_source()
        if _send_source is not None:
            stats["send_source"] = _send_source
        _clear_hold_open()
        _clear_listening_state()


def _record_audio_with_silence_detection(max_duration: float, disable_silence_detection: bool = False, min_duration: float = 0.0, vad_aggressiveness: Optional[int] = None, yield_check: Optional[Callable[[], bool]] = None, yield_state: Optional[dict] = None, pre_roll: Optional[np.ndarray] = None, pause_check: Optional[Callable[[], bool]] = None, step_away_state: Optional[dict] = None, checkin_callback: Optional[Callable[[], None]] = None, append_window_override_ms: Optional[int] = None, step_away_enabled_override: Optional[bool] = None, supersede_check: Optional[Callable[[], bool]] = None, listen_stats: Optional[dict] = None, _device_attempt: int = 0, _listen_t0: Optional[float] = None) -> Tuple[np.ndarray, bool]:
    """Record audio from microphone with automatic silence detection.

    Uses WebRTC VAD to detect when the user stops speaking and automatically
    stops recording after a configurable silence threshold.

    Args:
        max_duration: Maximum recording duration in seconds
        disable_silence_detection: If True, disables silence detection and uses fixed duration recording
        min_duration: Minimum recording duration before silence detection can stop (default: 0.0)
        vad_aggressiveness: VAD aggressiveness level (0-3). If None, uses VAD_AGGRESSIVENESS from config
        yield_check: Optional callable polled during the listen loop. When it
            returns True while the listen is still IDLE (no speech detected
            yet) AND at least CONCH_YIELD_GRACE_SECONDS have elapsed in this
            listen, recording ends early so the caller can yield the mic to
            another agent (vibedispatcher#132). Never fires once speech has
            been detected — an in-progress utterance always completes. The
            grace floor exists so a fresh listen can't be preempted before
            the human has had any chance to start replying.
        yield_state: Optional dict; when the listen ends because of
            yield_check or supersede_check, ``yield_state["yielded"]`` is set
            True. (Out-of-band so the 2-tuple return stays stable for
            existing callers.)
        supersede_check: Optional callable polled during the listen loop,
            same IDLE-only gating as yield_check (never fires once speech has
            been detected) but WITHOUT the grace floor. Distinct from
            yield_check because the two signals aren't equivalent: yield_check
            answers "is someone politely asking for the mic?" (a request,
            fine to make wait out the grace period); supersede_check answers
            "has the mic already been taken from us in place?" (a fact — the
            conch's epoch has already moved on, so by the time this fires two
            processes may already believe they hold it). Gating a fact behind
            a multi-second grace floor would leave that double-hold standing
            for the whole floor, which is the exact bug conch fail-closed
            (PR #12) exists to remove. Checked first and unconditionally
            (no grace, no config gate) every idle tick.
        pre_roll: Optional audio already captured BEFORE this call started (natural-mode
            barge-in: the mic audio that triggered the interruption). When provided, it
            seeds the recording as already-in-progress speech, so his interruption
            becomes the start of this turn instead of being discarded and re-prompted.
        pause_check: Optional callable (founder-os#11655). Polled while the listen is
            IDLE. When step-away is enabled and this returns True (convomode paused),
            the idle listen WAITS longer (up to STEP_AWAY_GRACE_SECONDS) instead of
            timing out at max_duration. Defaults to convomode_paused when step-away is
            enabled and none is passed. Inert when step-away is off.
        step_away_state: Optional dict; out-of-band signals set on exit —
            ``stepped_away`` (a pause happened this listen), ``resumed`` (pause
            cleared while still idle → he's back), ``checkin_done`` (the ONE check-in
            was spoken). Keeps the 2-tuple return stable for existing callers.
        checkin_callback: Optional no-arg callable invoked ONCE, after
            STEP_AWAY_CHECKIN_SECONDS of continuous idle-pause, to speak the "Still
            there?" check-in. None (default) = no spoken check-in (patience still
            extends). Failures are swallowed — a check-in never breaks the listen.
        append_window_override_ms: Optional ms to extend the trailing-silence
            threshold (append-to-turn, Gap 3). None → resolved from config/knob file.
            0 → OFF, byte-for-byte the prior single-shot behavior.
        step_away_enabled_override: Optional bool to force step-away on/off (tests).
            None → resolved from env flag / flag file. Default resolves to OFF.
        listen_stats: Optional dict filled out-of-band with ``end_reason``,
            ``held`` and chunk counters (see record_audio_with_silence_detection).
        _device_attempt, _listen_t0: internal -- device-error recovery reopens
            the stream by recursing with the attempt number and the original
            listen start, so the attempt cap, the hold ceiling and Send's
            wall-clock floor span the whole listen, not one stream.

    Returns:
        Tuple of (audio_data, speech_detected):
            - audio_data: Numpy array of recorded audio samples
            - speech_detected: Boolean indicating if speech was detected during recording
    """

    logger.info(f"record_audio_with_silence_detection called - VAD_AVAILABLE={VAD_AVAILABLE}, DISABLE_SILENCE_DETECTION={DISABLE_SILENCE_DETECTION}, min_duration={min_duration}")
    stats = listen_stats if listen_stats is not None else {}
    _t0 = _listen_t0 if _listen_t0 is not None else time.monotonic()

    # --- founder-os#11655: resolve pause/step-away/append config (all inert by default) ---
    _append_ms = append_window_override_ms if append_window_override_ms is not None else append_window_ms()
    _step_away_on = step_away_enabled_override if step_away_enabled_override is not None else step_away_enabled()
    if _step_away_on and pause_check is None:
        pause_check = convomode_paused  # reuse the existing pause primitive
    _step_away = StepAwayTracker(_step_away_on, max_duration, STEP_AWAY_GRACE_SECONDS, STEP_AWAY_CHECKIN_SECONDS)
    if _append_ms > 0:
        logger.info(f"🎙️ append-to-turn window enabled: +{_append_ms}ms after silence timer")
    if _step_away_on:
        logger.info(f"🎙️ step-away enabled: grace {STEP_AWAY_GRACE_SECONDS}s, check-in at {STEP_AWAY_CHECKIN_SECONDS}s")
    
    if not VAD_AVAILABLE:
        logger.warning("webrtcvad not available, falling back to fixed duration recording")
        if hold_active():
            return _record_held_until_send(max_duration, stats)
        # For fallback, assume speech is present since we can't detect
        stats["end_reason"] = "max_duration"
        _clear_stale_send()
        return (record_audio(max_duration, honor_send=True), True)

    if DISABLE_SILENCE_DETECTION or disable_silence_detection:
        if disable_silence_detection:
            logger.info("Silence detection disabled for this interaction by request")
        else:
            logger.info("Silence detection disabled globally via VOICEMODE_DISABLE_SILENCE_DETECTION")
        # Held: fixed-length record_audio() never checks signals, so Send
        # could not end it. Stream until Send instead.
        if hold_active():
            return _record_held_until_send(max_duration, stats)
        # For fallback, assume speech is present since we can't detect
        stats["end_reason"] = "max_duration"
        _clear_stale_send()
        return (record_audio(max_duration, honor_send=True), True)
    
    logger.info(f"🎤 Recording with silence detection (max {max_duration}s)...")
    
    try:
        # Initialize VAD with provided aggressiveness or default
        effective_vad_aggressiveness = vad_aggressiveness if vad_aggressiveness is not None else VAD_AGGRESSIVENESS
        vad = webrtcvad.Vad(effective_vad_aggressiveness)

        # Calculate chunk size (must be 10, 20, or 30ms worth of samples)
        chunk_samples = int(SAMPLE_RATE * VAD_CHUNK_DURATION_MS / 1000)
        chunk_duration_s = VAD_CHUNK_DURATION_MS / 1000

        # WebRTC VAD only supports 8000, 16000, or 32000 Hz
        # We'll tell VAD we're using 16kHz even though we're recording at 24kHz
        # This requires adjusting our chunk size to match what VAD expects
        vad_sample_rate = 16000
        vad_chunk_samples = int(vad_sample_rate * VAD_CHUNK_DURATION_MS / 1000)

        # Endpointing (Silero VAD, D5): OFF unless BOTH explicitly enabled via
        # VOICEMODE_ENDPOINTING AND Silero is actually available (onnxruntime +
        # vendored model loaded). When either is false, `use_endpointing` is
        # False and every branch below this point is the pre-existing
        # webrtcvad + silence-timer path, byte-for-byte unchanged.
        use_endpointing = ENDPOINTING_ENABLED and SILERO_AVAILABLE
        endpointer = None
        if use_endpointing:
            # append-to-turn (founder-os#11655) folds into the endpoint
            # threshold itself: Endpointer.update() resets its trailing-silence
            # counter on any above-threshold frame regardless of why the
            # window is longer, so extending min_endpoint_ms by _append_ms
            # reproduces the webrtcvad path's `_silence_stop_threshold =
            # SILENCE_THRESHOLD_MS + _append_ms` + resume-cancels-the-timer
            # behavior exactly. _append_ms == 0 (default) -> unchanged.
            endpointer = Endpointer(
                sample_rate=vad_sample_rate,
                speech_threshold=ENDPOINTING_SPEECH_THRESHOLD,
                min_endpoint_ms=ENDPOINTING_MIN_ENDPOINT_MS + _append_ms,
                min_speech_ms=ENDPOINTING_MIN_SPEECH_MS,
            )
            endpointer.reset()
            logger.info(
                f"🎙️ endpointing enabled (Silero): speech_threshold={ENDPOINTING_SPEECH_THRESHOLD}, "
                f"min_endpoint_ms={ENDPOINTING_MIN_ENDPOINT_MS}(+{_append_ms} append)={ENDPOINTING_MIN_ENDPOINT_MS + _append_ms}, "
                f"min_speech_ms={ENDPOINTING_MIN_SPEECH_MS}"
            )

        # Recording state -- seed from natural-mode barge-in pre-roll if given,
        # so his interruption becomes the start of this turn's speech rather
        # than being thrown away.
        has_pre_roll = pre_roll is not None and len(pre_roll) > 0
        chunks = [pre_roll] if has_pre_roll else []
        silence_duration_ms = 0
        recording_duration = (len(pre_roll) / SAMPLE_RATE) if has_pre_roll else 0
        speech_detected = has_pre_roll
        stop_recording = False

        if use_endpointing and has_pre_roll:
            # Feed the pre-roll's own audio through the endpointer (resampled
            # to its 16kHz working rate, same as every other chunk) so its
            # internal speech_started/timing state reflects the interruption
            # that already happened, rather than starting cold.
            from scipy import signal as _pre_roll_signal
            _pre_roll_flat = np.asarray(pre_roll).flatten()
            _resampled_len = int(len(_pre_roll_flat) * vad_sample_rate / SAMPLE_RATE)
            if _resampled_len > 0:
                _pre_roll_16k = _pre_roll_signal.resample(_pre_roll_flat, _resampled_len)
                endpointer.update(_pre_roll_16k)
        
        # Use a queue for thread-safe communication
        import queue
        audio_queue = queue.Queue()
        
        # Save stdio state
        import sys
        original_stdin = sys.stdin
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        logger.debug(f"VAD config - Aggressiveness: {effective_vad_aggressiveness} (param: {vad_aggressiveness}, default: {VAD_AGGRESSIVENESS}), "
                    f"Silence threshold: {SILENCE_THRESHOLD_MS}ms, "
                    f"Min duration: {MIN_RECORDING_DURATION}s, "
                    f"Initial grace period: {INITIAL_SILENCE_GRACE_PERIOD}s")
        
        if VAD_DEBUG:
            logger.info(f"[VAD_DEBUG] Starting VAD recording with config:")
            logger.info(f"[VAD_DEBUG]   max_duration: {max_duration}s")
            logger.info(f"[VAD_DEBUG]   min_duration: {min_duration}s")
            logger.info(f"[VAD_DEBUG]   effective_min_duration: {max(MIN_RECORDING_DURATION, min_duration)}s")
            logger.info(f"[VAD_DEBUG]   VAD aggressiveness: {effective_vad_aggressiveness}")
            logger.info(f"[VAD_DEBUG]   Silence threshold: {SILENCE_THRESHOLD_MS}ms")
            logger.info(f"[VAD_DEBUG]   Sample rate: {SAMPLE_RATE}Hz (VAD using {vad_sample_rate}Hz)")
            logger.info(f"[VAD_DEBUG]   Chunk duration: {VAD_CHUNK_DURATION_MS}ms")
        
        def audio_callback(indata, frames, time, status):
            """Callback for continuous audio stream"""
            if status:
                logger.warning(f"Audio stream status: {status}")
                # Check for device-related errors
                status_str = str(status).lower()
                if any(err in status_str for err in ['device unavailable', 'device disconnected', 
                                                      'invalid device', 'unanticipated host error',
                                                      'stream is stopped', 'portaudio error']):
                    # Signal that we should stop recording due to device error
                    audio_queue.put(None)  # Sentinel value to indicate error
                    return
            # Put the audio data in the queue for processing
            stats["chunks_delivered"] = stats.get("chunks_delivered", 0) + 1
            audio_queue.put(indata.copy())
        
        def duplex_audio_callback(indata, outdata, frames, time, status):
            """Full-duplex wrapper: silent output + the normal input callback.

            F1 (founder-os#13215): when the input is a Bluetooth headset, the
            listen window must be a two-way stream or macOS tears down HFP and
            the mic reads dead silence. Output writes silence; input feeds the
            same queue as the plain path.
            """
            outdata.fill(0)
            audio_callback(indata, frames, time, status)

        _loop_end_reason = None  # set once the listen loop has ended
        try:
            # Create continuous stream (full-duplex on BT input, input-only
            # otherwise). Construction opens the PortAudio stream, so it stays
            # inside this try to keep the device-error recovery path intact.
            if _bluetooth_input_active():
                logger.info("🎧 Bluetooth input — full-duplex listen stream (F1)")
                stream_ctx = sd.Stream(samplerate=SAMPLE_RATE,
                                       channels=CHANNELS,
                                       dtype=np.int16,
                                       device=_duplex_device_pair(),
                                       callback=duplex_audio_callback,
                                       blocksize=chunk_samples)
            else:
                stream_ctx = sd.InputStream(samplerate=SAMPLE_RATE,
                                            channels=CHANNELS,
                                            dtype=np.int16,
                                            callback=audio_callback,
                                            blocksize=chunk_samples)
            with stream_ctx:
                
                logger.debug("Started continuous audio stream")

                # Manual turn-end signal (push-to-talk "I'm done"). A reachable-while-
                # driving surface (Apple Shortcut → SSH → `touch`) drops this file to end
                # the listen window IMMEDIATELY, no VAD needed. Cleaned up on each entry so
                # a stale signal can't pre-end the next turn -- but not when reopening
                # after a device error: a Send pressed then belongs to THIS listen.
                _turn_end_signal = _turn_end_signal_path()
                if _device_attempt == 0:
                    _clear_stale_send()

                # Hold-open push-to-talk: latched -- once hold-mode or
                # hold-open.signal is seen, silence can no longer end this
                # listen; only Send (turn-end.signal) or the hold ceiling can.
                _held = hold_active()
                _hold_max = hold_max_seconds()
                _listen_started = _t0
                _stall_s = stream_stall_seconds()
                _last_audio_at = time.monotonic()
                _end_reason = None  # RECORDING_END diagnostics; see listen_stats
                stats["chunks_processed"] = 0
                if _held:
                    logger.info(f"✋ Held listen: silence will not end the turn (max {_hold_max:.0f}s)")

                while recording_duration < (_hold_max if _held else _step_away.effective_max()) and not stop_recording:
                    if not _held and hold_active():
                        _held = True
                        stats["held"] = True
                        _write_listening_state(stats.get("started_at", time.time()), True)
                        logger.info(f"✋ Hold engaged at {recording_duration:.1f}s: silence will not end the turn (max {_hold_max:.0f}s)")
                    # Wall-clock backstop for a held listen: a stream that stops
                    # delivering never advances recording_duration.
                    if _held and time.monotonic() - _listen_started >= _hold_max + 1.0:
                        logger.warning("Held listen hit its wall-clock ceiling - stopping")
                        _end_reason = "hold_ceiling"
                        break

                    # Graceful step-away (founder-os#11655): while IDLE and paused,
                    # extend patience (effective_max above) and speak ONE check-in.
                    # Fully inert unless step-away is enabled AND convomode is paused.
                    if pause_check is not None and not speech_detected:
                        try:
                            _paused_now = bool(pause_check())
                        except Exception:
                            _paused_now = False
                        if _step_away.observe(_paused_now, recording_duration, speech_detected) and checkin_callback is not None:
                            logger.info("⏸️ step-away: speaking one 'still there?' check-in")
                            # Off the loop thread: the check-in speaks TTS and
                            # waits on it (up to 20s), and the loop must keep
                            # polling Send meanwhile (B17).
                            threading.Thread(target=_run_checkin, args=(checkin_callback,),
                                             daemon=True, name="step-away-checkin").start()

                    # Honor a manual turn-end signal first (push-to-talk). If William
                    # tapped his "done" Shortcut, end the recording now (only after the
                    # min duration so a too-fast tap can't return empty audio). The
                    # signal stays on disk until the floor is reached, so an early
                    # press is deferred, never dropped. Held turns use only the tiny
                    # global floor: listen_duration_min must not delay a Send.
                    # The floor is measured on the wall clock as well as the audio
                    # clock: a stream that stops delivering never advances
                    # recording_duration, which deferred Send forever (B17).
                    _turn_end_floor = MIN_RECORDING_DURATION if _held else max(MIN_RECORDING_DURATION, min_duration)
                    _now = time.monotonic()
                    try:
                        if os.path.exists(_turn_end_signal) and max(recording_duration, _now - _listen_started) >= _turn_end_floor:
                            logger.info("✓ Manual turn-end signal received — stopping recording")
                            _consume_send()
                            _end_reason = "send"
                            stats["chunks_drained"] = _drain_delivered_audio(audio_queue, chunks)
                            stop_recording = True
                            speech_detected = True  # he spoke and signaled done; transcribe it
                            break
                    except Exception:
                        pass

                    # Stall watchdog: an input stream that delivers nothing (e.g.
                    # CoreAudio dropped the IOProc on an AirPods route change)
                    # must not hold the mic -- and the floor -- until the ceiling.
                    if _now - _last_audio_at >= _stall_s:
                        if _held and _device_attempt + 1 < DEVICE_RECOVERY_MAX_ATTEMPTS:
                            # AirPods can deliver audio intermittently: a held turn
                            # must not end mid-sentence on a stall. Reopen the stream
                            # via device recovery (captured audio carries as pre-roll,
                            # Send/hold state preserved); end only once the cap is hit.
                            logger.warning(f"Input stream delivered nothing for {_stall_s:.1f}s - reopening held listen")
                            raise _StreamStalled(f"input stream stalled for {_stall_s:.1f}s")
                        logger.warning(f"Input stream delivered nothing for {_stall_s:.1f}s - ending listen")
                        _end_reason = "stream_stalled"
                        break

                    # In-place supersession (conch fail-closed, PR #12): checked
                    # FIRST and UNCONDITIONALLY (no grace floor, no config gate).
                    # This is not a request that can be politely kept waiting —
                    # a preempter has already taken the lock in place under a
                    # bumped epoch, so we no longer hold what we think we hold.
                    # Gating this behind CONCH_YIELD_GRACE_SECONDS (like the
                    # request-path check below) would leave two processes both
                    # believing they hold the mic for the whole grace window —
                    # exactly the double-hold bug PR #12 exists to close. Still
                    # IDLE-only: once speech has been detected the in-progress
                    # utterance completes via VAD as usual (same as yield_check).
                    if supersede_check is not None and not speech_detected:
                        try:
                            if supersede_check():
                                logger.info("✓ Conch superseded in place — yielding idle listen immediately "
                                            "(no grace floor: this is a fact, not a request)")
                                if yield_state is not None:
                                    yield_state["yielded"] = True
                                _end_reason = "supersede"
                                stop_recording = True
                                break
                        except Exception:
                            pass

                    # MID-UTTERANCE supersession: do NOT cut the human off (the
                    # utterance completes via VAD and is delivered), but DO
                    # notice -- previously a holder superseded after speech
                    # began never learned of it until it next went idle. The
                    # flag lets converse() deliver the words and then hand the
                    # floor over instead of carrying on as the conch holder.
                    if (supersede_check is not None and speech_detected
                            and yield_state is not None
                            and not yield_state.get("superseded_mid_utterance")):
                        try:
                            if supersede_check():
                                yield_state["superseded_mid_utterance"] = True
                                logger.info("Conch superseded MID-UTTERANCE -- finishing this "
                                            "utterance, then standing down")
                        except Exception:
                            pass

                    # Yieldable listen (vibedispatcher#132): another agent is asking
                    # for the mic. Yield ONLY while idle — once speech has been
                    # detected, the in-progress utterance completes via VAD as usual.
                    #
                    # GRACE PERIOD (founder-os barge-in fix, 2026-08-04): a waiter's
                    # wait_for_conch poll loop can start requesting the mic BEFORE
                    # this listen even begins (it was polling during the holder's
                    # TTS). Without a minimum elapsed-time floor, yield_check() can
                    # return True on this loop's very first tick (recording_duration
                    # == 0), so a live human-AI turn gets cut off before the human
                    # has any chance to start replying to what was just said. This
                    # is a turn-boundary vs. conversation-boundary bug, not a real
                    # abandoned-mic case. Require CONCH_YIELD_GRACE_SECONDS of
                    # actual idle-listening before a yield request is honored. This
                    # grace applies ONLY to a request (yield_check) — supersession
                    # (supersede_check, above) is a fact and always bypasses it.
                    # Not while held: the user is holding the key to speak, so an
                    # idle mic is not an abandoned one. (Supersession above, a
                    # fact rather than a request, still applies.)
                    if (yield_check is not None and not speech_detected and not _held
                            and recording_duration >= CONCH_YIELD_GRACE_SECONDS):
                        try:
                            if yield_check():
                                logger.info("✓ Conch requested by another agent — yielding idle listen "
                                            f"(after {recording_duration:.1f}s grace)")
                                if yield_state is not None:
                                    yield_state["yielded"] = True
                                _end_reason = "yield"
                                stop_recording = True
                                break
                        except Exception:
                            pass
                    try:
                        # Get audio chunk from queue with timeout
                        chunk = audio_queue.get(timeout=0.1)
                        _last_audio_at = time.monotonic()
                        
                        # Check for error sentinel
                        if chunk is None:
                            logger.error("Audio device error detected - stopping recording")
                            _end_reason = "device_error"
                            # Raise an exception to trigger recovery logic
                            raise sd.PortAudioError("Audio device disconnected or unavailable")
                        
                        # Flatten for consistency
                        chunk_flat = chunk.flatten()
                        chunks.append(chunk_flat)
                        
                        # For VAD, we need to downsample from 24kHz to 16kHz
                        # Use scipy's resample for proper downsampling
                        # Calculate the number of samples we need after resampling
                        resampled_length = int(len(chunk_flat) * vad_sample_rate / SAMPLE_RATE)
                        vad_chunk = signal.resample(chunk_flat, resampled_length)
                        # Take exactly the number of samples VAD expects
                        vad_chunk = vad_chunk[:vad_chunk_samples].astype(np.int16)
                        chunk_bytes = vad_chunk.tobytes()
                        
                        if use_endpointing:
                            # --- Endpointing (Silero VAD) decision path (D5: only
                            # reached when ENDPOINTING_ENABLED and SILERO_AVAILABLE
                            # are both true) -- replaces the webrtcvad is_speech +
                            # manual silence_duration_ms timer below with the
                            # Endpointer state machine, fed the SAME 16kHz-resampled
                            # frame (vad_chunk) the webrtcvad path computed above.
                            endpoint_state = endpointer.update(vad_chunk)
                            if not speech_detected and endpoint_state.speech_started:
                                logger.info("🎤 Speech detected, starting active recording")
                                if VAD_DEBUG:
                                    logger.info(f"[VAD_DEBUG] STATE CHANGE: WAITING_FOR_SPEECH -> SPEECH_ACTIVE at t={recording_duration:.1f}s")
                                speech_detected = True
                            if VAD_DEBUG and int(recording_duration * 1000) % 500 == 0:
                                logger.info(f"[VAD_DEBUG] t={recording_duration:.1f}s: speech_prob={endpoint_state.speech_prob:.3f}, "
                                            f"speech_started={endpoint_state.speech_started}, endpointed={endpoint_state.endpointed}")

                            if endpoint_state.endpointed and not _held:
                                # Use the larger of MIN_RECORDING_DURATION (global) or
                                # min_duration (parameter) -- same floor as the
                                # webrtcvad path, so a too-early endpoint can't return
                                # a clipped recording. Never while held.
                                effective_min_duration = max(MIN_RECORDING_DURATION, min_duration)
                                if recording_duration >= effective_min_duration:
                                    logger.info(f"✓ Endpoint detected after {recording_duration:.1f}s of recording (Silero)")
                                    _end_reason = "endpoint"
                                    stop_recording = True
                                elif VAD_DEBUG and int(recording_duration * 1000) % 500 == 0:
                                    logger.info(f"[VAD_DEBUG] Min duration not met: {recording_duration:.1f}s < {effective_min_duration}s")
                        else:
                            # Check if chunk contains speech
                            try:
                                is_speech = vad.is_speech(chunk_bytes, vad_sample_rate)
                                # ENERGY GATE (driving profile): webrtcvad has no energy
                                # floor, so steady road/engine noise reads as "speech" and
                                # the silence counter never accumulates → the mic hangs.
                                # When an energy threshold is set, a chunk only counts as
                                # speech if it ALSO clears the RMS floor; below-floor chunks
                                # (road rumble) are treated as silence so end-of-turn is
                                # detected. Disabled (==0) → pure-webrtcvad, unchanged.
                                if is_speech and VAD_ENERGY_THRESHOLD > 0:
                                    chunk_rms = float(np.sqrt(np.mean(chunk.astype(float) ** 2)))
                                    if chunk_rms < VAD_ENERGY_THRESHOLD:
                                        is_speech = False
                                        if VAD_DEBUG and int(recording_duration * 1000) % 500 == 0:
                                            logger.info(f"[VAD_DEBUG] t={recording_duration:.1f}s: energy-gated (RMS={chunk_rms:.0f} < floor={VAD_ENERGY_THRESHOLD:.0f}) -> silence")
                                if VAD_DEBUG:
                                    # Log VAD decision every 500ms for less spam
                                    if int(recording_duration * 1000) % 500 == 0:
                                        rms = np.sqrt(np.mean(chunk.astype(float)**2))
                                        logger.info(f"[VAD_DEBUG] t={recording_duration:.1f}s: speech={is_speech}, RMS={rms:.0f}, floor={VAD_ENERGY_THRESHOLD:.0f}, state={'WAITING' if not speech_detected else 'ACTIVE'}")
                            except Exception as vad_e:
                                logger.warning(f"VAD error: {vad_e}, treating as speech")
                                is_speech = True

                            # State machine for speech detection
                            if not speech_detected:
                                # WAITING_FOR_SPEECH state
                                if is_speech:
                                    logger.info("🎤 Speech detected, starting active recording")
                                    if VAD_DEBUG:
                                        logger.info(f"[VAD_DEBUG] STATE CHANGE: WAITING_FOR_SPEECH -> SPEECH_ACTIVE at t={recording_duration:.1f}s")
                                    speech_detected = True
                                    silence_duration_ms = 0
                                # No timeout in this state - just keep waiting
                                # The only exit is speech detection or max_duration
                            else:
                                # We have detected speech at some point
                                if is_speech:
                                    # SPEECH_ACTIVE state - reset silence counter.
                                    # Append-to-turn (founder-os#11655): if speech resumes
                                    # AFTER the normal silence threshold but within the
                                    # extra append window, it continues the SAME turn.
                                    if _append_ms > 0 and silence_duration_ms >= SILENCE_THRESHOLD_MS:
                                        logger.info(f"➕ append-to-turn: speech resumed within window (was {silence_duration_ms}ms silent) — continuing same turn")
                                    silence_duration_ms = 0
                                else:
                                    # SILENCE_AFTER_SPEECH state - accumulate silence
                                    silence_duration_ms += VAD_CHUNK_DURATION_MS
                                    if VAD_DEBUG and silence_duration_ms % 100 == 0:  # More frequent logging in debug mode
                                        logger.info(f"[VAD_DEBUG] Accumulating silence: {silence_duration_ms}/{SILENCE_THRESHOLD_MS}ms, t={recording_duration:.1f}s")
                                    elif silence_duration_ms % 200 == 0:  # Log every 200ms
                                        logger.debug(f"Silence: {silence_duration_ms}ms")

                                    # Check if we should stop due to silence threshold
                                    # Use the larger of MIN_RECORDING_DURATION (global) or min_duration (parameter)
                                    effective_min_duration = max(MIN_RECORDING_DURATION, min_duration)
                                    # Append-to-turn (founder-os#11655): extend the trailing-
                                    # silence threshold by the append window. _append_ms=0
                                    # (default) → identical to prior behavior.
                                    _silence_stop_threshold = SILENCE_THRESHOLD_MS + _append_ms
                                    # Never while held (hold-open push-to-talk).
                                    if (recording_duration >= effective_min_duration
                                            and silence_duration_ms >= _silence_stop_threshold
                                            and not _held):
                                        logger.info(f"✓ Silence threshold reached after {recording_duration:.1f}s of recording")
                                        _end_reason = "silence"
                                        if VAD_DEBUG:
                                            logger.info(f"[VAD_DEBUG] STOP: silence_duration={silence_duration_ms}ms >= threshold={_silence_stop_threshold}ms")
                                            logger.info(f"[VAD_DEBUG] STOP: recording_duration={recording_duration:.1f}s >= min_duration={effective_min_duration}s")
                                        stop_recording = True
                                    elif VAD_DEBUG and recording_duration < effective_min_duration:
                                        if int(recording_duration * 1000) % 500 == 0:  # Log every 500ms
                                            logger.info(f"[VAD_DEBUG] Min duration not met: {recording_duration:.1f}s < {effective_min_duration}s")

                        recording_duration += chunk_duration_s
                        stats["chunks_processed"] += 1
                            
                    except queue.Empty:
                        # No audio data available, continue waiting
                        continue
                    except Exception as e:
                        logger.error(f"Error processing audio chunk: {e}")
                        _end_reason = _end_reason or "other"
                        break

                if _end_reason is None:
                    # The loop condition ended it: the duration ceiling.
                    if _held:
                        _end_reason = "hold_ceiling"
                    else:
                        _end_reason = "max_duration" if speech_detected else "no_speech"
                stats["end_reason"] = _end_reason
                stats["held"] = _held
                _loop_end_reason = _end_reason

            # Publish step-away signals out-of-band (founder-os#11655) so the
            # async caller can speak a graceful ending / prepend a resume recap.
            if step_away_state is not None:
                step_away_state["stepped_away"] = _step_away.stepped_away
                step_away_state["resumed"] = _step_away.resumed
                step_away_state["checkin_done"] = _step_away.checkin_done

            # Concatenate all chunks
            if chunks:
                full_recording = np.concatenate(chunks)
                
                if not speech_detected:
                    logger.info(f"✓ Recording completed ({recording_duration:.1f}s) - No speech detected")
                    if VAD_DEBUG:
                        logger.info(f"[VAD_DEBUG] FINAL STATE: No speech was ever detected during recording")
                else:
                    logger.info(f"✓ Recorded {len(full_recording)} samples ({recording_duration:.1f}s) with speech")
                    if VAD_DEBUG:
                        logger.info(f"[VAD_DEBUG] FINAL STATE: Speech was detected, recording complete")
                
                if DEBUG:
                    # Calculate RMS for debug
                    rms = np.sqrt(np.mean(full_recording.astype(float) ** 2))
                    logger.debug(f"Recording stats - RMS: {rms:.2f}, Speech detected: {speech_detected}")
                
                # Return tuple: (audio_data, speech_detected)
                return (full_recording, speech_detected)
            else:
                logger.warning("No audio chunks recorded")
                return (np.array([]), False)
                
        except Exception as e:
            if _loop_end_reason is not None:
                # The listen already ended (Send, silence, ...) and only closing
                # the stream failed: deliver it. Retrying would reopen the mic
                # after a consumed Send.
                logger.warning(f"Audio stream close failed after the listen ended ({_loop_end_reason}): {e}")
                if chunks:
                    return (np.concatenate(chunks), speech_detected)
                return (np.array([]), False)
            logger.error(f"Recording with VAD failed: {e}")
            _stalled = isinstance(e, _StreamStalled)
            stats["end_reason"] = "stream_stalled" if _stalled else "device_error"
            
            # Import here to avoid circular imports
            from voice_mode.utils.audio_diagnostics import get_audio_error_help
            
            # Check if this is a device error that might be recoverable
            error_str = str(e).lower()
            if _stalled or any(err in error_str for err in ['device unavailable', 'device disconnected', 
                                                 'invalid device', 'unanticipated host error',
                                                 'portaudio error']):
                # Audio captured before the failure survives the reopen (as
                # pre-roll) or the give-up. Idle silence on a non-held listen
                # is dropped: as pre-roll it would read as speech already begun.
                _keep_audio = bool(chunks) and (speech_detected or bool(stats.get("held")))
                _captured = np.concatenate(chunks) if _keep_audio else None

                def _captured_result():
                    if _captured is not None and len(_captured) > 0:
                        return (_captured, True)
                    return (np.array([], dtype=np.int16), False)

                if _device_attempt + 1 >= DEVICE_RECOVERY_MAX_ATTEMPTS:
                    # The device is gone: end the listen with what we have
                    # instead of retrying forever (B17).
                    logger.error(f"Audio device still failing after {DEVICE_RECOVERY_MAX_ATTEMPTS} attempts - ending listen")
                    return _captured_result()

                logger.info("Audio device error detected - attempting to reinitialize audio system")
                
                # Try to reinitialize sounddevice
                try:
                    # Get current default device info before reinit
                    try:
                        old_device = sd.query_devices(kind='input')
                        old_device_name = old_device.get('name', 'Unknown')
                    except:
                        old_device_name = 'Previous device'
                    
                    sd._terminate()
                    sd._initialize()
                    
                    # Get new default device info
                    try:
                        new_device = sd.query_devices(kind='input')
                        new_device_name = new_device.get('name', 'Unknown')
                        logger.info(f"Audio system reinitialized - switched from '{old_device_name}' to '{new_device_name}'")
                    except:
                        logger.info("Audio system reinitialized - retrying with new default device")
                    
                    # Wait a moment for the system to stabilize -- Send ends it
                    if _sleep_unless_send(DEVICE_RECOVERY_DELAY_SECONDS):
                        logger.info("✓ Manual turn-end signal received during device recovery — stopping recording")
                        _consume_send()
                        stats["end_reason"] = "send"
                        return _captured_result()
                    
                    # Retry with the new device. Recurse into the inner function
                    # (not the public wrapper, which would reset the listen's
                    # bookkeeping and clear a per-turn hold mid-listen).
                    logger.info(f"Retrying recording with new audio device (attempt {_device_attempt + 2}/{DEVICE_RECOVERY_MAX_ATTEMPTS})...")
                    return _record_audio_with_silence_detection(
                        max_duration, disable_silence_detection, min_duration, vad_aggressiveness,
                        yield_check=yield_check, yield_state=yield_state,
                        pre_roll=_captured if _captured is not None else pre_roll,
                        pause_check=pause_check, step_away_state=step_away_state,
                        checkin_callback=checkin_callback,
                        append_window_override_ms=append_window_override_ms,
                        step_away_enabled_override=step_away_enabled_override,
                        supersede_check=supersede_check,
                        listen_stats=stats,
                        _device_attempt=_device_attempt + 1,
                        _listen_t0=_t0,
                    )
                    
                except Exception as reinit_error:
                    logger.error(f"Failed to reinitialize audio: {reinit_error}")
                    # Fall through to normal error handling
            
            # Get helpful error message
            help_message = get_audio_error_help(e)
            logger.error(f"\n{help_message}")
            
            logger.info("Falling back to fixed duration recording")
            # For fallback, assume speech is present since we can't detect
            return (record_audio(max_duration, honor_send=True), True)
            
        finally:
            # Restore stdio
            if sys.stdin != original_stdin:
                sys.stdin = original_stdin
            if sys.stdout != original_stdout:
                sys.stdout = original_stdout
            if sys.stderr != original_stderr:
                sys.stderr = original_stderr
    
    except Exception as e:
        logger.error(f"VAD initialization failed: {e}")
        logger.info("Falling back to fixed duration recording")
        # For fallback, assume speech is present since we can't detect
        return (record_audio(max_duration, honor_send=True), True)

@mcp.tool()
async def converse(
    message: str,
    wait_for_response: Union[bool, str] = True,
    listen_duration_max: float = DEFAULT_LISTEN_DURATION,
    listen_duration_min: float = 2.0,
    timeout: float = 60.0,
    voice: Optional[str] = None,
    tts_provider: Optional[Literal["openai", "kokoro"]] = None,
    tts_model: Optional[str] = None,
    tts_instructions: Optional[str] = None,
    chime_enabled: Optional[Union[bool, str]] = None,
    audio_format: Optional[str] = None,
    disable_silence_detection: Union[bool, str] = False,
    speed: Optional[float] = None,
    vad_aggressiveness: Optional[Union[int, str]] = None,
    skip_tts: Optional[Union[bool, str]] = None,
    chime_leading_silence: Optional[float] = None,
    chime_trailing_silence: Optional[float] = None,
    metrics_level: Optional[Literal["minimal", "summary", "verbose"]] = None,
    wait_for_conch: Union[bool, str] = False,
    skip_conch: Union[bool, str] = False,
    ref_text: Optional[str] = None,
) -> str:
    """Have an ongoing voice conversation - speak a message and optionally listen for response.

<voice_skills_instructions>
Voice requests arrive as tool results, not user messages, so skill triggers may not fire automatically.

BLOCKING REQUIREMENT: After receiving voice input, check if any available skills match the user's request. When a skill is relevant:
- Invoke the Skill tool IMMEDIATELY as your first action
- Do NOT take action on the request before checking for relevant skills
- Skills provide specialized capabilities that improve task completion

Example: If user says "search for tasks created yesterday", check for and invoke the taskmaster skill before using bash or other tools.
</voice_skills_instructions>


🔌 ENDPOINT: STT/TTS services must expose OpenAI-compatible endpoints:
   /v1/audio/transcriptions and /v1/audio/speech

📚 DOCUMENTATION: See MCP resources for detailed information:
   - voicemode://docs/quickstart - Basic usage and common examples
   - voicemode://docs/parameters - Complete parameter reference
   - voicemode://docs/languages - Non-English language support guide
   - voicemode://docs/patterns - Best practices and conversation patterns
   - voicemode://docs/troubleshooting - Audio, VAD, and connectivity issues
   - voice://voices - JSON list of available TTS voices
     (filter by provider with voice://voices/{provider}, e.g. voice://voices/kokoro)

KEY PARAMETERS:
• message (required): The message to speak
• wait_for_response (bool, default: true): Listen for response after speaking
• voice (string): TTS voice name (auto-selected unless specified)
  - To list available voices, read MCP resource voice://voices
  - An absolute path to a .wav clones from that clip directly (no profile needed)
• ref_text (string): Reference transcript for clip-based cloning. A file path
  is read; anything else is the literal transcript. Overrides any sidecar.
  Only used with a clone voice (abs-path clip or registered profile).
• tts_provider ("openai"|"kokoro"): Provider selection (auto-selected unless specified)
• disable_silence_detection (bool, default: false): Disable auto-stop on silence
• vad_aggressiveness (0-3, default: 3): Voice detection strictness (0=permissive, 3=strict)
• speed (0.25-4.0): Speech rate (1.0=normal, 2.0=double speed)
• chime_enabled (bool): Enable/disable audio feedback chimes
• chime_leading_silence (float): Silence before chime in seconds
• chime_trailing_silence (float): Silence after chime in seconds
• metrics_level ("minimal"|"summary"|"verbose"): Output detail level
  - minimal: Just response text (saves tokens)
  - summary: Response + compact timing (default)
  - verbose: Response + detailed metrics breakdown
• wait_for_conch (bool, default: false): Multi-agent coordination
  - false: If another agent is speaking, return status immediately
  - true: Wait until the other agent finishes, then speak
• skip_conch (bool, default: false): Bypass conch entirely
  - false: Honour the conch lock (default multi-agent coordination)
  - true: Don't try to acquire or release the conch -- speak immediately
    regardless of whether another agent holds it. Use when you intentionally
    want to talk over other agents or run outside the coordination protocol.

TIMING PARAMETERS (usually leave at defaults):
  Silence detection handles most cases automatically. Only override these if
  silence detection is disabled or the user reports being cut off.
  Defaults are configurable by the user via ~/.voicemode/voicemode.env.
• listen_duration_max (number, default: 120): Max listen time in seconds
• listen_duration_min (number, default: 2.0): Min recording time before silence detection

PRIVACY: Microphone access required when wait_for_response=true.
         Audio processed via STT service, not stored.

RECOGNITION TIP: If specific words are consistently misrecognized, configure
   VOICEMODE_STT_PROMPT for vocabulary biasing - see voicemode://docs/parameters

VOICEMODE ECHO (default ON): Some hosts (e.g. newer Claude Code) collapse MCP
   tool calls, hiding voice turns from the visible transcript. To keep voice
   exchanges readable on screen, echo each converse turn as Markdown blockquotes:
       > **ASSISTANT (voicemode):** <message arg, verbatim>
       [voicemode:converse tool call]
       > **USER (voicemode):** <captured user message, verbatim>
   - ASSISTANT echo: always (incl. wait_for_response=false). Verbatim — the
     exact string passed to `message`, not a paraphrase or reformat.
   - USER echo: only when a user message was captured (skip on empty result
     or transcription failure). Verbatim, no truncation.
   - Visual aids (lists, tables, code) may follow AFTER the blockquote, not
     inside it — the blockquote stays a clean verbatim copy of what was spoken.
   - Don't double-echo content already visible as prose.
   - Disable on request — canonical phrase: "disable voicemode echo".

For complete parameter list, advanced options, and detailed examples,
consult the MCP resources listed above.
    """
    # Convert string booleans to actual booleans
    if isinstance(wait_for_response, str):
        wait_for_response = wait_for_response.lower() in ('true', '1', 'yes', 'on')
    if isinstance(disable_silence_detection, str):
        disable_silence_detection = disable_silence_detection.lower() in ('true', '1', 'yes', 'on')
    if isinstance(chime_enabled, str):
        chime_enabled = chime_enabled.lower() in ('true', '1', 'yes', 'on')
    if skip_tts is not None and isinstance(skip_tts, str):
        skip_tts = skip_tts.lower() in ('true', '1', 'yes', 'on')
    if isinstance(wait_for_conch, str):
        wait_for_conch = wait_for_conch.lower() in ('true', '1', 'yes', 'on')
    if isinstance(skip_conch, str):
        skip_conch = skip_conch.lower() in ('true', '1', 'yes', 'on')

    # Resolve ref_text override once (path-vs-inline auto-detect). None means
    # "no override" — fall back to the resolved profile/sidecar transcript.
    resolved_ref_text = resolve_ref_text(ref_text)

    # Convert vad_aggressiveness to integer if provided as string
    if vad_aggressiveness is not None and isinstance(vad_aggressiveness, str):
        try:
            vad_aggressiveness = int(vad_aggressiveness)
            # Validation will happen later in the function
        except ValueError:
            logger.warning(f"Invalid VAD aggressiveness value '{vad_aggressiveness}', using default")
            vad_aggressiveness = None
    
    # Determine whether to skip TTS
    if skip_tts is not None:
        # Parameter explicitly set, use it
        should_skip_tts = skip_tts
    else:
        # Use global setting
        should_skip_tts = SKIP_TTS
    
    # Convert string speed to float
    if speed is not None and isinstance(speed, str):
        try:
            speed = float(speed)
        except ValueError:
            return f"❌ Error: speed must be a number (got '{speed}')"

    # Apply default speed from config if not provided
    speed_from_config = False
    if speed is None:
        speed = TTS_SPEED
        speed_from_config = True

    # Validate speed parameter range
    if speed is not None:
        if not (0.25 <= speed <= 4.0):
            source = " from VOICEMODE_TTS_SPEED environment variable" if speed_from_config else ""
            return f"❌ Error: speed must be between 0.25 and 4.0 (got {speed}{source})"

    # Determine effective metrics level (parameter overrides config)
    effective_metrics_level = metrics_level if metrics_level else METRICS_LEVEL

    logger.info(f"Converse: '{message[:50]}{'...' if len(message) > 50 else ''}' (wait_for_response: {wait_for_response})")
    
    # Validate vad_aggressiveness parameter
    if vad_aggressiveness is not None:
        if not isinstance(vad_aggressiveness, int) or vad_aggressiveness < 0 or vad_aggressiveness > 3:
            return f"Error: vad_aggressiveness must be an integer between 0 and 3 (got {vad_aggressiveness})"
    
    # Validate duration parameters
    if wait_for_response:
        if listen_duration_min < 0:
            return "❌ Error: listen_duration_min cannot be negative"
        if listen_duration_max <= 0:
            return "❌ Error: listen_duration_max must be positive"
        if listen_duration_min > listen_duration_max:
            logger.warning(f"listen_duration_min ({listen_duration_min}s) is greater than listen_duration_max ({listen_duration_max}s), using listen_duration_max as minimum")
            listen_duration_min = listen_duration_max
    
    # Check if FFmpeg is available
    ffmpeg_available = getattr(voice_mode.config, 'FFMPEG_AVAILABLE', True)  # Default to True if not set
    if not ffmpeg_available:
        from ..utils.ffmpeg_check import get_install_instructions
        error_msg = (
            "FFmpeg is required for voice features but is not installed.\n\n"
            f"{get_install_instructions()}\n\n"
            "Voice features cannot work without FFmpeg."
        )
        logger.error(error_msg)
        return f"❌ Error: {error_msg}"
    
    # Run startup initialization if needed
    await startup_initialization()
    
    # Refresh audio device cache to pick up any device changes (AirPods, etc.)
    # This takes ~1ms and ensures we use the current default device
    import sounddevice as sd
    sd._terminate()
    sd._initialize()
    
    # Get event logger and start session
    event_logger = get_event_logger()
    session_id = None
    
    # Check time since last session for AI thinking time
    global last_session_end_time
    current_time = time.time()
    
    if last_session_end_time and wait_for_response:
        time_since_last = current_time - last_session_end_time
        logger.info(f"Time since last session: {time_since_last:.1f}s (AI thinking time)")
    
    # For conversations with responses, create a session
    if event_logger and wait_for_response:
        session_id = event_logger.start_session()
        # Log the time since last session as an event
        if last_session_end_time:
            event_logger.log_event("TIME_SINCE_LAST_SESSION", {
                "seconds": time_since_last
            })
    
    # Log tool request start (after session is created)
    if event_logger:
        # If we have a session, the event will be associated with it
        log_tool_request_start("converse", {
            "wait_for_response": wait_for_response,
            "listen_duration_max": listen_duration_max if wait_for_response else None
        })
    
    # Track execution time and resources
    start_time = time.time()
    if DEBUG:
        import resource
        start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        logger.debug(f"Starting converse - Memory: {start_memory} KB")
    
    result = None
    success = False
    conch = Conch(agent_name="converse")  # Named for event logging

    # AirPods press watcher for the whole call (TTS + listen); closed in finally.
    _press_session = airpods_press.session()
    _press_session.__enter__()

    try:
        # Try to acquire conch atomically (no race condition)
        # skip_conch=true bypasses coordination entirely: don't acquire, don't
        # check the holder, don't release. Speak regardless of who else has it.
        if CONCH_ENABLED and not skip_conch:
            acquired = conch.try_acquire()

            if not acquired:
                # Another agent has the conch
                holder = Conch.get_holder()
                holder_agent = holder.get('agent', 'unknown') if holder else 'unknown'

                if event_logger:
                    event_logger.log_event("CONCH_BLOCKED", {
                        "pid": os.getpid(),
                        "holder_pid": holder.get('pid') if holder else None,
                        "holder_agent": holder_agent,
                        "wait_for_conch": wait_for_conch
                    })

                if not wait_for_conch:
                    # Default: return immediately with status info
                    return (f"User is currently speaking with {holder_agent}. "
                            "Use wait_for_conch=true to queue, or try again later.")

                # Wait mode - poll with atomic retry
                if event_logger:
                    event_logger.log_event("CONCH_WAIT_START", {
                        "pid": os.getpid(),
                        "holder_agent": holder_agent,
                        "timeout": CONCH_TIMEOUT
                    })

                waited = 0.0
                while not conch.try_acquire() and waited < CONCH_TIMEOUT:
                    # Ask a merely-listening holder to hand the mic over
                    # (vibedispatcher#132). Refreshed every poll so the request
                    # stays fresh; an idle holder yields within seconds.
                    if CONCH_YIELD_ENABLED:
                        Conch.request_yield("converse", owner=conch.owner)
                    await asyncio.sleep(CONCH_CHECK_INTERVAL)
                    waited += CONCH_CHECK_INTERVAL

                if not conch._acquired and CONCH_YIELD_ENABLED:
                    # Hard timeout: preempt-and-acquire instead of failing — but
                    # NEVER mid-utterance (see _preempt_conch_after_tts_grace).
                    await _preempt_conch_after_tts_grace(conch)

                Conch.clear_yield_request(owner=conch.owner)

                if event_logger:
                    event_logger.log_event("CONCH_WAIT_END", {
                        "pid": os.getpid(),
                        "waited_seconds": waited,
                        "result": "acquired" if conch._acquired else "timeout"
                    })

                if not conch._acquired:
                    return f"Timed out waiting for conch ({CONCH_TIMEOUT}s). {holder_agent} is still speaking."

            # Successfully acquired
            if event_logger:
                event_logger.log_event("CONCH_ACQUIRE", {
                    "pid": os.getpid(),
                    "agent": "converse"
                })

            # Auto-focus tmux pane after conch acquisition, before audio playback
            if AUTO_FOCUS_PANE and is_tmux():
                focus_tmux_pane()
        elif CONCH_ENABLED and skip_conch:
            # Conch is enabled but the caller asked to bypass it.
            if event_logger:
                holder = Conch.get_holder()
                event_logger.log_event("CONCH_SKIPPED", {
                    "pid": os.getpid(),
                    "agent": "converse",
                    "holder_pid": holder.get('pid') if holder else None,
                    "holder_agent": holder.get('agent') if holder else None,
                })
            # Still auto-focus tmux pane -- pane focus is unrelated to the conch.
            if AUTO_FOCUS_PANE and is_tmux():
                focus_tmux_pane()

        # Local microphone approach with timing
        transport = "local"
        timings = {}
        try:
            async with audio_operation_lock:
                # Speak the message
                tts_start = time.perf_counter()
                _held_floor_at_start = we_hold_floor() is True
                barge_in_result = None  # set below only when natural mode actually armed a listener
                # Clear any barge-in flag left set by a prior turn before we
                # speak, so a stale trigger (e.g. from a previous natural-mode
                # turn that fired, then natural mode was switched off) can never
                # truncate THIS playback. reset is otherwise only reached inside
                # BargeInListener.start(), i.e. natural-mode turns only.
                audio_player.reset_barge_in_event()
                if should_skip_tts:
                    # Skip TTS entirely for faster response
                    tts_success = True
                    tts_metrics = {
                        'ttfa': 0,
                        'generation': 0,
                        'playback': 0,
                        'total': 0
                    }
                    tts_config = {'provider': 'no-op', 'voice': 'none'}
                else:
                    # Natural mode (Phase 1 barge-in): arm a concurrent mic
                    # listener for the duration of this TTS playback. Inert
                    # (never constructed) in turn mode, the default -- this
                    # branch only fires when the natural-mode flag file is
                    # present. See voice_mode/barge_in.py for the mechanism.
                    barge_in_listener = None
                    if barge_in.natural_mode_enabled():
                        barge_in_listener = barge_in.BargeInListener()
                        barge_in_listener.start()

                    # Duck DJ volume during TTS playback
                    with DJDucker():
                        tts_success, tts_metrics, tts_config = await text_to_speech_with_failover(
                            message=message,
                            voice=voice,
                            model=tts_model,
                            instructions=tts_instructions,
                            audio_format=audio_format,
                            initial_provider=tts_provider,
                            speed=speed,
                            ref_text=resolved_ref_text
                        )

                    if barge_in_listener is not None:
                        barge_in_result = barge_in_listener.stop()
                        if barge_in_result.triggered:
                            logger.info("🗣️ Natural mode: William spoke over the agent — treating it as the next turn")

                # Pause cut the speech: don't listen; wait for Proceed, then
                # play the not-yet-heard remainder (voice_mode/pause_resume.py).
                if not should_skip_tts and tts_metrics and tts_metrics.get('paused'):
                    async def _replay_remainder(text):
                        _ok, _m, _c = await text_to_speech_with_failover(
                            message=text,
                            voice=voice,
                            model=tts_model,
                            instructions=tts_instructions,
                            audio_format=audio_format,
                            initial_provider=tts_provider,
                            speed=speed,
                            ref_text=resolved_ref_text,
                        )
                        return _m
                    pause_outcome = await resume_after_pause(
                        message, tts_metrics, _replay_remainder,
                        turn_end_signal_path=_turn_end_signal_path(),
                        held_floor_at_start=_held_floor_at_start,
                        # The host's idle timer started with this call: spend only
                        # what's left of the budget after the TTS already played.
                        max_seconds=max(0.0, pause_max_seconds() - (time.perf_counter() - tts_start)),
                    )
                    if pause_outcome and pause_outcome.handed_off:
                        return (
                            f"[FLOOR HANDED OFF — not delivered: {pause_outcome.undelivered_text}] "
                            f"The floor moved to another session mid-message; the rest was not "
                            f"played and no listen was started."
                        )
                    if pause_outcome and pause_outcome.timed_out:
                        return (
                            f"[PAUSED — timed out after {pause_outcome.paused_for_seconds:.0f}s "
                            f"waiting for Proceed. Speech was cut mid-message and the rest was not "
                            f"played; no listen was started.]"
                        )
                    if pause_outcome and pause_outcome.final_metrics is not None:
                        tts_metrics = dict(pause_outcome.final_metrics)
                        tts_metrics['truncated'] = False  # not a barge-in: nothing to report as talked-over

                # Add TTS sub-metrics
                if tts_metrics:
                    timings['ttfa'] = tts_metrics.get('ttfa', 0)
                    timings['tts_gen'] = tts_metrics.get('generation', 0)
                    timings['tts_play'] = tts_metrics.get('playback', 0)
                timings['tts_total'] = time.perf_counter() - tts_start
                
                # Log TTS immediately after it completes
                if tts_success:
                    try:
                        # Format TTS timing
                        tts_timing_parts = []
                        if 'ttfa' in timings:
                            tts_timing_parts.append(f"ttfa {timings['ttfa']:.1f}s")
                        if 'tts_gen' in timings:
                            tts_timing_parts.append(f"gen {timings['tts_gen']:.1f}s")
                        if 'tts_play' in timings:
                            tts_timing_parts.append(f"play {timings['tts_play']:.1f}s")
                        tts_timing_str = ", ".join(tts_timing_parts) if tts_timing_parts else None
                        
                        conversation_logger = get_conversation_logger()
                        conversation_logger.log_tts(
                            text=message,
                            audio_file=os.path.basename(tts_metrics.get('audio_path')) if tts_metrics and tts_metrics.get('audio_path') else None,
                            model=tts_config.get('model') if tts_config else tts_model,
                            voice=tts_config.get('voice') if tts_config else voice,
                            provider=tts_config.get('provider') if tts_config else (tts_provider if tts_provider else 'openai'),
                            provider_url=tts_config.get('base_url') if tts_config else None,
                            provider_type=tts_config.get('provider_type') if tts_config else None,
                            is_fallback=tts_config.get('is_fallback', False) if tts_config else False,
                            fallback_reason=tts_config.get('fallback_reason') if tts_config else None,
                            timing=tts_timing_str,
                            audio_format=audio_format,
                            transport=transport,
                            # Add timing metrics
                            time_to_first_audio=timings.get('ttfa') if timings else None,
                            generation_time=timings.get('tts_gen') if timings else None,
                            playback_time=timings.get('tts_play') if timings else None,
                            total_turnaround_time=timings.get('total') if timings else None
                        )
                    except Exception as e:
                        logger.error(f"Failed to log TTS to JSONL: {e}")
                
                if not tts_success:
                    # Check if we have detailed error information
                    if tts_config and tts_config.get('error_type') == 'all_providers_failed':
                        error_lines = ["Error: Could not speak message. TTS service connection failed:"]
                        openai_error_shown = False

                        for attempt in tts_config.get('attempted_endpoints', []):
                            # Check if we have parsed OpenAI error details
                            if attempt.get('error_details') and not openai_error_shown and attempt.get('provider') == 'openai':
                                error_details = attempt['error_details']
                                error_lines.append("")
                                error_lines.append(error_details.get('title', 'OpenAI Error'))
                                error_lines.append(error_details.get('message', ''))
                                if error_details.get('suggestion'):
                                    error_lines.append(f"💡 {error_details['suggestion']}")
                                if error_details.get('fallback'):
                                    error_lines.append(f"ℹ️ {error_details['fallback']}")
                                openai_error_shown = True
                            else:
                                # Show raw error for non-OpenAI or if we already showed OpenAI error
                                endpoint_or_provider = attempt.get('endpoint', attempt.get('provider', 'unknown'))
                                error_lines.append(f"  - {endpoint_or_provider}: {attempt['error']}")

                        result = "\n".join(error_lines)
                    # Check if we have config info that might indicate why it failed
                    elif tts_config and 'openai.com' in tts_config.get('base_url', ''):
                        # Check if API key is missing for OpenAI
                        from voice_mode.config import OPENAI_API_KEY
                        if not OPENAI_API_KEY:
                            result = "Error: Could not speak message. OpenAI API key is not set. Please set OPENAI_API_KEY environment variable or use local services (Kokoro TTS)."
                        else:
                            result = "Error: Could not speak message. TTS request to OpenAI failed. Please check your API key and network connection."
                    else:
                        result = "Error: Could not speak message. All TTS providers failed. Check that local services are running or set OPENAI_API_KEY for cloud fallback."
                    return result

                # If speak-only mode, return success after TTS
                if not wait_for_response:
                    # Format timing info for speak-only mode
                    timing_info = ""
                    if tts_success and tts_metrics:
                        timing_info = f" (gen: {tts_metrics.get('generation', 0):.1f}s, play: {tts_metrics.get('playback', 0):.1f}s)"

                    # Create timing string for statistics
                    timing_str = ""
                    if tts_success and timings:
                        timing_parts = []
                        if 'ttfa' in timings:
                            timing_parts.append(f"ttfa {timings['ttfa']:.1f}s")
                        if 'tts_gen' in timings:
                            timing_parts.append(f"tts_gen {timings['tts_gen']:.1f}s")
                        if 'tts_play' in timings:
                            timing_parts.append(f"tts_play {timings['tts_play']:.1f}s")
                        timing_str = ", ".join(timing_parts)

                    # Track statistics for speak-only interaction
                    track_voice_interaction(
                        message=message,
                        response="[speak-only]",
                        timing_str=timing_str,
                        transport="speak-only",
                        voice_provider=tts_provider,
                        voice_name=voice,
                        model=tts_model,
                        success=tts_success,
                        error_message=None if tts_success else "TTS failed"
                    )

                    # Format result based on metrics level
                    if effective_metrics_level == "minimal":
                        result = "✓ Message spoken successfully"
                    else:
                        result = f"✓ Message spoken successfully{timing_info}"

                    # W3c: speak-only calls can still be barge-in'd (the
                    # natural-mode listener is armed independent of
                    # wait_for_response) — surface it here too.
                    truncation_note = _format_truncation_note(tts_metrics)
                    if truncation_note:
                        result = f"{truncation_note}\n{result}"

                    logger.info(f"Speak-only result: {result}")
                    return result

                natural_mode_barge_in = barge_in_result is not None and barge_in_result.triggered

                if natural_mode_barge_in:
                    # He was already mid-utterance when he interrupted -- a
                    # "listening" chime now would be a confusing non-sequitur
                    # (and a fresh 0.5s pause would just eat the start of what
                    # he's saying). Skip both; go straight to recording, seeded
                    # with the audio the barge-in listener already captured.
                    logger.info("🎤 Natural mode barge-in — continuing to listen without a chime")
                else:
                    # Brief pause before listening
                    await asyncio.sleep(0.5)

                    # Play "listening" feedback sound
                    await play_audio_feedback(
                        "listening",
                        openai_clients,
                        chime_enabled,
                        "whisper",
                        chime_leading_silence=chime_leading_silence,
                        chime_trailing_silence=chime_trailing_silence
                    )

                    # Record response
                    logger.info(f"🎤 Listening for {listen_duration_max} seconds...")

                # Log recording start
                if event_logger:
                    event_logger.log_event(event_logger.RECORDING_START)

                # Yieldable listen (vibedispatcher#132) + in-place supersession:
                # while we hold the conch and are merely LISTENING (idle), we end
                # the listen early either because another agent ASKED for the mic
                # (Conch.is_wanted, config-gated — a request, subject to the grace
                # floor down in record_audio_with_silence_detection) or because a
                # preempter has already TAKEN it from us in place (Conch.is_superseded,
                # always honoured, bypasses the grace floor). These are two DIFFERENT
                # callables, not Conch.should_yield's composed bool, because the grace
                # floor must apply to the request half only — see the supersede_check
                # docstring on record_audio_with_silence_detection for why gating a
                # fact behind grace would reopen the double-hold bug PR #12 fixed.
                # Only wired when we actually hold the conch (skip_conch never yields).
                yield_state = {"yielded": False}
                listen_yield_check = (
                    functools.partial(Conch.is_wanted, owner=conch.owner)
                    if (CONCH_ENABLED and CONCH_YIELD_ENABLED and conch._acquired)
                    else None
                )
                listen_supersede_check = (
                    conch.is_superseded
                    if (CONCH_ENABLED and conch._acquired)
                    else None
                )

                # Every listen-again call site in this function MUST route
                # through here rather than calling record_audio_with_silence_
                # detection directly. This is the one place yield_check and
                # supersede_check are bundled together, so a call site can
                # forward the yield half (subject to the grace floor) without
                # anyone having to remember the supersede half (never gated,
                # see the comment above) — the exact split that let a fix land
                # on the primary listen and silently miss the two listen-again
                # paths below (repeat/wait), which is precisely where a stale
                # holder mattered most. Extra keyword args (pre_roll,
                # step_away_state, checkin_callback, ...) are forwarded as-is
                # for the primary listen's richer wiring.
                def _listen_call(**overrides):
                    kwargs = dict(
                        yield_check=listen_yield_check,
                        yield_state=yield_state,
                        supersede_check=listen_supersede_check,
                    )
                    kwargs.update(overrides)
                    return functools.partial(
                        record_audio_with_silence_detection,
                        listen_duration_max, disable_silence_detection, listen_duration_min, vad_aggressiveness,
                        **kwargs,
                    )

                # Graceful step-away wiring (founder-os#11655). Inert unless step-away
                # is enabled (env flag or ~/.voicemode/step-away.enabled). The check-in
                # is spoken from the executor thread via run_coroutine_threadsafe →
                # play_system_audio (existing pre-recorded/​TTS path). All swallow errors
                # so a check-in can never break the listen.
                step_away_state = {}
                _sa_on = step_away_enabled()
                _sa_loop = asyncio.get_event_loop()
                _sa_futs = []

                def _stepaway_checkin():
                    try:
                        fut = asyncio.run_coroutine_threadsafe(
                            play_system_audio("still-there", fallback_text="Still there?"),
                            _sa_loop,
                        )
                        _sa_futs.append(fut)
                        fut.result(timeout=20)
                    except Exception as _e:
                        logger.debug(f"step-away check-in failed (ignored): {_e}")

                listen_stats = {}
                _record_call = _listen_call(
                    pre_roll=(barge_in_result.pre_roll if natural_mode_barge_in else None),
                    step_away_state=step_away_state,
                    checkin_callback=(_stepaway_checkin if _sa_on else None),
                    listen_stats=listen_stats,
                )

                record_start = time.perf_counter()
                logger.debug(f"About to call record_audio_with_silence_detection with duration={listen_duration_max}, disable_silence_detection={disable_silence_detection}, min_duration={listen_duration_min}, vad_aggressiveness={vad_aggressiveness}, natural_mode_barge_in={natural_mode_barge_in}")
                try:
                    audio_data, speech_detected = await asyncio.get_event_loop().run_in_executor(
                        None, _record_call
                    )
                finally:
                    # Send/silence ended the listen: a still-playing "Still there?"
                    # check-in must not talk over the next turn.
                    for _f in _sa_futs:
                        _f.cancel()
                timings['record'] = time.perf_counter() - record_start

                # Log recording end (with end_reason / hold / starvation diagnostics)
                _rec_end, _starved = recording_end_event_data(timings['record'], len(audio_data), listen_stats)
                if event_logger:
                    event_logger.log_event(event_logger.RECORDING_END, _rec_end)
                if "send_source" in _rec_end:
                    logger.info(f"Recording ended by Send from: {_rec_end['send_source']}")
                if _starved:
                    logger.warning(f"⚠️ Input starved: {_rec_end['audio_seconds']:.2f}s of audio in "
                                   f"{_rec_end['wall_seconds']:.2f}s of listening")
                    if event_logger:
                        event_logger.log_event("INPUT_STARVED", _rec_end)

                if yield_state["yielded"]:
                    # Hand the mic over: skip the finished chime (the requester is
                    # about to speak), release via the finally block, and tell the
                    # caller this turn ended without a response.
                    if event_logger:
                        event_logger.log_event("CONCH_YIELDED", {
                            "pid": os.getpid(),
                            "agent": "converse",
                            "listened_seconds": timings['record'],
                        })
                    success = True  # a clean hand-off, not an error
                    result = ("Yielded the mic — another agent requested the floor while "
                              "you were idle-listening. No response was captured. Re-call "
                              "converse (wait_for_conch=true) when you want to continue.")
                    return result

                # Step-away outcome (founder-os#11655). Inert unless step-away armed.
                _resume_recap = False
                if _sa_on and step_away_state.get("stepped_away"):
                    if not speech_detected:
                        # He stepped away and never came back within the grace. We
                        # already spoke ONE check-in during the wait — end GRACEFULLY,
                        # not silently. Clear the pause flag so the next turn is fresh.
                        try:
                            if convomode_paused():
                                os.remove(PAUSE_FLAG_PATH)
                        except Exception:
                            pass
                        success = True
                        result = ("Stepped away — you asked me to hold on, I waited and "
                                  "checked in once, and it's still quiet. Pausing here; "
                                  "say something (or tap resume) when you're back and I'll "
                                  "pick up with a quick recap of where we left off.")
                        return result
                    else:
                        # He came back and spoke → resume. Clear the pause flag and flag
                        # a one-sentence recap (composes with temporal-orientation).
                        try:
                            if convomode_paused():
                                os.remove(PAUSE_FLAG_PATH)
                        except Exception:
                            pass
                        _resume_recap = True

                # Play "finished" feedback sound
                await play_audio_feedback(
                    "finished",
                    openai_clients,
                    chime_enabled,
                    "whisper",
                    chime_leading_silence=chime_leading_silence,
                    chime_trailing_silence=chime_trailing_silence
                )

                # Mark the end of recording - this is when user expects response to start
                user_done_time = time.perf_counter()
                logger.info(f"Recording finished at {user_done_time - tts_start:.1f}s from start")
                
                if len(audio_data) == 0:
                    result = "Error: Could not record audio"
                    return result
                
                # Track STT-specific metrics (defined here to be in scope for event logging later)
                stt_metrics = None

                # Check if no speech was detected
                if not speech_detected:
                    logger.info("No speech detected during recording - skipping STT processing")
                    response_text = None
                    timings['stt'] = 0.0

                    # Still save the audio if configured
                    if SAVE_AUDIO and AUDIO_DIR:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        audio_path = os.path.join(AUDIO_DIR, f"no_speech_{timestamp}.wav")
                        write(audio_path, SAMPLE_RATE, audio_data)
                        logger.debug(f"Saved no-speech audio to: {audio_path}")
                else:
                    # Convert to text
                    # Log STT start
                    if event_logger:
                        event_logger.log_event(event_logger.STT_START)

                    stt_start = time.perf_counter()
                    stt_result = await speech_to_text(audio_data, SAVE_AUDIO, AUDIO_DIR if SAVE_AUDIO else None, transport)
                    timings['stt'] = time.perf_counter() - stt_start

                    # Handle structured STT result
                    if isinstance(stt_result, dict):
                        # Extract metrics if present
                        stt_metrics = stt_result.get("metrics")
                        if stt_metrics:
                            # Store in timings for later use
                            timings['stt_request_ms'] = stt_metrics.get('request_time_ms', 0)
                            timings['stt_file_size_bytes'] = stt_metrics.get('file_size_bytes', 0)
                            timings['stt_is_local'] = stt_metrics.get('is_local', False)
                            logger.debug(f"STT metrics: request={stt_metrics.get('request_time_ms')}ms, "
                                       f"file_size={stt_metrics.get('file_size_bytes')/1024:.1f}KB, "
                                       f"is_local={stt_metrics.get('is_local')}")

                        if "error_type" in stt_result:
                            # Handle connection failures vs no speech
                            if stt_result["error_type"] == "connection_failed":
                                # Build helpful error message
                                error_lines = ["STT service connection failed:"]
                                openai_error_shown = False

                                for attempt in stt_result.get("attempted_endpoints", []):
                                    # Check if we have parsed OpenAI error details
                                    if attempt.get('error_details') and not openai_error_shown and attempt.get('provider') == 'openai':
                                        error_details = attempt['error_details']
                                        error_lines.append("")
                                        error_lines.append(error_details.get('title', 'OpenAI Error'))
                                        error_lines.append(error_details.get('message', ''))
                                        if error_details.get('suggestion'):
                                            error_lines.append(f"💡 {error_details['suggestion']}")
                                        if error_details.get('fallback'):
                                            error_lines.append(f"ℹ️ {error_details['fallback']}")
                                        openai_error_shown = True
                                    else:
                                        # Show raw error for non-OpenAI or if we already showed OpenAI error
                                        error_lines.append(f"  - {attempt['endpoint']}: {attempt['error']}")

                                error_msg = "\n".join(error_lines)
                                logger.error(error_msg)

                                # Return error immediately
                                return error_msg

                            elif stt_result["error_type"] == "no_speech":
                                # Genuine no speech detected
                                response_text = None
                                stt_provider = stt_result.get("provider", "unknown")
                        else:
                            # Successful transcription
                            response_text = stt_result.get("text")
                            stt_provider = stt_result.get("provider", "unknown")
                            if stt_provider != "unknown":
                                logger.info(f"📡 STT Provider: {stt_provider}")
                    else:
                        # Should not happen with new code, but handle gracefully
                        response_text = None
                        stt_provider = "unknown"

                # Spoken step-away / resume keywords (founder-os#11655). Inert unless
                # step-away is armed. A whole-utterance "hold on" sets the pause flag
                # so the NEXT listen is a graceful step-away wait; "I'm back"/"resume"
                # clears it and flags a recap. Matched near-exact so a mid-sentence
                # mention never false-triggers; failure falls back to normal handling.
                if _sa_on and response_text:
                    if is_step_away_phrase(response_text):
                        logger.info(f"⏸️ step-away phrase heard: '{response_text}' — pausing the listen")
                        try:
                            os.makedirs(os.path.dirname(PAUSE_FLAG_PATH), exist_ok=True)
                            open(PAUSE_FLAG_PATH, "w").close()
                        except Exception:
                            pass
                        success = True
                        result = ("You said to hold on — I'll wait. The next time you "
                                  "speak (or tap resume) I'll pick up with a quick recap "
                                  "of where we left off.")
                        return result
                    if is_resume_phrase(response_text):
                        logger.info(f"▶️ resume phrase heard: '{response_text}'")
                        try:
                            if convomode_paused():
                                os.remove(PAUSE_FLAG_PATH)
                        except Exception:
                            pass
                        _resume_recap = True

                # Check for repeat phrase - if detected, replay the audio and listen again
                if response_text and should_repeat(response_text):
                    logger.info(f"🔁 Repeat requested: '{response_text}'")

                    # Play system message for repeat
                    await play_system_audio("repeating", fallback_text="Repeating")

                    # Replay the same audio
                    if transport == "local":
                        logger.info("Replaying audio...")

                        # Play the cached audio if available from tts_metrics
                        audio_path = tts_metrics.get('audio_path') if 'tts_metrics' in locals() and tts_metrics else None
                        if audio_path and os.path.exists(audio_path):
                            try:
                                import soundfile as sf

                                # Read and play the audio file using non-blocking player
                                data, samplerate = sf.read(audio_path)
                                player = NonBlockingAudioPlayer()
                                player.play(data, samplerate, blocking=True)
                                logger.info("Audio replay completed")
                            except Exception as e:
                                logger.warning(f"Failed to replay cached audio: {e}. Regenerating...")
                                # Fall back to regenerating TTS
                                with DJDucker():
                                    tts_success, new_tts_metrics, _ = await text_to_speech_with_failover(
                                        message=message,
                                        voice=voice,
                                        model=tts_model,
                                        instructions=tts_instructions,
                                        audio_format=audio_format,
                                        initial_provider=tts_provider,
                                        speed=speed,
                                        ref_text=resolved_ref_text
                                    )
                                if not tts_success:
                                    logger.error("Failed to replay audio via TTS regeneration")
                        else:
                            # No cached audio, regenerate TTS
                            logger.info("No cached audio available, regenerating...")
                            with DJDucker():
                                tts_success, new_tts_metrics, _ = await text_to_speech_with_failover(
                                    message=message,
                                    voice=voice,
                                    model=tts_model,
                                    instructions=tts_instructions,
                                    audio_format=audio_format,
                                    initial_provider=tts_provider,
                                    speed=speed,
                                    ref_text=resolved_ref_text
                                )
                            if not tts_success:
                                logger.error("Failed to replay audio via TTS regeneration")

                        # Listen again for response - reuse the recording logic
                        logger.info("Listening for response after repeat...")

                        # Play "listening" feedback sound
                        await play_audio_feedback(
                            "listening",
                            openai_clients,
                            chime_enabled,
                            "whisper",
                            chime_leading_silence=chime_leading_silence,
                            chime_trailing_silence=chime_trailing_silence
                        )

                        # Record audio
                        record_start = time.perf_counter()
                        audio_data, speech_detected = await asyncio.get_event_loop().run_in_executor(
                            None, _listen_call()
                        )
                        record_time = time.perf_counter() - record_start
                        timings['record'] = timings.get('record', 0) + record_time  # Accumulate timing

                        if yield_state["yielded"]:
                            if event_logger:
                                event_logger.log_event("CONCH_YIELDED", {
                                    "pid": os.getpid(),
                                    "agent": "converse",
                                    "listened_seconds": record_time,
                                })
                            success = True
                            result = ("Yielded the mic — another agent requested the floor while "
                                      "you were idle-listening. No response was captured. Re-call "
                                      "converse (wait_for_conch=true) when you want to continue.")
                            return result

                        # Play "finished" feedback sound
                        await play_audio_feedback(
                            "finished",
                            openai_clients,
                            chime_enabled,
                            "whisper",
                            chime_leading_silence=chime_leading_silence,
                            chime_trailing_silence=chime_trailing_silence
                        )

                        if len(audio_data) > 0 and speech_detected:
                            # Transcribe the audio
                            stt_start = time.perf_counter()
                            stt_result = await speech_to_text(audio_data, SAVE_AUDIO, AUDIO_DIR if SAVE_AUDIO else None, transport)
                            stt_time = time.perf_counter() - stt_start
                            timings['stt'] = timings.get('stt', 0) + stt_time  # Accumulate timing

                            # Process result
                            if isinstance(stt_result, dict) and not stt_result.get("error"):
                                response_text = stt_result.get("text")
                                stt_provider = stt_result.get("provider", "unknown")
                                logger.info(f"New response after repeat: {response_text}")

                # Check for wait phrase - if detected, pause for configured duration
                if response_text and should_wait(response_text):
                    logger.info(f"⏸️ Wait requested: '{response_text}'. Pausing for {WAIT_DURATION} seconds...")

                    # Play system message for wait
                    await play_system_audio("waiting-1-minute", fallback_text="Waiting one minute")

                    await asyncio.sleep(WAIT_DURATION)

                    # Play system message when ready to listen again
                    await play_system_audio("ready-to-listen", fallback_text="Ready to listen")

                    # After waiting, listen again
                    logger.info("Wait period ended. Listening for response...")
                    if transport == "local":
                        # Play "listening" feedback sound
                        await play_audio_feedback(
                            "listening",
                            openai_clients,
                            chime_enabled,
                            "whisper",
                            chime_leading_silence=chime_leading_silence,
                            chime_trailing_silence=chime_trailing_silence
                        )

                        # Record audio
                        record_start = time.perf_counter()
                        audio_data, speech_detected = await asyncio.get_event_loop().run_in_executor(
                            None, _listen_call()
                        )
                        record_time = time.perf_counter() - record_start
                        timings['record'] = timings.get('record', 0) + record_time  # Accumulate timing

                        if yield_state["yielded"]:
                            if event_logger:
                                event_logger.log_event("CONCH_YIELDED", {
                                    "pid": os.getpid(),
                                    "agent": "converse",
                                    "listened_seconds": record_time,
                                })
                            success = True
                            result = ("Yielded the mic — another agent requested the floor while "
                                      "you were idle-listening. No response was captured. Re-call "
                                      "converse (wait_for_conch=true) when you want to continue.")
                            return result

                        # Play "finished" feedback sound
                        await play_audio_feedback(
                            "finished",
                            openai_clients,
                            chime_enabled,
                            "whisper",
                            chime_leading_silence=chime_leading_silence,
                            chime_trailing_silence=chime_trailing_silence
                        )

                        if len(audio_data) > 0 and speech_detected:
                            # Transcribe the audio
                            stt_start = time.perf_counter()
                            stt_result = await speech_to_text(audio_data, SAVE_AUDIO, AUDIO_DIR if SAVE_AUDIO else None, transport)
                            stt_time = time.perf_counter() - stt_start
                            timings['stt'] = timings.get('stt', 0) + stt_time  # Accumulate timing

                            # Process result
                            if isinstance(stt_result, dict) and not stt_result.get("error"):
                                response_text = stt_result.get("text")
                                stt_provider = stt_result.get("provider", "unknown")
                                logger.info(f"New response after wait: {response_text}")

                # Log STT complete with metrics
                if event_logger:
                    stt_event_data = {}
                    if response_text:
                        stt_event_data["text"] = response_text
                    # Include metrics in event log (debug level data)
                    if stt_metrics:
                        stt_event_data["metrics"] = {
                            "file_size_bytes": stt_metrics.get('file_size_bytes', 0),
                            "request_time_ms": stt_metrics.get('request_time_ms', 0),
                            "is_local": stt_metrics.get('is_local', False),
                            "format": "wav",
                            "sample_rate_hz": SAMPLE_RATE,
                            "bitrate_kbps": (SAMPLE_RATE * 16 * CHANNELS) // 1000
                        }
                    if response_text:
                        event_logger.log_event(event_logger.STT_COMPLETE, stt_event_data)
                    else:
                        event_logger.log_event(event_logger.STT_NO_SPEECH, stt_event_data)
                
                # Log STT immediately after it completes (even if no speech detected)
                try:
                    # Format STT timing
                    stt_timing_parts = []
                    if 'record' in timings:
                        stt_timing_parts.append(f"record {timings['record']:.1f}s")
                    if 'stt' in timings:
                        stt_timing_parts.append(f"stt {timings['stt']:.1f}s")
                    stt_timing_str = ", ".join(stt_timing_parts) if stt_timing_parts else None
                    
                    conversation_logger = get_conversation_logger()
                    # Get STT config for provider info
                    stt_config = await get_stt_config()
                    
                    conversation_logger.log_stt(
                        text=response_text if response_text else "[no speech detected]",
                        model=stt_config.get('model', STT_MODEL),
                        provider=stt_config.get('provider', 'openai'),
                        provider_url=stt_config.get('base_url'),
                        provider_type=stt_config.get('provider_type'),
                        audio_format='mp3',
                        transport=transport,
                        timing=stt_timing_str,
                        silence_detection={
                            "enabled": not (DISABLE_SILENCE_DETECTION or disable_silence_detection),
                            "vad_aggressiveness": VAD_AGGRESSIVENESS,
                            "silence_threshold_ms": SILENCE_THRESHOLD_MS
                        },
                        # Add timing metrics
                        transcription_time=timings.get('stt'),
                        total_turnaround_time=None  # Will be calculated and added later
                    )
                except Exception as e:
                    logger.error(f"Failed to log STT to JSONL: {e}")
            
            # Calculate total time (use tts_total instead of sub-metrics)
            main_timings = {k: v for k, v in timings.items() if k in ['tts_total', 'record', 'stt']}
            total_time = sum(main_timings.values())
            
            # Format timing strings separately for TTS and STT
            tts_timing_parts = []
            stt_timing_parts = []
            
            # TTS timings
            if 'ttfa' in timings:
                tts_timing_parts.append(f"ttfa {timings['ttfa']:.1f}s")
            if 'tts_gen' in timings:
                tts_timing_parts.append(f"gen {timings['tts_gen']:.1f}s")
            if 'tts_play' in timings:
                tts_timing_parts.append(f"play {timings['tts_play']:.1f}s")
            
            # STT timings
            if 'record' in timings:
                stt_timing_parts.append(f"record {timings['record']:.1f}s")
            if 'stt' in timings:
                stt_timing_parts.append(f"stt {timings['stt']:.1f}s")
            # Add detailed STT metrics if available
            if 'stt_file_size_bytes' in timings and timings['stt_file_size_bytes'] > 0:
                stt_timing_parts.append(f"audio {timings['stt_file_size_bytes']/1024:.0f}KB")
            
            tts_timing_str = ", ".join(tts_timing_parts) if tts_timing_parts else None
            stt_timing_str = ", ".join(stt_timing_parts) if stt_timing_parts else None
            
            # Keep combined timing for backward compatibility in result message
            all_timing_parts = []
            if tts_timing_parts:
                all_timing_parts.extend(tts_timing_parts)
            if stt_timing_parts:
                all_timing_parts.extend(stt_timing_parts)
            timing_str = ", ".join(all_timing_parts) + f", total {total_time:.1f}s"
            
            # Resume recap (founder-os#11655): if he came back after a step-away,
            # prepend a one-sentence-recap instruction so the agent orients William
            # before responding (composes with temporal-orientation-protocol). Inert
            # unless a step-away actually happened.
            if locals().get("_resume_recap") and response_text:
                response_text = (
                    "[Resuming after a step-away pause — open your reply with ONE short "
                    "sentence recapping where we left off, then continue.] " + response_text
                )

            # Track statistics for full conversation interaction
            actual_response = response_text or "[no speech detected]"
            track_voice_interaction(
                message=message,
                response=actual_response,
                timing_str=timing_str,
                transport=transport,
                voice_provider=tts_provider,
                voice_name=voice,
                model=tts_model,
                success=bool(response_text),  # Success if we got a response
                error_message=None if response_text else "No speech detected"
            )
            
            # End event logging session
            if event_logger and session_id:
                event_logger.end_session()
            
            if response_text:
                # Save conversation transcription if enabled
                if SAVE_TRANSCRIPTIONS:
                    conversation_text = f"Assistant: {message}\n\nUser: {response_text}"
                    metadata = {
                        "type": "conversation",
                        "transport": transport,
                        "voice": voice,
                        "model": tts_model,
                        "stt_model": stt_config.get('model', STT_MODEL),
                        "timing": timing_str,
                        "timestamp": datetime.now().isoformat()
                    }
                    save_transcription(conversation_text, prefix="conversation", metadata=metadata)

                # Logging already done immediately after TTS and STT complete

                # Format result based on metrics level
                stt_info = f" (STT: {stt_provider})" if 'stt_provider' in locals() and stt_provider != "unknown" else ""
                if effective_metrics_level == "minimal":
                    result = f"Voice response: {response_text}"
                elif effective_metrics_level == "verbose":
                    # Build verbose metrics block
                    verbose_parts = [f"Voice response: {response_text}{stt_info}"]
                    verbose_parts.append(f"Timing: {timing_str}")
                    if 'stt_request_ms' in timings:
                        verbose_parts.append(f"STT request: {timings['stt_request_ms']:.0f}ms")
                    if 'stt_file_size_bytes' in timings:
                        verbose_parts.append(f"STT file: {timings['stt_file_size_bytes']/1024:.0f}KB")
                    if 'stt_is_local' in timings:
                        verbose_parts.append(f"STT local: {timings['stt_is_local']}")
                    result = " | ".join(verbose_parts)
                else:  # summary (default)
                    result = f"Voice response: {response_text}{stt_info} | Timing: {timing_str}"
                success = True
            else:
                if effective_metrics_level == "minimal":
                    result = "No speech detected"
                else:
                    result = f"No speech detected | Timing: {timing_str}"
                success = True  # Not an error, just no speech

            # W3c: surface barge-in truncation context (if any) ahead of the
            # voice response itself -- this is metadata about what WE said,
            # independent of whether his reply was captured, so it applies
            # to every branch above. See _format_truncation_note's docstring
            # + voice_mode/truncation.py for the precision this can and
            # can't claim.
            truncation_note = _format_truncation_note(tts_metrics if 'tts_metrics' in locals() else None)
            if truncation_note:
                result = f"{truncation_note}\n{result}"

            # Superseded while he was mid-utterance: his words are delivered
            # above (never discarded), but this agent no longer owns the floor.
            if locals().get("yield_state") and yield_state.get("superseded_mid_utterance"):
                if event_logger:
                    event_logger.log_event("CONCH_SUPERSEDED_MID_UTTERANCE", {
                        "pid": os.getpid(), "agent": "converse",
                    })
                result = (f"{result}\n[Floor lost mid-utterance: another agent took the "
                          f"conch while he was speaking. Do not continue the conversation; "
                          f"re-call converse (wait_for_conch=true) only if you still need the floor.]")

            return result
                
        except Exception as e:
            logger.error(f"Local voice error: {e}")
            if DEBUG:
                logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Track failed conversation interaction
            track_voice_interaction(
                message=message,
                response="[error]",
                timing_str=None,
                transport=transport,
                voice_provider=tts_provider,
                voice_name=voice,
                model=tts_model,
                success=False,
                error_message=str(e)
            )
            
            result = f"Error: {str(e)}"
            return result
        
    except asyncio.CancelledError:
        # Tool call was cancelled by the MCP client (e.g. user pressed ESC).
        #
        # We intentionally DO NOT re-raise. Under FastMCP 2.x stdio transport,
        # an uncaught CancelledError escaping the tool handler tears down the
        # MCP server process, leaving the client with a failed connection that
        # requires `/mcp` reconnect. That surfaces to the user as VoiceMode
        # "disappearing" after every ESC (see VM-1026 / GH issue #337).
        #
        # Swallowing cancellation here is safe because this function is a leaf
        # coroutine invoked by FastMCP -- there is no outer task that needs to
        # observe the cancellation signal. The `finally` block below still
        # releases the conch, logs TOOL_REQUEST_END, and updates timing state,
        # so cleanup invariants hold.
        logger.info("Converse cancelled by client (ESC or tool-call cancel)")
        if event_logger:
            event_logger.log_event("TOOL_CANCELLED", {
                "tool_name": "converse",
                "reason": "client_cancel",
            })
        result = "Cancelled by user."
        success = False
        return result

    except Exception as e:
        logger.error(f"Unexpected error in converse: {e}")
        if DEBUG:
            logger.error(f"Full traceback: {traceback.format_exc()}")
        result = f"Unexpected error: {str(e)}"
        return result

    finally:
        _press_session.__exit__(None, None, None)

        # Drop any yield request we wrote while waiting (own-pid guarded no-op
        # if we never requested or another waiter's request is newer).
        Conch.clear_yield_request(owner=conch.owner)

        # Release the conch to signal voice conversation has ended
        if CONCH_ENABLED and conch._acquired:
            held_seconds = conch.release()
            if event_logger:
                event_logger.log_event("CONCH_RELEASE", {
                    "pid": os.getpid(),
                    "held_seconds": held_seconds
                })
        else:
            # Don't call release() when not acquired — it would delete the lock
            # file belonging to the agent that IS holding the conch, defeating
            # the flock coordination (they'd end up locking different inodes).
            pass

        # Log tool request end
        if event_logger:
            log_tool_request_end("converse", success=success)
        
        # Update last session end time for tracking AI thinking time
        if wait_for_response:
            last_session_end_time = time.time()
        
        # Log execution metrics
        elapsed = time.time() - start_time
        logger.info(f"Converse completed in {elapsed:.2f}s")
        
        if DEBUG:
            import resource
            import gc
            end_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            memory_delta = end_memory - start_memory
            logger.debug(f"Memory delta: {memory_delta} KB (start: {start_memory}, end: {end_memory})")
            
            # Force garbage collection
            collected = gc.collect()
            logger.debug(f"Garbage collected {collected} objects")




