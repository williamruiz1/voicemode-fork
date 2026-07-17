"""Tests for turn-taking UX: append-to-turn + step-away pause (founder-os#11655).

These drive the pure decision helper directly — no audio hardware — simulating
the per-chunk events the converse.py listen loop feeds it, so the behaviour is
pinned deterministically.
"""

import os

import pytest

from voice_mode.turn_ux import TurnUxListenState, is_step_away_phrase


CHUNK_MS = 30
THRESHOLD_MS = 1000  # stock SILENCE_THRESHOLD_MS


def _state(tmp_path, *, append_window_ms=1200, step_away_max_s=180.0):
    return TurnUxListenState(
        append_window_ms=append_window_ms,
        step_away_max_duration_s=step_away_max_s,
        listen_pause_flag_path=str(tmp_path / "listen-pause.flag"),
        resumed_flag_path=str(tmp_path / "resumed.flag"),
        chunk_duration_ms=CHUNK_MS,
    )


# --- is_step_away_phrase -----------------------------------------------------

@pytest.mark.parametrize("text", [
    "hold on",
    "Hold on.",
    "hold on a sec",
    "give me a minute",
    "one moment",
    "hang on",
    "hold on, let me grab that",
    "  Give me a second!  ",
])
def test_step_away_phrase_matches(text):
    assert is_step_away_phrase(text) is True


@pytest.mark.parametrize("text", [
    "",
    None,
    "what's the status on the deploy",
    "I told him to hold on to the receipt",  # mid-sentence, not a leading cue
    "let's keep going",
    "okay sounds good",
])
def test_step_away_phrase_non_matches(text):
    assert is_step_away_phrase(text) is False


# --- append-to-turn ----------------------------------------------------------

def test_append_window_holds_stop_then_stops(tmp_path):
    """Once the base silence threshold is reached, the turn should NOT end
    immediately — it waits append_window_ms of further silence first."""
    st = _state(tmp_path, append_window_ms=300)
    # Silence has just crossed the base threshold.
    rec_s = 5.0
    # First chunk at/after threshold: arms the grace, does not stop.
    assert st.on_silence(recording_s=rec_s, silence_ms=THRESHOLD_MS,
                         effective_min_s=0.5, threshold_ms=THRESHOLD_MS) is False
    # Accumulate grace: 300ms / 30ms = 10 more silence chunks before stop.
    stops = []
    for i in range(1, 12):
        sil = THRESHOLD_MS + i * CHUNK_MS
        stops.append(st.on_silence(recording_s=rec_s, silence_ms=sil,
                                   effective_min_s=0.5, threshold_ms=THRESHOLD_MS))
    # It should eventually return True once grace >= 300ms, and not before.
    assert True in stops
    assert stops.index(True) >= 9  # ~300ms/30ms chunks of grace elapsed first


def test_append_window_resumed_speech_continues_turn(tmp_path):
    """Speaking again inside the append window cancels the stop — same turn."""
    st = _state(tmp_path, append_window_ms=300)
    rec_s = 5.0
    # Cross threshold, arm grace.
    st.on_silence(recording_s=rec_s, silence_ms=THRESHOLD_MS,
                  effective_min_s=0.5, threshold_ms=THRESHOLD_MS)
    # A couple more silence chunks (grace building).
    st.on_silence(recording_s=rec_s, silence_ms=THRESHOLD_MS + CHUNK_MS,
                  effective_min_s=0.5, threshold_ms=THRESHOLD_MS)
    # William resumes — speech chunk cancels the grace.
    st.on_speech()
    # A fresh silence period now must re-run the FULL grace, not fire immediately.
    assert st.on_silence(recording_s=rec_s + 1, silence_ms=THRESHOLD_MS,
                         effective_min_s=0.5, threshold_ms=THRESHOLD_MS) is False


def test_append_window_zero_is_stock_behaviour(tmp_path):
    """append_window_ms == 0 stops exactly when the base threshold is reached."""
    st = _state(tmp_path, append_window_ms=0)
    assert st.on_silence(recording_s=5.0, silence_ms=THRESHOLD_MS,
                         effective_min_s=0.5, threshold_ms=THRESHOLD_MS) is True


def test_below_min_duration_never_stops(tmp_path):
    st = _state(tmp_path, append_window_ms=1200)
    # recording shorter than the effective minimum: never stop regardless.
    assert st.on_silence(recording_s=0.2, silence_ms=THRESHOLD_MS,
                         effective_min_s=0.5, threshold_ms=THRESHOLD_MS) is False


# --- step-away pause ---------------------------------------------------------

def test_step_away_suspends_stop_and_extends_cap(tmp_path):
    st = _state(tmp_path, step_away_max_s=180.0)
    flag = tmp_path / "listen-pause.flag"
    flag.write_text("1")
    # While stepped away, silence NEVER ends the turn...
    assert st.on_silence(recording_s=5.0, silence_ms=5000,
                         effective_min_s=0.5, threshold_ms=THRESHOLD_MS) is False
    # ...and the hard cap is extended past the normal max_duration.
    assert st.effective_max_duration(120.0) == 180.0


def test_step_away_resume_clears_flag_and_arms_recap(tmp_path):
    st = _state(tmp_path)
    flag = tmp_path / "listen-pause.flag"
    resumed = tmp_path / "resumed.flag"
    flag.write_text("1")
    # Observe the step-away (as the while-guard does each iteration).
    st.effective_max_duration(120.0)
    assert flag.exists()
    # William returns and speaks.
    st.on_speech()
    # The pause flag is auto-cleared and the recap marker is armed.
    assert not flag.exists()
    assert resumed.exists()
    assert st.resumed_from_pause is True


def test_no_step_away_no_resume_marker(tmp_path):
    """A normal turn with no pause must not drop a resume marker."""
    st = _state(tmp_path)
    resumed = tmp_path / "resumed.flag"
    st.effective_max_duration(120.0)  # flag absent
    st.on_speech()  # ordinary first-speech
    assert not resumed.exists()
    assert st.resumed_from_pause is False


def test_effective_max_duration_normal_when_no_flag(tmp_path):
    st = _state(tmp_path, step_away_max_s=180.0)
    assert st.effective_max_duration(120.0) == 120.0
