"""Turn-taking UX: append-to-turn + step-away pause (founder-os#11655).

William's ask (voice, 2026-07-15): *"Convo mode works but it doesn't work well.
Sometimes I have to step away, sometimes I need to pause things, sometimes I want
to add something, sometimes it cuts me off."* This module owns the two of those
that are pure turn-loop UX gaps (design of record:
`wiki/research/natural-voice-mode-alternatives-2026-07-15.md` §4 items 5-6):

  1. **Append-to-turn** — after the silence timer fires, keep the SAME recording
     open for a short grace window; speech within it continues the turn instead
     of ending it. Fixes "I want to add something."
  2. **Step-away pause** — a flag file mirrors the existing TTS-side pause.flag
     onto the LISTEN side. While it exists the silence timer is suspended so
     William can step away; the moment he speaks again the flag is auto-cleared
     and a resume marker is dropped so the agent recaps rather than continuing
     mid-thought. Fixes "I have to step away" / "I need to pause."

The decision logic lives here — free of audio hardware and wall-clock reads, so
it is unit-testable. converse.py feeds it per-chunk events; this decides when
the turn actually ends. It is inert unless the caller constructs a
``TurnUxListenState`` at all, which converse.py only does when
``VOICEMODE_TURN_UX`` is set — so with the flag off, the listen loop is
byte-for-byte unchanged.
"""

import logging
import os

logger = logging.getLogger("voicemode.turn_ux")

# Spoken cues that mean "give me a moment" — canonical + testable in one place.
# The convomode skill layer matches a transcript against these to decide whether
# to raise the listen-pause flag; kept here (not in the skill prose) so the list
# is versioned with the mechanism and covered by tests.
STEP_AWAY_PHRASES = (
    "hold on",
    "hold on a sec",
    "hold on a second",
    "hold on a moment",
    "hold that thought",
    "give me a minute",
    "give me a sec",
    "give me a second",
    "give me a moment",
    "one moment",
    "one sec",
    "one second",
    "just a moment",
    "just a sec",
    "hang on",
    "hang on a sec",
    "wait a sec",
    "wait a second",
)


def is_step_away_phrase(text) -> bool:
    """True when a transcript is William asking to pause / step away.

    Matches the whole utterance or a leading cue ("hold on, let me grab that" →
    True), but not an incidental mid-sentence mention ("I told him to hold on to
    the receipt" → False), by only accepting the phrase at the start.
    """
    if not text:
        return False
    t = text.strip().lower().strip(" .,!?;:—-\"'")
    for p in STEP_AWAY_PHRASES:
        if t == p:
            return True
        # Leading cue followed by a boundary ("hold on, let me..." / "hold on —").
        if t.startswith(p) and t[len(p):len(p) + 1] in (" ", ",", ".", "!", "?", ";", ":", "-", "—"):
            return True
    return False


class TurnUxListenState:
    """Per-listen state for the append-window + step-away behaviours.

    Constructed once per ``record_audio_with_silence_detection`` call (only when
    the turn-UX flag is on). All timing is passed in by the caller in
    milliseconds/seconds; nothing here reads a clock, so a test can drive a full
    step-away / append sequence deterministically.
    """

    def __init__(
        self,
        *,
        append_window_ms: int,
        step_away_max_duration_s: float,
        listen_pause_flag_path: str,
        resumed_flag_path: str,
        chunk_duration_ms: int,
    ):
        self.append_window_ms = max(0, int(append_window_ms))
        self.step_away_max_duration_s = float(step_away_max_duration_s)
        self.listen_pause_flag_path = listen_pause_flag_path
        self.resumed_flag_path = resumed_flag_path
        self.chunk_duration_ms = int(chunk_duration_ms)
        # None → not in the post-silence append grace; else ms elapsed within it.
        self._append_grace_ms = None
        # True once we've observed the step-away flag during this listen, so we
        # know to arm the resume marker when speech comes back.
        self._stepped_away_seen = False
        # Read by the caller after the loop; also mirrored to the marker file.
        self.resumed_from_pause = False

    # --- step-away ----------------------------------------------------------

    def step_away_active(self) -> bool:
        """True while William has stepped away (the listen-pause flag exists)."""
        try:
            return os.path.exists(self.listen_pause_flag_path)
        except OSError:
            return False

    def effective_max_duration(self, base_max_s: float) -> float:
        """Hard listen cap for the while-guard.

        While stepped away, extend it to ``step_away_max_duration_s`` so a
        step-away doesn't hit the normal max_duration and end the turn. Recorded
        so that even after the flag clears we still know a pause happened.
        """
        if self.step_away_active():
            self._stepped_away_seen = True
            return max(base_max_s, self.step_away_max_duration_s)
        return base_max_s

    def on_speech(self) -> None:
        """A speech chunk arrived. Cancel any pending append grace, and if we were
        stepped away, clear the pause and arm the resume marker so the agent
        recaps instead of continuing mid-thought.
        """
        self._append_grace_ms = None
        if self._stepped_away_seen or self.step_away_active():
            self._clear_step_away()
            self._arm_resume_marker()

    def on_silence(
        self,
        *,
        recording_s: float,
        silence_ms: float,
        effective_min_s: float,
        threshold_ms: float,
    ) -> bool:
        """Decide whether the turn ends on this silence chunk.

        - Stepped away → never stop on silence (William asked for time).
        - Append window off → stop exactly as the stock loop would.
        - Append window on → once the base threshold is reached, wait
          ``append_window_ms`` more (accumulated across silence chunks) for a
          resumed thought before actually stopping.
        """
        if self.step_away_active():
            return False
        base_reached = recording_s >= effective_min_s and silence_ms >= threshold_ms
        if not base_reached:
            return False
        if self.append_window_ms <= 0:
            return True
        if self._append_grace_ms is None:
            # First chunk past the base threshold — start the grace, don't stop yet.
            self._append_grace_ms = 0
            return False
        self._append_grace_ms += self.chunk_duration_ms
        return self._append_grace_ms >= self.append_window_ms

    # --- internals ----------------------------------------------------------

    def _clear_step_away(self) -> None:
        try:
            if os.path.exists(self.listen_pause_flag_path):
                os.remove(self.listen_pause_flag_path)
        except OSError as e:
            logger.debug(f"could not remove listen-pause flag: {e}")
        self._stepped_away_seen = False

    def _arm_resume_marker(self) -> None:
        if self.resumed_from_pause:
            return
        self.resumed_from_pause = True
        try:
            with open(self.resumed_flag_path, "w") as f:
                f.write("1")
        except OSError as e:
            logger.debug(f"could not write resume marker: {e}")
