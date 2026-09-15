"""Regression test for the B1 sticky-barge-in-flag bug.

`voice_mode/audio_player.py`'s `_barge_in_event` is a process-global
`threading.Event()`. `trigger_barge_in()` sets it; `reset_barge_in_event()`
clears it. Before this fix, `reset_barge_in_event()` was reachable from
EXACTLY ONE call site: inside `BargeInListener.start()`
(`voice_mode/barge_in.py`), which only runs when natural mode is armed
(`barge_in.natural_mode_enabled()` is True). So once a barge-in fired
during a natural-mode turn, the flag stayed set forever -- across every
subsequent turn, and even after natural mode was switched back off -- and
the very next TTS playback (streamed or non-streamed) would see the stale
flag already set and self-truncate at 0 bytes before speaking a word.

The fix adds an UNCONDITIONAL `audio_player.reset_barge_in_event()` call in
`voice_mode/tools/converse.py`, right before the `should_skip_tts` branch
inside the per-call `audio_operation_lock`, so every turn starts with a
clean flag regardless of what mode fired (or didn't fire) on the turn
before it.
"""

from unittest.mock import patch

import pytest

import voice_mode.audio_player as audio_player
from voice_mode import barge_in


class TestBargeInEventSetAndClear:
    """The underlying primitives: set by trigger, cleared by reset."""

    def teardown_method(self):
        # Never leak a set flag into another test in this process.
        audio_player.reset_barge_in_event()

    def test_trigger_barge_in_sets_the_flag(self):
        audio_player.reset_barge_in_event()
        assert audio_player.barge_in_triggered() is False
        audio_player.trigger_barge_in()
        assert audio_player.barge_in_triggered() is True

    def test_reset_barge_in_event_clears_the_flag(self):
        audio_player.trigger_barge_in()
        assert audio_player.barge_in_triggered() is True
        audio_player.reset_barge_in_event()
        assert audio_player.barge_in_triggered() is False


class TestConverseResetsStaleFlagEvenWithNaturalModeOff:
    """The regression itself: a flag left set by a PRIOR turn (e.g. a
    natural-mode barge-in that fired, then natural mode was switched off)
    must not survive into the NEXT speak() call -- proven here with natural
    mode forced off, so the only place the reset can possibly be coming
    from is the new unconditional call in converse.py, not
    BargeInListener.start()."""

    @pytest.mark.asyncio
    async def test_reset_called_before_tts_with_natural_mode_off(self):
        from voice_mode.tools.converse import converse

        # Simulate exactly the sticky-flag scenario: a barge-in fired on some
        # earlier turn and nothing has cleared it since.
        audio_player.trigger_barge_in()
        assert audio_player.barge_in_triggered() is True, "test precondition: flag starts stale/set"

        try:
            with patch.object(barge_in, "natural_mode_enabled", return_value=False) as mock_natural_mode, \
                 patch(
                     "voice_mode.tools.converse.text_to_speech_with_failover",
                     return_value=(False, {}, {"provider": "test"}),
                 ):
                await getattr(converse, "fn", converse)(
                    message="Hello",
                    wait_for_response=False,
                    skip_conch=True,
                )

            # The BargeInListener.start() path (the ONLY pre-fix reset site)
            # never runs when natural mode is off -- confirms this test is
            # actually exercising the new call, not the old one.
            mock_natural_mode.assert_called()
            assert audio_player.barge_in_triggered() is False, (
                "a stale barge-in flag from a prior turn must be cleared before "
                "this turn's TTS plays, even with natural mode off"
            )
        finally:
            audio_player.reset_barge_in_event()

    @pytest.mark.asyncio
    async def test_reset_is_the_real_converse_module_function_not_a_stub(self):
        """Spies on the exact function converse.py calls (via the module
        reference it imports, `from voice_mode import audio_player`) to
        confirm the wiring is real, not just an observed side effect that
        could have come from somewhere else."""
        from voice_mode.tools.converse import converse

        audio_player.reset_barge_in_event()  # clean baseline
        with patch(
            "voice_mode.audio_player.reset_barge_in_event",
            wraps=audio_player.reset_barge_in_event,
        ) as spy, \
             patch.object(barge_in, "natural_mode_enabled", return_value=False), \
             patch(
                 "voice_mode.tools.converse.text_to_speech_with_failover",
                 return_value=(False, {}, {"provider": "test"}),
             ):
            await getattr(converse, "fn", converse)(
                message="Hello",
                wait_for_response=False,
                skip_conch=True,
            )

        spy.assert_called()
