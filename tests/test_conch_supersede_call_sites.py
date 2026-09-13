"""Regression coverage for supersede_check wiring across ALL of converse()'s
listen-again call sites (fix/conch-fail-closed-into-master, PR #14).

Background: record_audio_with_silence_detection() is called from three
places inside converse() -- the primary listen, and two "listen again"
paths reached after a repeat/wait phrase is heard. Only the primary site
originally forwarded `supersede_check` (the in-place-preemption signal that
must stand a holder down immediately, no grace floor -- see
tests/test_conch_yield.py for why that's distinct from the gated
yield_check). The two secondary sites forwarded `yield_check` but silently
dropped `supersede_check`, so a holder already superseded by another agent
would keep listening through an entire repeat/wait cycle before it could
possibly notice.

converse.py now routes every listen call through one local `_listen_call()`
helper that bundles yield_check + supersede_check together, so a call site
can no longer omit the supersede half by accident. These tests exercise
that BEHAVIOR (not the presence of a keyword in source): they drive
converse() through the repeat and wait paths with a real Conch, have a real
preempter steal the mic between listens, and assert the secondary listen
actually stands down.
"""

from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from voice_mode.conch import Conch


# Loud enough to clear the STT_SILENCE_RMS_FLOOR pre-STT gate (0.005 of full
# scale) so speech_to_text actually calls simple_stt_failover instead of
# short-circuiting on "silence".
_SPEECH_AUDIO = np.array([0, 3000, -3000, 3000, -3000, 0], dtype=np.int16)
_EMPTY_AUDIO = np.array([], dtype=np.int16)


@pytest.fixture
def clean_conch():
    """Ensure no conch/wanted files exist before/after tests."""
    for f in (Conch.LOCK_FILE, Conch.WANTED_FILE):
        if f.exists():
            f.unlink()
    yield
    for f in (Conch.LOCK_FILE, Conch.WANTED_FILE):
        if f.exists():
            f.unlink()


def _make_recorder(preempt_before_second_call: bool = True):
    """A fake record_audio_with_silence_detection with the SAME idle-loop
    contract as the real one: on an in-place supersession it sets
    yield_state['yielded'] and returns early. Call #1 is the primary
    listen (normal speech); call #2 is the secondary "listen again" site
    under test -- a real preempter steals the conch right before this call
    polls its checks, exactly like a genuine in-place takeover between
    turns.
    """
    calls = []

    def _record(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return (_SPEECH_AUDIO.copy(), True)

        # This is the listen-again call site under test.
        if preempt_before_second_call:
            assert Conch(agent_name="preempter").preempt_acquire("preempter") is True, (
                "setup: preempter could not steal the conch"
            )

        supersede_check = kwargs.get("supersede_check")
        assert supersede_check is not None, (
            "the listen-again call site dropped supersede_check -- an "
            "already-superseded holder would keep listening instead of "
            "standing down immediately"
        )
        assert callable(supersede_check)
        assert supersede_check() is True, (
            "supersede_check did not report the real supersession that just "
            "happened -- wiring is present but not actually connected to "
            "the live Conch instance"
        )

        yield_state = kwargs.get("yield_state")
        assert yield_state is not None, "yield_state must still be forwarded too"
        yield_state["yielded"] = True
        return (_EMPTY_AUDIO.copy(), False)

    return _record, calls


def _patch_tts_and_system_audio(stack):
    """Patches needed to drive converse() to a listen-again call site
    without touching TTS/STT providers, real audio playback, or the
    microphone (record_audio_with_silence_detection is patched by the
    caller separately, per-test, so it can inspect kwargs). Entered into
    the caller's ExitStack so every test applies the same two patches
    without repeating them.
    """
    stack.enter_context(patch(
        "voice_mode.tools.converse.text_to_speech_with_failover",
        new=AsyncMock(return_value=(True, {"duration_ms": 10}, {"provider": "test"})),
    ))
    stack.enter_context(patch("voice_mode.tools.converse.play_system_audio", new=AsyncMock(return_value=True)))


class TestSupersedeCheckOnListenAgainSites:
    """Behavioral coverage for the repeat-path and wait-path call sites."""

    @pytest.mark.asyncio
    async def test_repeat_path_stands_down_on_in_place_supersession(self, clean_conch):
        from voice_mode.tools.converse import converse

        fake_record, calls = _make_recorder()

        with ExitStack() as stack:
            stack.enter_context(patch(
                "voice_mode.tools.converse.record_audio_with_silence_detection",
                side_effect=fake_record,
            ))
            stack.enter_context(patch(
                "voice_mode.simple_failover.simple_stt_failover",
                new=AsyncMock(side_effect=[
                    {"text": "please repeat", "provider": "test"},
                    {"text": "should never be reached", "provider": "test"},
                ]),
            ))
            _patch_tts_and_system_audio(stack)
            result = await getattr(converse, "fn", converse)(
                message="Test message",
                wait_for_response=True,
                chime_enabled=False,
            )

        assert len(calls) == 2, f"expected exactly 2 record calls (primary + repeat-listen), got {len(calls)}"
        assert "yielded the mic" in result.lower(), (
            f"converse() did not stand down on the repeat-listen after in-place "
            f"supersession; got: {result!r}"
        )

    @pytest.mark.asyncio
    async def test_wait_path_stands_down_on_in_place_supersession(self, clean_conch):
        from voice_mode.tools.converse import converse

        fake_record, calls = _make_recorder()

        with ExitStack() as stack:
            stack.enter_context(patch(
                "voice_mode.tools.converse.record_audio_with_silence_detection",
                side_effect=fake_record,
            ))
            stack.enter_context(patch("voice_mode.tools.converse.WAIT_DURATION", 0.01))
            stack.enter_context(patch(
                "voice_mode.simple_failover.simple_stt_failover",
                new=AsyncMock(side_effect=[
                    {"text": "wait", "provider": "test"},
                    {"text": "should never be reached", "provider": "test"},
                ]),
            ))
            _patch_tts_and_system_audio(stack)
            result = await getattr(converse, "fn", converse)(
                message="Test message",
                wait_for_response=True,
                chime_enabled=False,
            )

        assert len(calls) == 2, f"expected exactly 2 record calls (primary + wait-listen), got {len(calls)}"
        assert "yielded the mic" in result.lower(), (
            f"converse() did not stand down on the wait-listen after in-place "
            f"supersession; got: {result!r}"
        )

    @pytest.mark.asyncio
    async def test_repeat_path_supersede_check_independent_of_conch_yield_enabled(self, clean_conch):
        """§CONCH_YIELD_ENABLED-independence, proven executably.

        listen_supersede_check is gated only on CONCH_ENABLED + conch._acquired
        -- never on CONCH_YIELD_ENABLED (that flag gates the REQUEST half,
        yield_check, only). With CONCH_YIELD_ENABLED off, yield_check is None
        at every call site, but supersede_check must still be wired and must
        still fire on the repeat-listen call site.
        """
        from voice_mode.tools.converse import converse

        fake_record, calls = _make_recorder()

        with ExitStack() as stack:
            stack.enter_context(patch(
                "voice_mode.tools.converse.record_audio_with_silence_detection",
                side_effect=fake_record,
            ))
            stack.enter_context(patch("voice_mode.tools.converse.CONCH_YIELD_ENABLED", False))
            stack.enter_context(patch(
                "voice_mode.simple_failover.simple_stt_failover",
                new=AsyncMock(side_effect=[
                    {"text": "please repeat", "provider": "test"},
                    {"text": "should never be reached", "provider": "test"},
                ]),
            ))
            _patch_tts_and_system_audio(stack)
            result = await getattr(converse, "fn", converse)(
                message="Test message",
                wait_for_response=True,
                chime_enabled=False,
            )

        assert len(calls) == 2
        # yield_check must be the disabled/None half...
        assert calls[1].get("yield_check") is None, (
            "CONCH_YIELD_ENABLED=False should disable yield_check, or this "
            "test isn't proving independence"
        )
        # ...while supersede_check still fired and stood the holder down.
        assert "yielded the mic" in result.lower(), (
            "supersede_check stopped working once CONCH_YIELD_ENABLED was "
            "turned off -- it must be independent of that flag"
        )
