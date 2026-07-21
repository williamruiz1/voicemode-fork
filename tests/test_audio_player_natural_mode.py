"""Tests for the natural-mode additions to voice_mode/audio_player.py:
the barge-in event (a second, separate trigger reusing the existing
mid-buffer stop mechanism), the in-process "is TTS speaking" accessor, and
the far-end reference ring buffer the AEC reads from.
"""

import numpy as np
import pytest
import sounddevice as sd

import voice_mode.audio_player as ap


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Each test starts from a clean slate regardless of test order."""
    ap.reset_barge_in_event()
    ap._pause_event.clear()
    yield
    ap.reset_barge_in_event()
    ap._pause_event.clear()


class TestBargeInEvent:
    def test_starts_unset(self):
        assert ap.barge_in_triggered() is False

    def test_trigger_sets_it(self):
        ap.trigger_barge_in()
        assert ap.barge_in_triggered() is True

    def test_reset_clears_it(self):
        ap.trigger_barge_in()
        ap.reset_barge_in_event()
        assert ap.barge_in_triggered() is False

    def test_independent_of_pause_event(self):
        """Barge-in and manual pause are separate signals -- setting one
        must not report as the other."""
        ap.trigger_barge_in()
        assert ap.convomode_paused() is False
        ap.reset_barge_in_event()
        ap._pause_event.set()
        assert ap.barge_in_triggered() is False
        ap._pause_event.clear()


class TestIsTtsSpeaking:
    def test_false_when_nothing_playing(self):
        assert ap.is_tts_speaking() is False

    def test_true_while_playback_ref_counted_up(self):
        ap._speaking_inc()
        try:
            assert ap.is_tts_speaking() is True
        finally:
            ap._speaking_dec()
        assert ap.is_tts_speaking() is False

    def test_refcounted_across_overlapping_playbacks(self):
        ap._speaking_inc()
        ap._speaking_inc()
        try:
            assert ap.is_tts_speaking() is True
            ap._speaking_dec()
            assert ap.is_tts_speaking() is True, "one of two overlapping playbacks finished -- should still be speaking"
        finally:
            ap._speaking_dec()
        assert ap.is_tts_speaking() is False


class TestFarEndReferenceRingBuffer:
    def test_reads_zero_before_anything_written(self):
        ap._ref_buffer_init(24000)
        out = ap.get_reference_audio(480)
        assert len(out) == 480
        assert np.all(out == 0)

    def test_round_trip_no_delay(self):
        ap._ref_buffer_init(24000)
        samples = np.linspace(-1, 1, 480, dtype=np.float32)
        ap._ref_buffer_write(samples)
        out = ap.get_reference_audio(480, delay_samples=0)
        assert np.allclose(out, samples)

    def test_delay_reads_older_window(self):
        ap._ref_buffer_init(24000)
        first = np.full(480, 0.1, dtype=np.float32)
        second = np.full(480, 0.9, dtype=np.float32)
        ap._ref_buffer_write(first)
        ap._ref_buffer_write(second)
        # "As of 480 samples ago" (delay_samples=480) should land on `first`,
        # not the just-written `second`.
        out = ap.get_reference_audio(480, delay_samples=480)
        assert np.allclose(out, first)

    def test_wraps_around_capacity(self):
        # Tiny buffer so we can exercise the wrap-around branch cheaply.
        ap._ref_buffer_init(24000)
        ap._ref_buffer_capacity = 1000
        ap._ref_buffer = np.zeros(1000, dtype=np.float32)
        ap._ref_write_pos = 0

        chunk = np.full(600, 0.5, dtype=np.float32)
        ap._ref_buffer_write(chunk)  # pos now 600
        chunk2 = np.full(600, -0.5, dtype=np.float32)
        ap._ref_buffer_write(chunk2)  # wraps: pos now 1200, writes indices [600:1000] then [0:200]

        # Most recent 600 samples should be all -0.5 (chunk2), no corruption.
        out = ap.get_reference_audio(600, delay_samples=0)
        assert np.allclose(out, -0.5)

    def test_reading_further_back_than_available_zero_fills_prefix(self):
        ap._ref_buffer_init(24000)
        samples = np.full(100, 0.3, dtype=np.float32)
        ap._ref_buffer_write(samples)
        # Ask for 500 samples ending now -- only the last 100 were ever
        # written, so the first 400 must come back as silence, not garbage.
        out = ap.get_reference_audio(500, delay_samples=0)
        assert len(out) == 500
        assert np.allclose(out[:400], 0.0)
        assert np.allclose(out[400:], 0.3)


class TestAudioCallbackHonorsBargeIn:
    """The realtime audio callback must stop playback within one buffer when
    EITHER the pause flag or the barge-in flag is set -- barge-in reuses the
    exact same stop mechanism the manual pause already has."""

    def _make_player(self):
        player = ap.NonBlockingAudioPlayer(buffer_size=256)
        player.audio_queue = __import__("queue").Queue()
        player.audio_queue.put(np.ones(256, dtype=np.float32))
        player.audio_queue.put(None)
        return player

    def test_stops_on_barge_in_event(self):
        player = self._make_player()
        outdata = np.zeros((256, 1), dtype=np.float32)
        ap.trigger_barge_in()
        try:
            with pytest.raises(sd.CallbackStop):
                player._audio_callback(outdata, 256, None, None)
            assert np.all(outdata == 0)
            assert player.playback_complete.is_set()
        finally:
            ap.reset_barge_in_event()

    def test_stops_on_pause_event(self):
        player = self._make_player()
        outdata = np.zeros((256, 1), dtype=np.float32)
        ap._pause_event.set()
        try:
            with pytest.raises(sd.CallbackStop):
                player._audio_callback(outdata, 256, None, None)
            assert np.all(outdata == 0)
        finally:
            ap._pause_event.clear()

    def test_plays_normally_when_neither_event_set(self):
        player = self._make_player()
        outdata = np.zeros((256, 1), dtype=np.float32)
        player._audio_callback(outdata, 256, None, None)
        assert np.allclose(outdata[:, 0], 1.0)
        assert not player.playback_complete.is_set()
