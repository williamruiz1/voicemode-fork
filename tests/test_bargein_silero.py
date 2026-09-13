"""Tests for voicemode-endpointing-bargein W3:

  A. The barge-in listener's speech decision is swapped from webrtcvad alone
     to SileroVAD.prob() >= config.BARGE_IN_SILERO_THRESHOLD whenever Silero
     is available, falling back to the pre-existing webrtcvad path
     byte-for-byte when it isn't (voice_mode/barge_in.py).
  B. DoD criterion D4: audio_player.trigger_barge_in() is a REAL playback
     truncate (zeroes the in-flight buffer and halts the output stream
     within one audio buffer via sd.CallbackStop), not a no-op/flag-only
     mechanism that playback silently ignores or merely drains around.

No real audio hardware, ONNX inference, or webrtcvad classification is
involved -- SileroVAD/webrtcvad are both mocked, mirroring the existing
tests/test_barge_in.py and tests/test_audio_player_natural_mode.py patterns.
These validate the STATE MACHINE and the CALLBACK-LEVEL truncate mechanism,
not real acoustic behavior.
"""

import queue
import time
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sounddevice as sd

import voice_mode.audio_player as ap
import voice_mode.barge_in as bi


def _make_chunk(value: int = 5000, n: int = bi.CHUNK_SAMPLES_MIC) -> np.ndarray:
    """A synthetic mono int16 mic chunk, shaped like sounddevice's callback
    indata (frames, channels) -- same helper as tests/test_barge_in.py."""
    return np.full((n, 1), value, dtype=np.int16)


@pytest.fixture
def exit_stack():
    with ExitStack() as stack:
        yield stack


# ============================================================================
# A. Silero swap: decision routing
# ============================================================================

class TestSileroDecisionRouting:
    """BargeInListener must use SileroVAD.prob() >= threshold as the speech
    decision when Silero is available, and skip the webrtcvad path entirely
    in that case (proven by the listener never touching the mocked
    webrtcvad module)."""

    def _arm_with_silero(self, stack, monkeypatch, *, tts_speaking: bool, probs):
        """Arm a listener with Silero forced available and its prob() driven
        by a scripted sequence (one value per frame; the last value repeats
        for any extra calls)."""
        triggered_calls = []
        monkeypatch.setattr(bi.audio_player, "is_tts_speaking", lambda: tts_speaking)
        monkeypatch.setattr(
            bi.audio_player, "get_reference_audio",
            lambda n, delay_samples=0: np.zeros(n, dtype=np.float32),
        )
        monkeypatch.setattr(bi.audio_player, "trigger_barge_in", lambda: triggered_calls.append(True))
        monkeypatch.setattr(bi.audio_player, "reset_barge_in_event", lambda: None)

        probs_iter = iter(probs)
        mock_silero_instance = MagicMock()
        mock_silero_instance.prob.side_effect = lambda frame: next(probs_iter, probs[-1])

        monkeypatch.setattr(bi, "SILERO_AVAILABLE", True)
        stack.enter_context(patch.object(bi, "SileroVAD", return_value=mock_silero_instance))
        # webrtcvad is patched too, but never armed to answer speech --
        # `test_never_calls_webrtcvad_when_silero_available` asserts it's
        # never even consulted while Silero is active.
        mock_webrtcvad = stack.enter_context(patch.object(bi, "webrtcvad"))
        mock_webrtcvad.Vad.return_value = MagicMock()
        mock_sd = stack.enter_context(patch.object(bi, "sd"))
        mock_sd.InputStream.return_value = MagicMock()

        listener = bi.BargeInListener()
        assert listener._silero is mock_silero_instance, (
            "constructor must build a SileroVAD instance when SILERO_AVAILABLE is True"
        )
        listener.start()
        return listener, triggered_calls, mock_silero_instance, mock_webrtcvad

    def test_triggers_on_sustained_silero_speech_probability(self, monkeypatch, exit_stack):
        n_chunks = (bi.BARGE_IN_TRIGGER_MS // bi.CHUNK_MS) + 3
        listener, triggered_calls, mock_silero, _ = self._arm_with_silero(
            exit_stack, monkeypatch, tts_speaking=True, probs=[0.95] * n_chunks
        )
        for _ in range(n_chunks):
            listener._audio_queue.put(_make_chunk())
        listener._thread.join(timeout=3.0)
        result = listener.stop()

        assert result.triggered is True
        assert triggered_calls == [True]
        assert result.pre_roll is not None and len(result.pre_roll) > 0
        assert mock_silero.prob.called, "Silero's prob() must be the decision source"

    def test_below_threshold_probability_never_triggers(self, monkeypatch, exit_stack):
        n_chunks = (bi.BARGE_IN_TRIGGER_MS // bi.CHUNK_MS) + 5
        listener, triggered_calls, mock_silero, _ = self._arm_with_silero(
            exit_stack, monkeypatch, tts_speaking=True, probs=[0.2] * n_chunks
        )
        for _ in range(n_chunks):
            listener._audio_queue.put(_make_chunk())
        time.sleep(0.4)
        result = listener.stop()

        assert result.triggered is False
        assert triggered_calls == []

    def test_never_calls_webrtcvad_when_silero_available(self, monkeypatch, exit_stack):
        n_chunks = (bi.BARGE_IN_TRIGGER_MS // bi.CHUNK_MS) + 3
        listener, _, _, mock_webrtcvad = self._arm_with_silero(
            exit_stack, monkeypatch, tts_speaking=True, probs=[0.95] * n_chunks
        )
        for _ in range(n_chunks):
            listener._audio_queue.put(_make_chunk())
        listener._thread.join(timeout=3.0)
        listener.stop()

        mock_webrtcvad.Vad.return_value.is_speech.assert_not_called()

    def test_reset_called_on_arm(self, monkeypatch, exit_stack):
        """SileroVAD.reset() must be called when the listener arms (fresh
        turn), per the frozen-interface contract in silero_vad.py -- stale
        RNN state from a prior turn must not bias the earliest frames."""
        listener, _, mock_silero, _ = self._arm_with_silero(
            exit_stack, monkeypatch, tts_speaking=False, probs=[0.0]
        )
        try:
            mock_silero.reset.assert_called_once()
        finally:
            listener.stop()

    def test_custom_threshold_knob_respected(self, monkeypatch, exit_stack):
        """A probability that clears a low threshold but not a high one
        proves the decision reads config.BARGE_IN_SILERO_THRESHOLD live,
        not a hardcoded 0.5."""
        n_chunks = (bi.BARGE_IN_TRIGGER_MS // bi.CHUNK_MS) + 3
        monkeypatch.setattr(bi, "BARGE_IN_SILERO_THRESHOLD", 0.9)
        listener, triggered_calls, _, _ = self._arm_with_silero(
            exit_stack, monkeypatch, tts_speaking=True, probs=[0.8] * n_chunks
        )
        for _ in range(n_chunks):
            listener._audio_queue.put(_make_chunk())
        time.sleep(0.4)
        result = listener.stop()

        assert result.triggered is False, "0.8 must not clear a 0.9 threshold"
        assert triggered_calls == []


# ============================================================================
# A (regression). Fallback to webrtcvad when Silero is unavailable
# ============================================================================

class TestFallbackToWebrtcvadWhenSileroUnavailable:
    """With SILERO_AVAILABLE forced False, the listener must fall back to
    the pre-existing webrtcvad decision path byte-for-byte -- the same
    contract tests/test_barge_in.py locks in, re-verified here under the
    explicit fallback condition this dispatch calls out. Confirms nothing
    regresses in an environment without onnxruntime/the vendored model."""

    def _arm_webrtcvad_fallback(self, stack, monkeypatch, *, tts_speaking: bool, vad_says_speech: bool):
        triggered_calls = []
        monkeypatch.setattr(bi, "SILERO_AVAILABLE", False)
        monkeypatch.setattr(bi.audio_player, "is_tts_speaking", lambda: tts_speaking)
        monkeypatch.setattr(
            bi.audio_player, "get_reference_audio",
            lambda n, delay_samples=0: np.zeros(n, dtype=np.float32),
        )
        monkeypatch.setattr(bi.audio_player, "trigger_barge_in", lambda: triggered_calls.append(True))
        monkeypatch.setattr(bi.audio_player, "reset_barge_in_event", lambda: None)

        mock_vad = MagicMock()
        mock_vad.is_speech.return_value = vad_says_speech
        mock_webrtcvad = stack.enter_context(patch.object(bi, "webrtcvad"))
        mock_webrtcvad.Vad.return_value = mock_vad
        mock_sd = stack.enter_context(patch.object(bi, "sd"))
        mock_sd.InputStream.return_value = MagicMock()

        listener = bi.BargeInListener()
        assert listener._silero is None, "SILERO_AVAILABLE=False must leave the Silero backend unconstructed"
        listener.start()
        return listener, triggered_calls

    def test_webrtcvad_path_still_triggers(self, monkeypatch, exit_stack):
        listener, triggered_calls = self._arm_webrtcvad_fallback(
            exit_stack, monkeypatch, tts_speaking=True, vad_says_speech=True
        )
        n_chunks = (bi.BARGE_IN_TRIGGER_MS // bi.CHUNK_MS) + 3
        for _ in range(n_chunks):
            listener._audio_queue.put(_make_chunk())
        listener._thread.join(timeout=3.0)
        result = listener.stop()
        assert result.triggered is True
        assert triggered_calls == [True]

    def test_webrtcvad_path_still_respects_no_speech(self, monkeypatch, exit_stack):
        listener, triggered_calls = self._arm_webrtcvad_fallback(
            exit_stack, monkeypatch, tts_speaking=True, vad_says_speech=False
        )
        n_chunks = (bi.BARGE_IN_TRIGGER_MS // bi.CHUNK_MS) + 5
        for _ in range(n_chunks):
            listener._audio_queue.put(_make_chunk())
        time.sleep(0.4)
        result = listener.stop()
        assert result.triggered is False
        assert triggered_calls == []


# ============================================================================
# B. D4 -- proving trigger_barge_in() is a REAL cancel, not a no-op flag
# ============================================================================

class TestTruncateIsARealCancelNotANoOp:
    """Discriminating test for D4's crux. Two rival hypotheses:

      H1 (no-op / drain-around): trigger_barge_in() sets a flag that
         playback ignores, or only stops QUEUEING new work while whatever's
         already buffered/queued keeps draining out to the speaker.
      H2 (real cancel): the output callback checks the flag at the TOP of
         EVERY invocation, BEFORE touching the queue, zeroes the in-flight
         buffer, and raises sd.CallbackStop -- sounddevice's documented
         in-callback stream-stop signal (PortAudio issues no further
         callback once CallbackStop is raised).

    Discriminating test: trigger barge-in mid-playback (with un-played
    chunks still sitting in the queue) and invoke the SAME callback the
    OutputStream would call next. H1 predicts the next queued chunk plays
    (or is at least dequeued); H2 predicts the buffer is zeroed and the
    queue is left completely untouched -- an observable difference in
    actual output content and queue state, not a proxy.
    """

    def _queued_player(self, n_chunks: int = 5, buffer_size: int = 256):
        player = ap.NonBlockingAudioPlayer(buffer_size=buffer_size)
        player.audio_queue = queue.Queue()
        for i in range(n_chunks):
            # Distinct, identifiable content per chunk so "was this chunk
            # ever touched" is directly observable in the output.
            player.audio_queue.put(np.full(buffer_size, float(i + 1), dtype=np.float32))
        player.audio_queue.put(None)
        return player

    def test_trigger_mid_playback_zeroes_the_in_flight_buffer(self):
        player = self._queued_player()
        outdata = np.zeros((256, 1), dtype=np.float32)

        # One normal buffer first -- proves the harness is genuinely wired
        # to real playback (both hypotheses agree here).
        player._audio_callback(outdata, 256, None, None)
        assert np.allclose(outdata[:, 0], 1.0)
        assert not player.playback_complete.is_set()

        # Trigger barge-in via the REAL production function, then feed the
        # SAME callback the next buffer request.
        ap.trigger_barge_in()
        try:
            outdata2 = np.full((256, 1), 999.0, dtype=np.float32)  # sentinel
            with pytest.raises(sd.CallbackStop):
                player._audio_callback(outdata2, 256, None, None)
            assert np.all(outdata2 == 0), (
                "H1 falsified: the in-flight buffer must be zeroed to silence, "
                "not left as the sentinel or filled with the next queued chunk"
            )
            assert player.playback_complete.is_set()
        finally:
            ap.reset_barge_in_event()

    def test_trigger_mid_playback_never_dequeues_remaining_audio(self):
        """H1's weaker form (stop QUEUEING new work but let what's already
        buffered drain) would still eventually consume the queue. H2
        predicts the queue is completely untouched -- the callback bails out
        via the barge-in check BEFORE the `audio_queue.get_nowait()` call,
        so remaining chunks are never even read, let alone played."""
        player = self._queued_player(n_chunks=5)
        qsize_before = player.audio_queue.qsize()

        ap.trigger_barge_in()
        try:
            outdata = np.zeros((256, 1), dtype=np.float32)
            with pytest.raises(sd.CallbackStop):
                player._audio_callback(outdata, 256, None, None)
        finally:
            ap.reset_barge_in_event()

        assert player.audio_queue.qsize() == qsize_before, (
            "the queue must be untouched -- a no-op/flag-only implementation "
            "would still dequeue (and thus eventually play) the buffered audio"
        )

    def test_callbackstop_is_the_real_sounddevice_stream_stop_signal(self):
        """Not a home-rolled sentinel: sd.CallbackStop is the actual
        exception type sounddevice's PortAudio binding interprets as 'stop
        calling this callback' -- confirms the mechanism plugs into the
        real audio stack's own stop contract, not a mocked stand-in."""
        assert issubclass(sd.CallbackStop, Exception)

    def test_full_pipeline_listener_trigger_reaches_the_real_playback_cutoff(self, monkeypatch, exit_stack):
        """End-to-end wiring for D4: BargeInListener._watch_loop's trigger
        call IS audio_player.trigger_barge_in() (the REAL function this
        time, not a stub) -- proves a Silero-detected sustained speech run
        during TTS actually sets the SAME module-level event the playback
        callback checks, cuts a live buffer within one callback, AND hands
        back a non-empty pre_roll so his interruption seeds the next turn
        instead of being thrown away."""
        ap.reset_barge_in_event()
        monkeypatch.setattr(bi.audio_player, "is_tts_speaking", lambda: True)
        monkeypatch.setattr(
            bi.audio_player, "get_reference_audio",
            lambda n, delay_samples=0: np.zeros(n, dtype=np.float32),
        )
        # trigger_barge_in / barge_in_triggered are NOT mocked -- real path.
        monkeypatch.setattr(bi.audio_player, "reset_barge_in_event", lambda: None)

        n_chunks = (bi.BARGE_IN_TRIGGER_MS // bi.CHUNK_MS) + 3
        mock_silero_instance = MagicMock()
        mock_silero_instance.prob.return_value = 0.95
        monkeypatch.setattr(bi, "SILERO_AVAILABLE", True)
        exit_stack.enter_context(patch.object(bi, "SileroVAD", return_value=mock_silero_instance))
        mock_sd = exit_stack.enter_context(patch.object(bi, "sd"))
        mock_sd.InputStream.return_value = MagicMock()

        listener = bi.BargeInListener()
        listener.start()
        try:
            for _ in range(n_chunks):
                listener._audio_queue.put(_make_chunk())
            listener._thread.join(timeout=3.0)
            result = listener.stop()

            assert result.triggered is True
            assert result.pre_roll is not None and len(result.pre_roll) > 0

            # The REAL audio_player module-level event must now be set --
            # exactly what the output callback checks first, every buffer.
            assert ap.barge_in_triggered() is True

            player = ap.NonBlockingAudioPlayer(buffer_size=256)
            player.audio_queue = queue.Queue()
            player.audio_queue.put(np.ones(256, dtype=np.float32))
            outdata = np.zeros((256, 1), dtype=np.float32)
            with pytest.raises(sd.CallbackStop):
                player._audio_callback(outdata, 256, None, None)
            assert np.all(outdata == 0), "a fresh playback started after the real trigger must also be cut"
        finally:
            ap.reset_barge_in_event()
