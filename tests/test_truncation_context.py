"""Tests for W3c: capturing where barge-in cut TTS playback off, and
surfacing what was/wasn't delivered to the next converse() turn.

Three things under test:

1. `voice_mode/truncation.py` -- the pure text/time estimator. No mocking
   needed; these lock down the honesty contract (sentence-boundary only,
   never word-level; confidence + basis always explain the number).

2. `voice_mode/streaming.py`'s `stream_pcm_audio()` -- the actual playback
   loop this depends on. Before this change, `audio_player.
   barge_in_triggered()` was never checked anywhere in the streamed-PCM
   playback path (only `NonBlockingAudioPlayer`'s callback, a different,
   unused-by-default path, checked it) -- so a barge-in trigger never
   actually stopped audio from continuing to play, and there was nothing
   to truncate. These tests assert the loop now stops, and that the
   reported numbers match what was actually written to the output stream.

3. `voice_mode/tools/converse.py`'s `_format_truncation_note()` -- the
   shape handed back through the converse tool result, which is the one
   place the calling agent actually reads this.

Sounddevice is fully mocked (`sd.OutputStream` never opens a real device),
matching tests/test_streaming_barge_in_wiring.py's existing pattern -- no
microphone or speaker port is touched by this file.
"""

import numpy as np
import pytest

import voice_mode.audio_player as ap
from voice_mode import streaming
from voice_mode.truncation import estimate_truncation
from voice_mode.tools.converse import _format_truncation_note


# ---------------------------------------------------------------------------
# 1. voice_mode.truncation.estimate_truncation -- pure function, no mocks
# ---------------------------------------------------------------------------

class TestEstimateTruncation:
    def test_partial_delivery_splits_at_a_sentence_boundary_never_mid_word(self):
        text = "This is the first sentence. This is the second one. This is the third."
        # 150 wpm default => ~13 words / (150/60) ≈ 5.2s for the whole text.
        # Give it enough time to plausibly cover sentence 1 only.
        result = estimate_truncation(text, delivered_seconds=1.8, wpm=150)

        assert result.delivered_text in (
            "This is the first sentence.",
            "",  # acceptable if the estimate rounds down to nothing
        )
        # Whatever it picked, it must be an exact prefix ending exactly at a
        # sentence boundary -- never a partial word.
        assert text.startswith(result.delivered_text)
        if result.delivered_text:
            assert result.delivered_text.rstrip()[-1] in ".!?"
        assert result.undelivered_text == text[len(result.delivered_text):].strip()
        assert 0.0 <= result.delivered_fraction <= 1.0
        assert result.confidence == "sentence-boundary-estimate"
        assert "sentence boundary" in result.basis
        assert "No word-level alignment is available" in result.basis  # explicit disclaimer present

    def test_full_delivery_when_elapsed_time_covers_the_whole_message(self):
        text = "Short message. Two sentences."
        result = estimate_truncation(text, delivered_seconds=9999.0, wpm=150)
        assert result.delivered_text == text
        assert result.undelivered_text == ""
        assert result.delivered_fraction == 1.0

    def test_nothing_delivered_when_elapsed_seconds_is_zero_or_negative(self):
        text = "Anything at all."
        for bad in (0.0, -1.0):
            result = estimate_truncation(text, delivered_seconds=bad, wpm=150)
            assert result.delivered_text == ""
            assert result.undelivered_text == text
            assert result.delivered_fraction == 0.0
            assert "no usable text/timing" in result.basis

    def test_empty_text_never_raises(self):
        result = estimate_truncation("", delivered_seconds=3.0, wpm=150)
        assert result.delivered_text == ""
        assert result.undelivered_text == ""

    def test_single_sentence_all_or_nothing(self):
        text = "Just one sentence with no internal period."
        # Any positive delivered_seconds against a single-sentence message
        # can only land on "none" or "all" -- there's no interior boundary.
        result = estimate_truncation(text, delivered_seconds=0.1, wpm=150)
        assert result.delivered_text in ("", text)

    def test_never_overstates_confidence_tier(self):
        """The dataclass default and every real code path must use the one
        documented confidence tier -- nothing claims word-level precision."""
        text = "One. Two. Three."
        result = estimate_truncation(text, delivered_seconds=1.0, wpm=150)
        assert result.confidence == "sentence-boundary-estimate"
        assert "word" not in result.confidence  # no "word-level" tier exists

    def test_basis_always_names_the_measured_and_estimated_quantities(self):
        text = "One sentence here. Another sentence here."
        result = estimate_truncation(text, delivered_seconds=2.0, wpm=150)
        assert "audio actually reached the speaker" in result.basis
        assert "estimated" in result.basis
        assert "words/min" in result.basis


# ---------------------------------------------------------------------------
# 2. stream_pcm_audio() -- the actual playback loop
# ---------------------------------------------------------------------------

class _FakeOutputStream:
    """Same shape as tests/test_streaming_barge_in_wiring.py's fake, plus
    `.abort()` tracking (the new truncation path calls abort(), not stop(),
    to match NonBlockingAudioPlayer's immediate mid-buffer cut)."""

    def __init__(self, *args, **kwargs):
        self.written = []
        self.started = False
        self.stopped = False
        self.aborted = False
        self.latency = 0.0

    def start(self):
        self.started = True

    def write(self, samples):
        self.written.append(np.asarray(samples).copy())

    def stop(self):
        self.stopped = True

    def abort(self):
        self.aborted = True

    def close(self):
        pass


class _FakeStreamingResponse:
    """Like test_streaming_barge_in_wiring's fake, but can fire a real
    audio_player.trigger_barge_in() mid-iteration -- simulating the
    concurrent barge-in listener thread setting the flag while
    stream_pcm_audio is between chunks, exactly as it does in production."""

    def __init__(self, chunks, trigger_before_index=None):
        self._chunks = chunks
        self._trigger_before_index = trigger_before_index

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def iter_bytes(self, chunk_size=None):
        for i, c in enumerate(self._chunks):
            if self._trigger_before_index is not None and i == self._trigger_before_index:
                ap.trigger_barge_in()
            yield c


class _FakeOpenAIClient:
    def __init__(self, chunks, trigger_before_index=None):
        self._chunks = chunks
        self._trigger_before_index = trigger_before_index
        self.audio = self
        self.speech = self
        self.with_streaming_response = self

    def create(self, **kwargs):
        return _FakeStreamingResponse(self._chunks, trigger_before_index=self._trigger_before_index)


@pytest.fixture(autouse=True)
def _reset_audio_player_state():
    """Every test starts from (and leaves) a clean audio_player state."""
    while ap.is_tts_speaking():
        ap._speaking_dec()
    ap.reset_barge_in_event()
    ap._ref_buffer_init(24000)
    ap._ref_write_pos = 0
    ap._ref_buffer[:] = 0
    yield
    while ap.is_tts_speaking():
        ap._speaking_dec()
    ap.reset_barge_in_event()


def _chunk(n_samples, value=100):
    """One PCM chunk: n_samples of int16, as bytes."""
    return np.full(n_samples, value, dtype=np.int16).tobytes()


class TestStreamPcmAudioTruncation:
    async def test_barge_in_mid_stream_truncates_and_reports_the_split(self, monkeypatch):
        monkeypatch.setattr(streaming.sd, "OutputStream", _FakeOutputStream)

        text = "This is the first sentence. This is the second sentence. This is the third one."
        chunks = [_chunk(2048) for _ in range(6)]
        # Fire the barge-in flag right before the 3rd chunk (index 2) would
        # be processed -- chunks 0 and 1 get written, chunk 2 onward never do.
        client = _FakeOpenAIClient(chunks, trigger_before_index=2)

        success, metrics = await streaming.stream_pcm_audio(
            text=text,
            openai_client=client,
            request_params={"response_format": "pcm"},
        )

        assert success is True
        assert metrics.truncated is True, "barge-in fired mid-stream but metrics.truncated stayed False"
        assert metrics.chunks_received == 2, (
            f"expected exactly the 2 chunks written before the trigger, got {metrics.chunks_received}"
        )

        # The reported audio-time must match what was ACTUALLY written to
        # the output stream -- not an assumption, the real fake-stream log.
        assert ap  # (import kept alive / sanity)
        expected_bytes = 2 * 2048 * 2  # 2 chunks * 2048 samples * 2 bytes/sample (int16)
        expected_seconds = expected_bytes / (streaming.SAMPLE_RATE * 2)
        # elapsed_audio_seconds is rounded to 3 decimal places before being
        # stored on the metrics object -- compare against that same rounding.
        assert metrics.elapsed_audio_seconds == pytest.approx(round(expected_seconds, 3), abs=1e-6)

        # Delivered/undelivered split must exist and be a clean partition.
        assert metrics.delivered_text is not None
        assert metrics.undelivered_text is not None
        assert text.startswith(metrics.delivered_text)
        assert metrics.truncation_confidence == "sentence-boundary-estimate"
        assert metrics.truncation_basis  # non-empty, explains the derivation
        assert "audio actually written to the output stream" in metrics.truncation_basis

        # Fraction must be a real 0..1 estimate, never fabricated exactness.
        assert 0.0 <= metrics.delivered_fraction <= 1.0

    async def test_barge_in_stops_via_abort_not_a_normal_drain(self, monkeypatch):
        """Truncated playback must cut immediately (abort()), not let
        whatever's still buffered in the output stream play out (stop())."""
        monkeypatch.setattr(streaming.sd, "OutputStream", _FakeOutputStream)
        captured_stream = {}

        real_output_stream = _FakeOutputStream

        class _Spy(_FakeOutputStream):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                captured_stream["instance"] = self

        monkeypatch.setattr(streaming.sd, "OutputStream", _Spy)

        chunks = [_chunk(1024) for _ in range(4)]
        client = _FakeOpenAIClient(chunks, trigger_before_index=1)

        await streaming.stream_pcm_audio(
            text="One sentence. Another sentence.",
            openai_client=client,
            request_params={"response_format": "pcm"},
        )

        assert captured_stream["instance"].aborted is True
        assert captured_stream["instance"].stopped is False

    async def test_normal_completion_never_claims_truncation(self, monkeypatch):
        """A playback that runs to completion (no barge-in) must NOT report
        truncated=True or populate any of the truncation-context fields --
        the false-positive direction is just as important as detecting a
        real cut."""
        monkeypatch.setattr(streaming.sd, "OutputStream", _FakeOutputStream)
        chunks = [_chunk(1024) for _ in range(3)]
        client = _FakeOpenAIClient(chunks, trigger_before_index=None)

        success, metrics = await streaming.stream_pcm_audio(
            text="This all plays out normally.",
            openai_client=client,
            request_params={"response_format": "pcm"},
        )

        assert success is True
        assert metrics.truncated is False
        assert metrics.delivered_text is None
        assert metrics.undelivered_text is None
        assert metrics.elapsed_audio_seconds is None
        assert metrics.truncation_confidence is None

    async def test_truncation_check_does_not_break_the_speaking_flag_bookkeeping(self, monkeypatch):
        """The barge-in break path must still go through the same
        speaking_marked -> playback_finished() cleanup as every other exit
        path (see test_streaming_barge_in_wiring.py) -- a truncation must
        not leave is_tts_speaking() permanently wedged True."""
        monkeypatch.setattr(streaming.sd, "OutputStream", _FakeOutputStream)
        chunks = [_chunk(512) for _ in range(3)]
        client = _FakeOpenAIClient(chunks, trigger_before_index=0)

        assert ap.is_tts_speaking() is False
        await streaming.stream_pcm_audio(
            text="Doesn't matter.",
            openai_client=client,
            request_params={"response_format": "pcm"},
        )
        assert ap.is_tts_speaking() is False


# ---------------------------------------------------------------------------
# 3. _format_truncation_note -- the shape surfaced through the converse
#    tool result
# ---------------------------------------------------------------------------

class TestFormatTruncationNote:
    def test_returns_none_when_not_truncated(self):
        assert _format_truncation_note({"truncated": False}) is None
        assert _format_truncation_note({}) is None
        assert _format_truncation_note(None) is None

    def test_renders_delivered_and_undelivered_with_confidence_and_basis(self):
        tts_metrics = {
            "truncated": True,
            "delivered_text": "This is the first sentence.",
            "undelivered_text": "This is the second sentence.",
            "delivered_fraction": 0.5,
            "truncation_confidence": "sentence-boundary-estimate",
            "truncation_basis": "some derivation string",
        }
        note = _format_truncation_note(tts_metrics)
        assert note is not None
        assert "INTERRUPTED" in note
        assert "This is the first sentence." in note
        assert "This is the second sentence." in note
        assert "50%" in note
        assert "sentence-boundary-estimate" in note
        assert "some derivation string" in note
        # The instruction to the calling agent must be present -- this is
        # the whole point of surfacing it.
        assert "fold" in note.lower() or "bring it back" in note.lower()

    def test_handles_missing_optional_fields_without_raising(self):
        note = _format_truncation_note({"truncated": True})
        assert note is not None
        assert "INTERRUPTED" in note
