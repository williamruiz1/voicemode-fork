"""Regression tests for the 2026-09-13 barge-in silent-capture investigation.

A real live-conversation trace (574 frames, the user talking over the whole
of the assistant's speech) showed the natural-mode barge-in listener's
`tts_speaking` flag (`voice_mode.audio_player.is_tts_speaking()`) FALSE on
every single frame, and the AEC far-end reference signal
(`voice_mode.audio_player.get_reference_audio()`) all-zeros throughout.

Root cause: `voice_mode/streaming.py`'s `stream_pcm_audio()` and
`stream_with_buffering()` -- the ACTUAL playback path for TTS whenever
`VOICEMODE_STREAMING_ENABLED` is true (the default) and the format is one of
opus/mp3/pcm/wav (pcm is also the default) -- each open their OWN
`sd.OutputStream` and write to it directly. Neither function ever touched
`voice_mode.audio_player`, which is the ONLY module that used to raise the
"TTS is speaking" ref count or feed the AEC reference ring buffer, and it
only did so from `NonBlockingAudioPlayer.play()` -- a class the streaming
path never uses. So `barge_in.BargeInListener._watch_loop` always took its
"nothing is playing right now" branch (see `voice_mode/barge_in.py`, the
`if not tts_speaking:` block), which hard-codes `is_speech=False`,
`rms_far=0.0`, `rms_clean=0.0` -- exactly the trace's "silence" signature --
regardless of what was actually happening on the mic.

These tests assert the two streaming functions now call the public wiring
helpers added to `voice_mode/audio_player.py` (`playback_started()`,
`playback_finished()`, `write_reference_audio()`), observed through their
real effect on `is_tts_speaking()` / `get_reference_audio()` -- not just
"was the function called" -- so a future refactor that keeps calling the
old (now-removed) names but breaks the actual flag/buffer state would still
be caught.

Sounddevice is fully mocked here (`sd.OutputStream` never opens a real
device) -- these tests do not exercise microphone capture at all, only
playback-side bookkeeping, but see tests/conftest.py:block_real_microphone
for the structural guard on the input side regardless.
"""

import numpy as np
import pytest

import voice_mode.audio_player as ap
from voice_mode import streaming


class _FakeOutputStream:
    """Stand-in for sd.OutputStream that records writes instead of touching
    real hardware. Constructor accepts (and ignores) whatever kwargs the
    real sd.OutputStream would."""

    def __init__(self, *args, **kwargs):
        self.written = []
        self.started = False
        self.latency = 0.0

    def start(self):
        self.started = True

    def write(self, samples):
        self.written.append(np.asarray(samples).copy())

    def stop(self):
        pass

    def close(self):
        pass


class _BoomOnWriteStream(_FakeOutputStream):
    """Fails mid-playback, to prove the speaking flag still gets cleared."""

    def write(self, samples):
        raise RuntimeError("boom")


class _FakeStreamingResponse:
    """Minimal async-context-manager + iter_bytes() stand-in for the OpenAI
    SDK's `client.audio.speech.with_streaming_response.create(...)` result."""

    def __init__(self, chunks, raise_in_iter=None):
        self._chunks = chunks
        self._raise_in_iter = raise_in_iter

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def iter_bytes(self, chunk_size=None):
        for c in self._chunks:
            yield c
        if self._raise_in_iter is not None:
            raise self._raise_in_iter


class _FakeOpenAIClient:
    """Just enough of the AsyncOpenAI shape for
    `openai_client.audio.speech.with_streaming_response.create(**kwargs)`."""

    def __init__(self, chunks, raise_in_iter=None):
        self._chunks = chunks
        self._raise_in_iter = raise_in_iter
        self.audio = self
        self.speech = self
        self.with_streaming_response = self

    def create(self, **kwargs):
        return _FakeStreamingResponse(self._chunks, raise_in_iter=self._raise_in_iter)


class _FakeAudioSegment:
    """Stand-in for pydub's AudioSegment -- avoids needing a real
    ffmpeg-decodable payload or an installed ffmpeg binary just to exercise
    the flag/reference-buffer wiring in stream_with_buffering()."""

    def __init__(self, frame_rate, sample_width, array):
        self.frame_rate = frame_rate
        self.sample_width = sample_width
        self._array = array

    def set_frame_rate(self, rate):
        self.frame_rate = rate
        return self

    def set_sample_width(self, width):
        self.sample_width = width
        return self

    def get_array_of_samples(self):
        return self._array


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Every test starts from (and leaves) a clean audio_player state,
    regardless of test order or a mid-test failure."""
    while ap.is_tts_speaking():
        ap._speaking_dec()
    ap._ref_buffer_init(24000)
    ap._ref_write_pos = 0
    ap._ref_buffer[:] = 0
    yield
    while ap.is_tts_speaking():
        ap._speaking_dec()


class TestStreamPcmAudioSpeakingFlag:
    """stream_pcm_audio() is the DEFAULT playback path (VOICEMODE_TTS_
    AUDIO_FORMAT=pcm, VOICEMODE_STREAMING_ENABLED=true by default, and
    Kokoro -- the default local TTS endpoint -- serves pcm)."""

    async def test_speaking_flag_up_during_playback_down_after(self, monkeypatch):
        monkeypatch.setattr(streaming.sd, "OutputStream", _FakeOutputStream)
        seen_during_playback = []

        real_write = ap.write_reference_audio

        def _spy(samples, sample_rate):
            # Called once per chunk WHILE stream_pcm_audio is still inside
            # its playback loop -- the flag must be up right now.
            seen_during_playback.append(ap.is_tts_speaking())
            return real_write(samples, sample_rate)

        monkeypatch.setattr(ap, "write_reference_audio", _spy)

        assert ap.is_tts_speaking() is False

        pcm_chunk = np.array([100, -100, 200, -200], dtype=np.int16).tobytes()
        client = _FakeOpenAIClient([pcm_chunk, pcm_chunk])

        success, metrics = await streaming.stream_pcm_audio(
            text="hello",
            openai_client=client,
            request_params={"response_format": "pcm"},
        )

        assert success is True
        assert seen_during_playback == [True, True], (
            "is_tts_speaking() was not True while stream_pcm_audio was "
            "writing chunks to the output stream -- the natural-mode "
            "barge-in listener can never detect a turn is in progress "
            "when TTS actually plays through this path."
        )
        assert ap.is_tts_speaking() is False, (
            "speaking flag was not cleared after playback finished -- would "
            "permanently wedge the barge-in listener 'always armed' for "
            "every future turn."
        )

    async def test_speaking_flag_cleared_even_on_mid_playback_failure(self, monkeypatch):
        monkeypatch.setattr(streaming.sd, "OutputStream", _BoomOnWriteStream)

        client = _FakeOpenAIClient([b"\x00\x00\x00\x00"])
        success, metrics = await streaming.stream_pcm_audio(
            text="hello",
            openai_client=client,
            request_params={"response_format": "pcm"},
        )

        assert success is False
        assert ap.is_tts_speaking() is False, (
            "a mid-stream exception left the speaking ref count stuck up -- "
            "is_tts_speaking() would stay wedged True forever, permanently "
            "disabling the barge-in listener's 'nothing playing' branch."
        )

    async def test_played_audio_reaches_aec_reference_buffer(self, monkeypatch):
        monkeypatch.setattr(streaming.sd, "OutputStream", _FakeOutputStream)

        loud = np.full(240, 30000, dtype=np.int16)  # far from silence
        client = _FakeOpenAIClient([loud.tobytes()])

        await streaming.stream_pcm_audio(
            text="hi",
            openai_client=client,
            request_params={"response_format": "pcm"},
        )

        ref = ap.get_reference_audio(240, delay_samples=0)
        assert np.any(ref != 0), (
            "get_reference_audio() is all-zeros after stream_pcm_audio played "
            "real (non-silent) audio -- the streaming path never fed the AEC "
            "far-end reference buffer, so the barge-in listener's echo "
            "canceller has nothing to subtract from the mic signal."
        )
        assert np.allclose(ref, 30000 / 32768.0, atol=1e-3)


class TestStreamWithBufferingSpeakingFlag:
    """stream_with_buffering() is the fallback path for opus/mp3/etc (any
    non-pcm streamable format), e.g. the OpenAI cloud TTS failover."""

    async def test_speaking_flag_up_during_playback_down_after(self, monkeypatch):
        monkeypatch.setattr(streaming.sd, "OutputStream", _FakeOutputStream)
        fake_segment = _FakeAudioSegment(
            frame_rate=24000, sample_width=2,
            array=np.full(240, 30000, dtype=np.int16),
        )
        monkeypatch.setattr(streaming.AudioSegment, "from_file", lambda *a, **kw: fake_segment)

        seen_during_playback = []
        real_write = ap.write_reference_audio

        def _spy(samples, sample_rate):
            seen_during_playback.append(ap.is_tts_speaking())
            return real_write(samples, sample_rate)

        monkeypatch.setattr(ap, "write_reference_audio", _spy)

        # Small payload -> never crosses the mid-loop 32KB decode threshold,
        # so this exercises the "process any remaining data" flush path
        # (the second of the two write_reference_audio call sites).
        client = _FakeOpenAIClient([b"\x00" * 100])

        success, metrics = await streaming.stream_with_buffering(
            text="hello",
            openai_client=client,
            request_params={"response_format": "mp3"},
        )

        assert success is True
        assert seen_during_playback == [True], (
            "is_tts_speaking() was not True while stream_with_buffering was "
            "writing decoded audio to the output stream."
        )
        assert ap.is_tts_speaking() is False

    async def test_speaking_flag_cleared_even_on_mid_playback_failure(self, monkeypatch):
        """stream.start() succeeds (so the speaking flag DOES get raised),
        then the HTTP stream itself blows up -- decode failures inside the
        function's own two inner try/excepts are deliberately swallowed
        (existing behaviour, not under test here), so the discriminating
        failure has to come from somewhere those can't catch: iterating the
        response body itself."""
        monkeypatch.setattr(streaming.sd, "OutputStream", _FakeOutputStream)
        client = _FakeOpenAIClient([], raise_in_iter=RuntimeError("boom"))

        success, metrics = await streaming.stream_with_buffering(
            text="hello",
            openai_client=client,
            request_params={"response_format": "mp3"},
        )

        assert success is False
        assert ap.is_tts_speaking() is False, (
            "a mid-stream exception left the speaking ref count stuck up in "
            "stream_with_buffering() too."
        )

    async def test_played_audio_reaches_aec_reference_buffer(self, monkeypatch):
        monkeypatch.setattr(streaming.sd, "OutputStream", _FakeOutputStream)
        fake_segment = _FakeAudioSegment(
            frame_rate=24000, sample_width=2,
            array=np.full(240, 30000, dtype=np.int16),
        )
        monkeypatch.setattr(streaming.AudioSegment, "from_file", lambda *a, **kw: fake_segment)

        client = _FakeOpenAIClient([b"\x00" * 100])
        await streaming.stream_with_buffering(
            text="hi",
            openai_client=client,
            request_params={"response_format": "mp3"},
        )

        # stream_with_buffering appends TTS_TRAILING_SILENCE worth of zero
        # padding AFTER the real decoded samples before its one stream.write()
        # call, and write_reference_audio() mirrors exactly what was written
        # -- so the most-recent samples in the ring buffer are that trailing
        # silence, not the real audio. Read back a window wide enough to
        # cover both the padding and the real content ahead of it, rather
        # than just the (silent) tail.
        ref = ap.get_reference_audio(20000, delay_samples=0)
        assert np.any(ref != 0), (
            "get_reference_audio() is all-zeros after stream_with_buffering "
            "played real (non-silent) audio -- the streaming path never fed "
            "the AEC far-end reference buffer."
        )


class TestAudioPlayerStreamingWiringHelpers:
    """Direct unit coverage of the new public helpers themselves, isolated
    from the streaming module."""

    def test_playback_started_finished_round_trip_ref_count(self):
        assert ap.is_tts_speaking() is False
        ap.playback_started()
        assert ap.is_tts_speaking() is True
        ap.playback_finished()
        assert ap.is_tts_speaking() is False

    def test_playback_started_stacks_like_speaking_inc(self):
        """Two overlapping streamed playbacks (e.g. a quick double-fire)
        must not let the first one's finish() drop the flag early."""
        ap.playback_started()
        ap.playback_started()
        ap.playback_finished()
        assert ap.is_tts_speaking() is True
        ap.playback_finished()
        assert ap.is_tts_speaking() is False

    def test_write_reference_audio_initializes_buffer_lazily(self):
        """Unlike NonBlockingAudioPlayer.play() (which always calls
        _ref_buffer_init() itself before playing), a streaming-only session
        may never call _ref_buffer_init() any other way -- write_reference_
        audio() must not silently no-op forever the way raw _ref_buffer_
        write() does when the buffer was never sized."""
        ap._ref_buffer = None
        ap._ref_buffer_capacity = 0

        samples = np.full(100, 0.5, dtype=np.float32)
        ap.write_reference_audio(samples, sample_rate=24000)

        out = ap.get_reference_audio(100, delay_samples=0)
        assert np.allclose(out, 0.5)

    def test_write_reference_audio_accepts_multichannel_input(self):
        """Streaming code sometimes hands over (n, 1)-shaped arrays;
        write_reference_audio must reduce to mono like _ref_buffer_write does."""
        ap._ref_buffer_init(24000)
        samples = np.full((50, 1), 0.25, dtype=np.float32)
        ap.write_reference_audio(samples, sample_rate=24000)
        out = ap.get_reference_audio(50, delay_samples=0)
        assert np.allclose(out, 0.25)

    def test_write_reference_audio_never_raises(self, monkeypatch):
        """Reference-buffer bookkeeping must never break playback -- mirrors
        the existing try/except around _ref_buffer_write in
        NonBlockingAudioPlayer._audio_callback."""

        def _boom(*a, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(ap, "_ref_buffer_write", _boom)
        ap.write_reference_audio(np.zeros(10, dtype=np.float32), 24000)  # must not raise
