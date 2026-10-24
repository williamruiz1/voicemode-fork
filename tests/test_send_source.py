"""B32: RECORDING_END says WHICH control ended a turn.

Writers put a one-line tag in ~/.voicemode/turn-end.signal ("widget",
"hotkey", "airpods", "script"); the listen reads it before deleting the file
and reports it as ``send_source``. An empty file (older writers, bare
``touch``) reads as "unknown".
"""

import pytest

import voice_mode.tools.converse as C
from voice_mode.tools.converse import record_audio_with_silence_detection

from tests.test_hold_open_ptt import (  # noqa: F401  (fixtures)
    SEND_AT, SPEECH_N, _FakeInputStream, _FakeVad, _speech_then_silence, home, sim,
)


def _write_at(path, at_chunk, text):
    def on_chunk(n):
        if n == at_chunk:
            path.write_text(text)
    return on_chunk


@pytest.fixture(autouse=True)
def _no_leftover_source():
    C._take_send_source()
    yield
    C._take_send_source()


class TestReadSendSource:
    @pytest.mark.parametrize("text,expected", [
        ("", "unknown"),
        ("\n", "unknown"),
        ("widget\n", "widget"),
        ("hotkey", "hotkey"),
        ("  AirPods  \nsecond line ignored\n", "airpods"),
        ("weird tag!$\n", "weirdtag"),
        ("x" * 100, "x" * 32),
    ])
    def test_tags(self, tmp_path, text, expected):
        p = tmp_path / "turn-end.signal"
        p.write_text(text)
        assert C._read_send_source(str(p)) == expected

    def test_missing_file_is_unknown(self, tmp_path):
        assert C._read_send_source(str(tmp_path / "nope")) == "unknown"


class TestConsume:
    def test_consume_reads_then_deletes(self, home):
        sig = home / "turn-end.signal"
        sig.write_text("hotkey\n")
        assert C._consume_send() == "hotkey"
        assert not sig.exists()
        assert C._take_send_source() == "hotkey"
        assert C._take_send_source() is None  # taken once

    def test_clear_stale_is_not_reported(self, home):
        (home / "turn-end.signal").write_text("widget\n")
        C._clear_stale_send()
        assert not (home / "turn-end.signal").exists()
        assert C._take_send_source() is None


class TestListenReportsSource:
    def _stats(self, **kw):
        stats = {}
        record_audio_with_silence_detection(listen_stats=stats, **kw)
        return stats

    def test_vad_loop_send_carries_tag(self, sim):
        (sim / "hold-mode").touch()
        _FakeInputStream.script = _speech_then_silence(SPEECH_N)
        _FakeVad.on_processed = _write_at(sim / "turn-end.signal", SEND_AT, "widget\n")
        stats = self._stats(max_duration=30.0)
        assert stats["end_reason"] == "send"
        assert stats["send_source"] == "widget"
        assert not (sim / "turn-end.signal").exists()

    def test_empty_file_is_unknown(self, sim):
        (sim / "hold-mode").touch()
        _FakeInputStream.script = _speech_then_silence(SPEECH_N)
        _FakeVad.on_processed = _write_at(sim / "turn-end.signal", SEND_AT, "")
        stats = self._stats(max_duration=30.0)
        assert stats["end_reason"] == "send"
        assert stats["send_source"] == "unknown"

    def test_held_no_vad_send_carries_tag(self, sim):
        (sim / "hold-mode").touch()
        _FakeInputStream.on_chunk = _write_at(sim / "turn-end.signal", 30, "airpods\n")
        stats = self._stats(max_duration=30.0, disable_silence_detection=True)
        assert stats["end_reason"] == "send"
        assert stats["send_source"] == "airpods"

    def test_stale_send_at_entry_is_not_this_listens_source(self, sim):
        (sim / "turn-end.signal").write_text("hotkey\n")  # left from an earlier turn
        _FakeInputStream.script = _speech_then_silence(SPEECH_N)
        stats = self._stats(max_duration=30.0)
        assert stats["end_reason"] != "send"
        assert "send_source" not in stats

    def test_source_from_outside_a_listen_is_dropped(self, sim):
        C._consume_send()  # e.g. a fixed-length path consumed one with no listen
        _FakeInputStream.script = _speech_then_silence(SPEECH_N)
        stats = self._stats(max_duration=30.0)
        assert "send_source" not in stats


class TestRecordingEndEventData:
    def test_send_source_reported(self):
        data, _ = C.recording_end_event_data(2.0, 48000, {"end_reason": "send", "send_source": "hotkey"})
        assert data["send_source"] == "hotkey"

    def test_send_without_source_is_unknown(self):
        data, _ = C.recording_end_event_data(2.0, 48000, {"end_reason": "send"})
        assert data["send_source"] == "unknown"

    def test_non_send_has_no_source(self):
        data, _ = C.recording_end_event_data(2.0, 48000, {"end_reason": "silence"})
        assert "send_source" not in data
