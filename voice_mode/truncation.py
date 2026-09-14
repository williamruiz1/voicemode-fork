"""
Barge-in truncation-context estimation (W3c).

When barge-in cuts TTS playback mid-message, `voice_mode/streaming.py` knows
EXACTLY how many bytes of audio it actually wrote to the output device
before it stopped (see `stream_pcm_audio`'s truncation branch). It does NOT
know -- and cannot know, given how the TTS APIs this project talks to
stream audio -- what fraction of the ORIGINAL text that represents:

  * the TTS response is streamed as it's generated; the total duration of
    the full, uninterrupted utterance is never known up front and is never
    reported by the API mid-stream;
  * neither Kokoro's nor OpenAI's streaming TTS endpoints return word- or
    phoneme-level timestamps for the audio they emit, so there is no
    ground-truth alignment between "N seconds of audio played" and "M
    characters of the source text" to read off.

So this module maps elapsed playback time back onto the text using an
assumed speaking rate (`config.TTS_ESTIMATED_WPM`), then snaps the
resulting character offset to the nearest SENTENCE boundary. Word-level
precision is never claimed anywhere in this module's output -- the
`confidence` field on every result says so explicitly, and every numeric
estimate is paired with the `basis` string documenting exactly how it was
derived, so a caller can decide how much to trust it rather than being
handed a bare number.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from .config import TTS_ESTIMATED_WPM

# Split on whitespace that follows a sentence-ending punctuation mark. This
# is a best-effort heuristic -- it doesn't special-case abbreviations like
# "Mr." or "e.g." -- but a wrong boundary here only shifts the estimate by
# part of one sentence, which is within the honesty budget this module
# already declares (see module docstring): the output is "roughly here,
# near a sentence boundary," never an exact word index.
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> List[str]:
    """Best-effort sentence split of `text`. Returns [] for empty/whitespace-only input."""
    stripped = text.strip()
    if not stripped:
        return []
    return [p for p in _SENTENCE_END_RE.split(stripped) if p]


@dataclass
class TruncationEstimate:
    """What we're willing to claim about a truncated TTS message.

    `delivered_text` / `undelivered_text` always split at a sentence
    boundary -- never mid-word, never at an arbitrary character offset --
    because that's the coarsest unit this estimate can honestly support.
    """
    delivered_text: str
    undelivered_text: str
    delivered_fraction: float          # 0.0-1.0, ESTIMATED (not measured)
    estimated_total_seconds: float     # ESTIMATED, from TTS_ESTIMATED_WPM
    confidence: str = "sentence-boundary-estimate"
    basis: str = ""


def estimate_truncation(
    full_text: str,
    delivered_seconds: float,
    wpm: Optional[int] = None,
) -> TruncationEstimate:
    """Estimate how much of `full_text` was audibly delivered given that
    approximately `delivered_seconds` of its audio reached the speaker
    before playback was cut.

    `delivered_seconds` should already be corrected for any known
    detection lag by the caller (e.g. the barge-in trigger's own
    speech-run threshold) -- this function just maps whatever value it's
    given onto the text; it has no way to know about that correction
    itself.
    """
    wpm = wpm if wpm is not None else TTS_ESTIMATED_WPM
    sentences = _split_sentences(full_text)

    if not sentences or delivered_seconds <= 0 or wpm <= 0:
        basis = (
            f"no usable text/timing to estimate from "
            f"(delivered_seconds={delivered_seconds:.2f}, wpm={wpm}) -- "
            f"treating nothing as confirmed-delivered"
        )
        return TruncationEstimate(
            delivered_text="",
            undelivered_text=full_text.strip(),
            delivered_fraction=0.0,
            estimated_total_seconds=0.0,
            basis=basis,
        )

    word_count = len(full_text.split())
    estimated_total_seconds = (word_count / wpm) * 60.0

    if estimated_total_seconds <= 0:
        delivered_fraction = 1.0
    else:
        delivered_fraction = max(0.0, min(1.0, delivered_seconds / estimated_total_seconds))

    approx_char_offset = round(delivered_fraction * len(full_text))

    # Sentence boundaries as cumulative end-offsets into the ORIGINAL text
    # (walk full_text rather than the stripped sentence list, so offsets
    # line up even with whatever whitespace the split regex consumed).
    boundaries = [0]
    cursor = 0
    try:
        for sentence in sentences:
            idx = full_text.index(sentence, cursor)
            cursor = idx + len(sentence)
            boundaries.append(cursor)
    except ValueError:
        # Should not happen (sentences are literal substrings of full_text
        # in order), but never let a text-alignment bug crash the estimate
        # -- fall back to "everything or nothing" at the midpoint.
        boundaries = [0, len(full_text)]
    if boundaries[-1] != len(full_text):
        boundaries.append(len(full_text))

    split_at = min(boundaries, key=lambda b: abs(b - approx_char_offset))

    delivered_text = full_text[:split_at].strip()
    undelivered_text = full_text[split_at:].strip()

    basis = (
        f"~{delivered_seconds:.1f}s of audio actually reached the speaker "
        f"vs. an estimated {estimated_total_seconds:.1f}s for the full "
        f"message at {wpm} words/min -- {delivered_fraction * 100:.0f}% by "
        f"estimated speaking time, split at the nearest sentence boundary "
        f"(char {split_at}/{len(full_text)}). No word-level alignment is "
        f"available from the TTS API this ran against."
    )

    return TruncationEstimate(
        delivered_text=delivered_text,
        undelivered_text=undelivered_text,
        delivered_fraction=round(delivered_fraction, 3),
        estimated_total_seconds=round(estimated_total_seconds, 2),
        basis=basis,
    )
