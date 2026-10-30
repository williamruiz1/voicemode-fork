"""Split a TTS message so the first audio does not wait on the whole message.

Local TTS servers (mlx-audio, Kokoro) synthesize a large first chunk before
they start streaming, so time-to-first-audio grows with message length. Asking
for a short first piece (one sentence, or a clause near the start of a long
one) and fetching the rest while that piece plays makes first audio
independent of message length.
"""

import re
from typing import List

# Messages shorter than this are fast already (0-60 chars: ~0.6s median) and a
# split would only add a prosody seam, so they pass through unchanged.
MIN_TOTAL_CHARS = 100
# Shortest first piece worth a request of its own.
MIN_FIRST_CHARS = 12
# Longest first piece; a longer first sentence is cut at a clause boundary.
MAX_FIRST_CHARS = 90
# Earliest clause cut inside a long first sentence.
MIN_CLAUSE_CHARS = 30
# Don't leave a trivially short tail as its own request.
MIN_REST_CHARS = 20

_ABBREVIATIONS = {
    "e.g", "i.e", "etc", "vs", "dr", "mr", "mrs", "ms", "prof", "sr", "jr",
    "st", "no", "inc", "ltd", "co", "approx", "fig", "cf", "al", "u.s", "u.k",
}

# Terminator run, optional closing quote/bracket, then whitespace (or newline).
_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*(?=\s)")
_CLAUSE_END = re.compile(r"[,;:—](?=\s)|\s[-–—]\s")


def _is_abbreviation(text: str, end: int) -> bool:
    """True if the terminator ending at ``end`` closes an abbreviation or an
    initial ("Dr.", "e.g.", "J.") rather than a sentence."""
    m = re.search(r"(?<![A-Za-z])([A-Za-z](?:\.[A-Za-z])*)\.+[\"')\]]*$", text[:end])
    if not m:
        return False
    word = m.group(1).lower()
    return word in _ABBREVIATIONS or len(word) == 1


def _sentence_boundaries(text: str) -> List[int]:
    """Indexes just past each real sentence end (and paragraph break)."""
    cuts = []
    for m in _SENTENCE_END.finditer(text):
        if m.group().startswith(".") and _is_abbreviation(text, m.end()):
            continue
        cuts.append(m.end())
    for m in re.finditer(r"\n\s*\n", text):
        cuts.append(m.start())
    return sorted(cuts)


def split_first_piece(text: str) -> List[str]:
    """Return ``[first, rest]`` or ``[text]`` when splitting would not help."""
    if len(text) < MIN_TOTAL_CHARS:
        return [text]

    cut = None
    for end in _sentence_boundaries(text):
        if end < MIN_FIRST_CHARS:
            continue
        if end <= MAX_FIRST_CHARS:
            cut = end
        break

    if cut is None:
        # First sentence is long: cut at the last clause boundary in range.
        window = text[:MAX_FIRST_CHARS]
        for m in _CLAUSE_END.finditer(window):
            if m.start() + 1 >= MIN_CLAUSE_CHARS:
                cut = m.start() + 1 if text[m.start()] in ",;:—" else m.start()
        if cut is None:
            return [text]

    first, rest = text[:cut].strip(), text[cut:].strip()
    if len(first) < MIN_FIRST_CHARS or len(rest) < MIN_REST_CHARS:
        return [text]
    return [first, rest]
