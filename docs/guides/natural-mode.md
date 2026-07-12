# Natural Mode (Phase 1 — barge-in turn-taking)

Natural mode lets you interrupt the agent mid-sentence, the way you can with
the ChatGPT/Claude mobile apps — instead of waiting for it to finish talking
before the mic opens. This is **Phase 1** of the natural-conversation-mode
work: barge-in on top of the existing voicemode architecture (see
`wiki/research/natural-voice-mode-research-2026-07-12.md` in the founder-os
repo for the feasibility research and phased plan this implements).

## Turn mode vs. natural mode

| | **Turn mode** (default, unchanged) | **Natural mode** |
|---|---|---|
| Mic during playback | Closed — opens only after TTS finishes | Open — listening the whole time |
| Interrupting the agent | Not possible; wait for it to finish | Speak over it; playback halts mid-word, your speech becomes the next turn |
| Risk | None (identical to today) | Depends on live tuning — see below |

## Turning it on

```bash
~/.local/bin/founder-os/convomode-natural-mode.sh on       # turn it on
~/.local/bin/founder-os/convomode-natural-mode.sh off      # back to turn mode
~/.local/bin/founder-os/convomode-natural-mode.sh toggle   # flip it
~/.local/bin/founder-os/convomode-natural-mode.sh status   # "on" or "off"
```

This writes/removes a flag file (`~/.voicemode/natural-mode.flag`) that
`converse()` checks directly at the start of each turn — **no server
restart needed**, and turning it off instantly returns to today's exact
turn-taking behavior.

## How it works (Phase 1 mechanism)

1. While TTS plays, a **second, concurrent** microphone stream opens
   alongside it (turn mode's listener only ever opens *after* playback).
2. Each mic frame is run through a **software echo canceller** (an NLMS
   adaptive filter, `voice_mode/aec.py`) against the *exact* reference
   audio being sent to the speaker — so the mic hearing the agent's own
   voice doesn't get mistaken for you talking.
3. The echo-cancelled signal goes through `webrtcvad` (the same VAD library
   turn mode already uses). After **300ms** (default,
   `VOICEMODE_BARGE_IN_TRIGGER_MS`) of sustained speech *while TTS is
   audible*, playback halts within one audio buffer — reusing the exact
   mechanism already built for the manual Pause flag — and what you said
   becomes the start of your turn (no chime, no re-prompt).

## Live-trial instructions (the part that needs YOU)

The mechanism has been validated with **synthetic echo signals** (see
`tests/test_aec.py`, `tests/test_barge_in.py`) — that proves the algorithm
converges and the state machine triggers correctly. It has **not** been
tuned against a real microphone + real speaker/AirPods acoustic path, because
that genuinely can't be done from code alone. To trial it:

1. Turn natural mode on: `convomode-natural-mode.sh on`.
2. Start (or continue) a convomode voice session.
3. Say **"natural mode"** so the agent knows the toggle is intentional, then
   just start talking over it mid-sentence.
4. Notice what goes wrong, if anything:
   - **Agent interrupts itself constantly / cuts off on silence** → the echo
     canceller isn't fully cancelling its own voice, so a echo residual is
     tripping the VAD. Try raising `VOICEMODE_AEC_FILTER_MS` (more filter
     taps = more acoustic delay it can model) or lowering
     `VOICEMODE_BARGE_IN_VAD_AGGRESSIVENESS` (stricter = fewer false
     positives), in `~/.voicemode/voicemode.env`.
   - **Takes too long / doesn't respond to real interruptions** → lower
     `VOICEMODE_BARGE_IN_TRIGGER_MS` (default 300ms).
   - **Real speech gets partially swallowed right at the interruption
     point** → this is a known limitation of the NLMS approach during
     "double-talk" (both the echo and your real voice present at once —
     see the comment on `AEC_STEP_SIZE` in `voice_mode/config.py`); try
     lowering `VOICEMODE_AEC_STEP_SIZE` further (default 0.15) for less
     aggressive cancellation, at the cost of slower echo convergence.
   - **A consistent lag between when you'd expect the echo and when it
     actually shows up in the mic** (e.g. Bluetooth codec delay) → set
     `VOICEMODE_AEC_REF_DELAY_MS` to the measured round-trip delay.
5. Record what worked in this doc / a follow-up note so the defaults can be
   updated once they're proven on real hardware.

None of this has been claimed as "working" — only as built and mechanism-
tested. The live trial IS the acceptance test for whether it actually feels
natural on your AirPods.

## Tunables reference

All optional env vars (set in `~/.voicemode/voicemode.env`; defaults shown
are conservative and require a server restart / fresh `/mcp` reconnect to
pick up, since they're read at config-import time, unlike the mode toggle
itself):

| Var | Default | What it controls |
|---|---|---|
| `VOICEMODE_BARGE_IN_TRIGGER_MS` | `300` | How many ms of sustained post-cancellation speech, while TTS is playing, counts as a genuine barge-in |
| `VOICEMODE_BARGE_IN_VAD_AGGRESSIVENESS` | same as `VOICEMODE_VAD_AGGRESSIVENESS` (3) | webrtcvad aggressiveness for the barge-in listener specifically |
| `VOICEMODE_AEC_FILTER_MS` | `200` | Adaptive filter length — how much acoustic delay it can model |
| `VOICEMODE_AEC_REF_DELAY_MS` | `0` | Fixed offset between "sample sent to speaker" and "echo reaches mic" — set this if echo consistently leaks through |
| `VOICEMODE_AEC_STEP_SIZE` | `0.15` | NLMS adaptation rate — lower = slower to converge but preserves more of your real voice during double-talk |
| `VOICEMODE_NATURAL_MODE_FLAG_PATH` | `~/.voicemode/natural-mode.flag` | Override the toggle flag's location |

## Known Phase 1 limitations (by design, not bugs)

- **Echo cancellation is a pure-Python NLMS filter, not the native WebRTC
  AEC3 library.** The research doc named `webrtc-audio-processing` /
  `aec-audio-processing` as options, but both ship as source-only PyPI
  packages requiring a C++ toolchain to compile — a much larger, more
  fragile dependency than an in-repo numpy filter for a Phase 1 patch. Same
  algorithm family (linear adaptive filtering against a known reference),
  different maturity level.
- **No double-talk detector.** While both you and the echo are present at
  once (the barge-in detection window itself), the filter can partially
  attenuate your real voice along with the echo — a known NLMS limitation,
  mitigated but not eliminated by a conservative default step size.
- **Concurrent DJ background music + natural mode isn't specifically
  handled.** The echo canceller's reference buffer reflects whichever
  player wrote most recently; if DJ music and TTS overlap, the reference
  won't be perfectly clean.
- **FOSCC iOS app voice mode is a separate codebase** and does not get
  natural mode from this change — that's an explicit follow-on task, not
  done here.
- **Phase 2** (a full Pipecat-based rewrite with SmartTurnDetection) is
  reserved for if Phase 1, once tuned, still doesn't clear the "feels
  natural" bar.
