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

## 2026-07-13 live trial: what happened, and what we now know (2026-07-14 update)

The first live trial (2026-07-13) failed silently: William tried it twice on
his AirPods, playback never got cut, and it was turned off. **There was no
usable evidence of why** — `barge_in.py`'s log lines only reached stderr
(`config.setup_logging` only adds a file handler when `VOICEMODE_DEBUG=true`,
which it wasn't that day), and no event types existed to record whether the
listener even armed. That gap is now closed (see `voice_mode/barge_in.py`'s
`log_barge_in_armed/unavailable/triggered/disarmed` events in
`~/.voicemode/logs/events/` + the opt-in per-frame trace below) — the *next*
trial, pass or fail, will leave a real record.

**A real-hardware acoustic test harness** (`scripts/barge_in_acoustic_test.py`)
was then built to stop guessing: it drives the ACTUAL production code
(`NonBlockingAudioPlayer` + `BargeInListener`) against this Mac's real
built-in speakers/mic (not a mock), playing a real TTS clip and, for
double-talk trials, a second real speech clip through a separate `afplay`
process so it mixes acoustically through real air into the real mic — the
same physical phenomenon barge-in has to survive.

**The finding, on built-in mic/speakers (not AirPods — see caveat below):**
the AEC provides only ~3-4dB of real echo cancellation (measured: mean
`rms_clean`/`rms_near` ratio ≈0.67 across an 11-second TTS-alone clip — for
reference, a usable AEC typically needs 15-30dB). As a direct result, **96.5%
of post-AEC frames during that clip were misclassified as "speech" by
webrtcvad at aggressiveness=3**, with a longest continuous false run of
**4.35 seconds**. A full sweep of every documented tunable failed to fix
this:

| Swept | Range | Effect on the false trigger |
|---|---|---|
| `AEC_STEP_SIZE` | 0.15 → 0.9 | none (trigger time identical ±0.04s across the whole range) |
| `AEC_REF_DELAY_MS` | 0 → 300ms | none (trigger time identical ±0.05s across the whole range) |
| TTS output buffer granularity | 2048 → 720 samples | none |
| `BARGE_IN_TRIGGER_MS` | 300 → 1800ms | delayed the self-interruption, never prevented it |

A fifth, new, **off-by-default** knob was added and tested —
`VOICEMODE_BARGE_IN_ENERGY_MARGIN` (an adaptive echo-floor gate, see
`voice_mode/config.py` for the mechanism) — and it also failed to find a
usable middle ground: `margin=3.0` eliminated the false positive across 7/7
repeated silence trials, but the SAME setting then missed 2 of 3 real
double-talk trials (and the one "hit" was itself a coincidental false
positive that fired before the injected interruption even started). Lower
margins reduced but didn't reliably eliminate the false positive. **The two
failure modes trade off against each other on this hardware — no single
value of any tunable, alone or combined, gets both "doesn't self-interrupt"
and "detects a real interruption."**

**What this does NOT prove:** this harness necessarily ran on the MacBook
Pro's built-in mic/speakers (two independent, physically-separated devices)
— not William's AirPods (one shared Bluetooth device serving both
directions, a fundamentally different and likely HARDER acoustic/codec
path per the concern already documented in `voice_mode/aec.py`). It is
possible AirPods behave differently (better OR worse); the instrumentation
above is what will tell us, the next time natural mode is turned on for a
real trial. But this result is strong evidence that Phase 1's linear
software AEC, as built, does not reach the cancellation quality barge-in
needs — on the friendlier of the two hardware paths. **The credible next
step is fixing the AEC's cancellation quality itself** (a real native AEC
library, or a nonlinear/learned echo suppressor) rather than continuing to
tune the existing knobs.

## Live-trial instructions (the part that needs YOU)

The mechanism has been validated with **synthetic echo signals** (see
`tests/test_aec.py`, `tests/test_barge_in.py`) — that proves the algorithm
converges and the state machine triggers correctly. Per the section above,
it has ALSO now been validated against real (built-in) hardware, and found
wanting — this section is preserved for when AirPods are actually trialed,
since that acoustic path is still unverified. To trial it:

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
5. **Whatever happens, this trial now leaves evidence** — check
   `~/.voicemode/logs/events/voicemode_events_<date>.jsonl` for
   `BARGE_IN_ARMED` / `BARGE_IN_UNAVAILABLE` / `BARGE_IN_TRIGGERED` /
   `BARGE_IN_DISARMED` entries (this answers "did it even try to listen" —
   the exact question 2026-07-13 couldn't answer). For frame-by-frame detail,
   set `VOICEMODE_BARGE_IN_TRACE=1` before the session starts; it writes one
   JSON line per processed mic frame (rms_near/rms_far/rms_clean/is_speech/
   speech_run_ms) to `~/.voicemode/logs/barge_in/trace_<date>.jsonl`. Record
   what worked in this doc / a follow-up note so the defaults can be updated
   once they're proven on real hardware.

None of this has been claimed as "working" — only as built and mechanism-
tested. The live trial IS the acceptance test for whether it actually feels
natural on your AirPods. **Per the 2026-07-14 update above, expect this to
likely self-interrupt** unless AirPods' acoustic path behaves meaningfully
better than the built-in mic/speakers already tested — treat a trial on
AirPods primarily as gathering AirPods-specific evidence, not as a
pass/fail usability test of the current build.

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
| `VOICEMODE_BARGE_IN_ENERGY_MARGIN` | `0` (disabled) | Adaptive echo-floor gate multiplier — added 2026-07-14, tested and found NOT to resolve the false-positive/false-negative trade-off on built-in hardware (see `voice_mode/config.py` docstring). Leave at 0 unless deliberately experimenting. |
| `VOICEMODE_BARGE_IN_TRACE` | unset (disabled) | Set to `1` to write a per-frame decision trace to `~/.voicemode/logs/barge_in/trace_<date>.jsonl` — the evidence trail added 2026-07-14 so a failed trial is diagnosable instead of a repeat of 2026-07-13's silence |
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
