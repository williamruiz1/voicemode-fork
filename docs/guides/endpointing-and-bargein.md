# Endpointing & Barge-In (Silero VAD)

Two related, independently-toggleable fixes, both built on the same shared
model: `voice_mode/silero_vad.py`, a vendored Silero VAD v5 ONNX model that
outputs a continuous `P(speech)` in `[0, 1]` per audio frame — unlike
`webrtcvad`'s binary speech/no-speech decision, which has no energy
awareness and reads steady noise or a trailing breath as "speech."

| | What it fixes | Status |
|---|---|---|
| **Endpointing** | The mic hanging open after you stop talking | Off by default, opt in with one env var |
| **Barge-in** ("natural mode") | Talking over the assistant to interrupt it | Off by default, opt in with a flag file (pre-existing toggle) |

Both are **inert by default** — with neither turned on, the code path is
byte-for-byte the current `webrtcvad` behavior. Turning either on requires
the optional `onnxruntime` runtime dependency to actually be installed
(the `silero` extra); if it isn't, the code silently falls back to the old
`webrtcvad` path and logs nothing broken — see
[SILERO_AVAILABLE](#how-the-fallback-works) below.

For **exact copy-paste steps to try this live**, see
[`HOW-WILLIAM-TRIES-IT.md`](../../HOW-WILLIAM-TRIES-IT.md) at the repo root.
This doc is the reference for what each knob does.

---

## 1. Endpointing — fixes the mic-hang

**Root cause it fixes:** `webrtcvad`'s speech/no-speech decision has no
energy floor. Steady road/room noise or a trailing breath after you stop
talking reads as "speech," so the silence timer that's supposed to end your
turn never accumulates — the mic stays open indefinitely.

**The fix:** `voice_mode/silero_vad.py`'s `Endpointer` — a small state
machine driven by Silero's continuous probability instead of a binary
decision. Noise/breath scores low even at high energy, so thresholding the
*probability* (not the raw signal) tells real speech apart from it.

It's wired into the recording loop in `voice_mode/tools/converse.py`. When
enabled, `Endpointer` replaces the existing `webrtcvad` + silence-timer
state machine one-for-one — same recording loop, same downstream code, only
the "has the person stopped talking" decision changes.

### Env knobs (all read at import time — a fresh server start / MCP reconnect is needed to pick up a change)

| Variable | Default | What it controls |
|---|---|---|
| `VOICEMODE_ENDPOINTING` | unset (off) | Master switch. Set to `true`/`1`/`yes`/`on` to enable. Endpointing only actually activates when this **AND** `SILERO_AVAILABLE` are both true — see [the fallback](#how-the-fallback-works) below. |
| `VOICEMODE_MIN_ENDPOINT_MS` | `700` | Consecutive milliseconds of sub-threshold (non-speech) probability required before the turn is declared over. Lower = ends your turn faster after you stop talking; higher = more tolerant of mid-sentence pauses. |
| `VOICEMODE_SILERO_SPEECH_THRESHOLD` | `0.5` | The `P(speech)` cutoff, 0–1. A frame counts as speech once its probability is `>=` this. Raise it if background noise is still triggering false "speech started" events; lower it if quiet/soft speech isn't being picked up. |
| `VOICEMODE_MIN_SPEECH_MS` | `200` | Consecutive milliseconds of above-threshold probability required before speech is considered to have genuinely *started* (debounces a single noisy blip). |

Source of truth for these defaults: `voice_mode/config.py`,
`ENDPOINTING_ENABLED` / `ENDPOINTING_MIN_ENDPOINT_MS` /
`ENDPOINTING_SPEECH_THRESHOLD` / `ENDPOINTING_MIN_SPEECH_MS`.

### Felt behavior

With `VOICEMODE_ENDPOINTING=true` and Silero available, the mic should close
roughly `VOICEMODE_MIN_ENDPOINT_MS` (700ms by default) after you actually
stop talking, even with steady background noise present — instead of
hanging open indefinitely the way the old webrtcvad-only path could.

---

## 2. Barge-in — talk over the assistant ("natural mode")

**What it does:** lets you interrupt the assistant mid-sentence by talking
over it, instead of waiting for it to finish. This is the pre-existing
"natural mode" feature (see `docs/guides/natural-mode.md` for the full
mechanism, live-trial history, and known Phase-1 limitations); this branch's
change is narrow — it swaps the *decision* barge-in uses to detect your
voice from `webrtcvad` + an energy-margin heuristic over to the same Silero
model endpointing uses.

**Root cause it fixes:** the previous decision path
(`webrtcvad.is_speech()` gated by `VOICEMODE_BARGE_IN_ENERGY_MARGIN`) is a
binary decision with no amplitude-invariant probability — see
`docs/guides/natural-mode.md`'s 2026-07-14 trial log for the documented
false-positive/false-negative trade-off that heuristic couldn't resolve on
real hardware. Silero's `P(speech)` doesn't have that failure mode.

**Enable it:** barge-in is armed by a **flag file**, not an env var — it's
a runtime session choice, not deployment config. Presence of the file =
natural mode ON.

```bash
~/.local/bin/founder-os/convomode-natural-mode.sh on       # turn it on
~/.local/bin/founder-os/convomode-natural-mode.sh off      # back to turn mode
~/.local/bin/founder-os/convomode-natural-mode.sh status   # "on" or "off"
```

This is the existing, already-installed toggle script — it reads/writes the
same flag path `voice_mode/config.py` reads (`NATURAL_MODE_FLAG_PATH`,
override via `VOICEMODE_NATURAL_MODE_FLAG_PATH`, default
`~/.voicemode/natural-mode.flag`). No server restart needed to flip the
toggle itself — `barge_in.py` checks the flag file directly at the top of
each turn.

### The new knob this branch adds

| Variable | Default | What it controls |
|---|---|---|
| `VOICEMODE_BARGE_IN_SILERO_THRESHOLD` | `0.5` | The `P(speech)` cutoff, 0–1, used by the barge-in listener's Silero decision (separate from `VOICEMODE_SILERO_SPEECH_THRESHOLD`, which is endpointing-only). |

Source: `voice_mode/config.py`, `BARGE_IN_SILERO_THRESHOLD`.

### How the fallback works

`voice_mode/barge_in.py` checks `voice_mode.silero_vad.SILERO_AVAILABLE` at
listener-construction time:

- **Silero available** → uses `VOICEMODE_BARGE_IN_SILERO_THRESHOLD` against
  the model's continuous probability. `VOICEMODE_BARGE_IN_ENERGY_MARGIN`
  (the old heuristic) is **not consulted** on this path.
- **Silero unavailable** (onnxruntime not installed, or the vendored model
  failed to load) → falls back entirely to the pre-existing `webrtcvad` +
  `VOICEMODE_BARGE_IN_ENERGY_MARGIN` path, unchanged.

`SILERO_AVAILABLE` is a single process-wide flag computed once at import
(`voice_mode/silero_vad.py`): `True` only if `onnxruntime` imports **and**
the vendored ONNX model (`voice_mode/resources/models/silero_vad.onnx`)
loads successfully. It never raises — a missing dependency or model just
degrades this flag to `False`, so an install without the `silero` extra
keeps working exactly as it does today.

### Unchanged knobs (still apply on both the Silero and fallback paths)

These are the pre-existing barge-in/AEC tunables, not new — see
`docs/guides/natural-mode.md`'s "Tunables reference" for the full table and
the live-trial history behind each one:

- `VOICEMODE_BARGE_IN_TRIGGER_MS` (default `300`) — sustained speech
  duration, while TTS is playing, before playback is cut.
- `VOICEMODE_AEC_FILTER_MS`, `VOICEMODE_AEC_REF_DELAY_MS`,
  `VOICEMODE_AEC_STEP_SIZE` — the software echo canceller's tunables. These
  still run **before** the VAD decision on both paths (Silero's job is only
  to classify the *post-AEC* signal, not to replace echo cancellation
  itself).
- `VOICEMODE_BARGE_IN_ENERGY_MARGIN` — now fallback-path-only (see above);
  only consulted when Silero is unavailable.

### Felt behavior

Talking over the assistant while it's speaking should cut playback within
about `VOICEMODE_BARGE_IN_TRIGGER_MS` (300ms default) of sustained speech,
and what you said becomes the start of your next turn. As documented in
`docs/guides/natural-mode.md`, the previous (webrtcvad + energy-margin)
decision path was tested on real hardware and found to trade off false
self-interruption against missed real interruptions — the Silero swap is
intended to fix that trade-off, **but has not itself been live-trialed
yet**. See the honesty note below.

---

## What has and hasn't been validated

- The Silero model, `Endpointer` state machine, and barge-in Silero
  decision path all have **passing automated tests** (unit tests against
  the real ONNX model, plus the existing synthetic-signal AEC/VAD test
  suite).
- **Neither feature has been validated live** — on William's actual
  hardware (AirPods), in a real convomode session. Automated tests prove
  the mechanism; they do not prove it feels right or reliably fixes the
  hang/self-interruption in practice.
- Do not treat this document, or any status message from this branch, as a
  claim that either feature "works" outside of automated testing. The live
  try described in `HOW-WILLIAM-TRIES-IT.md` is the actual acceptance test.
