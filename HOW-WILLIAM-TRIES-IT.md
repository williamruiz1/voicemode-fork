# How William Tries It — Endpointing & Barge-In Fix

This branch fixes two things in convomode voice:

1. **The mic hanging open** after you stop talking (endpointing fix).
2. **Not being able to talk over the assistant** to interrupt it (barge-in
   fix — an upgrade to the existing "natural mode" feature).

Both are **off by default** and independently toggleable. Neither has been
tried live yet — it passed automated tests only. **This live try is the
real acceptance test.** Nothing below claims it works; it only tells you
how to turn it on and what to look for.

Full technical reference (every knob, defaults, mechanism):
`docs/guides/endpointing-and-bargein.md`.

---

## Step 0 — one-time setup: point convomode at this branch

Your live convomode currently runs an **installed** copy of voice-mode (a
`uv tool`), not this checked-out branch — so the fix won't be active until
you install this branch's code in its place. This one command does that:

```bash
uv tool install --force --editable "/Users/williamruiz/code/worktrees/voicemode-endpointing-bargein[silero]"
```

What this does, in plain terms:
- Installs this branch as your `voice-mode` command, **in editable mode**
  (further edits to the worktree take effect without reinstalling).
- Pulls in `onnxruntime` — the one extra runtime piece the Silero model
  needs — via the `[silero]` bit at the end. Without it, both features
  silently stay off and everything behaves exactly like today.
- `--force` overwrites your current install. `uv` does **not** back this up
  for you automatically — but the exact file your current install points
  to has already been captured below in **Rollback**, so you can restore it
  exactly.

This has been verified end-to-end in an isolated test install (not your
live one): the command succeeds, the `voice-mode` command still works
afterward, and the Silero model loads correctly.

After running it, **reconnect the voicemode MCP** so the new server binary
gets used — in Claude Code, that's disconnecting and reconnecting the
`voicemode` MCP server (a manual step in the Claude Code UI; there's no
command-line equivalent). If you're not sure it picked up the change, the
simplest confirmation is starting a fresh convomode session after
reconnecting.

---

## Try 1 — the mic-stops-hanging fix (endpointing)

**Turn it on** — add this line to `~/.voicemode/voicemode.env` (the file
convomode reads its settings from):

```
VOICEMODE_ENDPOINTING=true
```

Then reconnect the voicemode MCP (same step as above — this setting is read
once at server start).

**Try it:** start a convomode session, talk normally, then stop and wait.

**What good looks like:** the mic closes roughly 700ms after you actually
stop talking — even with background noise (traffic, room noise) present —
instead of staying open indefinitely. It should feel like the assistant
picks up your turn promptly without you having to go dead-silent for
several seconds first.

**If it's not right:** see the "Env knobs" table in
`docs/guides/endpointing-and-bargein.md` for the tuning knobs
(`VOICEMODE_MIN_ENDPOINT_MS`, `VOICEMODE_SILERO_SPEECH_THRESHOLD`,
`VOICEMODE_MIN_SPEECH_MS`).

---

## Try 2 — also try talking over the assistant (barge-in)

This is a separate toggle, layered on top of Try 1 — try it after
endpointing feels right, or independently, whichever you prefer.

**Turn it on** — use the existing toggle script (not a manual file edit):

```bash
~/.local/bin/founder-os/convomode-natural-mode.sh on
```

No MCP reconnect needed for this one — it's checked live at the start of
each turn.

**Try it:** start (or continue) a convomode session, say "natural mode" so
the assistant knows the toggle is intentional, then just start talking over
it mid-sentence while it's speaking.

**What good looks like:** playback cuts off within about 300ms of you
starting to talk, and what you said becomes the start of your next turn —
no waiting for it to finish, no chime, no re-prompt.

**Known history worth knowing before you try:** this exact feature was
live-trialed once before (2026-07-13/07-14) on built-in Mac speakers/mic
and found to self-interrupt on its own voice/noise — documented in
`docs/guides/natural-mode.md`. This branch swaps the detection model
(Silero, the same one behind Try 1) in hopes of fixing that, but that swap
itself has **not** been live-trialed. Your AirPods are also a different,
untested acoustic path (one shared Bluetooth device instead of two
separate ones) — so this try is gathering real evidence, not confirming a
known-working feature.

**Turn it back off any time:**
```bash
~/.local/bin/founder-os/convomode-natural-mode.sh off
```

---

## What good looks like — summary

| Feature | On | Working |
|---|---|---|
| Endpointing | `VOICEMODE_ENDPOINTING=true` in `~/.voicemode/voicemode.env` | Mic closes ~1s after you stop talking, even with background noise |
| Barge-in | `convomode-natural-mode.sh on` | You can interrupt the assistant mid-sentence by talking; it stops within ~300ms |

**Neither is claimed to work live yet.** Both passed automated tests only.
This try is the real test — whatever you find (works great / self-interrupts
/ doesn't close the mic / anything else), that result is the actual signal,
not anything written here.

---

## Rollback — back to today's exact behavior

Two independent layers to roll back, depending on what you changed:

### Just the settings (keep this branch installed)

```bash
# turn off endpointing — either remove the line from ~/.voicemode/voicemode.env
# or set it to false:
VOICEMODE_ENDPOINTING=false

# turn off barge-in / natural mode:
~/.local/bin/founder-os/convomode-natural-mode.sh off
```

With `VOICEMODE_ENDPOINTING` unset or false, the endpointing code path is
never reached — the recording loop runs the exact same `webrtcvad` +
silence-timer logic as before this branch existed, byte-for-byte. Barge-in
with the flag file absent is the pre-existing default (turn mode only,
mic closed during playback) — also unchanged. Reconnect the voicemode MCP
after changing `VOICEMODE_ENDPOINTING` for it to take effect (barge-in's
flag toggle doesn't need a reconnect).

### The install itself (go back to the released tool)

Your live install, before Step 0's command runs, is:

```
version:  8.6.1
source:   /Users/williamruiz/.local/share/voicemode-wheels/voice_mode-8.6.1-py3-none-any.whl
```

(confirmed by reading `~/.local/share/uv/tools/voice-mode/uv-receipt.toml`
before this branch touched anything — this is NOT a PyPI install, it's a
locally-built wheel, so restoring it means pointing back at that exact
file, not reinstalling from PyPI).

To restore it exactly:

```bash
uv tool install --force /Users/williamruiz/.local/share/voicemode-wheels/voice_mode-8.6.1-py3-none-any.whl
```

Then reconnect the voicemode MCP. This returns your install to the identical
wheel that was live before any of this branch's changes — none of this
branch's code remains active.

If that exact wheel file is somehow gone by the time you need it, two
backup copies of the same 8.6.1 wheel already exist alongside it in
`~/.local/share/voicemode-wheels/` (`.bak-20260722`, `.bak-20260804-conch-grace`)
— any of the three restores the same pre-branch behavior.
