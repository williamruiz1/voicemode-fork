#!/usr/bin/env python3
"""Real-hardware acoustic test harness for the natural-mode barge-in listener.

Drives the ACTUAL production code path -- voice_mode.audio_player's real
NonBlockingAudioPlayer + voice_mode.barge_in's real BargeInListener, against
whatever the system's REAL default input/output devices are (queried via
sounddevice, not mocked) -- to answer the question the July-13 live trial
left unanswered: does the barge-in mechanism actually fire on real speech
during real playback, over a real speaker->room->mic acoustic path?

Two trial types:
  --trial silence   : play the TTS clip ALONE. A trigger here is a FALSE
                       POSITIVE -- the agent interrupting itself off its own
                       echo (the self-interruption failure mode named in
                       docs/guides/natural-mode.md).
  --trial interrupt : play the TTS clip, then partway through (--interrupt-at)
                       play a SECOND, different-voice speech clip through the
                       SAME output device via a separate `afplay` process (so
                       it is NOT written into audio_player's reference ring
                       buffer -- only NonBlockingAudioPlayer.play() writes
                       there -- it mixes with the TTS acoustically, exactly
                       like a real interruption). A trigger here, and its
                       latency from interrupt-onset, is the real signal.

One trial = one subprocess. AEC/VAD tunables are read from environment
variables ONCE at voice_mode.config import time (exactly like the real MCP
server does on a restart per docs/guides/natural-mode.md), so sweeping
different knob combinations means a fresh interpreter per combination -- this
script sets os.environ BEFORE importing anything under voice_mode.

Prints one JSON result line to stdout. Caller (a sweep driver) accumulates
these across trials to compute trigger-latency and false-positive stats.
"""
import argparse
import json
import os
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", choices=["silence", "interrupt"], required=True)
    ap.add_argument("--tts-wav", required=True)
    ap.add_argument("--interrupt-wav")
    ap.add_argument("--interrupt-at", type=float, default=2.5,
                     help="seconds into TTS playback to start the interruption clip")
    ap.add_argument("--trigger-ms", type=int, default=300)
    ap.add_argument("--vad-aggr", type=int, default=None)
    ap.add_argument("--aec-filter-ms", type=int, default=200)
    ap.add_argument("--aec-ref-delay-ms", type=int, default=0)
    ap.add_argument("--aec-step-size", type=float, default=0.15)
    ap.add_argument("--energy-margin", type=float, default=0.0,
                     help="VOICEMODE_BARGE_IN_ENERGY_MARGIN -- 0 disables the adaptive "
                          "echo-floor gate (pre-2026-07-14 behavior); >1.0 enables it")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--tts-buffer-size", type=int, default=2048,
                     help="NonBlockingAudioPlayer buffer_size in samples (default 2048 == "
                          "production default, ~85ms @ 24kHz -- coarser than the barge-in "
                          "listener's 30ms mic chunk; smaller values tighten the reference "
                          "window's time resolution to test the granularity-mismatch hypothesis)")
    args = ap.parse_args()

    if args.trial == "interrupt" and not args.interrupt_wav:
        print("ERROR: --trial interrupt requires --interrupt-wav", file=sys.stderr)
        sys.exit(2)

    # Must be set BEFORE `import voice_mode.config` (module-level constants
    # are computed once at import time -- same load-bearing constraint the
    # real server has).
    os.environ["VOICEMODE_BARGE_IN_TRIGGER_MS"] = str(args.trigger_ms)
    if args.vad_aggr is not None:
        os.environ["VOICEMODE_BARGE_IN_VAD_AGGRESSIVENESS"] = str(args.vad_aggr)
    os.environ["VOICEMODE_AEC_FILTER_MS"] = str(args.aec_filter_ms)
    os.environ["VOICEMODE_AEC_REF_DELAY_MS"] = str(args.aec_ref_delay_ms)
    os.environ["VOICEMODE_AEC_STEP_SIZE"] = str(args.aec_step_size)
    os.environ["VOICEMODE_BARGE_IN_ENERGY_MARGIN"] = str(args.energy_margin)
    if args.trace:
        os.environ["VOICEMODE_BARGE_IN_TRACE"] = "1"

    import numpy as np
    import scipy.io.wavfile as wavfile

    from voice_mode import audio_player
    from voice_mode.barge_in import BargeInListener
    from voice_mode.config import SAMPLE_RATE
    from voice_mode.utils.event_logger import initialize_event_logger

    initialize_event_logger()  # so BARGE_IN_* lifecycle events actually land

    sr, tts_data = wavfile.read(args.tts_wav)
    if sr != SAMPLE_RATE:
        print(f"ERROR: tts wav sample rate {sr} != config.SAMPLE_RATE {SAMPLE_RATE}", file=sys.stderr)
        sys.exit(2)
    tts_float = (tts_data.astype(np.float32) / 32768.0)

    listener = BargeInListener()
    player = audio_player.NonBlockingAudioPlayer(buffer_size=args.tts_buffer_size)

    t0 = time.monotonic()
    listener.start()
    player.play(tts_float, sample_rate=SAMPLE_RATE, blocking=False)

    interrupt_proc = None
    interrupt_started_at = None
    triggered_at = None

    max_wait = (len(tts_float) / SAMPLE_RATE) + 3.0
    while True:
        elapsed = time.monotonic() - t0

        if (args.trial == "interrupt" and interrupt_proc is None
                and elapsed >= args.interrupt_at):
            interrupt_proc = subprocess.Popen(
                ["afplay", args.interrupt_wav],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            interrupt_started_at = elapsed

        if audio_player.barge_in_triggered() and triggered_at is None:
            triggered_at = elapsed

        if not audio_player.is_tts_speaking():
            break
        if elapsed > max_wait:
            break
        time.sleep(0.02)

    result = listener.stop()

    if interrupt_proc is not None:
        try:
            interrupt_proc.wait(timeout=5)
        except Exception:
            interrupt_proc.kill()

    latency = None
    if triggered_at is not None and interrupt_started_at is not None:
        latency = triggered_at - interrupt_started_at

    out = {
        "trial": args.trial,
        "triggered": bool(result.triggered),
        "error": result.error,
        "trigger_ms": args.trigger_ms,
        "vad_aggr": args.vad_aggr,
        "aec_filter_ms": args.aec_filter_ms,
        "aec_ref_delay_ms": args.aec_ref_delay_ms,
        "aec_step_size": args.aec_step_size,
        "energy_margin": args.energy_margin,
        "interrupt_started_at_s": interrupt_started_at,
        "triggered_at_s": triggered_at,
        "trigger_latency_s": latency,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
