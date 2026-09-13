#!/usr/bin/env bash
# Canonical installer for the LIVE voice-mode tool environment
# (~/.local/share/uv/tools/voice-mode/), which backs long-running voice
# servers. See docs/dev/live-environment-pinning.md for the incident and the
# discriminating tests that led to this script.
#
# DO NOT run `uv tool install --editable ... --force` by hand for this
# project. That command re-resolves the ENTIRE dependency graph from
# scratch every time, and pins committed to pyproject.toml or a uv.lock are
# NOT honoured by `uv tool install` (verified empirically — see the doc).
# The only thing uv actually honours here is `--constraints`/`-c` on the
# command line (or UV_CONSTRAINT), which is what this script supplies from
# the committed constraints-live.txt.
#
# Usage:
#   scripts/install-live-tool-env.sh                  # editable install from this checkout, pinned
#   scripts/install-live-tool-env.sh --upgrade-package foo   # deliberate, scoped upgrade (still pinned elsewhere)
#
# After ANY run of this script, running voice servers are stale-until-
# reconnected: they hold already-imported modules from the OLD environment
# in memory. Treat them as needing a restart/reconnect check, per the
# operating rule in docs/dev/live-environment-pinning.md — this script does
# not restart or signal any process itself.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONSTRAINTS="$REPO_ROOT/constraints-live.txt"

if [[ ! -f "$CONSTRAINTS" ]]; then
  echo "error: $CONSTRAINTS not found — refusing to install unpinned." >&2
  exit 1
fi

echo "Installing voice-mode tool env, pinned via: $CONSTRAINTS"
exec uv tool install --editable "${REPO_ROOT}[silero]" --with pyaec \
  --constraints "$CONSTRAINTS" \
  --force \
  "$@"
