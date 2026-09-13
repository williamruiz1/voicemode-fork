# Pinning the live voice-mode tool environment

## The incident (2026-09-13)

`~/.local/share/uv/tools/voice-mode/` is a `uv tool install --editable`
environment that backs several long-running voice servers. A builder ran:

```
uv tool install --editable "<worktree>[silero]" --with pyaec --force
```

to add one package. Because nothing constrained resolution, `--force`
re-resolved the ENTIRE dependency graph: ~85 of ~90 packages moved,
including `openai` 2.53 → 3.13 (a major version) and `onnxruntime` 1.28 →
1.30. The 5 already-running voice servers held half-old modules in memory
and started failing on symbols that had moved.

## What does NOT fix this (verified, not assumed)

Two mechanisms look like the obvious fix and do nothing for this exact
command. Both were tested by setting `cryptography` to an intentionally
wrong version and checking what actually got installed:

1. **`[tool.uv] constraint-dependencies` in the editable package's own
   `pyproject.toml`.** Setting `cryptography==40.0.2` there and running
   `uv tool install --editable ...` (with `--no-cache`, so no stale cache
   could hide the effect) installed `cryptography==50.0.1` anyway.
   `uv --verbose` resolver logs confirm it: the resolver only ever consults
   `idna>=2.8`-style requirement ranges from the dependency graph, never
   the constraint. Same result via plain `uv pip install -e ".[silero]"`
   into a throwaway venv — this is not `uv tool`-specific, it's that
   `uv pip install`/`uv tool install` (the pip-compatible installers) do
   not read `[tool.uv]` project settings from a local editable source at
   all. Those settings are read by `uv sync`/`uv lock`/`uv add` — i.e. the
   *contributor* `.venv` — not by installing a package as a *tool*.
2. **A committed `uv.lock`.** `uv tool install --help` has no `--locked`/
   `--frozen` flag and no mention of a lockfile at all; `uv lock` explicitly
   documents itself as updating "the project's lockfile" for `uv sync`/
   `uv run`. A lockfile is never consulted on this path.

Both are "fixes that look right and do nothing" — the exact failure shape
this doc exists to prevent.

## What DOES work (also verified)

`uv tool install` (and `uv pip install`) DO honour a constraints file passed
explicitly: `-c <file>` / `--constraints <file>`, or the `UV_CONSTRAINT` env
var. Proof, in an isolated `UV_TOOL_DIR`/`UV_TOOL_BIN_DIR` scratch location
(never the live env):

- `-c constraints.txt` with `cryptography==40.0.2` → installs `40.0.2`.
- `-c constraints.txt` with `idna==3.7` (a version too old to satisfy other
  packages' own requirements) → the resolver reports a hard, unsatisfiable
  conflict naming `idna==3.7` — proof it is a real constraint, not a hint.
- `-c constraints.txt` with `certifi==2024.2.2` (a version old but still
  compatible) → cleanly installs `2024.2.2` instead of today's `2026.7.22`.
- A full resolve with all 91 currently-installed versions pinned via `-c`
  reproduces the live environment exactly — zero-line diff against
  `uv pip list --python ~/.local/share/uv/tools/voice-mode/bin/python
  --format=freeze`.

## The mechanism this repo uses

- **`constraints-live.txt`** (repo root) — every non-`voice-mode` package
  from the live environment's freeze, pinned with `==`, regenerated only on
  a deliberate upgrade.
- **`scripts/install-live-tool-env.sh`** — the ONLY supported way to
  (re)install the live tool env. It resolves `constraints-live.txt`
  relative to itself and always passes it via `--constraints`, then runs
  `uv tool install --editable "<repo>[silero]" --with pyaec --force`.
- **`[tool.uv] constraint-dependencies` in `pyproject.toml`** is kept, but
  only protects `uv sync`/the contributor `.venv` — it is NOT the guard for
  the live tool env. Its comment says so explicitly so nobody re-derives
  the wrong mental model from reading it in isolation.

## The operating rule

- This environment backs LIVE voice servers. Never run
  `uv tool install --editable ... --force` by hand for this project — run
  `scripts/install-live-tool-env.sh` instead (it forwards extra flags, e.g.
  `--upgrade-package <name>` for a deliberate, scoped bump).
- After ANY reinstall — via the script or otherwise — treat every already-
  running voice server as **stale-until-reconnected**: it holds old modules
  in memory and may fail on the next import of something that moved.
  Restarting/reconnecting them is a separate, deliberate step, not implied
  by a successful install.
- Bumping a pin in `constraints-live.txt` is fine and expected over time —
  do it one line at a time, on purpose, and re-verify the live servers
  afterward. Deleting `constraints-live.txt` or bypassing the wrapper
  script to "let uv resolve freely" reopens this exact incident.
- The `silero` extra (`onnxruntime`, for model-based VAD) must stay defined
  in `pyproject.toml`'s `[project.optional-dependencies]` — the live install
  depends on it for noise handling; losing it silently degrades to the
  binary webrtcvad decision.
