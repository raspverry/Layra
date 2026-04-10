#!/bin/bash
# ============================================================
# Layra — Claude Code on the web SessionStart hook
# ============================================================
# Prepares the remote container so `make ci` works out of the box:
#   - Creates .venv with Python 3.12
#   - Installs Layra in editable mode with [dev] extras
#   - Persists venv activation to $CLAUDE_ENV_FILE so all subsequent
#     shells in this session see the venv on PATH
#
# Idempotent: safe to re-run. Local (non-remote) sessions exit immediately.
# ============================================================

set -euo pipefail

# Only run in Claude Code on the web. Local Mac dev sessions skip this.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
    exit 0
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="${LAYRA_PYTHON:-python3.12}"

log() { printf '[layra-hook] %s\n' "$*"; }

# ------------------------------------------------------------
# 1. Python 3.12 사용 가능 확인
# ------------------------------------------------------------
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    log "ERROR: $PYTHON_BIN not found in PATH"
    log "Set LAYRA_PYTHON env var to the correct interpreter."
    exit 1
fi
log "Using interpreter: $($PYTHON_BIN --version)"

# ------------------------------------------------------------
# 2. Prefer uv for speed; fall back to pip
# ------------------------------------------------------------
if command -v uv >/dev/null 2>&1; then
    INSTALLER="uv"
    log "Using uv ($(uv --version | head -n1))"
else
    INSTALLER="pip"
    log "uv not found, falling back to pip"
fi

# ------------------------------------------------------------
# 3. Create / reuse virtualenv
# ------------------------------------------------------------
if [ ! -x "$VENV_DIR/bin/python" ]; then
    log "Creating virtualenv at $VENV_DIR"
    if [ "$INSTALLER" = "uv" ]; then
        uv venv --python "$PYTHON_BIN" "$VENV_DIR"
    else
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
else
    log "Reusing existing virtualenv at $VENV_DIR"
fi

# ------------------------------------------------------------
# 4. Persist venv activation for the session
# ------------------------------------------------------------
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
    {
        echo "export VIRTUAL_ENV=\"$VENV_DIR\""
        echo "export PATH=\"$VENV_DIR/bin:\$PATH\""
        echo "export PYTHONPATH=\"$PROJECT_DIR:\${PYTHONPATH:-}\""
    } >> "$CLAUDE_ENV_FILE"
    log "Exported VIRTUAL_ENV / PATH / PYTHONPATH to CLAUDE_ENV_FILE"
fi

# ------------------------------------------------------------
# 5. Install Layra in editable mode with dev extras
# ------------------------------------------------------------
# MLX is gated by platform marker (Darwin/arm64) in pyproject.toml,
# so it is skipped automatically on Linux remote containers.
# Heavy deps (torch, diffusers, onnxruntime) are installed so that
# `make test`, `make lint`, and `make typecheck` all work.
log "Installing Layra with [dev] extras..."
if [ "$INSTALLER" = "uv" ]; then
    uv pip install \
        --python "$VENV_DIR/bin/python" \
        --editable ".[dev]"
else
    "$VENV_DIR/bin/python" -m pip install --upgrade pip --quiet
    "$VENV_DIR/bin/python" -m pip install --editable ".[dev]" --quiet
fi

# ------------------------------------------------------------
# 6. Sanity check
# ------------------------------------------------------------
log "Sanity-checking core imports..."
"$VENV_DIR/bin/python" - <<'PY'
import importlib
missing = []
for mod in ["numpy", "PIL", "psd_tools", "pydantic", "loguru", "typer", "pytest", "ruff", "mypy"]:
    try:
        importlib.import_module(mod)
    except Exception as e:
        missing.append(f"{mod}: {e}")
if missing:
    print("MISSING:", *missing, sep="\n  ")
    raise SystemExit(1)
print("core imports OK")
PY

log "✅ Layra session environment ready"
log "Try:  make ci    (lint + typecheck + test-fast)"
