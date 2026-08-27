#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ENV_NAME="${1:-toolweave-gemma-synthesis}"
PYTORCH_INDEX_URL="${TOOLWEAVE_PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu129}"

if [[ ! "$ENV_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "Invalid Conda environment name: $ENV_NAME" >&2
    exit 2
fi

CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
    echo "Conda is required. Set CONDA_EXE or add conda to PATH." >&2
    exit 2
fi
if "$CONDA_BIN" run --name "$ENV_NAME" python -c "pass" >/dev/null 2>&1; then
    echo "Conda environment already exists; refusing to modify it: $ENV_NAME" >&2
    exit 2
fi

"$CONDA_BIN" run --name base python "$SCRIPT_DIR/verify.py" --lock-only
"$CONDA_BIN" create --yes --name "$ENV_NAME" \
    --file "$SCRIPT_DIR/conda-linux-64.explicit"
"$CONDA_BIN" run --name "$ENV_NAME" python -m ensurepip --upgrade
"$CONDA_BIN" run --name "$ENV_NAME" python -m pip install --no-cache-dir \
    "pip==26.2.1" "setuptools==80.10.2"
"$CONDA_BIN" run --name "$ENV_NAME" python -m pip install --no-cache-dir \
    --no-deps \
    -r "$SCRIPT_DIR/requirements-pypi-linux-x86_64.lock"
"$CONDA_BIN" run --name "$ENV_NAME" python -m pip install --no-cache-dir \
    --no-deps \
    --index-url "$PYTORCH_INDEX_URL" \
    -r "$SCRIPT_DIR/requirements-pytorch-cu129-linux-x86_64.lock"
"$CONDA_BIN" run --name "$ENV_NAME" python -m pip check
"$CONDA_BIN" run --name "$ENV_NAME" python "$SCRIPT_DIR/verify.py"

SYNTH_PYTHON="$("$CONDA_BIN" run --name "$ENV_NAME" \
    python -c "import sys; print(sys.executable)")"
printf '\nEnvironment created and verified. Add this exact value to environment/env.local.sh:\n'
printf 'export TOOLWEAVE_SYNTH_PYTHON=%q\n' "$SYNTH_PYTHON"
