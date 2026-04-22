#!/usr/bin/env bash
# Creates a Python virtual environment and installs dependencies.
#
# Options:
#   VENV_DIR=/some/path bash setup.sh     # default: ./.venv
#   PYVER=3.11 bash setup.sh              # pin python version (default: 3.10)
set -euo pipefail

PYVER=${PYVER:-3.10}
HERE=$(cd "$(dirname "$0")" && pwd)
VENV=${VENV_DIR:-"$HERE/.venv"}

PY=""
for cand in "python${PYVER}" python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        PY=$(command -v "$cand"); break
    fi
done
[[ -z "$PY" ]] && { echo "no python 3 found on PATH" >&2; exit 1; }

echo "python: $PY ($($PY --version 2>&1))"
echo "venv:   $VENV"

[[ -d "$VENV" ]] || "$PY" -m venv "$VENV"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip wheel setuptools >/dev/null
pip install -r "$HERE/requirements.txt"

python -c "import torch, numpy, sklearn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo
echo "activate:   source $VENV/bin/activate"
echo "next:       python scripts/run_dataset.py --dataset Beauty --backbone BSARec --seed 42"
