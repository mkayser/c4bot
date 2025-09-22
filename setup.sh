#!/usr/bin/env bash
set -euxo pipefail
PYVER="${PYVER:-3.12}"

# Create/activate venv
if [ ! -d ".venv" ]; then
  "python${PYVER}" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -U pip wheel setuptools
python -m pip install pip-tools

# Compile and install locked deps
python -m piptools compile -o requirements.txt requirements.in
python -m pip install -r requirements.txt
