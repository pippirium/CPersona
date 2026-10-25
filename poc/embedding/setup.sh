#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
uv venv --python 3.12 --allow-existing .venv-poc
uv pip install --python .venv-poc/bin/python -r poc/embedding/requirements.lock
uv pip install --python .venv-poc/bin/python --no-deps -e .
