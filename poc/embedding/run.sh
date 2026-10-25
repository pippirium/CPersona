#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
exec .venv-poc/bin/python poc/embedding/evaluate.py "$@"
