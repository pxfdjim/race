#!/bin/sh
set -eu
exec python3 /app/agent/main.py "$@"
