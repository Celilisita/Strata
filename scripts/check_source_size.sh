#!/usr/bin/env bash
set -eu
python -X utf8 "$(dirname "$0")/check_source_size.py" "$@"
