#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${MEDVLM_PYTHON:-python3}" "$script_dir/kiosk_terminal.py" "$@"
