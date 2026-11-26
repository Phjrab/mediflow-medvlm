#!/usr/bin/env python3
"""Run the explicit unattended workflow from a checkout."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mediflow_medvlm.kiosk.autorun import main

if __name__ == "__main__":
    raise SystemExit(main())
