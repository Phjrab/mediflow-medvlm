#!/usr/bin/env python3
"""Run the foreground V7 client from its reviewed source archive."""
from pathlib import Path
import sys

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from mediflow_medvlm.campaign_terminal import main

if __name__=="__main__": raise SystemExit(main())
