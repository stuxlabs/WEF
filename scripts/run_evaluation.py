#!/usr/bin/env python3
"""
Convenience script to run WiFi LLM agent evaluation
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.main import main

if __name__ == "__main__":
    main()
