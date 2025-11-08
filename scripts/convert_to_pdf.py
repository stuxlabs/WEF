#!/usr/bin/env python3
"""
Convert all PNG figures to PDF format
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.pdf_converter import convert_all_to_pdf

if __name__ == "__main__":
    convert_all_to_pdf()
