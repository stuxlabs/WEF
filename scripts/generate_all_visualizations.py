#!/usr/bin/env python3
"""
Generate all paper visualizations
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("Generating all visualizations...")

print("\n1. Radar charts...")
import subprocess
subprocess.run([sys.executable, "src/visualization/generate_radar_charts.py"])

print("\n2. Graph variants...")
subprocess.run([sys.executable, "src/visualization/generate_all_graph_variants.py"])

print("\n3. Paper figures...")
subprocess.run([sys.executable, "src/visualization/generate_paper_figures.py"])

print("\nAll visualizations generated!")
