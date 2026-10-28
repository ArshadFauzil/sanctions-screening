"""Project-root-anchored paths, so scripts work from any working directory
(PyCharm runs a script from its own folder by default)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"            # sdn.xml, sdn_advanced.xml (not committed)
BENCHMARK_DIR = ROOT / "benchmark"  # generated queries, candidates, reports
