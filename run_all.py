"""Run the full reproduction pipeline: verification, scenarios, refinement, figures."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for script in ("run_verification.py", "run_scenarios.py", "run_refinement.py", "make_figures.py"):
    print(f"\n=== {script} ===", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True)
