"""Entry point: python hub/run_hub.py (configuration through QVDE_* environment variables)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "hub")]

from qvde_hub.service import main  # noqa: E402

main()
