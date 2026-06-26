import sys
from pathlib import Path


EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "satellite_usv"
sys.path.insert(0, str(EXAMPLE_PATH))
