from pathlib import Path
import sys

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src" / "abm"))

from merged_abm_desktop_app import main

main()
