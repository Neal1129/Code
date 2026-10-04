from pathlib import Path
import runpy

runpy.run_path(Path(__file__).parent / "src" / "esm2" / "esm2_masked_marginal_score.py", run_name="__main__")
