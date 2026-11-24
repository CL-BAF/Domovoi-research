import json
import math
import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
calibration = root / "calibration-v2"
results = json.loads((calibration / "campaign.json").read_text(encoding="utf-8"))
verification = json.loads((root / "cuda-resume-v2/verification.json").read_text(encoding="utf-8"))
assert len(results) == 3 and all(record["exit_code"] == 0 for record in results)
assert verification["weights_and_adamw_restore_exact"] and verification["max_final_weight_delta"] == 0.0
seconds = {}
for record in results:
    run = calibration / record["id"]
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "COMPLETED" and summary["steps_completed"] == 200
    assert (run / "evals-step-00000200-val_nll.json").is_file()
    seconds[record["arm"]] = 10 * math.ceil(summary["wall_clock_s"] * 6104 / 200 / 10)
print("Projected wall seconds per arm:", seconds, flush=True)
script = Path("scripts/real_data_campaign.py")
sys.argv = [str(script), "--data", str(root / "packed/tinystories-v1"), "--root",
            str(root / "extended-v1"), "--steps", "6104", "--seconds", "1800",
            "--first-id", "8", "--seed", "1234", "--calibrated-seconds", json.dumps(seconds)]
runpy.run_path(str(script), run_name="__main__")
