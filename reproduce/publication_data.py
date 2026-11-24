"""Read captured experiment evidence and derive like-for-like measures."""

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "experiments/data/archive"
CAMPAIGNS = ("calibration-v1", "calibration-v2", "extended-v1")
ARMS = ("B02", "B08", "R02x4")
STEPS = (610, 1526, 3052, 6104)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def records(path):
    return [json.loads(row) for row in path.read_text(encoding="utf-8-sig").splitlines()
            if row.strip()]


def parameter_count(model):
    d, f, v, c, n = (model[k] for k in ("d_model", "ffn_dim", "vocab_size", "context_length",
                                        "n_layers"))
    return n * (4*d*d + 2*d*f + 9*d + f) + (v+c)*d + 2*d + (0 if model["tie_embeddings"] else v*d)


def load_campaign(campaign):
    path = ARCHIVE / campaign
    result = {}
    for run in read(path / "campaign.json"):
        folder = path / run["id"]
        metadata = read(folder / "metadata.json")
        summary = read(folder / "summary.json")
        train, val = (records(folder / f"{name}.jsonl") for name in ("metrics", "val_metrics"))
        env = metadata["env"]
        cfg = metadata["config"]["training"]
        tokens_per_update = cfg["batch_size"] * cfg["context_length"] * cfg["gradient_accumulation"]
        steps = summary["steps_completed"]
        assert run["exit_code"] == 0 and summary["status"] == metadata["status"] == "COMPLETED"
        assert not metadata["git_dirty"] and metadata["git_commit"] == env["git_commit"]
        assert summary["tokens_trained"] == train[-1]["tokens_seen"] == steps * tokens_per_update
        assert all(row["tokens_seen"] == row["step"] * tokens_per_update for row in train)
        assert all(math.isfinite(row[key]) for row in train for key in ("train_loss", "grad_norm"))
        assert all(math.isfinite(row["val_loss"]) for row in val)
        assert val[-1]["val_loss"] == summary["final_val_loss"]
        assert math.isclose(math.exp(summary["final_val_loss"]), summary["final_val_ppl"], rel_tol=2e-6)
        assert parameter_count(metadata["config"]["model"]) == summary["params_total"]
        rates = [row for row in train if row["step"] > 20]
        evals = {int(p.name.split("-")[2]): read(p)["tasks"]["val_nll"]
                 for p in sorted(folder.glob("evals-step-*-val_nll.json"))}
        if campaign != "calibration-v1":
            assert set(evals) == (set(STEPS) if campaign == "extended-v1" else {200})
            assert all(e["n"] == 4783349 and e["n_documents"] == 21990 for e in evals.values())
        result[run["arm"]] = {
            "id": run["id"], "meta": metadata, "summary": summary, "train": train, "val": val,
            "eval": evals, "tokens_per_update": tokens_per_update,
            "gpu": records(path / f"{folder.name}.gpu.jsonl"),
            "step_rate": statistics.median(row["step_tokens_per_s"] for row in rates),
            "step_ms": 1000 * statistics.median(row["mean_step_s"] for row in rates),
        }
    assert tuple(result) == ARMS
    return result


def differences(data, step):
    base, deep, rec = (data[name] for name in ARMS)
    _, b08, r = (a["eval"][step]["nll"] for a in (base, deep, rec))
    b02 = base["eval"][step]["nll"]
    deep_vram, rec_vram = (a["summary"]["vram"]["peak_allocated_mb"] for a in (deep, rec))
    memory_reduction = deep_vram - rec_vram
    return {
        "b08_minus_r02x4_peak_mib": memory_reduction,
        "reduction_pct_of_b08_peak": memory_reduction * 100 / deep_vram,
        "r02x4_throughput_increase_pct_over_b08": (rec["step_rate"] / deep["step_rate"] - 1) * 100,
        "r02x4_minus_b08_nll": r - b08,
        "r02x4_minus_b08_nll_pct_of_b08": (r / b08 - 1) * 100,
        "r02x4_minus_b02_nll": r - b02,
        "fraction_of_b08_gain_over_b02_recovered": (b02-r)/(b02-b08) if b02 > b08 else None,
    }
