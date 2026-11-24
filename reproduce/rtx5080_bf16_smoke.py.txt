"""Run from the Domovoi Labs implementation checkout. Writes only temporary synthetic data."""

import argparse
import gc
import sys
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from tokenizers import Tokenizer, models

sys.path.insert(0, str(Path.cwd()))

from domovoi.config import clone_config, load_config
from domovoi.datasets.tokenizer import sha256_file
from domovoi.training import Trainer
from scripts.plan_recurrent_experiments import build_arms, PILOT_ARMS

parser = argparse.ArgumentParser(description="Reproduce the bounded RTX 5080 synthetic smoke.")
parser.add_argument("--reverse", action="store_true", help="reverse the observed arm order")
args = parser.parse_args()
if not torch.cuda.is_available():
    parser.error("CUDA PyTorch is required")
device = torch.device("cuda")
print("GPU", torch.cuda.get_device_name(device), "capability", torch.cuda.get_device_capability(device),
      "torch", torch.__version__, "CUDA", torch.version.cuda, "bf16", torch.cuda.is_bf16_supported(), flush=True)
torch.set_num_threads(1)

def run_all(root):
    tok_path = root / "synthetic-tokenizer.json"
    vocab = {f"token{i}": i for i in range(4096)}
    Tokenizer(models.BPE(vocab, [])).save(str(tok_path))
    generator = np.random.default_rng(1234)
    counts = {"train": 410_400, "val": 16_416}
    files = {}
    for split, count in counts.items():
        path = root / (split + ".bin")
        generator.integers(0, 4096, size=count, dtype=np.uint16).tofile(path)
        files[split + "_bin"] = str(path)
    (root / "meta.json").write_text(json.dumps({
        "complete": True, "dataset": {"name": "synthetic-gpu-smoke", "version": "local"},
        "tokenizer": {"sha256": sha256_file(tok_path)}, "tokens": counts, "files": files,
    }), encoding="utf-8")
    base = clone_config(load_config("configs/baseline_small.yaml"),
        tokenizer={"path": str(tok_path)},
        dataset={"name": "synthetic-gpu-smoke", "processed_dir": str(root)},
        training={"steps": 24, "warmup_steps": 4, "gradient_accumulation": 2,
                  "sampler": "disjoint_segments", "pilot_stop_checks": True,
                  "eval_interval": 24, "eval_batches": 1, "log_interval": 8})
    results = {}
    arms = list(build_arms(base, only=PILOT_ARMS))
    if args.reverse:
        arms.reverse()
    for name, cfg in arms:
        trainer = Trainer(cfg, device=device)
        summary = trainer.train()
        assert summary["status"] == "COMPLETED", summary
        assert summary["tokens_trained"] == 24 * 16384
        results[name] = {k: summary[k] for k in ("tokens_trained", "avg_tok_per_s", "wall_clock_s", "vram")}
        print(name, json.dumps(results[name]), flush=True)
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
    print("RESULTS", json.dumps(results), flush=True)

with TemporaryDirectory(prefix="domovoi-5080-smoke-") as temp:
    run_all(Path(temp))
