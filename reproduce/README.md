# Reproduction package

Structured logs, configs, plots and digests are public. For private retained model, optimizer, packed corpora and tokenizer files, ask the project operator or independently reconstruct using pinned publisher inputs. A hash establishes byte identity without making the underlying artifact retrievable. Saved optimizer/RNG files use trusted-only deserialization.

The legacy executable examples below record prior invocations. Any new training, replay, evaluation or measurement needs a committed plan and a maintained launcher. Static validation and plots of stored metrics remain ordinary archive checks. [Experiment protocol](../docs/EXPERIMENT_PROTOCOL.md)

## Source and environment

| Record | Implementation and evidence |
|---|---|
| DML-0001 | [898ef3d](https://github.com/CL-BAF/Domovoi-Labs/commit/898ef3d0c48a6f4d4fd1a0addc41d7a3b220d377) from historical metadata. The original packed data, tokenizer and checkpoint are unavailable. |
| Independent audits | [5e76fc7](https://github.com/CL-BAF/Domovoi-Labs/commit/5e76fc78bbe9a5c12690e84d7c29e558248e6feb) pinned for the original baseline. Early recurrence rounds lack an exact per-round commit. |
| RTX synthetic smoke and repeat | [Byte-preserved historical harness](rtx5080_bf16_smoke.py.txt) and [repeat fingerprint](../provenance/RTX5080-BF16-SMOKE-001-source.json). Neither execution has an exact source commit. |
| Calibration V1 and default/manual recovery | [f5fac7a](https://github.com/CL-BAF/Domovoi-Labs/commit/f5fac7a34a211adf93e0dc8ea576c8d973ef87ac), DML-0002–0004. |
| Calibration V2, accepted CUDA recovery and extended campaign | [d914684](https://github.com/CL-BAF/Domovoi-Labs/commit/d914684dc730cd32132aeff389d6291c447ed7e4), DML-0005–0010. |

Real-data executions used Windows 11 build 26200, Python 3.13.17 x64, RTX 5080 driver 621.01, PyTorch 2.15.0+cu132, CUDA runtime 13.2 and reported cuDNN 92600. The host CPU was documented in the synthetic smoke. [V1 environment](../experiments/data/archive/calibration-v1/environment.json) · [V2 environment](../experiments/data/archive/calibration-v2/environment.json) · [V2 pip freeze](../experiments/data/archive/calibration-v2/pip-freeze.txt)

From a matching implementation checkout, create a separate CUDA environment. Align dependency versions with the observed freeze; install the pinned checkout in place of the redacted package path.

```powershell
python -m venv .venv-cuda
$CudaPython = (Resolve-Path '.venv-cuda/Scripts/python.exe').Path
& $CudaPython -m pip install 'torch==2.15.0+cu132' --index-url https://download.pytorch.org/whl/cu132
& $CudaPython -m pip install '.[corpora,dev]'
& $CudaPython -m pip check
nvidia-smi
```

## Hydrate and prepare

Set `$NasRoot` to the pinned raw dataset tree or mounted read-only source, `$RealDataRoot` to a fresh local data root, `$ResearchRoot` to this clone, and `$ImplementationRoot` to the pinned implementation. Run from that clean implementation checkout with Git on PATH.

```powershell
& $CudaPython scripts/audit_corpora.py --root $NasRoot --report "$RealDataRoot/new-inventory.json"
& $CudaPython "$ResearchRoot/reproduce/render_paths.py" "$ResearchRoot/datasets/data/raw-specifications/tinystories.json" "$RealDataRoot/tinystories.json" --nas-root $NasRoot --realdata-root $RealDataRoot --implementation-root $ImplementationRoot
& $CudaPython scripts/prepare_streaming.py --spec "$RealDataRoot/tinystories.json" --out "$RealDataRoot/packed/tinystories-v1"
```

Use the corresponding [source specifications](../datasets/data/raw-specifications/) for WikiText and FineWeb preview. Compare [packed manifests](../datasets/data/published-packing/) and [artifact hashes](../provenance/artifact-hashes/2026-11-24-local.json) before training. If another tokenizer or preprocessing environment produces different bytes, record and resolve the difference before treating it as the same corpus. Keep official splits and benchmark inputs apart. Rerun the same source specification and target path to resume staging before publication of a ready directory. [Detailed data contract](../datasets/preparation-and-tokenization.md)

## Recreate runs in fresh directories

The following documents commands for an independent replay. The publication executed saved-checkpoint validation and CPU checks. It did not start this training sequence.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
& $CudaPython scripts/real_data_campaign.py --data "$RealDataRoot/packed/tinystories-v1" --root "$RealDataRoot/calibration-v2" --steps 200 --seconds 900 --first-id 5
& $CudaPython scripts/verify_real_resume.py --config "$RealDataRoot/calibration-v2/DML-0007-R02x4.yaml" --out "$RealDataRoot/new-cuda-resume"
```

For default-CUDA V1 checkout f5fac7a, remove the workspace pin, write to a new `calibration-v1` and use `--first-id 2`. Deterministic V2 and extended use d914684, which explicitly enables the deterministic policy and verifies checkpoint hashes. The manual deterministic wrapper selected CUBLAS workspace and deterministic PyTorch before running the V1 harness. [Standalone incident](../findings/CUDA-RESUME-INCIDENT.md)

The exact extended invocation is in the [portable launch script](../experiments/data/archive/extended-v1/launch-extended-v1.py), with the copied calibration and verification prerequisites. It calls `scripts/real_data_campaign.py --steps 6104 --seconds 1800 --first-id 8 --seed 1234` on the packed TinyStories output and supplies the per-arm projected seconds as JSON `{"B02":180,"B08":480,"R02x4":410}`. The projections were calibration Trainer wall indexed by 6104/200 and rounded upward to the next 10 seconds. Use the original configs in [calibration-v1](../experiments/data/archive/calibration-v1/), [calibration-v2](../experiments/data/archive/calibration-v2/) and [extended-v1](../experiments/data/archive/extended-v1/). The archive includes original child command strings with private roots substituted. Monotonic deadline numbers are process-clock references and must be recomputed for a new run.

For publication validation on trusted copies of saved checkpoints:

```powershell
foreach ($step in 610,1526,3052,6104) {
  foreach ($id in 'DML-0008','DML-0009','DML-0010') {
    & $CudaPython scripts/evaluate.py --experiment "$RealDataRoot/extended-v1/$id" --checkpoint-step $step --tasks val_nll
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $id at $step" }
  }
}
```

Use a new restored working directory to avoid overwriting an existing evaluation result. Evaluate the accepted calibration with the same command at step 200, IDs DML-0005–0007. Do not use held-out test scores to tune. To repeat the independent test environment, run `python -m pytest -q -p no:cacheprovider` on CPU and `pytest tests/test_pilot.py -k cuda_profile` separately on CUDA.

## Monitor and recover safely

Tail the active arm's console log and GPU JSONL. Request a clean stop by creating `<campaign>/STOP` from another process. Wait for the supervisor to exit. Verify status, writer lock and checkpoint hashes before a same-config resume. Restore to the same data bytes, architecture, runtime and scheduler horizon. A new timeout should be recorded explicitly.

```powershell
$deadline = & $CudaPython -c 'import time; print(time.monotonic() + 1800)'
& $CudaPython scripts/train.py --config $OriginalArmYaml --resume --experiment-dir $InterruptedRun --pilot-deadline $deadline --stop-file $FreshStopPath
```

Direct arm resume retains trainer limits, with no campaign-level NVIDIA/disk watcher. Monitor those separately for that mode. Reject a bad SHA-256, missing/stale index or active writer condition and inspect before recovery. Hard-kill stale-lock cleanup requires confirming the old writer is gone. [Checkpoint state and recovery](../methodology/training-and-recovery.md)

## Output schemas and verification

| Source | Keys and purpose |
|---|---|
| `campaign.json`, `environment.json`, arm YAML | Arm, ID, sanitized argv, source/driver/packages, pin and exit status. |
| `metrics.jsonl`, `val_metrics.jsonl`, `sessions.jsonl` | Step/attempt, target totals, LR, window train loss/gradient, validation/PPL, rate and duration. Reconcile attempt suffixes. |
| `*.gpu.jsonl` | NVIDIA timestamp, temperature, utilization, power and device-wide memory. |
| `summary.json`, `checkpoint.json` | Train wall duration, status, reason/deviation, bounds, peak counters, optimizer/graph/input provenance, per-file checksums. |
| `evals-step-*-val_nll.json` | Validation NLL and document bootstrap, dtype/context, exact document/token counts and input hashes. |
| [Bulk artifact manifest](../provenance/artifact-hashes/2026-11-24-local.json) | Original sizes/SHA-256 and access category. Sanitized checkpoint metadata has a separate published checksum. |

From this research repository run `python reproduce/verify_publication.py` to verify links, numerical derivations, manifest consistency, published digests and privacy exclusions. Run `python reproduce/plot_results.py` with matplotlib 3.11.2 to regenerate the SVGs. The archival script `python reproduce/collect_local_evidence.py --data <root> --implementation <checkout> --nas-root <raw-dir> --research <checkout>` hashes locally retained evidence and writes path-sanitized records. Inspect staged changes before publishing. The archive preserves different measurement definitions rather than substituting results from one protocol for another.
