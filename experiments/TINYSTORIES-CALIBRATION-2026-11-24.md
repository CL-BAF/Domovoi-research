# TinyStories RTX 5080 calibration · 24 November 2026

**Accepted for infrastructure calibration.** Three models trained for 200 optimizer updates on the same prepared TinyStories corpus. This stage checks finite learning, measurements and recovery. The run is one paired seed with a short LR schedule and a narrow synthetic-story corpus. It gives no capability ranking beyond the observed calibration and no usable 8K or 4 GiB inference result.

## Research question and protocol

The efficiency hypothesis was that R02×4 would use less persistent parameter and AdamW memory than B08 at eight applied blocks, while B02 would provide a shallower, equal-stored-parameter control. All arms use width 640, ten heads, MLP width 1728, a tied 4096-token head, 512 learned positions and zero dropout. B02 is two stored and two applied blocks. B08 is eight stored and eight applied. R02×4 is two stored blocks with cycle schedule `0,1,0,1,0,1,0,1`.

Each update uses two 16×512 microbatches, or 16,384 scored tokens. Each arm consumed 3,276,800 tokens using a shared no-repeat stream with training seed 1234 and sampler seed 1235. AdamW uses LR 0.0006, betas (0.9, 0.95), matrix weight decay 0.1, norm clipping 1.0, 10 warmup steps and cosine decay to 10%. Validation ran every 50 steps on 25 reproducible fixed sampled batches. Checkpoints were retained at steps 50, 100, 150 and 200. After training, FP32 full-validation scoring used all 4,783,349 eligible within-document targets. The reserved test split was not scored.

TinyStories v1 uses [the audited manifest](../datasets/NAS-INVENTORY-2026-11-24.md). Tokenizer SHA-256 is `a0e552000a2acdeefddb1f2757f587e5b75e787585eb4d34cdb89f8c62dd144a`. Packed train SHA-256 is `a8bd40a12efa94aa48a48c8ddf5c67113df570791538cc0c41d76993c7dffa71` and validation SHA-256 is `7a5f9b4757b91742e57fad0820ac06cd2b6acfbcfd77c47382fd6c31282f373f`. Checkpoint metadata binds these inputs and the optimizer protocol on resume.

Environment was Windows 11 build 26200, Python 3.13.17, RTX 5080, driver 621.01, CUDA 13.2, PyTorch 2.15.0+cu132 and cuDNN reported as 92600. BF16 autocast used FP32 model and AdamW state. The accepted run uses [implementation commit `d914684dc730cd32132aeff389d6291c447ed7e4`](https://github.com/CL-BAF/Domovoi-Labs/commit/d914684dc730cd32132aeff389d6291c447ed7e4) and deterministic CUDA with `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Packed data and checkpoints stayed on local NVMe.

## Evidence and correction

The first calibration (DML-0002 through DML-0004) completed on [`f5fac7a34a211adf93e0dc8ea576c8d973ef87ac`](https://github.com/CL-BAF/Domovoi-Labs/commit/f5fac7a34a211adf93e0dc8ea576c8d973ef87ac). A separate R02×4 recovery check restored the checkpoint weights and AdamW state exactly but diverged from an uninterrupted run at a max absolute final parameter delta of **0.001584119**. Even the pre-interruption losses differed slightly between executions. This is direct evidence of nonrepeatability with the first CUDA settings. It does not identify the particular kernel responsible.

Enabling deterministic CUDA eliminated the observed discrepancy in the bounded control. The second calibration reran every arm without overwriting the first. Its R02×4 interruption at step 2 restored weights, AdamW tensors and the disjoint sampler's position 64, then completed at position 128. Reconciled steps were `[1, 2, 3, 4]` and final parameter delta was **0.0**. Checkpoint checksums reject later file alterations. Final checkpoints were loaded for full-validation scoring. Tests reported **275 CPU passes, one CUDA skip in the CPU environment**, and the separate profiling smoke passed on CUDA. The same short recovery test does not establish reproducibility across different drivers or devices.

## Deterministic calibration measurements

| Run | Unique parameters | Block applications | Train loss at 200 | Sampled val loss / PPL | Full val NLL | Synced tok/s | ms/update | Peak allocated / reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DML-0005 · B02 | 10,665,856 | 2 | 3.7074 | 3.7275 / 41.58 | 3.7147 | 773,561 | 21.18 | 911.57 / 1110 |
| DML-0006 · B08 | 33,812,224 | 8 | 3.8333 | 3.8492 / 46.95 | 3.8418 | 260,022 | 63.01 | 2257.50 / 2450 |
| DML-0007 · R02x4 | 10,665,856 | 8 | 3.8700 | 3.8870 / 48.77 | 3.8873 | 285,290 | 57.43 | 1851.17 / 2050 |

| Arm | Logged pre-clip gradient norm range | Sampled max GPU utilization % | Sampled max temp °C | Final optimizer MiB | Final checkpoint MiB |
|---|---:|---:|---:|---:|---:|
| B02 | 0.5788–1.9951 | 90 | 39.34 | 81.40 | 122.11 |
| B08 | 0.5411–18.315 | 94 | 44.22 | 258.05 | 387.06 |
| R02x4 | 0.6457–4.0631 | 95 | 46.33 | 81.40 | 122.11 |

Throughput is the median of synchronized 10-update windows after excluding updates 1–20, with each update timing synchronizing before and after its work. It includes batch retrieval and gradient accumulation and excludes validation, checkpointing and log I/O. End-to-end wall times, complete metric rows, validation bootstrap intervals, host-level GPU samples and checkpoint file sizes/hashes are in [results-v1](data/TINYSTORIES-CALIBRATION-v1.json) and [results-v2](data/TINYSTORIES-CALIBRATION-v2.json). NVIDIA telemetry was sampled roughly every two seconds and its maximum is a sampled observation. PyTorch allocator peaks describe training and validation in the process. The displayed device memory includes other workloads.

### Interpretation

R02×4 repeated the earlier 406.33 MiB allocated-memory advantage over B08, approximately 18%. Its measured median step throughput was about 9.7% higher. Its full validation NLL was 0.0455 worse than B08 and 0.1726 worse than B02. B02 had the best early validation NLL and the highest rate. This is evidence against claiming an early quality advantage for recurrence under this schedule. The tied arm has fewer trainable tensors to update, and the checkpoint size difference is consistent with that mechanism. The run does not isolate optimizer time or activation residency. Initialization, optimization and the narrow dataset can change the loss ranking as training proceeds. Longer controlled training and multiple paired seeds are the next tests of whether the observed trade-off persists.

## Commands and extended run

Run from the pinned Domovoi Labs checkout with Git on PATH and the separate CUDA environment. The inventory report gives the source revisions and file names. Rebase the portable manifest paths onto the local raw tree before preparation.

```powershell
python scripts/audit_corpora.py --root <raw-datasets> --report <new-inventory.json>
python scripts/prepare_streaming.py --spec <tinystories-spec.json> --out <packed-version>
python scripts/real_data_campaign.py --data <packed-version> --root <new-calibration-dir> --steps 200 --seconds 900 --first-id 5
python scripts/verify_real_resume.py --config <calibration-dir>/DML-0007-R02x4.yaml --out <new-verification-dir>
python scripts/evaluate.py --experiment <calibration-dir>/DML-0005 --tasks val_nll
python scripts/evaluate.py --experiment <calibration-dir>/DML-0006 --tasks val_nll
python scripts/evaluate.py --experiment <calibration-dir>/DML-0007 --tasks val_nll
```

An exploratory, single-seed extended TinyStories comparison started after calibration and recovery passed. It reserves DML-0008–0010, uses up to 6,104 steps per arm (100,007,936 scored tokens), has a 30-minute campaign wall limit, verifies checkpoints and records synchronized metrics. The initial per-arm projections were 180, 480 and 410 seconds from the calibration timings. GPU telemetry, free disk monitoring, stop criteria and the operator stop file are active. An extended result will be a separate record. This launch does not freeze multi-seed or 8K research decision thresholds.
