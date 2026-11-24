# Publication coverage checklist

## Documentation and evidence

| Document or folder | Coverage | Status and explicit limit |
|---|---|---|
| [README](README.md) and [archive chronology](experiments/ARCHIVE-INDEX.md) | Executive context, inventory, updates and provenance scope | Complete for available records. Earlier milestones retained. |
| [Mathematics](architecture/mathematical-foundations.md) and [recurrence](architecture/recurrence-and-gradient-flow.md) | Tensor shapes, operators, weights, gradients, cycle/sequence, alternatives, source/tests | Implemented architecture described. Proposed alternatives flagged. |
| [Memory and inference](architecture/memory-complexity.md) | Decoding/cache/dtypes, logical FLOPs and bytes, target workload | Derived versus measured values labeled. No 8K or 4 GiB qualifying run. |
| [Data inventory](datasets/NAS-INVENTORY-2026-11-24.md), [prep contract](datasets/preparation-and-tokenization.md) and [dataset data](datasets/data/) | Raw inventory, revision/licence, counts, staging, splits, tokenizer, prepared binaries and contamination | Original raw audit complete. Fresh raw rehash blocked by disconnected SMB; copied manifest and actual local packed files verified. Approximate contamination remains open. |
| [Training and recovery](methodology/training-and-recovery.md) | Seed, accumulation, LR, AdamW, status, frozen run limits, integrity and recovery | Complete for production implementation; hardware/OS limitation explicit. |
| [Control](methodology/experimental-controls.md), [measurement](methodology/measurement-methodology.md), [evaluation](methodology/evaluation-methodology.md) | Matched budgets, NVML vs allocator, exact NLL and method | Matched full validation done; single seed and test reserved. |
| [DML-0001](experiments/DML-0001.md), [CPU audits](experiments/CPU-AUDITS.md), [synthetic RTX](experiments/RTX5080-BF16-SMOKE-001.md) | Original baseline, test chronology and smoke observations | Retained source/output links; documented historical gaps. |
| [Calibration](experiments/TINYSTORIES-CALIBRATION-ANALYSIS.md), [extended campaign](experiments/TINYSTORIES-EXTENDED-2026-11-24.md), [individual runs](experiments/runs/) | Every new run's config, status, measurements, evidence and limitations | Nine completed real-data IDs preserved including superseded V1. Extended final rather than in-progress. |
| [Campaign data](experiments/data/archive/) and [figures](experiments/figures/) | Train/val/sessions, telemetry, config, validation, summary, checkpoint metadata, console and audit logs, three plots | Raw timestamps/values maintained. Paths sanitized. Unavailable stderr/benchmark material is called out. |
| [CUDA incident](findings/CUDA-RESUME-INCIDENT.md), [failure supplement](findings/failures-and-corrections.md), [hypotheses](findings/architecture-hypotheses.md) | Negative results, exact/nondeterministic controls, corrections, remaining tests | Prior [ledger](findings/ledger.md) preserved. Hypotheses limited to measured scope. |
| [Reproduction package](reproduce/README.md), [provenance](provenance/) | Source pins, portable data/commands/configs, environments, artifact digests, fail-safe recovery, numerical/plot validators | Large/raw artifacts excluded and retrieval requirements stated. |

## Checks

- Checked exact V1/V2/extended source pins and the historical implementation commit remotely. Preserved both repository histories with additive research commits.
- Recomputed counts, throughput medians, matched training token budgets, loss/resource effects and hash agreements from original local evidence. Scored twelve saved extended checkpoints at four common budgets without starting any new training.
- Verified published link targets, structured inputs, copied metadata hashes, Python syntax, figures and common private-path/token exclusions. Fresh CPU suite returned 275 passes and one CUDA-profile skip, with the scheduler-test monkeypatch warning. Historical CUDA smoke and deterministic same-device recovery remain available without reinterpretation.
- Open research and documentation dependencies include a restored raw NAS recheck, independently matched implementation provenance for the original smoke if ever available, replicate seeds, decontamination checks, external benchmarks, and qualified 8K inference.
