# Evidence inventory and chronology

## Future research plans after independent review (no approved experiments)

The second Claude roadmap review produced a **REVISE BEFORE APPROVAL** recommendation. The [full original second review](../reviews/2026-11-24-claude-roadmap-review.md) has since been archived exactly, with [source SHA-256 and overlooked findings](../reviews/2026-11-25-CLAUDE-FULL-REPORT-INTEGRATION.md); [the earlier summary-derived disposition](../reviews/2026-11-25-CLAUDE-SECOND-REVIEW-DISPOSITION.md) is retained for chronology. The [new phased roadmap](../architecture/PHASED-ARCHITECTURE-RESEARCH-ROADMAP.md) identifies distinct hardware, evaluation and architecture decision gates. The [six-run BF16/INT8 instrument](proposals/C0-C2-INSTRUMENT-ONLY-2026-11-25.md) and [physical hardware/capability ladder preflights](proposals/HARDWARE-AND-CAPABILITY-PREFLIGHT-2026-11-25.md) are **proposals only**. The [original larger C0/C2 plan](proposals/C0-C2-DEPLOYED-BYTE-STUDY.md) is retained but superseded for scheduling. None is a completed run, approved DRP or capability claim.


Publication date is 24 November 2026, Adelaide. This is a permanent research archive. Earlier measurements and review records remain accessible, including superseded workflows and failures.

| Date and unit | Status | Record and exact evidence |
|---|---|---|
| Sep 2026 · DML-0001 | Partial historical run; metadata retains RUNNING, recorded logs stop at 520/3000. | [Historical record](DML-0001.md), [original tracked files](data/archive/DML-0001/). Exact source metadata commit confirmed remotely. Original checkpoint and prepared dataset bytes unavailable. |
| Nov 2026 · CPU audit / recurrence correctness study | Initial baseline/recurrent suites completed. Several corruption and provenance assumptions failed initially; regression audits followed. | [Audit chronology](CPU-AUDITS.md), [repros and mutation outputs](data/archive/cpu-audit/), [correction history](../findings/failures-and-corrections.md). Some early test stderr and local working-tree commits were never retained. |
| 23 Nov · RTX5080-BF16-SMOKE-001 A/B | Two completed local synthetic checks with reversed model order. | [Original smoke report](RTX5080-BF16-SMOKE-001.md), [observations](data/RTX5080-BF16-SMOKE-001.csv), [historical harness](../reproduce/rtx5080_bf16_smoke.py.txt). Exact implementation commits missing. No saved checkpoint. |
| 24 Nov · DML-0002–0004 | Completed 200-step V1 training, superseded for acceptance after a recovery discrepancy. | [B02](runs/DML-0002.md) · [B08](runs/DML-0003.md) · [R02×4](runs/DML-0004.md) · [V1 source logs and configs](data/archive/calibration-v1/). Only sampled validation was scored. |
| 24 Nov · RCV-DEFAULT and RCV-DET-MANUAL | Default mode failed the trajectory gate; manually configured deterministic run matched. | [Incident](../findings/CUDA-RESUME-INCIDENT.md) · [Default archive](data/archive/cuda-resume-v1/) · [Manual mode archive](data/archive/cuda-resume-deterministic-v1/). |
| 24 Nov · DML-0005–0007 | Deterministic 200-step calibration completed and validated. | [B02](runs/DML-0005.md) · [B08](runs/DML-0006.md) · [R02×4](runs/DML-0007.md) · [V2 source logs and configs](data/archive/calibration-v2/) · [Comparison](TINYSTORIES-CALIBRATION-ANALYSIS.md). |
| 24 Nov · RCV-DET-PINNED | Accepted same-stack deterministic recovery, exact final weight match. | [Recovery result](data/archive/cuda-resume-v2/verification.json) · [Checkpoint receipts](../provenance/artifact-hashes/2026-11-24-local.json). |
| 24 Nov · DML-0008–0010 | Completed 6,104 updates per arm. Publication validated existing checkpoints at matched budgets. | [B02](runs/DML-0008.md) · [B08](runs/DML-0009.md) · [R02×4](runs/DML-0010.md) · [Final campaign](TINYSTORIES-EXTENDED-2026-11-24.md). |

Count convention: **16 documented experimental units** = ten DML run IDs (one partial and nine completed), two synthetic smoke executions, three CUDA recovery controls, and one aggregated chronological CPU audit series. Calibration/campaign analysis tables consolidate runs and are not counted again. Corpus preparation is an engineering record. It adds no training-run ID.

## Preregistered experiments

The [independent architecture investigation](../reviews/2026-11-24-independent-architecture-research.md) and [verification notes](../reviews/2026-11-24-independent-research-verification.md) review the existing evidence. They add no experimental runs.

Future plans and outcomes will appear here under their unique DRP IDs. Protocol IDs are separate from run and artifact IDs. The [execution ledger](protocol/state.json) and [research protocol](../docs/EXPERIMENT_PROTOCOL.md) govern new work. The historical rows above retain their original evidence and statuses.

The [phased architecture roadmap](../architecture/PHASED-ARCHITECTURE-RESEARCH-ROADMAP.md) and [first C0/C2 study](proposals/C0-C2-DEPLOYED-BYTE-STUDY.md) are unapproved drafts. See the [qualification decisions](../methodology/27B-4GIB-QUALIFICATION-SPECIFICATION.md) for criteria awaiting human review.

## What was inspected and verified before publication

- Implementation source, configs and tests at [d914684](https://github.com/CL-BAF/Domovoi-Labs/tree/d914684dc730cd32132aeff389d6291c447ed7e4), including its architecture explanation, independent audit and prior remediation narratives. Diffs to [f5fac7a](https://github.com/CL-BAF/Domovoi-Labs/compare/f5fac7a34a211adf93e0dc8ea576c8d973ef87ac...d914684dc730cd32132aeff389d6291c447ed7e4) distinguish deterministic policy, trajectory checks and per-file checksum changes. Prior research branch and [draft PR](https://github.com/CL-BAF/Domovoi-research/pull/2) remain intact.
- Existing DML summaries, per-attempt metrics, sampled validation, GPU telemetry, environment manifests, portable configs, checkpoint indexes, raw command logs and full-validation outputs. Privacy substitutions apply only to exported copies. Underlying local files are not rewritten.
- SHA-256 digests and byte sizes for hundreds of retained artifacts, including 84 checkpoint directories. A stored internal hash was checked wherever present. Older V1 digests establish publication-time identity without an intrinsic previous checksum. [Bulk receipt](../provenance/artifact-hashes/2026-11-24-local.json)
- NAS revision and SHA evidence was read from the original verification; the share was unavailable during publication. Actual packed bytes/tokenizers, sizes and manifests were compared with first-published counts and hashes. [Data evidence](../datasets/preparation-and-tokenization.md)
- Parameter totals, budget, calibration/extended medians, NLL gaps, VRAM units and bootstrap token counts were independently recalculated from archived observations. [Verification script](../reproduce/verify_publication.py)

## What is unavailable or still unpublished

No qualifying inference benchmark exists for 8K, 4 GiB and ≥8 tokens/s. CUDA smoke output verifies positive synchronized measurements and accumulation of peak statistics but has no retained publishable TTFT or generation rate. The extended records report training peaks and ordinary document validation with one seed; no held-out test or broader benchmark scores were created. A clean original DML-0001 checkpoint, several historic terminal transcripts, exact early uncommitted source snapshots and kernel traces are absent. Raw corpora, binary packs, tokenizers, staging databases and model/optimizer checkpoint weights are retained privately or at publisher origins. Readers need operator access or must independently reconstruct matching inputs. The archive never substitutes a hash or later snapshot for missing evidence.
