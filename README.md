# Domovoi Research

This repository records the architecture, methods, experiments and failures of Domovoi. The implementation lives in [Domovoi Labs](https://github.com/CL-BAF/Domovoi-Labs).

The research question is whether shared computation and memory-efficient model design can improve useful language-model capability within a hard 4 GiB GPU inference budget. The qualifying workload requires batch 1, usable 8K context, no CPU or disk offload and at least 8 output tokens per second on the reference GPU. The RTX 5080 is the development and training GPU.

## First milestone · 23 November 2026

| Record | Scope |
|---|---|
| [Architecture and systems](architecture/shared-core-transformer.md) | Computational graph, training, checkpointing, cache and complexity |
| [DML-0001 historical baseline](experiments/DML-0001.md) | Partial tinyshakespeare run, source commit and artifact checks |
| [RTX 5080 BF16 synthetic smoke](experiments/RTX5080-BF16-SMOKE-001.md) | Two bounded checks of training execution and memory with reversed arm order |
| [Failure and correction ledger](findings/ledger.md) | Reproduced bugs, changes and outstanding limitations |

The recurrent graph passed bounded CPU correctness checks. The 5080 profiling smoke passed, and the three 512-context configurations completed synthetic training steps. These measurements establish execution and resource use for the tested workloads. Capability, usable 8K context and total 4 GiB inference fit remain open.

## Real-data milestone · 24 November 2026

- [NAS inventory and packed corpus versions](datasets/NAS-INVENTORY-2026-11-24.md)
- [TinyStories RTX 5080 calibration and CUDA resume](experiments/TINYSTORIES-CALIBRATION-2026-11-24.md)

## Complete publication archive · 24 November 2026

**Result.** The three extended TinyStories arms completed 100,007,936 targets each. Full validation NLL was **1.5181 for B08**, **1.6612 for R02×4**, and **1.7185 for B02**. At equal applied depth R02×4 saved **406.33 MiB (18.00%)** of B08's peak allocated training memory, with lower validation quality than B08 and higher quality than B02 at this final budget. B02 led at the short calibration budget. All quality statements are scoped to one paired seed and this validation set. The held-out test and 4 GiB inference objective remain open.

| Area | Publication |
|---|---|
| Evidence availability, chronology and experiment IDs | [Archive index](experiments/ARCHIVE-INDEX.md) · [CPU audits](experiments/CPU-AUDITS.md) |
| Architecture | [Mathematics](architecture/mathematical-foundations.md) · [Recurrence and gradients](architecture/recurrence-and-gradient-flow.md) · [Memory and 8K qualification](architecture/memory-complexity.md) |
| Data | [Initial NAS inventory](datasets/NAS-INVENTORY-2026-11-24.md) · [Preparation and tokenizer contract](datasets/preparation-and-tokenization.md) |
| Analysis | [200-step calibration](experiments/TINYSTORIES-CALIBRATION-ANALYSIS.md) · [Final extended run](experiments/TINYSTORIES-EXTENDED-2026-11-24.md) |
| Failures and hypotheses | [Original ledger](findings/ledger.md) · [Full correction history](findings/failures-and-corrections.md) · [CUDA incident](findings/CUDA-RESUME-INCIDENT.md) · [Hypotheses](findings/architecture-hypotheses.md) |
| Methods | [Controls](methodology/experimental-controls.md) · [Training and recovery](methodology/training-and-recovery.md) · [Measurement](methodology/measurement-methodology.md) · [Evaluation](methodology/evaluation-methodology.md) |
| Reproduction and public review | [Reproduction package](reproduce/README.md) · [Artifact receipts](provenance/artifact-hashes/2026-11-24-local.json) · [Coverage checklist](PUBLICATION-CHECKLIST.md) |

The earlier milestone documents retain their original statuses and measurements. The completed extended record supersedes its initial in-progress status only. No earlier calibration or failed recovery was removed.

## Research execution protocol

Future experiments require a pushed, approved plan and a pushed analysis of the previous outcome. The implementation launcher verifies the live ledger and immutable commits before work begins. The ledger starts empty because all completed work is preserved in the historical archive. [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md) · [Execution ledger](experiments/protocol/state.json) · [Contributing](CONTRIBUTING.md)

The earlier architecture proposal compares selective state, token-adaptive computation, learned memory consolidation and procedural representations under the same 4 GiB and usable 8K objective. [Original architecture proposal](architecture/NEXT-ARCHITECTURE-PROPOSAL.md)

## Independent architecture investigation

Claude's complete independent report is archived with its original evidence labels and conclusions. Codex verified the visible references and checked material calculations in a separate record. [Independent report](reviews/2026-11-24-independent-architecture-research.md) · [Verification and annotations](reviews/2026-11-24-independent-research-verification.md)

## Reconciled architecture roadmap · draft

The proposed order is a strong native scalar low-bit control (C0), a bounded test of native learned codes (C2), then factored memory and reasoning (C1) if measured constraints justify it. Existing TinyStories experiments remain single-seed baselines. Factual density, natural-data transfer, and physical 4 GiB throughput are open. S1–S7, the reference arrangement and all experiment plans await human approval.

| Review document | Purpose |
|---|---|
| [Reconciliation](architecture/RESEARCH-RECONCILIATION-2026-11-24.md) | Shows where the investigations agree and why their priorities differ |
| [Efficiency evidence matrix](architecture/AI-EFFICIENCY-EVIDENCE-MATRIX.md) · [Candidate assessments](architecture/ARCHITECTURE-CANDIDATE-ASSESSMENT.md) | Compares measured prior art, costs, mechanisms and failure tests |
| [Qualification specification](methodology/27B-4GIB-QUALIFICATION-SPECIFICATION.md) | Proposes exact model references, physical measurement and benchmark decisions S1–S7 |
| [Phased roadmap](architecture/PHASED-ARCHITECTURE-RESEARCH-ROADMAP.md) · [First experiment proposal](experiments/proposals/C0-C2-DEPLOYED-BYTE-STUDY.md) | Stages discriminating controls, costs and stop rules. The proposal has no ledger entry. |
| [Claude review package](reviews/CLAUDE-ROADMAP-REVIEW-PACKAGE.md) | Provides the review prompt and sources for manual submission. The new tool-assisted review timed out. |

## Record policy

- Append a dated correction or a new record when evidence changes. Keep the original measurements and their limitations visible.
- For each run, record the implementation commit, effective config, hardware/runtime, input hashes, seed, commands, outcomes and raw outputs. Link large artifacts with checksums where available.
- Mark missing provenance explicitly. The first 5080 run used an uncommitted working tree and has no exact commit. The repeat has a per-file source fingerprint. Both remain preliminary and are excluded from commit-pinned research evidence until a new run is performed against an immutable implementation commit.
- Keep measurements, derived calculations and hypotheses distinct. Record failed, interrupted and inconclusive work alongside successful runs.
