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

## Record policy

- Append a dated correction or a new record when evidence changes. Keep the original measurements and their limitations visible.
- For each run, record the implementation commit, effective config, hardware/runtime, input hashes, seed, commands, outcomes and raw outputs. Link large artifacts with checksums where available.
- Mark missing provenance explicitly. The first 5080 run used an uncommitted working tree and has no exact commit. The repeat has a per-file source fingerprint. Both remain preliminary and are excluded from commit-pinned research evidence until a new run is performed against an immutable implementation commit.
- Keep measurements, derived calculations and hypotheses distinct. Record failed, interrupted and inconclusive work alongside successful runs.
