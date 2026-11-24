# Second independent Claude review: provisional disposition

25 November 2026 — **Review summary only; not the complete reviewer report. No experiment authorised.**

## Provenance and scope

The project owner supplied Claude's second-review conclusions in the conversation on 25 November 2026. Claude reports that its full review exists locally at C:\Users\User\Desktop\Domovoi-reports\2026-11-24-claude-roadmap-review.md, **outside the repositories**. That file was not accessible to this documentation update. This record faithfully distinguishes reported findings from independently inspected source evidence. Do **not** treat it as a publication of the uninspected full Claude report; archive the full report with a hash and original wording when made available, then reconcile any differences by dated addendum. The earlier independent investigation is already archived at [the original independent report](2026-11-24-independent-architecture-research.md).

Documents reviewed for this disposition: [original C0/C2 draft](../experiments/proposals/C0-C2-DEPLOYED-BYTE-STUDY.md), [24 November roadmap](../architecture/PHASED-ARCHITECTURE-RESEARCH-ROADMAP.md), [qualification draft](../methodology/27B-4GIB-QUALIFICATION-SPECIFICATION.md), [research protocol](../docs/EXPERIMENT_PROTOCOL.md) and published TinyStories record. This is a planning update, not a reproduced GPU experiment or an independent verification of all literature claims.

## Reviewer verdict and blocking issues

**REVISE BEFORE APPROVAL** for the original C0/C2 deployed-byte campaign. The review does not itself approve a revised study.

| Severity | Reported concern | Required disposition before an experiment can be approved |
|---|---|---|
| Blocking | The original fixed two-stage VQ initialisation may reduce to a rotated scalar quantiser if stages share the same basis; this is a **conditional algebraic objection**, not yet a verified result for code execution. | Prove or refute the degeneracy from exact code definitions and decoded codepoint sets. Add a dedicated rotated-scalar control and a competitive fixed E8 lattice code at matched effective rate, including transforms, scales and metadata. Do not describe a rotated scalar construction as evidence for vector coding. |
| Blocking | A 100-exposure endpoint can conflate learning rate with maximal retained information. The reviewer cites a separate study reaching its capacity regime at about 1,000 exposures, which may not transfer to this apparatus. | Include an exposure-to-saturation curve with an explicit stopping/saturation criterion; preserve earlier checkpoints for efficiency-of-learning results. Select an appropriate fact-world size before costing the campaign. |
| Blocking | Optional per-arm hyperparameter tuning can unfairly disadvantage low-bit controls. | Predefine equal tuning budget and protocol for every arm including codebooks, learning rates, optimizer and clipping; use disjoint calibration worlds; record all attempts and costs. |
| Blocking | Building seven fully packed GPU decoding kernels for a ~0.5 MB model is premature and may say little about deployment speed. | First validate byte layouts, exact unpack/reconstruct and logits equivalence using a CPU reference implementation. Benchmark realistic packed kernels in a separately approved larger-scale hardware study. |
| Major | The first plan risks producing an improved compressor, not a new neural architecture. | Call the result a representation/codec finding unless a distinct neural operator or learning mechanism and capability transfer are demonstrated against strong existing controls. |
| Major | The physical 4 GiB hardware, Windows system-memory fallback and inference bandwidth are not qualified. | Identify the actual board/laptop/TGP/driver, verify dedicated-memory residency with no fallback and then run a separately approved device experiment. |
| Major | A benchmark comparison to an artificially reasoning-capped 27B reference can make an easy target. | Give the pinned reference a fair, high-quality configuration; report both quality-first and matched-budget analyses without tuning by inspecting outcomes. |
| Major | Context floors, TTFT, statistical uncertainty and human approval policy are not fully reconciled. | Amend S1–S7 through explicit human review; predeclare a first-token latency target; define contextual absolute gates separately from normalized benchmark floors; protect canonical ledger and review named approvers. |

### Arithmetic and estimates: reported, not remeasured here

Claude reports reproducing the original proposal's geometry, payload bytes, synthetic entropy, 57 model-seed runs and 30.8 PFLOP. With changes, Claude estimates **about 123 runs and 192 PFLOP**, with **4.5–9 GPU-hours** only if substantial batching/acceleration assumptions hold. These are *proposed revised-scope scenarios*, **not** an approved or frozen schedule and **not** a measurement of this workload. A 192 PFLOP workload completed in 4.5–9 hours implies approximately 11.9–5.9 effective TFLOP/s; verify actual end-to-end rate before any runtime commitment. A saturation regime on the full original world could require far more exposures than the 100-exposure draft; recalculate after freezing scale, schedules and arms. Do not extrapolate the current 18 GPU-hour envelope.

## Candidate ranking after second review (not an architecture commitment)

1. **C0: strong native scalar low-bit hybrid** — main implementation reference and mandatory comparison.
2. **C2: rigorously controlled scalar-vs-vector/lattice native-training study** — a focused hypothesis, not a nominated final architecture.
3. **Parallel reuse of each loaded weight read / multi-token computation** — add a distinct bandwidth-aware research question and prior-art comparison (including ParScale and multi-token prediction). Distinguish increased arithmetic from increased weight traffic, and evaluate batch-one, acceptance and latency costs.
4. **C1: factored GPU-resident memory + bounded looped computation** — only if measured recall, memory traffic or multi-hop behaviour merits additional read passes.
5. Selective state / context compression — retain as controls and components.
6. Procedural playbooks — defer absent an independently measured need and differentiated operator.

The reviewer says current directions are largely existing compression methods or combinations of known modules. A positive byte-density result is valuable but does not, by itself, establish an invented architecture or 27B-class capability. Keep mechanism invention as an explicit *separate* scientific milestone.

## Required pre-training investigations

**Hardware feasibility, evaluation only:** identify an available *physical* 4 GiB device and exact model/TGP/driver, count all GPU-resident allocations, verify real non-fallback execution, measure bandwidth, low-bit decode and cold-start/prefill/steady decode at representative scale. A memory cap on RTX 5080 is a development screen, not equivalent qualification. No hardware measurements are authorised by this note.

**Capability ladder, evaluation only:** evaluate pinned small and large reference checkpoints under a frozen battery to discover the factual/reasoning/instruction/coding gap before investing in a specific novel mechanism. Use Qwen as proposed primary and Gemma 3 27B as an intermediate comparison milestone, with matched *fair* conditions, separate token-budget curves and post-training effects. Model downloads, external weights, benchmarks and reference GPU access need separate approval and experiment records.

**Revised first C0/C2 work:** split CPU-only correctness fixtures (which are not research measurements) from a small preregistered **six-model-seed BF16/INT8 instrument** and later, independently approved saturation and controlled representation comparisons. See [revised instrument proposal](../experiments/proposals/C0-C2-INSTRUMENT-ONLY-2026-11-25.md) and [two proposed pre-training studies](../experiments/proposals/HARDWARE-AND-CAPABILITY-PREFLIGHT-2026-11-25.md).

## S1–S7 and research-governance disposition

- **S1** named physical 4 GiB GPU including exact laptop/TGP and availability; RTX 5080 is nonqualifying screening hardware.
- **S2** absolute GPU device-wide cold-load-to-decode peak plus per-process and allocator measures; independently rule out Windows shared/system-memory fallback, hidden offload and transient overflow.
- **S3** assess the reference with adequate reasoning space; freeze a quality-first reference configuration and also matched output-token and measured-compute comparisons. Do not hobble the reference to 1,536 reasoning tokens solely to make it fit an arbitrary proposed cap.
- **S4** freeze reference-text / byte equality as well as native-token and generated-token room; distinguish absolute 8K-usefulness gates from normalized score calculations.
- **S5** regenerate all missing reference scores, on exact pinned revisions and contamination-reviewed data; no score access before approval.
- **S6** paired and familywise-corrected uncertainty with thresholds defined before scoring; stringent lower-bound parity can require a substantially larger *point* estimate (review cites roughly 0.975 as an illustration), not just a 0.95 point estimate.
- **S7** explicit category and long-tail floors; no aggregate-score compensation for a collapsed category. Fix any conflict between context category floors and absolute context qualification.
- **Governance** named human reviewer and approval record, protected canonical branch, verified cleanliness including untracked files, pinned analysis code and controlled crash recovery without arbitrary hand-written successful terminal entries.

These are recommendations **pending explicit human approval**. No S1–S7 policy or experiment status changes are authorised here.

## Disposition and audit trail

- Original full study: **not approved; needs redesign**. Retain [its historical draft](../experiments/proposals/C0-C2-DEPLOYED-BYTE-STUDY.md) with an explicit caution; do not erase its arithmetic or methodology.
- New instrument and preflight designs: **unapproved proposals**, no experiment IDs, GPU workloads, downloads or remote claims.
- Open issues: full Claude report not yet archived; model and reference availability; exact E8 implementation and fair rate; saturation plan and tuning; source-based novelty; actual backend rates; human/branch safeguards.
- When available, publish the untouched Claude report and its SHA-256 separately, then add a dated discrepancy ledger instead of editing this account retroactively.
