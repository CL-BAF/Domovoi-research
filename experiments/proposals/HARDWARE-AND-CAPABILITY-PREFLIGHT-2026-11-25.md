# Two proposed prerequisite studies: physical hardware and capability ladder

25 November 2026 — **Planning designs only. No running, model download, experiment ID or human approval.** These are independent proposals. Do not combine them into one uncontrolled execution. [Second-review disposition](../../reviews/2026-11-25-CLAUDE-SECOND-REVIEW-DISPOSITION.md) · [Research protocol](../../docs/EXPERIMENT_PROTOCOL.md).

## H0 — Physical 4 GiB deployment feasibility

### Question

What can a specifically identified **physical 4 GiB GPU** truly sustain in memory, bandwidth and batch-one, 8K-context inference without any hidden CPU/shared-system-memory fallback?

### Preconditions and inventory

- Identify actual device **ownership/availability**, vendor, notebook manufacturer/model, exact RTX 2050 SKU if selected, rated and observed TGP, total dedicated VRAM, driver, Windows/WDDM or Linux configuration, PCIe/power/thermals and compute capability. A generic RTX 2050 specification is not an exact qualification device.
- If the user does not have that device, list procurement/borrowing options for approval; **do not run on an imaginary card**. The RTX 5080, even with an allocator cap, is a separate development screen and cannot be called physical qualification.
- Establish a documented, verifiable policy and telemetry showing that Windows GPU shared memory/system-memory fallback is disabled or definitely not exercised. Driver support and control-panel UI may vary; do not assume a setting exists for all drivers. Sample dedicated/shared process allocations, process/device-wide counters, OS residency events and performance. Measure cold load through prefill and decode, including short-lived peaks, and record any ambiguous instrumentation as **unqualified**, not a pass.
- Verify all weights, embeddings, cache/state and intermediate compute remain on the GPU; ordinary CPU tokenization/harness work is permissible but offloaded model computation is not.

### Measurement conditions proposed for a future approved experiment

1. Device baseline/driver reservation and true available memory; reconcile dedicated physical memory versus allocator/reserved/device-wide values.
2. Representative packed BF16, INT8, INT4 and native low-bit decode/bandwidth cases on **realistic sizes**, with exact packed backend, no expanded full-precision persistent shadow, cold and warm paths.
3. Memory-traffic and arithmetic rooflines from measured bandwidth and decode kernels; distinguish memory-read amortisation across output tokens from weight-looping across iterations.
4. End-to-end representative batch-one prompts at several lengths including 8K reference-equivalent context and at least 256 emitted tokens; instrument prefill, first-token latency, steady output tok/s, p95 token latency, power/throttle and all allocations.
5. A failed overflow/no-offload check must be visible as a fail, regardless of reported speed; a slower-than-8 tok/s run is not qualified.
6. Freeze a **numerical TTFT threshold** and its percentile, cold/warm definition and prompt length **before scoring**; this threshold remains open for human decision.

### Gate

A named human approves exact device, data and kernels, bounded run time, telemetry and interpretation rules under the normal experiment protocol. Do not equate "the kernel works" with a full model passing the 4 GiB / 8K / 8 tok/s specification. No device profiling is authorised by this proposal.

## E0 — Matched-condition model capability ladder

### Question

Where does capability actually degrade when the model size and inference footprint shrink? Is the gap predominantly factual knowledge, reasoning, instruction following, mathematics, coding, or contextual retrieval?

### Proposed comparison families

- Pinned, licensed small Qwen-family checkpoints covering available sizes (the prior roadmap suggests Qwen3.5 0.8B, 2B, 4B and 9B, subject to confirming names, revisions and suitability).
- Pinned Qwen3.8-27B primary reference per [qualification draft](../../methodology/27B-4GIB-QUALIFICATION-SPECIFICATION.md).
- Pinned Gemma 3 27B IT as an independent 27B reference and **intermediate target** if its performance is lower on defined tasks; use measured scores, not an assumed permanent 2x gap or incompatible publisher protocols.

The precise ladder is not final. Different model generations, modalities, tokenizers, instruction tuning and training sets are confounds; include family-specific comparisons and flag incomparable checkpoints. The 27B runs need fit-capable, separately authorised hardware; **do not quietly quantize/offload and present results as native-reference numbers**. If only reduced-precision runs fit, describe those comparisons separately with their own calibration.

### Evaluation method

- Freeze reproducible software, prompt templates/chat formats, revision hashes, item identities, dataset licenses and cutoffs, benchmark exclusion and contamination controls.
- Use a broad battery: MMLU-Pro plus long-tail factual heldout; GPQA Diamond; MATH-500; contemporaneously filtered LiveCodeBench; IFBench strict; 8K RULER plus multi-hop and natural delayed-evidence. Separate raw scores and calibrated normalized/composite indices.
- Compare strong reference configurations with sufficient reasoning/output room to avoid accidental truncation; preregister **quality-first**, equal max-token and measured-compute conditions separately. Do not retrofit an arbitrarily low 1,536-token reasoning cap to degrade the Qwen result.
- Handle generated-token and tokenizer differences by comparing identical source text and also normalizing output workload by bytes; include end-of-generation quality, timeout and refusal categories.
- Evaluate the effect of instruction/post-training. Keep pretrained/base versus instructed comparisons separate, and add **post-training** as an explicit roadmap stage for Domovoi rather than expecting pretrained NLL to explain IFEval or coding.
- Protect sealed task outcomes; report one-sided intervals, simultaneous category floors and variance across items/questions and seeds. Repeated samples on the same question are clustered, not independent examples.
- Publish failure cases, task distribution shifts, denominator uncertainty and performance-vs-size/memory curves. Ladder outcomes do not establish per-weight mutual information.

### Approval boundary

No third-party weight downloads, evaluation-only model runs, benchmark scoring, extra computing hardware or new dataset acquisition without named approval. The first approved run may be a **small bounded ladder slice** if the full battery or reference is too expensive. Record its limitations; do not extrapolate to the 27B target.

## Independence and outputs

H0 and E0 answer different questions. Neither requires C2 implementation. Prefer these studies before an expensive representation campaign, but the order depends on actual hardware access and the approved limits. Each requires its own DRP plan (or explicitly frozen subarms), implementation/source and analysis hashes, published approvals, remote claim and published post-run interpretation. If one is blocked, document it and pursue non-experimental reading/design work; do not bypass the protocol.
