# Two proposed prerequisite studies: physical hardware and capability ladder

25 November 2026 — **Planning designs only. No running, model download, experiment ID or human approval.** These are independent proposals. Do not combine them into one uncontrolled execution. [Second-review disposition](../../reviews/2026-11-25-CLAUDE-SECOND-REVIEW-DISPOSITION.md) · [Research protocol](../../docs/EXPERIMENT_PROTOCOL.md).

## H0 — Physical 4 GiB deployment feasibility

### Question

What can a specifically identified **physical 4 GiB GPU** truly sustain in memory, bandwidth and batch-one, 8K-context inference without any hidden CPU/shared-system-memory fallback?

### Preconditions and inventory

- Record actual access, notebook model and configuration, exact GPU and PCI/subsystem IDs, VBIOS, rated and sustained TGP, dedicated VRAM and idle dedicated reservation, OS build, driver model/version, CUDA runtime and backend compatibility on its compute capability. Record AC mode, power limits, clocks, thermal throttling, display attachments and competing processes. The historical RTX 2050 record does not establish current access or sustained bandwidth.
- If the user does not have that device, list procurement/borrowing options for approval; **do not run on an imaginary card**. The RTX 5080, even with an allocator cap, is a separate development screen and cannot be called physical qualification.
- Record the effective NVIDIA CUDA system-memory fallback setting if the installed driver exposes it, plus WDDM or OS residency and process/device-wide dedicated and shared usage. Validate that instrumentation sees eviction and spill on a separately approved control and captures short-lived peaks. Trace model/state movement from cold load through decode. If the switch is absent, establish absence of fallback with reliable independent telemetry. Missing or ambiguous telemetry leaves physical qualification unresolved. [NVIDIA fallback policy](https://nvidia.custhelp.com/app/answers/detail/a_id/5490)
- Verify all weights, embeddings, cache/state and intermediate compute remain on the GPU; ordinary CPU tokenization/harness work is permissible but offloaded model computation is not.

### Measurement conditions proposed for a future approved experiment

The [full Claude review](../../reviews/2026-11-24-claude-roadmap-review.md) recommends a **provisional 60-second time-to-first-token ceiling at 8K**. This is not an approved value; freeze a numerical threshold, percentile, prompt length and warm/cold interpretation before any run. If driver menus lack the named fallback switch, do not pretend it is set: prove actual lack of shared-memory overflow via appropriate device/OS counters and fail closed on ambiguous evidence.

1. Device baseline/driver reservation and true available memory; reconcile dedicated physical memory versus allocator/reserved/device-wide values.
2. Sustained bandwidth by direction and realistic batch-one model matrix sizes and group layouts. Compare BF16, INT8, INT4 and the selected native low-bit backend with exact kernel/layout revisions. Record effective bytes read, median and dispersion, scale/codebook traffic, transient expansion, and whether unpack and matmul are fused. Account for every retained representation and cache. Measure paths from cold setup and warm steady state separately.
3. Memory-traffic and arithmetic rooflines from measured bandwidth and decode kernels; distinguish memory-read amortisation across output tokens from weight-looping across iterations.
4. Use batch one at 2K, 4K and 8K common-text lengths and include a run at 8,192 actual native prompt tokens. Account for wrappers and at least 256 emitted native tokens. Report cold-load-through-first-token and warm request-to-first-token separately, plus prefill, synchronized post-first-token rate, p95 inter-token latency, UTF-8 bytes/s, full device peak and throttling. Freeze prompts, repeats and aggregate rules. Treat the 8 tok/s requirement as a full-workload gate and record the failure if memory overflows, fallback occurs or output stops early.
5. A failed overflow/no-offload check must be visible as a fail, regardless of reported speed; a slower-than-8 tok/s run is not qualified.
6. Freeze a **numerical TTFT threshold** and its percentile, cold/warm definition and prompt length **before scoring**; this threshold remains open for human decision.

### Gate

A named human approves exact device, data and kernels, bounded run time, telemetry and interpretation rules under the normal experiment protocol. Do not equate "the kernel works" with a full model passing the 4 GiB / 8K / 8 tok/s specification. No device profiling is authorised by this proposal.

## E0 — Matched-condition model capability ladder

### Question

Where does capability actually degrade when the model size and inference footprint shrink? Is the gap predominantly factual knowledge, reasoning, instruction following, mathematics, coding, or contextual retrieval?

### Proposed comparison families

- Pinned, licensed small Qwen-family checkpoints covering available sizes (the prior roadmap suggests Qwen3.5 0.8B, 2B, 4B and 9B, subject to confirming names, revisions and suitability).
- **BitNet b1.58 2B4T** as the native ternary comparator, with the released codec/backend and exact checkpoint pinned separately from any conversion. Include eligible native-fit and post-training quantized (PTQ) checkpoints. Where a single checkpoint has reproducible precision variants, compare its BF16 versus PTQ performance on fit-capable hardware within-family and report its physical-fit condition separately. [BitNet release](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
- Pinned Qwen3.8-27B primary reference per [qualification draft](../../methodology/27B-4GIB-QUALIFICATION-SPECIFICATION.md).
- Pinned Gemma 3 27B IT as an independent 27B reference and **intermediate target** if its performance is lower on defined tasks; use measured scores, not an assumed permanent 2x gap or incompatible publisher protocols.

The precise ladder is not final. Different model generations, modalities, tokenizers, instruction tuning and training sets are confounds; include family-specific comparisons and flag incomparable checkpoints. The 27B runs need fit-capable, separately authorised hardware; **do not quietly quantize/offload and present results as native-reference numbers**. If only reduced-precision runs fit, describe those comparisons separately with their own calibration.

### Evaluation method

- Freeze software, model/tokenizer revisions and licenses, prompt templates, item IDs and cutoffs, benchmark licenses, exclusion and contamination rules, and CPU/GPU operating conditions. Pair exact task content and item order. Blind the final set and exclude it from prompt and decoding selection.
- Separate factual retrieval (sealed long-tail short answers with no retrieved evidence) from mixed knowledge/reasoning (MMLU-Pro), scientific reasoning (GPQA Diamond), math (MATH-500), coding (date-filtered LiveCodeBench expected pass@1), instruction adherence (IFBench strict), and contextual retrieval/composition (8K RULER, multi-hop and natural delayed evidence). Include a matched evidence-present factual diagnostic to distinguish poor recall from failure to use supplied evidence. Report task-level raw scores before the composite.
- Compare strong reference configurations with sufficient reasoning/output room to avoid accidental truncation; preregister **quality-first**, equal max-token and measured-compute conditions separately. Do not retrofit an arbitrarily low 1,536-token reasoning cap to degrade the Qwen result.
- Handle generated-token and tokenizer differences by comparing identical source text and also normalizing output workload by bytes; include end-of-generation quality, timeout and refusal categories.
- Keep base and instructed cohorts separate. Where a verified base/instruct companion from one lineage exists, use a paired contrast under a matched format and report remaining training differences. Keep short constrained-instruction failures, wrong answers, refusals and truncation distinct. Domovoi needs a future matched post-training stage for both C0 and any candidate before interpreting instruction and coding gaps.
- Protect sealed task outcomes; report one-sided intervals, simultaneous category floors and variance across items/questions and seeds. Repeated samples on the same question are clustered, not independent examples.
- Publish failure cases, task distribution shifts, denominator uncertainty and performance-vs-size/memory curves. Ladder outcomes do not establish per-weight mutual information.

### Approval boundary

No third-party weight downloads, evaluation-only model runs, benchmark scoring, extra computing hardware or new dataset acquisition without named approval. The first approved run may be a **small bounded ladder slice** if the full battery or reference is too expensive. Record its limitations; do not extrapolate to the 27B target.

## Independence and outputs

H0 and E0 answer different questions. Neither requires C2 implementation. Prefer these studies before an expensive representation campaign, but the order depends on actual hardware access and the approved limits. Each requires its own DRP plan (or explicitly frozen subarms), implementation/source and analysis hashes, published approvals, remote claim and published post-run interpretation. If one is blocked, document it and pursue non-experimental reading/design work; do not bypass the protocol.
