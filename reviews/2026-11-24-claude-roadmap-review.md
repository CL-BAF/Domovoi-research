# Domovoi: independent roadmap and architecture review

**Reviewer:** Claude (Opus 5.5), acting as an independent reviewer.
**Date:** 24 November 2026.

**Sources reviewed (pinned):**
- `CL-BAF/Domovoi-research@6d6a90d7cf34771ec57b579e358e4f11a03143b5`
- `CL-BAF/Domovoi-Labs@8252fbf361b387dc21d17fd6ff7443d9dc7274ad`

**Status:** advice only. No repository was modified. No training, evaluation, benchmarking or model download took place. Nothing in this report approves execution.

**Disclosure.** I wrote the earlier independent report that the reconciliation draws on (`reviews/2026-11-24-independent-architecture-research.md`, SHA-256 `77f35370…dd62`, byte-identical to my local copy). I have tried to review it as critically as the Codex documents. Where I now disagree with my own earlier conclusions, I say so (§E.4, §C).

### Evidence labels
| Label | Meaning |
|---|---|
| **[V]** | Directly verified in this session against the pinned repository file, the publisher's page or the arXiv abstract or paper text. |
| **[C]** | Independently calculated by me. The inputs are shown. |
| **[P]** | Supported by published research. I verified the abstract or the paper text unless I say otherwise. |
| **[A]** | Assumed. An engineering estimate that must be measured. |
| **[S]** | Speculative. |
| **[U]** | Unable to verify. |

---

## A. Executive assessment

1. **The direction is careful, well documented and honest about its limits. As currently sequenced, though, it is very unlikely to discover a fundamentally new architecture, and it is not yet aimed at the questions that decide whether the target is reachable.**
   - The roadmap is mostly a study of how to represent weights at low bit width (C0/C2), followed by recombining known components (C1, selective state, routing).
   - That is a legitimate research programme, and it can produce a useful capability-per-GiB frontier.
   - The documents should stop implying that it is on a path to a new architecture. At best it is on a path to a well-measured, efficiently compressed conventional hybrid.

2. **"Weights dominate the memory budget" holds up. "Better weight codes are the main remaining lever" only partly does.**
   - For any plausible 4 GiB hybrid, 8K sequence state is about 60–130 MiB, 2–3% of the budget, against about 78% for weights [C, §D.4]. So prioritising weight representation over context memory is correct.
   - However, recent 2-bit QAT papers, checked in the paper text, already map much of the representation frontier. All of them start from a pretrained model with a post-training-quantised (PTQ) initialisation:
     - **LC-QAT** [P, 2606.10531]: Qwen3 1.7B/8B and LLaMA-3 3B/8B, about 4B FineWeb tokens. It reports that 2-bit vector codes clearly beat 2-bit *scalar* QAT.
     - **E8-QAT** [P, 2610.27699]: Qwen3.6-35B-A3B, with only the expert FFNs (92.1% of weights) at 2.03 bits and attention and embeddings left in BF16. It matched the BF16 average across four tasks. Its *fixed* E8 lattice beat LC-QAT's *learned* code.
     - Native ternary trained from scratch at 2B parameters and 4T tokens is reported competitive [P, 2504.12285].
   - So there is evidence for "vector beats scalar at about 2 bits" in the PTQ-initialised regime. There is evidence *against* "a learned codebook beats a fixed lattice". There is **no evidence yet for native from-scratch training**.
   - The open question is therefore narrower than "can codes help": **how many usable knowledge bits per resident bit a native 2-bit-class format retains, and whether vector or lattice structure beats a strong scalar format from scratch**. That is worth measuring. Its likely effect is modest and cannot by itself close a 2–4× gap.

3. **The first experiment (the C0/C2 deployed-byte study) is cheap, its arithmetic is correct and it is mostly well controlled. It has design flaws that could produce a false positive or an uninterpretable null.** My verdict is **REVISE BEFORE APPROVAL**, with the instrument block split out as its own first experiment (§F). The blocking problems are:
   - **(a)** As specified, the "fixed VQ" arm is a rotated 4-level scalar quantiser if both stages share one rotation, so the study may never test a real vector code. It also lacks the strongest known fixed code, the E8 lattice.
   - **(b)** The primary endpoint at 100 exposures measures how fast facts are memorised, not how much can be stored.
   - **(c)** Learning-rate tuning is optional, which risks weak low-bit baselines.
   - **(d)** It requires building seven packed GPU decoders for a 0.5 MB model. That proves nothing about the RTX 2050 and delays the science.

4. **The two most decisive questions need no training, and the roadmap treats both as optional or late.**
   - **(i)** What do the RTX 2050's real memory overhead, bandwidth and low-bit kernel throughput look like? This includes Windows' silent system-memory fallback, which can hide forbidden offloading [V, §D.6].
   - **(ii)** On the matched-budget benchmark battery, how far is the best existing ≤4 GiB deployment (for example Qwen3.5-4B/9B compressed after training, or BitNet b1.58 2B4T) from the reference?
   - Together these tell the project whether the 95% target is a frontier problem, a recipe problem or out of reach. I recommend running them first, each with its own approval.

5. **The capability specification (S1–S7) is strong and much better than the original `EVALS.md`. Three things need fixing:**
   - S3's 1,536-token reasoning cap would set the reference's denominator by a truncation policy rather than by capability;
   - S4 and S7 contain inconsistent context gates;
   - the statistical bar in S6 and S7 effectively requires a point estimate of up to about 0.975 on the composite and up to about 0.92 on noisy categories. These are conservative upper bounds; pairing lowers them [C, §D.7].

   The roadmap also has **no post-training (SFT/RL) stage**. IFBench, LiveCodeBench and GPQA under a thinking budget cannot be reached by pretraining-only architecture screens.

6. **The "27B-class" label is ill-defined in practice.** On the publisher-reported numbers, the two proposed 27B references differ by about 2× on GPQA Diamond (89.2 against 42.4) and about 3× on LiveCodeBench (90.3 against 29.7 in Gemma's Table 6) [V].
   - Capability is set by the training recipe far more than by the parameter count.
   - Gemma 3 27B IT parity under the matched budget is therefore a scientifically meaningful and far more tractable intermediate milestone. It should be adopted explicitly.

7. **An overlooked direction has a stronger theoretical case than sequential recurrence (C1): parallel computation that reuses each weight read.**
   - At batch 1 the RTX 2050 is bound by bandwidth, with roughly 12–25× more arithmetic capacity than a 2-bit matrix-vector product (GEMV) uses [C/A, §D.6].
   - Parallel streams (ParScale [P, 2505.10475]) and multi-token prediction with self-speculation (MTP) [P, 2404.19737] spend that idle compute without extra weight reads. Sequential looping re-reads every weight each pass.
   - This is prior art, not novelty, but a combination built for this hardware regime is an under-explored, testable hypothesis (§C, §G).

8. **The TinyStories results establish four things:** a working, reproducible pipeline; an equal-token weight-sharing trade with gap closure 0.286 (single seed); a training-memory saving fully explained by parameter-state bytes; and nothing about inference, 8K or capability. They do not justify a standalone recurrence programme (§9 of the brief; §B, M-9).

**Bottom line:**
- Approve the S1–S7 revisions and the two zero-training characterisation studies first, each with its own approval.
- Run a revised C0/C2 instrument block, then a revised main screen.
- Re-scope the public goal to "maximum capability per GiB on a physical 4 GiB device, with matched-budget Gemma-3-27B-IT parity as the first serious milestone and Qwen3.8-27B parity as the moonshot".
- Do not promise a new architecture. Let one emerge only if a mechanism beats a strong C0 at equal measured bytes and speed.

---

## B. Critical findings (ranked)

### BLOCKING (resolve before the first experiment is approved)

**B-1. The "fixed VQ" arm may be mathematically a rotated scalar quantiser, so the study may not test vector codes at all.**
*Where:* `experiments/proposals/C0-C2-DEPLOYED-BYTE-STUDY.md`, §Backbone ("deterministic orthogonally rotated eight-dimensional hypercube tables, with second-stage initial scale one-half"; greedy nearest-code stage 1, then nearest residual).

*Proof [C]:*
- Let stage 1 be C₁ = {R·s : s ∈ {±1}⁸}. Then argmin_s ‖u − Rs‖ = argmin_s ‖Rᵀu − s‖, and the Euclidean distance separates by coordinate. So s₁ = sign(Rᵀu), applied coordinate by coordinate.
- If stage 2 is C₂ = {½·R·s} with the **same** R, the residual in rotated coordinates is x − s₁ with x = Rᵀu. Its nearest point is s₂ = sign(x − s₁).
- The reconstruction is R(s₁ + ½s₂). Coordinate by coordinate this takes values in {±0.5, ±1.5}, with decision thresholds at 0 and ±1. That is exactly nearest-level rounding to a uniform 4-level grid, the same {−3, −1, 1, 3}/2 grid as the "scalar 2b" arm, applied in a rotated basis.
- So fixed VQ against scalar 2b isolates only the rotation (the incoherence idea behind QuIP#). Learned VQ starts from a scalar-equivalent initial state, and its 4,096 trainable codebook values have to move it away.
- If the two stages use different rotations, the code is a genuine two-stage residual VQ, but greedy and sub-optimal.

*The proposal does not say which construction it uses* ("publish transformations… in the future plan").

*Remedy:*
- Specify the tables exactly.
- Add an **E8-lattice arm** (an implicit codebook at 2 bits per weight, the densest 8-dimensional lattice packing) as the strong fixed vector code [P, E8-QAT 2610.27699; QuIP# 2402.04396].
- Add a **Hadamard-rotated scalar 2b** arm so that the effect of rotation is separated from the effect of vector structure.
- Rename the current "fixed VQ" arm honestly if it is separable.

**B-2. The primary endpoint (100 exposures) confounds storage density with memorisation speed.**
*Where:* the proposal's §Question ("at high load and 100 exposures").
- The cited study measures about 2 bits per parameter at **1,000** exposures. At 100 exposures it measures about 1 bit per parameter for GPT-2, and architectures differ by up to 1.3× there [V, 2404.05405 Results 4–6].
- At 100 exposures, a format that learns more slowly under a straight-through estimator (STE) will look less dense even if its eventual capacity is higher, and the reverse is also possible. The proposal's own text says it wants to "separate capacity from memorization at finite exposures", but only 20 and 100 exposures are scheduled.

*Remedy:* add a saturation point. Either:
- (a) 1,000 exposures for BF16, INT8, the best scalar 2-bit-class arm, E8 and learned VQ. That is about 14.5 PFLOP per run at the 2-bit width [C], affordable but larger than the whole current plan; or
- (b) reduce the world size (for example to 65,536 facts) and model width together, keeping all arms below saturation.

Make the endpoint a **pair**: density at saturation (capacity) and density at 100 exposures (sample efficiency). Pre-register which one decides advancement.

**B-3. Optional tuning risks weak scalar baselines.**
*Where:* the proposal's §Backbone: "Allow equal prespecified tuning on a disjoint calibration world **or none** for every arm."
- One shared peak learning rate (3e-4) for BF16 through ternary and VQ at d = 64 is not neutral. Low-bit STE training is known to be more fragile at small scale. The tuple study itself notes training instability for gated MLPs below 100M parameters [V, 2404.05405].
- An untuned ternary or 2-bit scalar arm is the classic way an experiment "proves" that codes beat scalars.

*Remedy:* make an equal, pre-registered tuning budget mandatory for every arm: at least learning rate × {3 values} and warmup × {2 values}, on the calibration world. Report the trial counts. Count the tuning compute.

**B-4. The packed GPU kernel precondition is premature and uninformative.**
*Where:* the proposal's §Question ("requires an executable backend and logits parity through the compact path") and §Compute ("verified GPU packed decoder").
- The resident bytes of these formats are exact arithmetic (§D.2). A capacity screen needs (i) exact byte accounting and (ii) bit-exact pack/unpack round-trip parity on CPU.
- Writing seven GPU decoders for 64 × 2,368 matrices costs engineering time. It says nothing about RTX 2050 decode speed, because tiny matrices are limited by kernel-launch latency.

*Remedy:*
- Replace the GPU precondition with a CPU round-trip and logit-parity fixture.
- Move kernel feasibility to a separate microbenchmark at realistic shapes (for example 4,096 × 11,008 GEMV at batch 1) on the RTX 5080, then on the RTX 2050.
- Include existing open-source low-bit kernels as the bar: Marlin-class INT4 targets SM 8.0 and above [A, based on its published claims]; QuIP#-style E8 decoding has published CUDA kernels [P/A].

### MAJOR

**M-1. Non-MLP tensors store knowledge, and their contribution is not separated.**
- 37.6% of resident bytes (197,760 of about 526–539 KB) are BF16 embeddings, attention and norms [C].
- The cited study finds that even models **without MLP layers** reach about 2 bits per parameter at 1,000 exposures [V, 2404.05405 Result 5].
- So the attention parameters (65,536 values, about 131 Kbit at 2 bits per parameter) contribute knowledge K independently of the arm. This shrinks every ratio towards 1, a false-negative bias, and makes the "BF16 exceeds 5% of world entropy" floor partly a property of the fixed tensors.

*Remedy:*
- Add a reference arm with fixed tensors only (f → 0, or the MLP frozen at initialisation) and three seeds.
- Report ΔK attributable to the coded matrices and the density per **coded-MLP byte** alongside the whole-model density.

**M-2. Ternary is handicapped by its packing.**
- Ternary carries log₂3 = 1.585 bits in a 2-bit slot. That is 79% efficient, so the arm has 21% fewer weights than base-3 packing (5 trits per byte = 1.6 bits) would allow [C].
- With equal bytes and identical width, ternary is strictly dominated by scalar 2b, which has 4 levels.

*Remedy:* add a 1.6-bit packed ternary variant (byte-indexed lookup decode), or report ternary at both packings.

**M-3. The estimator's interpretation and calibration.** (Details in §D.3.)
- K = N_f[16 + mean log₂ q(Y|X)] is a valid *expected* lower bound on the information the resident bytes must contain, so D ≤ 1. That holds only if every world-dependent bit is counted and the decoder is independent of the world.
- K is sensitive to calibration: confidently wrong predictions cost more than 16 bits each. Low-bit arms may differ systematically in logit scale.

*Remedy:*
- Report K\* = max over a fixed 16-point temperature grid. Its extra description cost is at most 4 bits, which is negligible. Report it alongside K at T = 1.
- Treat any D > 1 (beyond seed noise) as a **leakage alarm** that invalidates the run.
- Generate labels with a cryptographic pseudo-random function (for example HMAC-SHA-256 or ChaCha20 keyed per seed), so the generator cannot be learned.

**M-4. Random-fact density is not general capability, and the proposal's transfer check comes too late.**
- A format that memorises random tuples better can still generalise worse. Memorisation capacity and inductive bias pull in different directions.
- The proposal defers transfer checks to a later natural-data study.

*Remedy:* add two cheap probes to the same runs (§G, Stage 1):
- **(a) Structured world:** a fraction of attributes is determined by latent rules over identity features. Measure accuracy on held-out identities, which tests rule extraction.
- **(b) Two-hop manipulation:** queries composed from two stored facts, following Physics of LMs 3.2 [P, 2309.14402].

C2 should advance only if its density gain does not come with a loss on (a) or (b).

**M-5. The Stage 2 natural-text screen is far too short to rank weight formats.**
*Where:* `PHASED-ARCHITECTURE-RESEARCH-ROADMAP.md`, Stage 2, which uses 20M tokens per arm at 10M, 30M and 100M logical parameters.
- That is 2, 0.67 and 0.2 tokens per parameter [C]. Precision effects interact with the amount of training data [P, Scaling Laws for Precision 2411.04330], so rankings found this early may reverse.
- The compute assumption (0.5–5 TFLOP/s) is roughly 10–100× too pessimistic for 30–100M models. The archived TinyStories run achieved about 58 TFLOP/s effective at 33.8M parameters [V/C].
- At 20 tokens per parameter, a 100M model needs 2B tokens: 6 × 1e8 × 2e9 = 1.2e18 FLOPs, about 6 h at 58 TFLOP/s [C/A].

*Remedy:* train the 30M and 100M sizes to at least 20 tokens per parameter. That requires a 2B-token FineWeb-Edu pack (currently 5.04M tokens are packed [V]), which is a data-preparation dependency to resolve first.

**M-6. The roadmap has no post-training stage.**
- IFBench, LiveCodeBench and thinking-budget GPQA scores depend heavily on SFT, RL and reasoning distillation. Pretraining NLL screens do not predict them.
- Without a fixed post-training recipe applied equally to C0 and each candidate, Stage 7 cannot be interpreted.

*Remedy:* add a fixed, versioned post-training recipe as a dependency before any milestone evaluation, and report "architecture effect after identical post-training".

**M-7. S3's reasoning cap makes the reference denominator a truncation artefact.**
- S3 allows 1,536 reasoning tokens for Qwen3.8-27B, which thinks by default. Unclosed thoughts will be common, and the score will depend on how truncation is handled rather than on capability.
- This lowers the bar in a way that is hard to interpret.

*Remedy:*
- Give the reference **its best compliant configuration** within the same 8,192-token window: the maximum over {thinking off; thinking with a budget B ∈ {1,536, 3,072, the largest that fits}} of the pre-registered metric, chosen on a disjoint development split.
- Report the score-against-budget curve for both models.

**M-8. Inconsistent context gates.**
- S4 sets **absolute** gates: at least 80% RULER macro at 8K and at most 10 points of drop from 2K.
- S7 sets a **relative** floor: at least 0.80 of Qwen at 8K.
- These can disagree. With Qwen at about 95% on RULER 8K [A], 0.80× is about 76% absolute.

*Remedy:* pick one primary gate (I recommend the relative floor plus the S4 drop limit) and make the other a diagnostic.

**M-9. Interpreting R02×4 as evidence for recurrence.**
- The reconciliation is right to keep the measurements as baselines. The extended run supports a narrow equal-token claim only (§9 of the brief).
- Under the iso-depth law (φ = 0.46 [P, 2604.21106]) and the bandwidth arithmetic in §D.6, standalone sequential recurrence is the weakest way to trade compute for bytes on a 112 GB/s card.
- No further standalone recurrence experiment should be funded.

**M-10. Windows system-memory fallback can silently break the no-offload rule.**
- The NVIDIA Windows driver has a "CUDA – Sysmem Fallback Policy". Several reports state that from driver branch 536 onwards, allocations beyond VRAM silently spill into shared system memory under WDDM instead of failing [P/U, forum and vendor guides; no official NVIDIA release note found].
- S2 mentions disabling managed-memory spill, but not this policy.

*Remedy:*
- Require "Prefer No Sysmem Fallback", set per application, in the qualification record.
- Verify that the "Shared GPU memory" counter stays at 0 and that PCIe traffic (`nvidia-smi dmon -s t`) shows no weight streaming during decode.
- Repeat qualification on headless Linux where possible.

**M-11. Gemma reference numbers are inconsistent.**
- The spec quotes Gemma 3 27B IT at LiveCodeBench 39.0. The Gemma 3 report gives **29.7** in Table 6 and **39.0** in Table 18 for the same model [V, 2503.19786]. The two tables presumably use different LiveCodeBench windows.
- The other values check out: MMLU-Pro 67.5, GPQA 42.4 and MATH 89.0 (Tables 6 and 18), IFEval 90.4 (Table 18), and RULER 91.1 at 32K (Table 15) [V].

*Remedy:* cite both values with their table windows. This does not matter much, because S5 already requires the project to re-measure.

**M-12. Protocol gaps that matter for reproducibility.** (Details in §10/I.)
- The ledger has no approver-identity enforcement, and the canonical ref is unprotected [V, reconciliation §Protocol].
- The clean-tree check ignores untracked files (`git status --porcelain --untracked-files=no`) [V, `launch.py:check_command`].
- No analysis code is pinned.
- The environment and dependencies are not part of the gate.

### MINOR

- **m-1.** My earlier report claimed 66 arXiv papers. It contains **65** distinct IDs [V, recount]. Codex's correction is right.
- **m-2.** My C3 adapter row said 1,318 MiB "int4". The raw INT4 value is 1,280 MiB, and about 1,320 MiB includes group-128 scales [C]. Codex's correction is right.
- **m-3.** "No test split" in my earlier report meant no test *score*. Codex's correction is right.
- **m-4.** The FLOP model's attention term 12·L·d·T is twice Kaplan's causal approximation. It is conservative and below 1.4% of total, so it does not matter [C].
- **m-5.** The 20-exposure points are checkpoints inside the 100-exposure cosine schedule, with the learning rate not yet annealed. Treat them as diagnostics, or run separate 20-exposure schedules.
- **m-6.** Gradient accumulation of 16 × 256 queries for a d = 64 model only multiplies kernel launches. Use a single 4,096-query batch with CUDA graphs or `torch.compile` (§D.5).
- **m-7.** The proposed 5–10 ms NVML polling can miss short peaks. On a physical 4 GiB card, *completion without OOM and with fallback disabled* is the binding proof. The polled peak is supporting evidence.

### UNRESOLVED

- **U-1.** Whether native 2-bit-class formats reach more than 0.25 knowledge bits per stored bit at saturation. That is the point of the revised Stage 1. No published native from-scratch measurement exists [U].
- **U-2.** How much benchmark-relevant knowledge Qwen3.8-27B uses. The 53.8 Gbit figure in my earlier report is an extrapolation, as the reconciliation correctly says.
- **U-3.** The RTX 2050's real sustained bandwidth, CUDA-context overhead, idle VRAM and thermal or power operating point. The 112 GB/s figure is nominal [V spec / U sustained].
- **U-4.** Whether the reported 2-bit QAT results (E8-QAT, LC-QAT) carry over to **from-scratch** training. Both start from strong PTQ initialisation [V, abstracts].
- **U-5.** Whether ParScale-style parallel streams keep their reported gains at about 1–10B scale with low-bit weights, and what multiplying the KV cache by P costs at 8K.

---

## C. Independent architecture ranking

The ranking criterion is the expected contribution to **capability per resident byte and per byte read at batch 1 on a 112 GB/s, Ampere SM 8.6, 4 GiB device**, weighted by plausibility and the cost of testing. Novelty is scored separately.

| Rank | Direction | Mechanism | What it addresses | Likely effect | Strongest evidence against | Smallest falsification |
|---:|---|---|---|---|---|---|
| **1** | **C0+: strong native low-bit hybrid with a mature recipe.** Ternary, 2-bit scalar or E8 lattice weights; DeltaNet/GQA hybrid; distillation and post-training | w = s·Q(u) trained from initialisation; hybrid mixing | Bytes per parameter (to about 1.6–2.25 bits); quality comes from the recipe | **Largest and most certain.** About 12–14B weight slots at 2–2.25 bits in about 3.3 GiB [C] | Not novel. Kernel throughput on SM 8.6 is unproven. Native low-bit at scale is reported only for 2B parameters / 4T tokens [P] | Equal-byte NLL at 30M/100M against BF16/INT4 at 20 tokens per parameter, plus a kernel microbenchmark |
| **2** | **C2 narrowed: vector or lattice versus scalar, trained natively from scratch** (scalar 2b, rotated scalar, ternary 1.6, E8, learned codes) | w = s·Σₘ Cₘ[iₘ] or a lattice point, trained from initialisation | Knowledge and skill per stored bit | Small to moderate. Ceiling of about 4× over INT8's 0.25 bits per bit; realistically I expect ≤1.5× [S] | **For:** LC-QAT finds VQ ≫ scalar at 2 bits (PTQ-initialised) [P, paper text]. **Against:** fixed E8 beat the learned code [P], so the "learned" part is weak. E8-QAT quantised only MoE expert FFNs and tested four tasks [V]. No from-scratch evidence | Revised Stage 1 (§F) |
| **3** | **Weight-stationary parallel computation** (new to this list): P parallel streams (ParScale) and/or MTP self-speculation | Reuse each weight read for P activation vectors (prefix-tuned streams with learned aggregation, or verification of draft tokens) | Compute per byte read; O(log P) "parameter-equivalent" scaling [P] | Moderate. ParScale reports 22× less memory increase and 6× less latency increase than parameter scaling at **batch 1** for a 1.6B model at P = 8. Models were 0.5–4.7B, trained from scratch on 42B tokens [V, paper text]. KV cache grows P× [V] | O(log P) gains are modest. Tested at full precision on large GPUs. At 8K, KV × P is material (§D.6) | P ∈ {1, 2, 4, 8} at equal resident bytes **including KV × P** on 30M–100M models: NLL plus measured batch-1 tokens/s on the 5080, then the 2050 |
| **4** | **C1:** resident associative memory plus a small looped core | Sparse top-k reads and sequential latent passes | Reasoning per byte read; possibly multi-hop | Uncertain | φ = 0.46 [P]. Ouro: gains come not from capacity [P]. Engram's evidence is against an MoE, not dense, comparator [V]. Sequential passes re-read the core, and the 2050 allows only about 16 passes of a 0.6 GB core [C] | Four-arm equal-byte and equal-bytes-read ablation on 2/3-hop tasks |
| **5** | **Selective state plus sparse exact attention** | Mamba-2/DeltaNet state with a few GQA layers | Context memory | **Needed as a component, not as research.** The 8K state is ≤ about 3% of the budget [C] | The reference already uses it [V]. Context memory is not the binding constraint | Use a mature hybrid; test only if 8K RULER fails |
| **6** | **Adaptive computation and a latent KV cache** (MoD/MoR) | Learned per-token depth | Average compute and latency | At most 25% fewer applications in the cited example, before overhead [V, doc] | Batch-1 divergence and tail latency | Learned against equal-budget random routing, with measured p95 |
| **7** | **Learned consolidation** (Titans/ATLAS-like) | Gated write into slot memory | Long-context state | Not binding at 8K | Context is not the bottleneck at 8K | Only if 8K delayed-evidence tests fail |
| **8** | **C3 procedural playbooks** | Routed low-rank adapters or a typed interpreter | Procedural reuse | Unproven | Modularity alone does not give systematic composition [P, 2206.02713]. Close prior art | Deferred, as the reconciliation says |

**Why this differs from both earlier rankings.**
- I previously ranked C2 first because bytes are the binding constraint. That reasoning holds.
- I read the 2026 QAT papers' text (LC-QAT, E8-QAT; TVQ abstract only). They narrow C2 to "vector or lattice versus scalar, from scratch", and they predict that the *learned* codebook will not beat a fixed lattice. C2 remains the cheapest, best-targeted measurement, but it is a measurement more than a large capability lever.
- C0+ ranks first because it is the most likely to deliver capability per GiB. Admitting that is the honest consequence of a "capability under a budget" objective.
- C2 and weight-stationary parallelism (ranks 2 and 3) are close. C2 goes first because it is cheaper and attacks the binding term directly.
- Parallel computation ranks above C1 because it spends the compute the hardware has idle at batch 1 (§D.6). C1's sequential passes spend the bandwidth it lacks.

**A stronger alternative framing.** The single largest lever on small-model capability in recent practice is the training recipe: data curation, long training, distillation and RL post-training. The two "27B" references differ by about 2–3× on GPQA and LiveCodeBench [V]. The architecture programme should be judged by **how much it adds on top of the best achievable recipe at equal measured bytes and speed**. That requires a strong recipe-matched C0 as the permanent control.

---

## D. Mathematical verification

### D.1 First-experiment geometry and parameter counts [C, all reproduced]
- **Fixed tensors:**
  - embedding 512 × 64 = 32,768;
  - attention 4 layers × 4 × 64² = 65,536;
  - RMSNorm (2 × 4 + 1) × 64 = 576;
  - **total 98,880 values, 197,760 B in BF16.** ✔
- **MLP:** 4 layers × 2 × 64 × f = 512f. ✔

### D.2 Resident bytes [C]
| Arm | f | MLP values | Payload bytes | Scales (FP16 per 64) | Books | Total including fixed tensors | Proposal |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 320 | 163,840 | 327,680 | 0 | 0 | **525,440** | 525,440 ✔ |
| INT8 | 640 | 327,680 | 327,680 | 10,240 | 0 | **535,680** | ✔ |
| INT4 | 1,216 | 622,592 | 311,296 | 19,456 | 0 | **528,512** | ✔ |
| Ternary (2-bit packing) | 2,368 | 1,212,416 | 303,104 | 37,888 | 0 | **538,752** | ✔ |
| Scalar 2b | 2,368 | 1,212,416 | 303,104 | 37,888 | 0 | **538,752** | ✔ |
| Fixed/learned VQ2 | 2,304 | 1,179,648 | 294,912 | 36,864 | 8,192 | **537,728** | ✔ |
| *Ternary at 1.6 bits (suggested)* | *2,816* | *1,441,792* | *288,359* (rounded up) | *45,056* | 0 | *531,175* | — |
| *E8 lattice at 2 bits (suggested)* | *2,368* | *1,212,416* | *303,104* | *37,888* | *0 (implicit)* | *538,752* | — |

- The maximum spread is 13,312 B, which is 2.53% of the smallest arm and inside the 3% rule ✔.
- The effective rates are 8.25, 4.25, 2.25, 2.25 and 2.25 bits per MLP weight ✔.
- The "528 KiB envelope" is 540,672 B, and all arms fit ✔.
- **Hidden-byte checks:**
  - The scales and books are counted ✔.
  - E8 needs a few extra bits for the sign/parity extension. E8-QAT reports less than 0.1% index overhead [V abstract].
  - The 1.6-bit ternary arm needs a 243-entry lookup table (negligible).
  - Not counted: allocator alignment and tile padding, which the proposal correctly says to measure; and the latent FP32 weights and optimiser state, which are correctly excluded from deployment.

### D.3 Entropy and the estimator
- **World:** 65,536 identities × 4 attributes = **262,144 facts** ✔, each with 2 independent uniform bytes, so H = 262,144 × 16 = **4,194,304 bits** ✔ [C].
- **Load:** the resident payload is about 4.2–4.3 Mbit, while the world holds 4.19 Mbit. Full storage would therefore need about 1 knowledge bit per stored bit. At published densities:
  - BF16 at 1–2 bits per parameter × 262,720 parameters is 0.26–0.53 Mbit, which is **6.3–12.5% of H**;
  - the INT8 arm (426,560 parameters) is 0.43–0.85 Mbit, which is **10–20%** [C/P].
- **Consequence:** the "BF16 at least 5% of world entropy" floor at 100 exposures has only about a 1.25× margin, and less if a d = 64 model underperforms GPT-2's density. **The instrument may fail its own floor.** Measure this in the instrument block before freezing the main study.
- **What the estimator means [C/P].** For a world W drawn uniformly at random:
  - Arithmetic coding with q_θ encodes W in |θ| + Σ −log₂q(y|x) bits, where |θ| is the length of every world-dependent bit the decoder needs.
  - Since H(W) = 16N, this gives **E[K] ≤ E|θ| ≤ 8·B_resident**, as long as the architecture, fixed books and any seed-derived tensors are independent of W.
  - This matches the "bit complexity lower bound" approach of 2404.05405 (Theorem 3.2). I verified the definition, not every detail of the theorem [V/U].
  - Implications: D is an expected lower bound on the fraction of resident bits carrying world information. Per-world fluctuation is negligible at N = 262,144. Restricting the softmax to the 256 allowed byte tokens is legitimate, because the decoder knows the alphabet. The teacher-forced two-byte factorisation is exact by the chain rule.
- **Calibration:** K can be negative and is penalised by overconfidence. The temperature-maximised K\* fixes this (§B, M-3).
- **Statistics:** K is evaluated on the whole population of trained facts, so all of its uncertainty comes from seeds and world draws. Clustering by identity matters only for prompt-variant probes ✔.
- **Seed power:** unknown until the instrument block runs. If the seed standard deviation of D is above about 4%, three seeds cannot reliably support a 10% learned-versus-fixed margin [C/A]. Size the main block from the instrument's variance.

### D.4 Exposures, positions, updates and FLOPs [C]
**Positions and updates:**
- At 100 exposures: 26,214,400 examples × 7 = **183,500,800 positions** ✔ and 52,428,800 value targets ✔.
- At 20 exposures: 36,700,160 positions ✔.
- The instrument world: 16,384 × 4 × 20 × 7 = 9,175,040 ✔.
- One update is 4,096 × 7 = **28,672 positions** ✔, so 100 exposures take **6,400 updates**.

**FLOPs per position**, from F = 6P + 12·L·d·T with 12·4·64·7 = 21,504:
- BF16 (P = 262,720): 1,597,824
- INT8 (426,560): 2,580,864
- INT4 (721,472): 4,350,336
- Ternary/2b (1,311,296): 7,889,280
- VQ (1,278,528): 7,692,672
- VQ search: 1,024·P_f per update ≈ 1.21e9, which is 0.5%.

**Compute-matched update counts:** 2.932e14 / (F × 28,672) = **2,351 / 1,296 / 1,322** for INT4 / 2b / VQ ✔. That corresponds to only 36.7, 20.3 and 20.7 exposures, so Block C compares a narrow model at 100 exposures with wide models at about 20.

**Totals:**
| Block | FLOPs |
|---|---:|
| Instrument | 8.8e13 |
| A (21 runs) | 6.157e15 |
| B (18 runs) | 2.102e16 |
| C (12 runs) | 3.518e15 |
| **Total** | **3.078e16 = 30.8 PFLOP** ✔ |

The total is **57 runs** ✔ and **270,393 optimiser updates** [C].

### D.5 Runtime realism [C/A]
- The FLOP-based estimate (1.7–17.1 h) uses the wrong model. A d = 64, 4-layer network trains at a rate limited by **kernel-launch and synchronisation latency**, not FLOPs.
- **Eager PyTorch with accumulation of 16:**
  - about 300 launches per microbatch × 16 microbatches × 8–20 µs per launch (WDDM launch latency is higher [A]);
  - about 40–100 ms per update;
  - × 270,393 updates ≈ **3–7.5 h**, plus fake-quantisation and VQ assignment kernels.
- **Full batch with CUDA graphs or `torch.compile`:** about 2–5 ms per update, so **roughly 0.2–0.4 h** of compute plus evaluation.
- **For wide arms (f ≥ 2,304)**, the step is not launch-bound. Each update is about 2.3e11 FLOPs through K = 64 GEMMs, plus GELU over [28,672 × 2,368] activations (about 136 MB per BF16 intermediate). Those are memory-bound, so **about 10–20 ms per update, ×1.5 with fake-quantisation or VQ ops** [A].
- **The original design fits the 18 GPU-hour cap** either way: about 0.5–2 h with the fixes and 3–8 h eager.

**The revised design (§F) needs its own budget [C, with timings A]:**

Arms: BF16, INT8, INT4, ternary-1.6, scalar 2b, Hadamard-rotated 2b, E8 and learned VQ. The hypercube "fixed VQ" is dropped if it is equivalent to rotated 2b. Ternary-1.6 runs at f = 2,816 with 9,265,536 FLOPs per position.

| Block | Runs | PFLOP | Updates | Wall time (fixes applied) |
|---|---:|---:|---:|---:|
| A: equal geometry, w = 320, 100 exposures | 24 | 7.04 | 153,600 | 0.2–0.4 h |
| B: equal bytes, 7 arms | 21 | 26.20 | 134,400 | 0.5–0.9 h |
| C: matched compute (INT4, 2b, E8, learned VQ) | 12 | 3.52 | 18,795 | 0.1–0.2 h |
| Fixed-tensors-only control | 3 | 0.34 | 19,200 | under 0.05 h |
| Mandatory tuning (6 configurations × 8 arms, low-load world, 20 exposures) | 48 | 2.71 | 15,360 | about 0.1 h |
| **Subtotal without saturation** | **108** | **39.8** | **341,355** | **about 0.9–1.7 h** |
| Saturation option (a): 1,000 exposures for BF16, INT8, 2b, E8 and learned VQ × 3 seeds | 15 | 152.4 | 960,000 | 2.9–5.7 h |
| **Total with (a)**, +20% for evaluation and checkpoints | **123** | **192.2** | **1,301,355** | **about 4.5–9 h** |

**Verdict on budget:**
- With full-batch execution and CUDA graphs or `torch.compile`, the revised design fits the 18 GPU-hour cap.
- Run eagerly with accumulation of 16 (3–5× slower), it would not.
- If the instrument block's measured time per update projects above 18 h, use **saturation option (b)**: a 4× smaller world with all MLP widths scaled down 4×, which costs about 1/16 of (a). Do not drop the saturation endpoint.
- The instrument block should replace all of these timing assumptions with measured milliseconds per update for each width.

### D.6 The 4 GiB hardware arithmetic [C unless noted]
- **Units:** 4 GiB = 4,294,967,296 B = 4,096 MiB ✔. DML-0001 recorded 4,095.5 total on the RTX 2050 [V]. On a physical card the 4,096 MiB cap therefore cannot be exceeded; the real limit is *usable* memory after driver, context and display.
- **Reference (Codex-verified, rechecked):**
  - text subtotal 26,893,352,960 parameters, 50.09 GiB in BF16;
  - embedding and head 2,542,796,800 (4.737 GiB);
  - 8K state 512 + 144 = 656 MiB;
  - weights-only rate to fit in 4 GiB: 1.278 bits per parameter;
  - after state and the 676 MiB overhead assumption: (4,096 − 656 − 676) MiB × 8,388,608 / 26.893e9 = **0.862 bits per parameter** ✔.
- **Plausible 4 GiB hybrid:**

  | Item | Size |
  |---|---:|
  | Weights: about 12.5B at 2.25 bits | about 3.27 GiB |
  | Embedding and head: 65,536 × 4,096 tied, INT8 | 256 MiB |
  | KV: 8 GQA layers × 2 × 4 × 128 × FP8-stored at 8K | 64 MiB |
  | DeltaNet: 24 layers × 16 × 128² × FP32 | 24 MiB |
  | Runtime and context | 300–550 MiB [A] |

  This is tight. Realistic designs are about 10–12B weight slots at about 2.25 bits, or about 6–7B at 4.25 bits.
- **FP8 on SM 8.6:** Ampere has no FP8 arithmetic. FP8 KV is storage only, converted in-kernel [A, from architecture knowledge]. MXFP4 and FP4 paths (MXCode) are unavailable on SM 8.6, so MXCode's deployment results do not transfer [P/A].
- **Decode bandwidth:**
  - RTX 2050: 112 GB/s nominal [V spec]. At 50–75% efficiency for low-bit GEMV [A], 56–84 GB/s.
  - A dense 3.4 GB read per token gives a **16–25 tokens/s roofline**. 8 tokens/s is **credible for dense C0**, but unproven until low-bit kernels are measured on SM 8.6 at laptop power limits (the 2050 ships in 30–45 W configurations [V spec pages]).
- **Compute headroom:**
  - Assume about 12–24 dense FP16 tensor TFLOP/s for GA107 [A; not verified from an official spec].
  - A 2-bit GEMV does 2 FLOPs per 0.28 B (including scales), about 7 FLOP per byte. The card supplies roughly 100–200 FLOP per byte at its bandwidth.
  - That is **12–25× idle arithmetic** at batch 1, which is the basis for ranking parallel streams above sequential loops.
  - Each sequential pass re-reads the weights, so a 0.62 GB core allows about 16 passes at 8 tokens/s (unchanged from my earlier report).
- **8K prefill:** 2 × 12e9 × 8,192 ≈ 2.0e14 FLOPs, about 10–20 s at 10–20 TFLOP/s effective [A]. S2 has no time-to-first-token (TTFT) limit. I recommend a pre-registered TTFT ceiling of 60 s at 8K so the speed requirement cannot be gamed.
- **Weight reconstruction:** a design that expands weights to BF16 for every decode step reads more bytes than it stores. On a 4 GiB card a fully expanded duplicate cannot be resident anyway. The real risk is **tile-wise decode overhead** making kernels compute-bound below the roofline. Only a microbenchmark can settle this.

### D.7 Statistics for S6 and S7 [C/A]
Assume binomial item variance, the reference near its published level and Domovoi near parity. The per-category standard errors of the chance-adjusted ratio r_j are then roughly:

| Category | Benchmark | SE of r_j |
|---|---|---:|
| General knowledge | MMLU-Pro (about 12k items) | 0.7% |
| Science | GPQA Diamond (198 items) | 3.4% |
| Maths | MATH-500 | 1.8% |
| Coding | Fresh LiveCodeBench (about 150 items, assumed) | 3.5% |
| Instruction following | IFBench (about 300 items, assumed) | 3% |
| Long context | RULER 8K | 1% |

- **Composite:** DCI SE ≈ √Σwⱼ²SEⱼ² ≈ **1.1%**, or about 1.5% once the reference's own sampling noise is included.
- **Composite bar:** a one-sided 95% lower bound ≥ 0.95 needs a **point estimate of about 0.975**.
- **Category floors:** simultaneous one-sided floors at 0.80 across 7 endpoints (Bonferroni z ≈ 2.45), with per-category SE of about 4.8% on GPQA and LiveCodeBench, need **point ratios of about 0.92**.
- **These are conservative upper bounds.** The √2 inflation treats the two models' item outcomes as independent. In the paired design, item difficulty is shared and outcomes are positively correlated, which reduces the variance of the ratio. The true required point estimates lie somewhere between the raw targets (0.95 and 0.80) and these figures. Measure the item-level correlation in CAP-0, then compute the actual requirement.
- **Verdict:** defensible, but the documents should state it plainly. A model at true parity passes; a true-0.95 model may not. GPQA's 198 items make the science floor the noisiest gate.

### D.8 Representation options for brief §4 [C, with kernel status A unless stated]
| Format | Bits per weight including metadata (group-64 FP16 scales) | SM 8.6 (RTX 2050) kernel status | Evidence regime | Include in Stage 1? |
|---|---:|---|---|---|
| INT8 scalar | 8.25 | Native INT8 tensor cores; weight-only GEMV is mature | PTQ keeps about 2 bits per parameter [P, 2404.05405] | Yes (control) |
| INT4 scalar | 4.25 | Mature weight-only kernels (Marlin-class SM 8.0+ [A]) | PTQ GPTQ: 0.7 bits per parameter in the tuple study [P] | Yes |
| Scalar 2b, plain or Hadamard-rotated | 2.25 | Custom unpack, simple | LC-QAT: scalar 2-bit QAT degrades severely when PTQ-initialised [P] | Yes (both) |
| Ternary at 1.6 / 2.0 packing | 1.85 / 2.25 | Custom lookup or unpack (BitNet kernels exist [P]) | Native from scratch at 2B/4T [P, 2504.12285] | Yes (1.6 packing) |
| E8 lattice (implicit) | about 2.25 (+<0.1% extension) | Custom decode (QuIP#/E8-QAT kernels exist [P/A]) | PTQ-initialised QAT, expert FFNs only [V] | **Yes (add)** |
| Learned additive VQ (2 × 8-bit / 8 dims) | 2.25 + books | Custom gather and decode | PTQ-initialised (LC-QAT, TVQ) [P] | Yes |
| 2:4 structured sparse INT4 | about 3.1 (2 of 4 kept × (4-bit value + 2-bit index) = 3.0, + scales) | Ampere sparse tensor cores exist on GA10x. Which dtypes cuSPARSELt supports on SM 8.6 is **unverified** [U]. At batch 1 the gain is bytes, not tensor-core FLOPs | Mostly post-training pruning (SparseGPT [P]) | Optional. A good *hardware-native* comparator for HW-0, not essential for Stage 1 |
| 2:4 sparse INT8 | about 5.1 | As above | As above | No (dominated by INT4 on bytes) |
| Low-rank factors, rank r at b bits | b·r(m+n)/(mn); for example 4,096², r = 512, INT4 ≈ 1.0 | Two dense GEMVs, so mature | Post-training truncation degrades at high compression [P, SVD-LLM] | No for knowledge. Low rank discards rare directions, which is where long-tail facts live [S] |
| Entropy-coded (Huffman/ANS) | Below the nominal rate | Needs random-access blocks; decode is costly at batch 1 | Deep Compression [P] | No (kernel risk) |

### D.9 TinyStories results (brief §9) [V/C]
- **Training memory against inference memory:** the 406.33 MiB B08 − R02×4 peak difference is training memory. Parameter-state accounting predicts 353.19 MiB (+44.15 MiB if BF16 casts are resident). The residual cannot be attributed to activations without allocator traces, as Codex correctly notes. The equal-precision *inference* weight difference is 44.15 MiB in BF16. **No inference peak was ever measured.**
- **Equal tokens against equal FLOPs:** at equal tokens, R02×4 closes 0.286 of the B02→B08 gap and trails B08 by 0.143 nats. My earlier "equal-FLOP" comparison used checkpoints taken mid-way through cosine schedules with different horizons, so it is **tentative**; I accept Codex's caveat.
- **Parameter sharing against novelty:** tied depth is established prior art (Universal Transformers, ALBERT, Relaxed Recursive Transformers). R02×4 is a baseline, not an architecture contribution.
- **Short context against 8K:** everything was trained at 512 positions with a learned 512-row table. Nothing bears on 8K.
- **Single seed against reproducible:** one paired seed, with document-bootstrap intervals conditional on that seed. The B02/R02×4 crossover (between 10M and 25M tokens) is an observation, not an established effect.
- **What is legitimately established:**
  - a deterministic, recoverable, well-audited training and evaluation pipeline on the RTX 5080;
  - an equal-token tying trade consistent in direction with the published iso-depth law;
  - the 58 TFLOP/s effective throughput reference at about 34M parameters.
- **No further standalone recurrence research is justified** (M-9).

---

## E. Prior art and novelty

### E.1 Established, verified in this session
- **Scalar QAT and native low bit:** BitNet 2310.11453, b1.58 2402.17764, 2B4T 2504.12285, Spectra 2407.12327, Scaling Laws for Precision 2411.04330.
- **Vector and lattice PTQ and QAT:** AQLM 2401.06118, QuIP# 2402.04396, QTIP 2406.11235, **LC-QAT 2606.10531** (2-bit VQ-QAT from PTQ initialisation, reporting that scalar quantisation degrades severely at 2 bits), **E8-QAT 2610.27699** (implicit E8 codebook, 2-bit QAT, reports parity with full precision on Qwen3.6-35B-A3B), **TVQ 2610.41822** (STE-trainable VQ), **MXCode 2611.00403** (2-bit VQ with MXFP4 reconstruction), DKM 2108.12659.
- **Sequence mixing:** Mamba-2 2405.21060, Gated DeltaNet 2412.06464. The reference itself uses DeltaNet.
- **Depth and compute:** iso-depth law 2604.21106, Ouro 2510.25741, MoD 2404.02258, MoR 2507.10524, **ParScale 2505.10475**, MTP 2404.19737.
- **Memory:** PKM 1907.05242, Memory Layers at Scale 2412.09764, Engram 2601.07372, Titans 2501.00663, ATLAS 2505.23735.
- **Modularity:** NMN 1511.02799, DreamCoder 2006.08381, Mittal et al. 2206.02713, LoraHub 2307.13269.

### E.2 What is not new
Every component in C0–C3 and the earlier A–D proposals: native ternary or 2-bit training, learned or lattice codebooks under QAT, hybrid state/attention, looping, sparse memory, routing, consolidation, and routed adapters.

### E.3 Where a defensible contribution could come from
These are specific technical distinctions, not claims of novelty.
1. **A measured knowledge rate-distortion curve for native from-scratch formats:** knowledge bits per *resident* bit against bits per weight, for scalar, ternary, E8 and learned codes, at saturation and at finite exposure, with the structured-world and two-hop controls. To my knowledge no published study measures this. The QAT papers above start from pretrained or PTQ initialisation [V abstracts]. This is a **measurement contribution**.
2. **A bandwidth-aware, weight-stationary compute design for batch-1 Ampere-class devices:** parallel streams and MTP verification sized to the device's FLOP-per-byte headroom, possibly with streams spent on latent hypotheses rather than input transformations. It is new only if it beats ParScale and MTP baselines at equal bytes and measured latency [S].
3. **A joint byte, bytes-read and latency frontier on a physical 4 GiB card**, open and reproducible. This is an engineering and science contribution, not an architecture.

### E.4 Correcting my earlier report
- I accept Codex's corrections: 65 IDs, not 66; the C3 MiB figure; and "test score", not "test split".
- I also accept two more corrections:
  - "4× decode compute" for R02×4 applies to block compute only, because the shared embedding and head are not multiplied.
  - My equal-FLOP "B02 wins" reading compares mid-schedule checkpoints with different cosine horizons, so it is tentative.
- My ranking of C2 first did not account for the 2026 VQ-QAT literature that Codex found (LC-QAT, E8-QAT, TVQ). I did not find those papers then. Their text supports vector-over-scalar (PTQ-initialised) and fixed-lattice-over-learned.
- My statement that "Engram's gains are against an MoE" stands and is consistent with Codex.

---

## F. First-experiment verdict: **REVISE BEFORE APPROVAL**

The question is good, the arithmetic is right and the cost is small. As written, though, the study (i) may not contain a genuine fixed vector code, (ii) measures memorisation speed rather than capacity, (iii) allows untuned low-bit baselines and (iv) is gated on engineering that does not inform the decision.

**Required changes:**
1. **Split the experiments.** Make the instrument block DRP-0001: BF16 and INT8 at width 320, the low-load world, the oracle/uniform/permuted checks, and the K\* calibration check. Add timing calibration (wall time per update) and a seed-variance estimate. The main study is a separate DRP that uses the instrument's measured variance and runtime.
2. **Specify the VQ tables exactly.** If both stages share a rotation, relabel the arm "rotated scalar 2b". Add an **E8 lattice arm** and a **Hadamard-rotated scalar 2b** arm.
3. **Pack ternary at 1.6 bits**, or report both packings.
4. **Add a saturation exposure point** (about 1,000 exposures) for BF16, INT8, the best scalar 2-bit-class arm, E8 and learned VQ, or a correspondingly reduced world. Pre-register capacity (at saturation) and sample efficiency (100 exposures) as two separate endpoints.
5. **Make equal hyperparameter tuning mandatory** for every arm on the calibration world, and report the trial counts.
6. **Add a fixed-tensors-only control** and report ΔK per coded-MLP byte.
7. **Use K\*** (temperature-maximised) alongside K. Treat D > 1 as a hard leakage alarm. Generate labels with a cryptographic pseudo-random function.
8. **Add structured-world and two-hop probes** to every arm (§B, M-4). Advancement requires no loss on them.
9. **Replace the GPU packed-decoder gate** with a CPU bit-exact pack/unpack and logit-parity fixture. Move kernel feasibility to a separate microbenchmark study.
10. **Remove gradient accumulation** and use CUDA graphs or `torch.compile`. The revised design is about 123 runs, about 192 PFLOP and about 1.3M updates, an estimated 4.5–9 GPU-hours with these fixes [C/A, §D.5]. Keep the 18 GPU-hour cap. If the instrument block's measured time per update projects above it, switch saturation to the reduced-world option (b) rather than dropping it.
11. **State in advance what success would mean.** A 1.2× density gain changes 4 GiB knowledge capacity from about 7 to about 8.5 Gbit at the earlier extrapolation [C]. That is a mechanism signal, not a path to parity.
12. **Keep** the proposal's existing three-seed screen, eight-seed confirmation, paired bootstrap and "drop if fixed equals learned" rules.

---

## G. Revised experimental sequence (priority order)

Each step needs its own pre-registration and approval under the protocol.

| # | Study | Training? | Controls | Decision gate |
|---:|---|---|---|---|
| **0** | **Specification freeze:** S1–S7 with the §H revisions; the reference device chosen and physically available | No | — | Owner sign-off |
| **1** | **HW-0: physical 4 GiB device characterisation.** Idle VRAM; CUDA context with lazy loading; usable maximum allocation; sysmem fallback disabled and verified; sustained copy bandwidth; existing low-bit GEMV kernels (INT4 Marlin-class, E8/QuIP#-style, ternary lookup) at representative shapes on **random weights** | No (benchmark, needs approval) | The same suite on the RTX 5080 | Measured usable memory and per-format tokens/s rooflines replace every [A] in §D.6. **If no 2-bit-class kernel reaches about 50% of bandwidth at batch 1, the C2/E8 deployment line pauses.** |
| **2** | **CAP-0: deployment frontier and capability ladder.** Existing models: Qwen3.5 0.8B/2B/4B/9B and BitNet b1.58 2B4T, natively or compressed after training to fit, on the RTX 2050 where they fit. Matched-budget battery including the long-tail factual guardrail | No (evaluation, needs approval; downloads need licence review) | Same harness, same budget | (a) The real gap to Gemma-3-27B-IT-level and Qwen-level ratios; (b) whether knowledge or reasoning items saturate first; (c) the first measured "strongest native-fit baseline" for M1/M2. The 27B references need ≥80 GB hardware, as a separate approval. |
| **3** | **DRP: C0/C2 instrument block** (§F.1) | Yes, under 1 GPU-hour | Oracle, uniform and permuted | Estimator calibrated; floors pass; seed SD measured |
| **4** | **Revised C0/C2 main screen** (§F) | Yes, under 18 GPU-hours | Scalar, ternary 1.6, rotated 2b, E8, learned | Capacity and sample-efficiency gates; no structured or two-hop regression |
| **5** | **Natural-text transfer** at 30M and 100M parameters, **20 tokens per parameter** (needs a 2B-token pack) | Yes | Best scalar / E8 / surviving code at equal resident bytes | ≥0.02 nats/target at both sizes; long-tail recall not worse |
| **6** | **Weight-stationary parallelism:** P ∈ {1, 2, 4, 8} streams and/or MTP heads on the winning format at equal resident bytes | Yes | P = 1 control; equal total bytes including KV × P | Lower NLL at equal bytes, **and** measured batch-1 tokens/s within 20% of P = 1 on the 5080, then the 2050 |
| **7** | **C1 four-arm ablation**, only if CAP-0 or Stage 5 shows a multi-hop deficit that more storage does not fix | Yes | Memory-only, loop-only, single-query | ≥5 points of held-out multi-hop at equal bytes **and** equal measured bytes read |
| **8** | **Fixed post-training recipe** (SFT, then RL or distillation with a fixed teacher) applied identically to C0 and any survivor at about 0.5–1B | Yes (likely external compute) | Teacher and no-teacher 2×2 | Architecture effect survives identical post-training |
| **9** | **Scale confirmation and milestone qualification** (M1/M2, then Gemma-parity milestone, then Qwen moonshot) | External | S1–S7 | As specified |

**Premature now:** C1, consolidation, adaptive routing, C3, Stage 6 of the roadmap (1B/3B scale), and teacher factorials before a post-training recipe exists.

**Cheapest invalidating tests:** HW-0 (kernel throughput) and CAP-0 (the size of the gap). Either could redirect the programme before any architecture training.

---

## H. Capability and hardware qualification: S1–S7

| Item | Assessment | Recommended correction |
|---|---|---|
| **S1 reference GPU** | Correct choice of a physical device ✔. Laptop RTX 2050s vary in TGP (30–45 W), clocks and thermals, and run on Optimus/WDDM. | Name the exact laptop SKU and TGP, the VBIOS, and power and clock locks. Prefer headless Linux, with Windows as a secondary. Keep a spare unit or name an approved substitute. The 5080 is a development screen only, because SM 12.0 has FP8/FP4 features absent on SM 8.6 and kernels written there may not port. |
| **S2 memory and speed** | Strong: device-wide, three cold processes, all byte domains ✔. Misses sysmem fallback. On a physical card the cap cannot be exceeded; what matters is completion without fallback. | Add the "Prefer No Sysmem Fallback" requirement, a shared-memory counter of 0, and PCIe traffic audits (M-10). Add a TTFT ceiling (proposed: 60 s at 8K). Keep NVML polling as supporting evidence and run completion as the proof. |
| **S3 reasoning budget** | Fair in principle. The 1,536-token reasoning cap handicaps a model that thinks by default, so the denominator becomes a truncation artefact (M-7). | The reference uses its **best compliant configuration** within the 8,192-token window (off or budgeted thinking, chosen on a dev split). Report both models' score-against-budget curves. Keep Qwen-tokenizer accounting. |
| **S4 context** | Good tokenizer-neutral design ✔. Conflicts with S7 (M-8). | Make relative-to-Qwen the primary gate, keep the ≤10-point 2K→8K drop, and make the absolute 80% a diagnostic. Add the natural delayed-evidence set as a gate, not only a diagnostic, because RULER is synthetic. |
| **S5 independent scores** | Strong ✔. LiveCodeBench `release_v6` ends April 2025 [V, spec] and is likely contaminated for 2026 models. GPQA and MMLU-Pro are public and contamination-prone. | Require fresh coding items authored or released after every cutoff, and **teacher-output overlap checks**. Record Gemma's two LiveCodeBench figures (29.7 / 39.0) with their windows (M-11). |
| **S6 statistics** | Correct method ✔. | State its practical meaning (§D.7): point estimates of about 0.975 composite and about 0.92 per category are required. Pre-register GPQA item expansion, for example an additional sealed science set, or accept that the science category stays unresolved. |
| **S7 floors** | 0.80 floors plus a factual guardrail ✔. Each category rests on a single benchmark. | Add a second benchmark per category as a diagnostic. Keep the long-tail guardrail. It is the gate most likely to fail and the most honest one. |
| **Reference choice** | Qwen3.8-27B is pinned and verified [V]. It is agentic and multimodal, and its published strengths (Terminal Bench, SWE-bench) fall outside the battery. Gemma 3 27B IT `005ad340…` is verified [V] and gated. | Keep Qwen as the primary. **Adopt matched-budget Gemma-3-27B-IT parity as an explicit intermediate milestone** (between M3 and M4), because it is a 27B model at a much more reachable level. |
| **Composite hiding deficits** | The floors prevent most hiding. Missing domains (multilingual, agentic, safety) mean "broad" applies only to this battery. | Phrase every claim as "on the frozen Domovoi battery under the matched budget". |

---

## I. Main risks and overlooked opportunities

**Scientific risks**
- The target may be unreachable for any 4 GiB model if the benchmarks are knowledge-limited (U-2). CAP-0 is the cheapest test.
- Synthetic-density wins that do not transfer (M-4).
- Undertrained natural-text screens (M-5).
- No post-training stage (M-6).
- Composite noise on 198-item GPQA (D.7).

**Engineering risks**
- Low-bit kernel throughput on SM 8.6.
- WDDM sysmem fallback (M-10).
- Laptop power and thermal variance.
- Kernels developed on SM 12.0 that use unavailable instructions.
- Cold-load staging that duplicates weights.
- Untracked-file and dependency gaps in the gate (I.3).

**Overlooked opportunities**
1. **Weight-stationary parallelism** (ParScale and MTP) sized to the batch-1 FLOP-per-byte headroom (§C, rank 2).
2. **Existing-model frontier measurement** on the real device before building anything (CAP-0).
3. **Vocabulary economics.** Tied INT8 64K × 4,096 is 256 MiB, about 6% of the budget. Vocabulary size and tying are cheap, measurable levers.
4. **Recipe-first baselines.** Most of the 2–3× spread between the two 27B references comes from the training recipe.
5. **Gemma-parity milestone**, which gives the project a meaningful and achievable scientific target.

**I.3 Protocol adequacy (§10 of the brief)**
- **Strengths:** the plan is pinned at approval; the command must match the plan exactly; the implementation SHA must be published; input hashes are checked; history is append-only and single-active; prior publication is required before the next experiment [V, `protocol.py`, `launch.py`].

| Gap | Remedy |
|---|---|
| (1) Any pusher can set APPROVED; there is no approver identity | GitHub branch protection with a required human review (CODEOWNERS) on `experiments/protocol/**`, plus signed commits |
| (2) The canonical ref is unprotected; a force-push could replace history consistently [V, reconciliation] | Protect the ref and periodically anchor the ledger head hash in a second location |
| (3) `--untracked-files=no` lets untracked modules or configs affect a run | Run from a fresh clone, or include untracked files outside the declared output directories |
| (4) The environment is not gated | Pin a lockfile hash and the driver/CUDA versions in `inputs`; compare against a runtime capture |
| (5) No analysis code is pinned, so endpoint choice can drift after the data are seen | Pin the evaluation and analysis scripts and the primary-endpoint definition in the plan |
| (6) "Verification records" and external harnesses (lm-eval, vLLM) can produce informal GPU evidence outside the gate | Define that any GPU measurement whose numbers are cited is an experiment; wrap external harnesses in the gate |
| (7) Extended-v1 was announced about 87 s after launch (historical) | Already addressed by the gate for future work |
| (8) Crash recovery lets a reviewer publish an "audited terminal correction" by hand when the receipt is lost [V, Labs `docs/EXPERIMENT_PROTOCOL.md`]. `verify_history` checks identity fields, not which outcome is chosen | Restrict manual corrections to INTERRUPTED or FAILED. Never allow a hand-written COMPLETED without the original receipt |

---

## J. Recommended next steps

**Human owner**
1. Decide S1–S7 with the §H corrections. Name and acquire the physical RTX 2050 SKU.
2. Decide whether the public goal is "a new architecture" or "maximum capability per GiB on a physical 4 GiB device". I recommend the latter, with Gemma-parity and Qwen-parity milestones.
3. Separately authorise HW-0 and CAP-0, including the licence and download review for the third-party models.
4. Enable branch protection and required human review on the ledger.

**Codex (no experiment approval needed)**
1. Revise the C0/C2 proposal per §F. Split out DRP-0001 (instrument).
2. Implement and CPU-test:
   - the cryptographic-PRF world generator;
   - the K/K\* estimator, with oracle and permuted fixtures;
   - exact byte accounting;
   - bit-exact pack/unpack for all formats, including E8 and 1.6-bit ternary;
   - structured-world and two-hop probe generators.
3. Draft the HW-0 and CAP-0 plans. Plan the 2B-token FineWeb-Edu pack; that is data preparation, so check whether it counts as an experiment under the protocol.
4. Fix the protocol gaps I.3 (3)–(6). Correct M-11 in the spec.

**Claude**
1. Review the revised DRP-0001, HW-0 and CAP-0 plans before approval, checking the arithmetic, controls and gates.
2. Review the E8 and ternary 1.6-bit packing specifications.
3. After each published outcome, write an independent interpretation before the next plan is drafted.

**Needs explicit approval:** every GPU run, microbenchmark, model evaluation, model download, dataset download and teacher run.

---

## K. Sources and verification record

**Repository documents, all read in full at the pinned SHAs [V]**

Domovoi-research `6d6a90d`:
- `architecture/RESEARCH-RECONCILIATION-2026-11-24.md`, `AI-EFFICIENCY-EVIDENCE-MATRIX.md`, `ARCHITECTURE-CANDIDATE-ASSESSMENT.md`, `PHASED-ARCHITECTURE-RESEARCH-ROADMAP.md`, `NEXT-ARCHITECTURE-PROPOSAL.md`
- `experiments/proposals/C0-C2-DEPLOYED-BYTE-STUDY.md`, `experiments/protocol/state.json` (empty ledger)
- `methodology/27B-4GIB-QUALIFICATION-SPECIFICATION.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `reviews/2026-11-24-independent-architecture-research.md` (byte-identical to my local copy, SHA-256 `77f35370…`), `reviews/2026-11-24-independent-research-verification.md`, `reviews/CLAUDE-ROADMAP-REVIEW-PACKAGE.md`

Domovoi-Labs `8252fbf`:
- `domovoi/experiments/protocol.py`, `domovoi/experiments/launch.py` (`check_command`), `scripts/experiment_protocol.py`

The TinyStories records, `memory-complexity.md` and the pilot plan were read in full at earlier SHAs (`91616af`/`d914684`) earlier in this session. The file list shows them present at `6d6a90d`.

**Other checks:**
- `reviews/independent-report-reference-audit.json`: I checked its structure. It covers 65 IDs with abstract-metadata resolution, matching my recount [V].
- Labs `docs/EXPERIMENT_PROTOCOL.md`: I read the crash-recovery section [V].
- **Pin drift:** the GitHub compare of `91616af…6d6a90d` lists only README, ARCHIVE-INDEX and newly added review and architecture files as changed. The TinyStories, methodology, memory-complexity and pilot records I cite are unchanged [V].

**Not read in full [U for content]:** `mathematical-foundations.md`, `experiments/CPU-AUDITS.md`.

**Paper text read this session [V]:**
- E8-QAT 2610.27699: Qwen3.6-35B-A3B; 32.34B expert-FFN weights (92.1%) at 2.03 bits, the other 2.77B in BF16; PTQ (LDLQ with block-Hadamard) initialisation; four tasks; throughput compared at each representation's largest batch.
- LC-QAT 2606.10531: Qwen3 1.7B/8B and LLaMA-3 3B/8B; about 4B FineWeb tokens; PTQ initialisation; compared with ParetoQ scalar QAT and BitNet 2B4T.
- ParScale 2505.10475: 0.5–4.7B models from scratch on 42B tokens; P ≤ 8; the 22× / 6× figures are at batch 1 for a 1.6B model; KV cache grows P×.

**Publisher and model metadata [V]**
- Qwen3.8-27B `37561816…`: config, card and benchmark table, from my earlier fetch in this session.
- Gemma 3 27B IT `005ad3404e59d6023443cb575daa05336842228a`: HF API (lastModified 2025-03-21, gated "manual"); arXiv 2503.19786 HTML, Tables 6, 15 and 18.

**Primary research, abstract or text verified this session [P/V]**
- Knowledge capacity: 2404.05405, paper text (Results 4–6 and 8, GPTQ post-training method, bit-complexity definition).
- Vector/lattice codes and QAT: 2606.10531, 2610.27699, 2610.41822 and 2611.00403 (abstracts).
- Parallel streams and precision: 2505.10475, 2411.04330 and 2504.12285 (abstracts).
- Multi-token prediction: 2404.19737 (abstract).
- Looping and memory: 2604.21106, 2510.25741 and 2601.07372, from the earlier session.

**Titles verified only:** 2210.17323, 2306.00978, 1510.00149, 1812.07520, 2403.07378, 2301.00774, 1608.05859, 2404.06654, 2404.02258, 2503.19786, plus the 65 IDs in my earlier report.

**Hardware**
- RTX 2050 (4 GB GDDR6, 64-bit, up to 112 GB/s): NVIDIA comparison page and third-party specification pages [V spec / U sustained].
- Sysmem fallback: NVIDIA developer forum threads and third-party guides. No official NVIDIA release note was found [P/U].

**Independently calculated [C]:** every table in §D. Inputs are stated inline.

**Assumed [A]:** CUDA context overhead (250–550 MiB); GA107 tensor throughput (about 12–24 TFLOP/s); 50–75% low-bit GEMV bandwidth efficiency; launch latency (8–20 µs); item counts for fresh LiveCodeBench and IFBench; Qwen RULER at 8K (about 95%).

**Speculative [S]:** the expected C2 effect size (≤1.5×); the latent-hypothesis variant of parallel streams; reachability of the Gemma-parity milestone.

**Unable to verify [U]:** the RTX 2050's sustained bandwidth and context overhead; ParScale behaviour at low bit width and 8K; whether 2-bit QAT results carry over from PTQ initialisation to from-scratch training; the reference's used-knowledge content.
