# Domovoi: independent architecture research report

**Author:** Claude (independent reviewer). **Date:** 24 November 2026.
**Scope:** This is research only. No repository was modified, no model was trained and no experiment was run.

**Sources inspected:**
- `Domovoi-Labs@d914684`: `RESEARCH.md`, `EVALS.md`, `docs/DOMOVOI_PILOT_PLAN.md`.
- `Domovoi-research@research/real-data-calibration`:
  - `experiments/TINYSTORIES-EXTENDED-2026-11-24.md`, `TINYSTORIES-CALIBRATION-2026-11-24.md` (commit `eecec3a`) and `TINYSTORIES-CALIBRATION-ANALYSIS.md`
  - `experiments/ARCHIVE-INDEX.md` and `experiments/runs/DML-0010.md`
  - `findings/architecture-hypotheses.md` and `findings/ledger.md`
  - `methodology/experimental-controls.md`
  - `architecture/memory-complexity.md` and `datasets/preparation-and-tokenization.md`
  - the packed-dataset `meta.json` files and `experiments/data/archive/extended-v1/{campaign.json, launch-extended-v1.py}`
- The Qwen3.8-27B Hugging Face card and `config.json` at revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`.
- 66 arXiv papers. Each title and first author was checked against arXiv in this session (§3).

### Evidence labels used throughout

| Label | Meaning |
|---|---|
| **[M]** | Measured. A number in a published Domovoi record or in an artifact fetched in this session. |
| **[D]** | Derived. My arithmetic from [M], [L] or [A] inputs. The inputs are stated. |
| **[L]** | Literature-supported. Stated in a cited paper. I verified the abstract or the title, not the full experiments, unless noted. |
| **[A]** | Assumed. An engineering estimate that must be measured before anyone relies on it. |
| **[S]** | Speculative. A hypothesis with no direct evidence. |
| **[R]** | Recalled from the literature but not verified against an abstract in this session. |

---

## 0. Executive summary

1. **Weight bytes are the binding constraint, not context memory.** The reference, Qwen3.8-27B, is already a hybrid of linear attention and attention: 48 Gated DeltaNet layers and 16 gated-attention layers with 4 KV heads [M, config]. Its whole 8K sequence state is about 656 MiB [D]. Its text weights are about 26.9B parameters, about 50 GiB in BF16 [D]. They would need 1.28 bits per parameter to fit in 4,096 MiB before any state or runtime memory [D]. Efficient sequence mixing is therefore not a Domovoi differentiator, because the reference already has it.

2. **Matching 27B-class factual knowledge (objective B) is very unlikely under current evidence.** In my main budget, about 3,380 MiB is available for weights. That is 28.4 Gbit [D].
   - The best measured knowledge density is about 2 bits of knowledge per parameter, and that holds down to int8 [L, Allen-Zhu & Li 2404.05405]. That is about 0.25 bits of knowledge per stored bit, so the budget holds roughly 7 Gbit of knowledge [D].
   - A 26.9B model at 2 bits per parameter has a capacity of about 54 Gbit [D].
   - Reaching even half of that would need about 0.95 bits of knowledge per stored bit, close to the information-theoretic ceiling of 1.0 [D].
   - The main uncertainty is that the 27B model's *used* knowledge, and the knowledge the benchmarks actually need, may be far below its capacity (§11).

3. **Trading compute for bytes is real but weak, and the reference GPU caps it.**
   - **Loop value:** looping a block r times is worth only r^0.46 unique blocks in validation loss [L, 2604.21106]. Looping also does not raise knowledge capacity [L, 2510.25741].
   - **Bandwidth cap on a 4 GiB card:** the historic reference GPU is the RTX 2050, with 112 GB/s of memory bandwidth [L, vendor spec]. Without speculative decoding, at 8 tokens/s it can stream a 3.3 GiB weight set only about 2.9 times per token [D]. That is worth about 1.6× in loss-equivalent parameters [D].
   - **Multi-token prediction:** self-speculative decoding with multi-token prediction (MTP) spreads each weight pass across several accepted tokens. The reference has an MTP layer [M]. Inference up to 3× faster has been reported [L, 2404.19737]. That would raise the budget to roughly 4–9 passes [D/A], depending on the task-dependent acceptance rate.
   - **On the RTX 5080:** with the same 4 GiB cap and 960 GB/s, about 25 passes would be possible without MTP [D/A].

   **The choice of reference device changes the feasible design space by about 8×. It is the most important open item in the specification.**

4. **The R02×4 results do not support continuing recurrence as a primary direction.**
   - At matched applied compute, the untied B08 is 0.143 nats better [M].
   - At matched *training* FLOPs, the shallow B02 trained on more tokens beats both deep arms at every comparable checkpoint [D].
   - The 18% training-memory saving is almost entirely parameter-state bytes [D]. At inference in BF16, the saving is 44 MiB [D].
   - Under the project's own draft pilot criteria, the extended result reads as **inconclusive**, not as "advance". The gap closure is 0.286, against an advance threshold of 0.50 [D].

5. **The pre-registration question resolves differently from what I said in my previous message, so I am correcting it.**
   - The extended campaign was announced as a separate, "exploratory, single-seed" run. It explicitly did not freeze any decision thresholds.
   - The pilot plan was never frozen (P7 was never carried out). **There was no pilot-plan violation, and I retract item F of my earlier briefing review.**
   - What is missing is different. The announcement was committed about 90 seconds *after* the launch, by git timestamps [D]. It contained no predicted outcome and no decision rule.
   - I found no lifecycle-enforcement files (CI, hooks or policy) on either audited branch [M].

6. **Playbooks are a weak primary architecture for this target.** Once they are made neural, they become sparse, low-rank expert routing over latent iterations, which is mostly prior art (§7). They cannot store facts more cheaply than other parameters. Their plausible benefit is procedural reuse, which dense models already get partly through superposition. I rank them last.

7. **Recommended first investigation: "bytes-first".** First measure knowledge and skill per stored bit across native representations: int8, int4, ternary and learned vector codes (candidate C2). Then measure reasoning per byte and per byte read for a factored memory and looped-core design (C1), using the C2 representation.
   - The first decisive experiment fits on the RTX 5080 in days.
   - If no native representation beats about 0.25 bits of knowledge per stored bit, **C2 is falsified**. That alone would *not* show objective B is infeasible.
   - B's feasibility turns on how much knowledge the reference actually *uses* for the benchmarks. The evaluation-only stage 0b (§9.2) measures that by comparing existing Qwen checkpoints of different sizes.

---

## 1. Assessment of the existing Domovoi results

### 1.1 What was measured
These are full FP32 validation NLLs from `TINYSTORIES-EXTENDED-2026-11-24.md`, with one seed (1234), TinyStories v1 and a 4,096-token BPE vocabulary [M].

| Step | Tokens | B02 | B08 | R02×4 |
|---:|---:|---:|---:|---:|
| 610 | 9.99M | 2.6053 | 2.5076 | 2.6528 |
| 1,526 | 25.0M | 2.1471 | 1.9706 | 2.1125 |
| 3,052 | 50.0M | 1.9023 | 1.7174 | 1.8562 |
| 6,104 | 100.0M | 1.7185 | 1.5181 | 1.6612 |

### 1.2 Theoretical versus measured training-memory saving (B08 minus R02×4)
| Term | Bytes per extra parameter | MiB for Δ = 23,146,368 parameters | Label |
|---|---:|---:|---|
| FP32 weights + gradients + AdamW m and v | 16 | 353.19 | D (the archive reports the same figure) |
| BF16 autocast weight copies, if resident at peak | 2 | 44.15 | D/A |
| **Predicted from parameter state alone** | 18 | **397.34** | D |
| Measured peak-allocated difference | | **406.33** | M |
| Residual for activations, workspaces and noise | | 8.99 (2.2%) | D |

**Conclusion [D]:** The measured saving matches what parameter state predicts to within about 9 MiB. Activation memory is the same for both arms, because both have 8 applications by construction. Tying does not reduce activation memory, and it gave no saving beyond the expected one. The archive itself states this as a "calculable prediction" (H-01). The briefing's framing of it as a finding overstates it.

### 1.3 Expected inference-memory saving at equal precision
Weights differ by Δ × bytes per weight, and the KV cache is identical at equal applied depth (A = 8): 2·A·S·d·b.

| Precision | Weight saving | KV, both arms, 512 positions | KV, both arms, 8,192 positions | Logical saving at 8K |
|---|---:|---:|---:|---:|
| FP32 | 88.30 MiB | 20.0 MiB | 320 MiB | 88.3 / 449.0 = 19.7% |
| BF16 | 44.15 MiB | 10.0 MiB | 160 MiB | 44.2 / 224.5 = 19.7% |
| INT4 (no scales) | 11.04 MiB | | | |

All rows are [D]. The 8K columns are hypothetical, because both models were trained with a 512-row learned position table. Adding about 350 MiB of CUDA context [A] cuts the BF16 device-wide saving at 8K to about 7%. No inference peak has ever been measured for these checkpoints [M, ARCHIVE-INDEX].

### 1.4 Matched applied computation: R02×4 versus B08
- At equal tokens and equal modelled matmul FLOPs, R02×4 is **0.1432 nats (9.43%) worse** [M]. B08 leads at all four checkpoints [M].
- **Crude estimate of the loop exponent [D/S]:** assume validation NLL is linear in log(block parameters) between B02 (7.72M block parameters) and B08 (30.86M).
  - R02×4's 28.6% gap closure then corresponds to an effective 7.72M × 4^0.286 ≈ 11.5M block parameters.
  - That implies r^φ ≈ 1.49, so φ ≈ 0.29.
  - The published iso-depth law measures φ = 0.46 [L, 2604.21106]. That is better, but still far below 1.
  - The two estimates broadly agree: four loops are worth roughly 1.5–1.9 unique blocks, at a 4× compute cost.

### 1.5 Matched unique parameters: R02×4 versus B02, with FLOPs accounted for
The analytic training cost is 0.066 GFLOP per token for B02 and 0.216 GFLOP per token for R02×4 and B08, a ratio of 3.27× [M, pilot-plan accounting].

| Comparison at roughly equal training FLOPs | FLOPs | NLL | Label |
|---|---:|---:|---|
| B02 at 100M tokens | 6.59e15 | **1.7185** | D (FLOPs) / M (NLL) |
| R02×4 at 25M tokens | 5.40e15 | 2.1125 | D/M |
| B08 at 25M tokens | 5.40e15 | 1.9706 | D/M |
| R02×4 at 50M tokens (1.64× B02's FLOPs) | 1.08e16 | 1.8562 | D/M |
| B08 at 50M tokens (1.64× B02's FLOPs) | 1.08e16 | 1.7174 | D/M |

- **Conclusion [D]:** at matched training compute, the shallow model trained on more data wins. B08 needs 1.64× B02's compute just to tie it.
- **Caveat:** the intermediate checkpoints sit mid-way through a cosine schedule, at a higher learning rate than a schedule that ended there. This biases the comparison against the deep arms. The size of the gap (about 0.14–0.39 nats) is larger than that bias is likely to be, but this is not proven.
- **Wall clock agrees:** 135 s for B02 against 348 s for R02×4 [M].
- **What this means:** all three models are heavily under-trained, at 9.4 tokens per parameter or fewer. In that regime, data beats depth. It says nothing about inference capability per byte, which is a different question.

### 1.6 Does the longer training establish anything beyond TinyStories?
Very little.
- TinyStories is a synthetic corpus with a narrow vocabulary and templated text.
- There is one seed, no test split, a 4,096-token vocabulary, width 640 and context 512, and no benchmark of any kind.

**What it does establish [M]:**
- the B02/R02×4 ranking reverses between 10M and 25M tokens on this corpus;
- the pipeline is reproducible and deterministic enough to measure differences of a few hundredths of a nat.

**What it does not establish:**
- anything about knowledge, reasoning, long context, behaviour at 8K, inference memory, throughput, or how these trends scale.

### 1.7 What the experiments imply for capability per GiB
- At equal stored bytes, R02×4 has lower NLL than B02 (1.661 against 1.719). It costs 4× the decode compute and 4× the KV cache [D].
- That is not capability per GiB. The right control is an untied model at the *same total measured inference memory at 8K*, for example a wider B02 or a B03. The pilot plan itself says so in §12.
- The experiments therefore leave capability per GiB unmeasured. Combined with the iso-depth law, they suggest that full-model looping is a poor exchange rate on a bandwidth-limited 4 GiB device (§4.3).

### 1.8 Pilot plan versus the published extended protocol
| Item | Pilot plan (`DOMOVOI_PILOT_PLAN.md`, "PLAN ONLY") | Extended-v1 as run |
|---|---|---|
| Corpus | FineWeb-Edu document subset | TinyStories v1 |
| Seeds | 1234, 1235, 1236 (escalating to 5 if inconclusive) | 1234 only |
| Primary metric | Held-out **test** NLL | Validation NLL; the test split is sealed |
| Decision criteria | §11 A1–A4 and B1–B2, frozen only at P7 | None declared |
| Inference profile (A4) | Required | Not done |
| Learning rate and warmup | 6e-4 with 200 warmup steps | Same |
| Status | Never frozen. Methodology says "thresholds have not been frozen" | Declared "exploratory, single-seed" |

**Determination:**
- **It was registered separately.** The calibration record (research commit `eecec3a`) announces the extended run as exploratory and single-seed. It reserves DML-0008 to 0010 and states that it "does not freeze multi-seed or 8K research decision thresholds".
- **No violation occurred.** The pilot plan was never frozen.
- **Timing [D]:**
  - `eecec3a` is timestamped 2026-11-23T21:38:12Z, which is 08:08:12 ACDT on 24 November (UTC+10:30).
  - The extended record gives the launch as about 08:06:45 ACDT.
  - The announcement was therefore committed about 87 seconds after launch.
  - These are commit timestamps. Local clock skew and the actual push time are unknown, so the most I can claim is that announcement and launch were concurrent.

**Missing evidence:**
1. No pre-execution prediction or decision rule for extended-v1.
2. No recorded owner authorisation in the archive. It may exist outside the archive.
3. No recorded rationale for substituting TinyStories for FineWeb-Edu. The FineWeb-Edu pack holds only about 5.04M training tokens [M, `meta.json`], which is the likely reason.
4. No inference profile.
5. No test-split score.

**If the pilot's §11 rules were applied to this one seed [D]:**
- the control is valid (B08 beats B02 by 0.200, above the 0.05 requirement);
- A1 passes (d = 0.057, above 0.02);
- A2 fails (gap closure 0.286, below 0.50);
- B2 is not triggered, because it needs a gap closure below 0.25.

The outcome would be **inconclusive, add seeds**. The extended result therefore does not meet the project's own draft bar for advancing recurrence.

**Lifecycle enforcement [M]:** neither audited branch contains `.github` workflows, hooks or policy files. `domovoi/experiments/registry.py` exists, but I did not find enforcement of publish-before-execute. If Codex has implemented it elsewhere, it is not on these branches.

---

## 2. Specification and benchmark audit

### 2.1 Verifying the owner's summary against the repository
| Claim in the brief | Repository text | Verdict |
|---|---|---|
| The reference is Qwen3.8-27B | `RESEARCH.md` §1 and §9 | ✔ No revision is pinned. Proposed pin: `37561816e66b5b9153b1b64c10968c18c1e6f4a0` (lastModified 2026-11-07) [M] |
| The moonshot is 95% of the chance-adjusted text composite | `RESEARCH.md` §5, `EVALS.md` §4 (DCI) | ✔ |
| The ceiling is 4,096 MiB peak GPU memory | `RESEARCH.md` §3.1: "Peak GPU memory is no greater than 4096 MiB" | ✔ It does not say whether this means the allocator, the process or the whole device |
| Native low-bit representations are allowed | §3, native-fit rule | ✔ |
| Training-time distillation is allowed | §2 "distillation for training experiments", §4 "distil knowledge during research" | ✔ It is not explicit that distillation is allowed *for the qualifying model*. I recommend saying so |
| Post-hoc compression does not count | §3, native-fit rule | ✔ |
| Training may use more powerful hardware | §4 | ✔ |
| Batch 1, at least 8K, at least 8 tokens/s, no offload | §3.2–3.5 | ✔ Speed is measured "on the reference 4 GiB GPU", which is never named |
| (Not in the brief) no retrieval and no external tools | `EVALS.md` §6 | **This matters.** Any memory must reside on the GPU inside the budget |

### 2.2 The reference model as actually published [M, card and config]
- **Model:** `Qwen3_5ForConditionalGeneration`. It is a native vision-language model under the Apache-2.0 licence.
- **Layout:** 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN)). That is 64 layers with hidden size 5,120 and FFN size 17,408.
- **Attention layers:** 24 query heads, 4 KV heads, head dimension 256.
- **DeltaNet layers:** 16 key heads and 48 value heads of 128 dimensions, with FP32 state (`mamba_ssm_dtype: float32`).
- **Vocabulary:** 248,320, with untied embeddings.
- **Other:** native context of 262,144 tokens, one multi-token-prediction (MTP) layer, and a 27-layer vision tower of width 1,152.
- **Text parameters [D, my count from the config]:** about 26.89B, excluding norms, convolutions, the MTP layer and the vision tower. Of these, the embedding and head alone are 2.54B, which is **4.74 GiB in BF16**. That alone exceeds the whole Domovoi budget.
- **Mode:** thinking is on by default. The recommended reasoning output length is up to 262,144 tokens. The thinking-mode sampling settings are temperature 1.0, top_p 0.95 and top_k 20.

**Published text scores at this revision [M]:**

| Benchmark | Score |
|---|---:|
| IFBench | 79.5 |
| GPQA Diamond | 89.2 |
| HLE | 30.8 |
| LiveCodeBench v6 | 90.3 |
| Terminal Bench 2.1 | 73.0 |
| SWE-bench Pro | 61.7 |

- **The numbers match `EVALS.md`.** The card was modified on 2026-11-07, after the specification was written on 2026-09-15. The three numbers used by the specification are unchanged, but this should be re-checked whenever the pin changes.
- **No generation budget or thinking setting is given for GPQA, LiveCodeBench or IFBench.** Several agentic rows use a 256K context window. One in-house row uses max_tokens 32,768.

### 2.3 Problems with the specification
| # | Problem | Why it matters | Recommendation (needs owner approval) |
|---|---|---|---|
| S1 | **The reference GPU is unnamed.** `memory-complexity.md` calls the RTX 2050 the "historic" reference. | Feasibility of compute-for-bytes designs changes about 8× between an RTX 2050 (112 GB/s) and an RTX 5080 (960 GB/s) (§4.3). | Choose one: **(a)** a physical 4 GiB-class card, the honest deployment claim, which forbids heavy looping; or **(b)** "4,096 MiB device-wide cap on a named GPU", which is easier but less meaningful. I lean towards (a), with the RTX 2050 as the device. |
| S2 | "Peak GPU memory" scope is undefined. The RTX 2050 reported 4,095.5 MB in total [M, DML-0001]. | On a physical 4 GiB card a 4,096 MiB peak can never be exceeded, so the constraint is really what is usable after the driver and display take their share. Allocator peaks miss the CUDA context, which is about 250–550 MiB [A]. | Define it as the **process's device-wide peak** (NVML polling plus allocator peaks), with both reported. If S1(a) is chosen, also require completion on the physical card. |
| S3 | **Context versus thinking budget.** The reference's scores rely on thinking traces that are probably far longer than 8K tokens [S, because the card gives no budget]. A qualifying model has 8K of context. | Comparing against published scores would demand capability under budgets Domovoi cannot use. `EVALS.md` §6 already requires "reasoning budget fixed across directly compared models". | **Re-run the reference** at the pinned revision in BF16 with the **same total token budget** as Domovoi. Report thinking-on and thinking-off. Use the matched-budget run as the official 1.00. This is fairer, and it is already implied by §6. |
| S4 | **"8K tokens" depends on the tokenizer.** The current pipeline uses a 4,096-entry BPE; Qwen uses 248K entries. | 8,192 tokens of a 4K vocabulary hold much less text, perhaps 1.5–2× less [A]. That makes "8K" cheaper to meet than intended. | Define usable context as **≥ 8,192 Qwen3.8-tokenizer tokens' worth of text**, or as a fixed byte count, held by the model natively. |
| S5 | **Most DCI categories have no published reference score.** MMLU-Pro, math (GSM8K or MATH-500) and RULER do not appear on the card. | 50% of the DCI weight (25 + 15 + 10) must be produced by Domovoi itself. | Reproduce all DCI reference scores with one harness (S3), and pin the harness version. |
| S6 | **The 95% threshold has no statistical definition.** GPQA Diamond has 198 items. At p = 0.89 the standard error is about 2.2 points. After chance adjustment ((89.2 − 25) = 64.2), one run's ratio has a standard error of about 3.4% [D]. | A single run cannot tell 95% from 90%. | Score sampled benchmarks as avg@k with k ≥ 8. Use paired item-level bootstrap for the DCI ratio. **Qualify on the lower bound of a one-sided 95% CI ≥ 0.95**, or, if that is too strict, on a point estimate ≥ 0.95 with a lower bound ≥ 0.90, both stated in advance. |
| S7 | "No major category collapsing" is undefined for the moonshot. The M4 rule is each category ≥ 60%. | The moonshot cannot be judged as written. | Define it, for example as each category ≥ 0.80 normalised, before any milestone comparison. |
| S8 | **Distillation contaminates through the teacher.** Qwen3.8 has probably seen GPQA, MMLU-Pro and older LiveCodeBench problems [S]. | Distilled students inherit memorised answers. | Use LiveCodeBench problems released after both the teacher's and the student's data cutoff. Run teacher-output n-gram overlap checks. Add a contamination-robust diagnostic subset. |
| S9 | No knowledge-specific diagnostic. | Objectives A, B and C (§4.5) cannot be told apart. | Add diagnostic-only sets that are not weighted in the DCI: a SimpleQA-style long-tail factual recall set, and a procedurally generated reasoning set with no knowledge load. |
| S10 | The reference model is multimodal and "27B" includes the vision tower. | This affects parameter-based claims. | State the text-only parameter count, about 26.9B [D], whenever comparing. |

### 2.4 Recommended benchmark protocol
- **Official DCI.** Keep the existing six categories and weights, with these settings:
  - **Benchmarks:** MMLU-Pro (chance 0.10), GPQA Diamond (0.25), MATH-500 (0), LiveCodeBench v6 restricted to a post-cutoff window (0), IFBench (0), and RULER at 8K (0).
  - **Reference run:** the pinned revision, matched to Domovoi's total token budget.
- **Uncertainty.** Paired bootstrap across items, and three or more sampling seeds for generative tasks (S6).
- **Progress during research, below 1B parameters.** The fast tier in `EVALS.md` is right as it stands. Add these:
  1. Knowledge bits per stored bit (KBPB) on a synthetic biography suite, following Allen-Zhu & Li.
  2. A synthetic multi-hop composition accuracy test, following Physics of LMs 3.2 (2309.14402).
  3. FineWeb-Edu test NLL at equal measured inference bytes.

  These three separate knowledge, manipulation and language-modelling effects at small scale. HellaSwag-style scores are mostly noise below about 300M parameters [A].

---

## 3. Literature review (all arXiv IDs verified against arXiv in this session)

The organising question for each line of work: **does it buy capability per stored byte, per FLOP, or per byte read?** Under the 4 GiB, no-offload, bandwidth-bound regime, only the first and third count.

| Area | Key works (verified IDs) | What it actually improves | Relevance to 4 GiB |
|---|---|---|---|
| SSMs and linear attention | Mamba 2312.00752; Mamba-2/SSD 2405.21060; Gated DeltaNet 2412.06464; TTT layers 2407.04620; Titans 2501.00663; ATLAS 2505.23735 | Memory for context length and decode speed [L] | **Already in the reference** [M]. Necessary, but not a differentiator. The 8K state is ≤ 0.7 GiB even for the 27B |
| Weight sharing and looping | Universal Transformers 1807.03819; ALBERT 1909.11942; Relaxed Recursive Transformers 2410.20672 (shared plus per-loop LoRA); Huginn recurrent depth 2502.05171; Mixture-of-Recursions 2507.10524; Ouro/LoopLM 2510.25741; iso-depth law 2604.21106; Parcae 2604.12946; Looped MoE scaling 2609.40316; Looped models design study 2610.23036 | Bytes per unit of loss. The exchange rate is r^0.46 [L], and capacity is unchanged at about 2 bits per parameter [L, Ouro] | A modest per-byte gain, paid for with per-token bandwidth. Relaxed Recursive Transformers is the nearest prior art to R02×4 plus adapters |
| Adaptive computation | ACT 1603.08983; PonderNet 2107.05407; MoR 2507.10524 | Average compute per token | Helps throughput, not storage |
| Latent reasoning | Coconut 2412.06769; Huginn 2502.05171; HRM/TRM (TRM 2510.04871; critique 2512.11847); depth-recurrent compositional reasoning 2603.21676 | Test-time compute without generating tokens | Relevant to objective A under an 8K cap. TRM's ARC results depend on puzzle-ID conditioning and heavy test-time compute [L, 2512.11847], so they do not transfer to language models |
| Sparse lookup memory | PKM 1907.05242; Memory Layers at Scale 2412.09764; UltraMem 2411.12364; PEER 2407.04153; Engram 2601.07372; Fast-weight PKM 2601.00671; End-to-End Memory Networks 1503.08895 | Parameters per FLOP [L] | **All parameters still count against 4 GiB.** Engram's gains are measured against a *strictly iso-parameter, iso-FLOPs MoE* [L, abstract], not against a dense model of equal bytes. They show nothing about value per stored byte compared with dense parameters. Its host-memory prefetching concerns serving cost, which the no-offload rule would forbid. These methods help per byte *read*, not per byte *stored* |
| MoE | Switch 2101.03961; DeepSeek-V2 2405.04434 | Parameters per FLOP | Same caveat. Total parameters must fit |
| Native low-bit | BitNet 2310.11453; b1.58 2402.17764; BitNet 2B4T 2504.12285; Spectra/TriLM 2407.12327; Scaling Laws for Precision 2411.04330 | Bytes per parameter [L] | Directly relevant. **The open question is knowledge per stored bit, not parameters per byte** |
| Vector and codebook compression | DKM (training-time clustering) 2108.12659; "And the Bit Goes Down" 1907.05686; AQLM 2401.06118; QuIP# 2402.04396; QTIP 2406.11235 | Capability per stored bit at about 2 bits per weight, mostly after training [L] | The post-training versions are excluded as the primary result, but **training natively with codes is allowed** and largely unexplored for language models [S] |
| Knowledge capacity | Physics 3.3 2404.05405; Physics 3.2 (manipulation) 2309.14402; Morris et al. 2505.24832 | Measurement | Verified in the paper body (ar5iv), Results 4, 5 and 8:<br>• 2 bits per parameter at 1,000 exposures per fact, about 1 bit at 100 exposures.<br>• int8 keeps 2 bits per parameter, but int4 by post-training GPTQ drops to 0.7 bits per parameter.<br>• Junk-heavy data (1:7 useful to junk) cuts capacity for useful knowledge by up to 20×.<br>Morris et al. estimate about 3.6 bits per parameter of memorisation for GPT-family models [L], with a different definition, so the figures are not directly comparable |
| Alternative objectives and tokenisation | Multi-token prediction 2404.19737; DeepSeek-V3 (MTP objective) 2412.19437; Byte Latent Transformer 2412.09871 | MTP improves sample efficiency and enables self-speculative decoding, up to 3× faster inference [L, 2404.19737 abstract]. BLT matches tokenizer-based models at scale with dynamic byte patches [L, abstract] | MTP spreads the cost of weight reads across tokens (§4.3). BLT removes the vocabulary embedding cost, which matters: the reference spends 4.74 GiB on it. It also makes "8K usable context" independent of the tokenizer (S4) |
| Distillation and scaling | Hinton 1503.02531; Distillation Scaling Laws 2502.08606; Gemma 2 2408.00118; DeepSeek-R1 distilled students 2501.12948; Chinchilla 2203.15556; Kaplan 2001.08361 | Training efficiency of a fixed-size student. A teacher that is too strong can hurt (the capacity gap) [L, 2502.08606] | Raises a student towards *its own* capacity. It cannot exceed the information that fits in the bytes |
| Test-time compute | Snell et al. 2408.03314 | Compute substitutes for parameters on some reasoning tasks [L] | Limited by the 8K context and 8 tokens/s |
| Modular and procedural | Neural Module Networks 1511.02799; NPI 1511.06279; DreamCoder 2006.08381; NTM 1410.5401; Modular Deep Learning survey 2302.11529; "Is a Modular Architecture Enough?" 2206.02713; LoRA 2106.09685; LoraHub 2307.13269; EWC 1612.00796; HyperNetworks 1609.09106 | Compositional generalisation, inconsistently [L, 2206.02713] | This is the basis for the playbook assessment (§7) |
| Retrieval | RETRO 2112.04426 | Knowledge per parameter, using an external store | **Excluded** by `EVALS.md` §6 (no retrieval) unless the store is on the GPU and counted |
| Small-model design | MobileLLM 2402.14905 | Depth over width below 1B; block sharing | Relevant to the C0 baseline |

**Claims I recalled but did not verify against an abstract in this session** (all marked [R] in place of [L]):
- that memory-layer values need a separate learning rate;
- the batch-1 decode speeds of QuIP#, QTIP and AQLM kernels;
- Huginn's and Parcae's findings on loop stability;
- the figure from Ouro's v1 text that looped and non-looped models both store about 2 bits per parameter. That figure comes from a search summary. The abstract confirms only that Ouro's gains do "not [come] from increased knowledge capacity".

---

## 4. The main obstacles to 27B-class capability under 4 GiB

### 4.1 Reference memory accounting [D, from config]
| Item | Size |
|---|---:|
| Text weights, BF16 (about 26.89B parameters) | about 50.1 GiB |
| Embedding and head only (2 × 248,320 × 5,120) | 4.74 GiB |
| Attention KV at 8,192 positions, BF16 (16 layers × 2 × 4 × 256 × 2 B) | 512 MiB |
| Gated DeltaNet state, FP32 (48 × 48 heads × 128 × 128 × 4 B) | 144 MiB |
| Bits per parameter needed for text weights alone to fit in 4,096 MiB | 1.28 |

### 4.2 A common device-wide budget template (every row is [A] and must be measured on the reference device)
| Item | MiB |
|---|---:|
| CUDA context, driver and runtime libraries (Windows WDDM may be higher) | 350 (range 250–550) |
| GEMM and attention kernel workspaces | 64 |
| Prefill activations (1,024-token chunks, fused attention, last-position logits only) | 96 |
| Decode activations, logits and sampling | 16 |
| Allocator fragmentation slack | 150 |
| **Fixed overhead** | **676** |
| **Remaining for weights and sequence state** | **3,420** |

### 4.3 The bandwidth constraint on trading compute for bytes [D]
At batch 1, decode is memory-bandwidth-bound. The bytes read per token, W_read, must satisfy W_read ≤ BW_eff / 8 tokens/s.

| Device | Peak bandwidth | Effective, at 75% [A] | Bytes per token at 8 tok/s | Full passes over a 3.3 GiB (3.54 GB) weight set | Loss-equivalent gain from looping everything, r^0.46 [L × D] |
|---|---:|---:|---:|---:|---:|
| RTX 2050, 4 GB, 64-bit bus | 112 GB/s [L, NVIDIA spec] | 84 GB/s | 10.5 GB | **about 2.9** | about 1.6× |
| RTX 5080 capped at 4 GiB | 960 GB/s [A, vendor spec not re-verified] | 720 GB/s | 90 GB | about 25 | about 4.4× (extrapolated far beyond the measured r ≤ 8) [S] |

These pass counts assume **one weight pass per generated token**.
- With multi-token prediction and self-speculative verification, one pass verifies several draft tokens. The pass budget then rises by the mean number of accepted tokens per pass.
- [L, 2404.19737] reports up to 3× faster inference.
- On the RTX 2050 that would give roughly 4–9 full passes [D/A], and about 1.9–2.7× loss-equivalent parameters.
- Acceptance rates depend on the task, and latent looping may lower them [S]. Measure them; do not assume them.

**Implications:**
- On a 4 GiB-class card, looping the whole model gives at most about 1.6× loss-equivalent parameters, or about 2.7× with optimistic MTP.
- Larger loop counts need a **small looped core with storage that is read sparsely** (candidate C1).
- Even then, Ouro reports that looping does not add knowledge capacity [L].

### 4.4 The knowledge-capacity constraint [D, with L inputs]
The weight budget is about 3,380 MiB, which is 28.35 Gbit.

| Assumption | Knowledge that fits | Share of the 27B's 2-bits-per-parameter capacity (53.8 Gbit) |
|---|---:|---:|
| Measured best: int8 at 2 bits per parameter, so 0.25 bits per stored bit [L] | 7.1 Gbit | 13% |
| BF16 at 2 bits per parameter, so 0.125 bits per stored bit [L] | 3.5 Gbit | 7% |
| int4 by post-training GPTQ at 0.7 bits per parameter, so 0.175 bits per stored bit [L, Result 8] | 5.0 Gbit | 9% |
| Hypothetical native code at 0.5 bits per stored bit [S] | 14.2 Gbit | 26% |
| Information-theoretic ceiling, 1 bit per stored bit (hard bound) | 28.35 Gbit | 53% |

Ternary weights carry at most log₂3 ≈ 1.58 bits per parameter, so the 2-bits-per-parameter figure **cannot** hold for ternary [D].

### 4.5 Separating the three objectives
| Objective | Feasibility | Reasoning |
|---|---|---|
| **A. Reasoning parity** (math, code, procedural or composition tasks) | **Most tractable, but still far off.** | Reasoning draws on manipulation more than storage [L, 2309.14402, 2510.25741]. Compute can partly substitute [L, 2408.03314]. The limits are the bandwidth ceiling (§4.3) and the 8K budget. LiveCodeBench and MATH also carry knowledge load. |
| **B. Factual-knowledge parity** | **Infeasible if the 27B uses most of its capacity; open if it does not.** | 7 Gbit fits at the measured density against a 53.8 Gbit capacity. The decisive unknown is how much knowledge the benchmarks *need* (§11). Retrieval is excluded. |
| **C. Broad parity at 95% DCI with no category collapse** | **Least tractable.** | It needs B for MMLU-Pro and GPQA (45% of DCI weight, mostly knowledge-loaded [A]) as well as A. |

---

## 5. Candidate architectures

There are three materially different mechanisms, plus the control they must all beat. Each section separates **existing**, **proposed** and **speculative** elements.

### C0. The control: a conventional hybrid trained natively at low bit width, with distillation
This is not a candidate. **It is the null hypothesis every candidate must beat at equal measured device bytes.**
- **Design:** about a 6.4B-parameter hybrid in the reference's style (3 DeltaNet layers to 1 GQA layer). Weights are trained natively at int4 with group-128 scales (4.125 bits) or ternary (about 2 bits packed, about 13.5B parameters). Vocabulary of 65,536 (it fits the uint16 pipeline). Distilled from a teacher.
- **Everything here is existing technique** [L: BitNet, Spectra, Gated DeltaNet, MobileLLM, distillation].

| Budget [A/D] | MiB |
|---|---:|
| Weights: 6.4B params at about 4.2 bits on average (int8 embeddings) | 3,204 |
| State: 8 GQA layers × 2 × 4 × 128 × FP8 at 8K, plus 24 DeltaNet layers × 16 × 128² × FP32 | 64 + 24 = 88 |
| Fixed overhead (§4.2) | 676 |
| **Total** | **3,968** |

- **RTX 2050 decode:** about 3.46 GB per token, so about 24 tokens/s [D/A].
- **Prefill:** 8K tokens × 2 × 6.4e9 ≈ 105 TFLOP, so about 10 s at an assumed 10 TFLOP/s [A].

### C1. Factored memory and compute: a small looped reasoning core plus a large sparsely read associative memory

1. **Hypothesis.** On a bandwidth-bound device, capability depends on two things: knowledge bytes, which are mostly stored but sparsely read, and applied reasoning compute, which needs dense reads of a *small* core. Separating them makes it possible to have both about 3 GiB of knowledge and more than 10 latent iterations per token at 8 tokens/s. A dense model has to trade one against the other. [S]

2. **Graph and maths.**
   - Prelude P (non-looped, holds all sequence mixing): hybrid DeltaNet and GQA layers, giving h₀ = P(x≤t).
   - Loop for k = 1…r_t:
     - qₖ = W_q·hₖ₋₁
     - mₖ = Σ_{i ∈ topK(PK(qₖ))} softmax(qₖ·Kᵢ)·Vᵢ, using product keys, so lookup cost is O(√N·d + K·d)
     - hₖ = Core(hₖ₋₁ + U·mₖ, h₀)
   - Halting: a halting head gives r_t ∈ [1, r_max] (ACT or MoR-style routing).
   - Coda C: a few layers, then the logits.
   - Memory is **queried again at every iteration**, so retrieval can be multi-hop: the fact retrieved in step k conditions the query in step k+1.

3. **Stored knowledge.** Factual associations live mostly in V ∈ ℝ^{N×d_v} (int8 or int4). Linguistic and procedural competence lives in the prelude, core and coda.

4. **Reasoning and state.** Latent iteration of the core. Sequence state exists only in P and C, so KV does not grow with r. That is a design choice, compared with MoR's per-recursion KV.

5. **Training.**
   - Full backpropagation through r_max, or truncated backpropagation.
   - Truncation lowers φ to 0.38 [L, 2604.21106], so prefer full backpropagation with activation checkpointing.
   - Memory values get sparse gradients (embedding-bag style) and probably need a separate learning rate [R, 2412.09764].
   - A ponder cost regularises the halting head.
   - Distillation targets come from the teacher's final logits.

6. **Inference.** Chunked prefill with r capped low for prompt tokens. Decode uses r_t, chosen adaptively.

7. **Budget [A/D].**

   | Item | MiB |
   |---|---:|
   | Core: 1.2B parameters, int4 | 590 |
   | Prelude and coda: 0.6B parameters, int4 | 295 |
   | Embedding: 65,536 × 2,048, int8, tied | 128 |
   | Memory: about 2.3B values, int8 (keys about 2√N·d_k, negligible) | 2,200 |
   | **Weights subtotal** | **3,213** |
   | State: 6 GQA layers at FP8 plus 12 DeltaNet layers | 60 |
   | Fixed overhead | 676 |
   | **Total** | **3,949** |

8. **Complexity.**
   - Decode bytes per token: 0.31 GB (P and C) + r × 0.62 GB (core) + r × 2 × 32 rows × 2 KB (memory, negligible) + 0.06 GB (state).
   - **On the RTX 2050, r ≤ about 16** without speculative decoding [D]. With MTP self-speculation it could be about 24–48 [D/A].
   - Decode FLOPs: 2 × (0.6 + 1.2r)·1e9 per token, about 40 GFLOP at r = 16. That is compute-light.
   - Prefill FLOPs scale with 0.6 + 1.2r billion parameters. At r = 4 that is 5.4B, about 0.84× C0, so about 9 s at 8K [D/A]. At r = 16 it is 19.8B, about 3.1× C0, so about 32 s.
   - Capping r for prompt tokens keeps time to first token near C0. Time to first token is not a specification criterion, but it should be.

9. **Precision.** The looped core is sensitive to error that compounds over iterations. Keep it at int4 or higher with FP16 activations and normalise the residual each iteration [R, Huginn 2502.05171 and Parcae 2604.12946 on loop stability]. Memory values could go to int4 or ternary; this is untested [S].

10. **Proposed advantage.** More reasoning iterations per byte read than C0 at the same stored knowledge [S]. The loss-equivalent estimate is core 1.2B × 16^0.46 ≈ 4.3B, plus 0.6B, plus memory. That is *comparable* to C0's 6.4B, not dramatically better [D/S]. **The bet is on composition and multi-hop tasks, not on loss.**

11. **Failure modes.**
    - φ may fall further at large r.
    - Memory layers' value per parameter is below that of dense parameters.
    - The halting head may collapse to r_max or to 1.
    - Product-key retrieval may be unstable when re-queried.
    - Throughput depends on custom kernels.

12. **Prior art and novelty.** Every component exists: memory layers, Engram, looped models, MoR, and multi-hop memory reads (End-to-End Memory Networks).
    - The re-query-per-loop combination is close to MemNN-style multi-hop reading inside a Huginn-style loop.
    - Novelty is **low to moderate**.
    - The genuinely new part, if any, is a **scaling law for allocating bytes under a device-byte and bandwidth constraint**. Engram's allocation law is per FLOP and per parameter [L].

13. **Smallest decisive experiment.** At about 60M stored parameters on RTX 5080, compare at **equal stored bytes and equal bytes read per token**:
    - C0-mini;
    - looped core without memory;
    - memory without loop;
    - C1-mini.

    Use three seeds and test on FineWeb-Edu NLL, the synthetic biography KBPB suite and a 2-hop/3-hop composition suite.

14. **Falsification.** C1 fails if:
    - it does not beat both the looped-only and the memory-only arms on multi-hop composition by a pre-registered margin (for example 5 accuracy points, in all seeds); or
    - its multi-hop advantage disappears when the per-loop memory query is replaced by a single query.

### C2. Natively coded weights: learned vector or trellis codes trained from scratch

1. **Hypothesis.** The 2 bits of knowledge per parameter that holds at int8 suggests BF16 weights carry about 8× more stored bits than the knowledge they encode [D from L]. A representation *trained from the start* with learned multi-dimensional codes can raise knowledge and skill per stored bit above about 0.25, towards the information limit. This is the only mechanism examined here that directly attacks objective B. [S]

2. **Graph and maths.**
   - Each weight matrix W ∈ ℝ^{m×n} is split into g-dimensional sub-vectors w_j ≈ s_{⌊j/G⌋} · C[c_j].
   - C is a learned codebook shared across layers of one type, with 2^b entries.
   - c_j are integer codes.
   - Storage: (b/g) bits per weight, plus scales s, plus codebook bytes.
   - Variants: additive multi-codebook codes as in AQLM, or trellis codes as in QTIP, with tiny code tables.
   - Training uses differentiable soft assignment that anneals to hard assignment (DKM-style), or latent full-precision weights with a straight-through estimator and periodic codebook EMA.

3. **Stored knowledge.** In the codes themselves. The codebook acts as a learned basis for weight space.

4. **Reasoning and state.** Unchanged. C2 is a *representation* that can wrap C0 or C1.

5. **Training.** Training needs more memory (latent weights and assignment state), which is allowed. Codebook collapse and dead codes are the known risks [L, VQ literature].

6. **Inference.** Decode kernels fetch codes and expand them through a codebook held in shared memory or L2. A 1 MiB codebook does not fit the RTX 2050's L2 [A], so use g = 8 with b = 8 per stage (a 4 KiB codebook) and 2 additive stages, giving 2 bits per weight.

7. **Budget [A/D].**

   | Item | MiB |
   |---|---:|
   | Codes: 12.5B parameters × 2.0 bits | 2,980 |
   | Scales: FP16 per 128 weights (0.125 bits per weight) | 186 |
   | Codebooks | about 16 |
   | **Weights subtotal** | **3,182** |
   | State: 12 GQA layers at FP8 plus 36 DeltaNet layers | 132 |
   | Fixed overhead | 676 |
   | **Total** | **3,990** |

8. **Complexity.** Bytes read per token are about the same as C0, so about 24 tokens/s on the RTX 2050 *if* decoding codes is cheap. My recollection is that post-training work reports near-FP16 batch-1 decode speed for well-designed 2-bit VQ kernels on large GPUs [R, QuIP# and QTIP]. On the RTX 2050 this is unknown [A].

9. **Precision.** Mixed: embeddings and the head may need 4 or more bits. Activations stay in FP16 or BF16.

10. **Proposed advantage.** If knowledge per stored bit reaches 0.5, objective B rises from 13% to 26% of the reference's capacity (§4.4), and C0's parameter count doubles at equal bytes [D/S].

11. **Failure modes.**
    - **The one existing data point is against C2.** In Allen-Zhu & Li, int4 by post-training GPTQ drops to 0.7 bits per parameter, which is 0.175 bits per stored bit, *below* int8's 0.25 [L, Result 8]. That was post-training scalar quantisation. Native training with vector codes is untested, which is exactly why the native measurement is the experiment.
    - Natively trained codes may behave like low precision in the Scaling Laws for Precision sense: effective parameters shrink as precision drops [L, 2411.04330], erasing the gain.
    - The optimiser may be unstable.
    - Kernels may be slow on small GPUs.
    - Gains may appear only in post-training regimes (quantising a trained model), not in native training.

12. **Prior art and novelty.** Codebook compression of trained models is well established (AQLM, QuIP#, QTIP). Training-time clustering exists for CNNs (DKM, "Bit Goes Down"). Native ternary pretraining exists (BitNet, Spectra).
    - **I found no verified study that measures knowledge bits per stored bit across native representations.** This is the most defensible contribution: a measurement, not an architecture.
    - Novelty is **moderate as a measurement and low as a mechanism**.

13. **Smallest decisive experiment (the KBPB sweep).**
    - **Data:** a synthetic biography corpus in the Allen-Zhu & Li style, with N individuals × 6 attributes and entropy computable exactly.
    - **Models:** 10–60M parameters, sized so each arm's *stored bits* are equal.
    - **Representations:** BF16, int8-QAT, int4-QAT, ternary, VQ-2-bit and VQ-1.5-bit.
    - **Exposures:** 100 and 1,000 per fact.
    - **Measure:** extracted knowledge bits divided by stored weight bits. Three seeds.
    - **Compute:** hours on the RTX 5080 (§8.1).

14. **Falsification.** C2 fails if no native coded representation exceeds int8-QAT's KBPB by ≥ 1.3× in all seeds, at both exposure levels and at two model sizes. If the advantage shrinks between 10M and 60M parameters, treat it as not scaling.

### C3. Neural playbooks: a compositional library of low-rank procedures over latent steps
1. **Hypothesis.** Procedural competence is redundantly duplicated across contexts in dense networks. A library of small, reusable low-rank operators selected sequentially over latent iterations can represent the same procedures in fewer bytes and compose them systematically. [S]

2. **Graph and maths.**
   - Core with shared weights W.
   - At latent step k, a router picks top-k playbooks πₖ = topK(R·hₖ₋₁) from a library {(Aᵢ, Bᵢ)}, i = 1…M, of rank-ρ adapters.
   - hₖ = Core_{W + Σ_{i∈πₖ} αᵢ AᵢBᵢᵀ}(hₖ₋₁).
   - A "program register" uₖ, a small recurrent vector, conditions the router so that sequences of playbooks can form.

3. **Stored knowledge.** Procedures live in the adapters, general competence in W, and facts in W. C3 has no separate fact store unless combined with C1.

4. **Reasoning.** Explicit sequential composition of selected operators.

5. **Training.** End-to-end with load balancing and a router entropy penalty. Sparse gradients localise updates (an MoE-like guard against interference). EWC-style penalties are only needed for continual learning [L, 1612.00796].

6. **Inference.** As C1, plus routing at each step.

7. **Budget [A/D].**

   | Item | MiB |
   |---|---:|
   | Core: 2.4B parameters, int4 | 1,180 |
   | Prelude and coda: 0.6B parameters | 295 |
   | Library: 16,384 adapters × (4 projections × 2 × 2,560 × rank 8) = 2.68B parameters, int4 | 1,318 |
   | Embedding | 128 |
   | **Weights subtotal** | **2,921** |
   | State | 60 |
   | Fixed overhead | 676 |
   | **Total** | **3,657** |

8. **Complexity.** Core reads of 1.24 GB per pass allow r ≤ about 8 on the RTX 2050 without speculative decoding [D], or about 12–24 with MTP [D/A]. Adapter reads are negligible.

9. **Precision.** As C1.

10. **Proposed advantage.** Systematic compositional generalisation on procedure-heavy tasks [S].

11. **Failure modes.**
    - Router collapse.
    - Modules that are not specialised in practice, so learned modularity does not yield compositional generalisation [L, 2206.02713].
    - Formally, it is a looped low-rank MoE whose scaling is already being studied [L, 2609.40316].

12. **Prior art and novelty.** Neural Module Networks, NPI, LoraHub (dynamic LoRA composition), PEER (a million tiny experts), Relaxed Recursive Transformers (per-loop LoRA), MoR and Looped MoE. Novelty is **low**. The "program register" sequencing is the only uncommon element, and it resembles NPI.

13. **Smallest decisive experiment.** On a procedurally generated composition suite (for example multi-step string, arithmetic and list programs with held-out *combinations* of seen primitives), compare C3-mini with an equal-byte looped MoE without the register and with a dense looped core. Three seeds.

14. **Falsification.** C3 fails if:
    - it does not beat the equal-byte looped MoE on held-out combinations by a pre-registered margin; or
    - ablating a single playbook does not selectively break the procedures routed to it (no real modularity).

---

## 6. Quantitative comparison of candidates (4,096 MiB device-wide; RTX 2050 as reference device)

| | C0 control | C1 factored | C2 coded | C3 playbooks |
|---|---:|---:|---:|---:|
| Stored weights (MiB) | 3,204 | 3,213 | 3,182 | 2,921 |
| Sequence state at 8K (MiB) | 88 | 60 | 132 | 60 |
| Fixed overhead [A] (MiB) | 676 | 676 | 676 | 676 |
| **Total (MiB)** | **3,968** | **3,949** | **3,990** | **3,657** |
| Parameters | 6.4B | 1.8B dense + 2.3B memory | 12.5B | 3.0B dense + 2.7B adapters |
| Bytes read per token at minimum compute | 3.46 GB | 0.99 GB | 3.47 GB | 1.65 GB |
| Maximum latent passes at 8 tok/s, RTX 2050, without speculative decoding (MTP range in brackets) | 1 (about 24 tok/s) | about 16 (24–48) | 1 (about 24 tok/s, if decode is cheap) | about 8 (12–24) |
| Knowledge, assuming 0.25 bits per stored bit [D/A] | about 6.7 Gbit | about 6.7 Gbit (unless memory is denser [S]) | 6.7 Gbit, or 13.3 if 0.5 [S] | about 6.1 Gbit |
| Main lever | none (baseline) | reasoning per byte read | knowledge per stored bit | procedural reuse |

**Distinctions:**
- **Theoretical:** every number in this table.
- **Allocator-measured:** none yet.
- **Whole-device-measured:** none yet.

The 676 MiB fixed overhead is the largest single assumption. On a physical 4 GiB card under Windows, the usable remainder could be 200–400 MiB smaller [A]. Every candidate would then have to shed about 10% of its weights.

---

## 7. The playbook concept, assessed in detail

**Software, neural, or both?** As originally described (recent working memory, a factual store and reusable procedures with consolidation), it is a **system-level agent memory design**: a retrieval-augmented prompt library.
- As software it is excluded from qualification: `EVALS.md` §6 forbids retrieval.
- It would also consume the 8K context.
- It only becomes architecture when procedures are *parameters* that the model selects and applies internally (C3).

**What playbooks could replace.**
- Duplicated procedural circuits: the same algorithm learned separately for different surface formats or domains.
- The redundancy is real in principle, but its size in large models is unmeasured. Superposition already lets dense models share circuits. [S]

**What they cannot replace.**
- **Facts.** They are incompressible in the Allen-Zhu & Li sense: one tuple, about one set of bits. No procedure regenerates them.
- **Lexical and semantic knowledge, and language competence.** These are distributed and enormous.
- **Calibration and world knowledge used inside procedures.**

Even a perfect procedure library leaves objective B untouched.

**Encoding options and their cost per procedure:**
1. Low-rank weight deltas (C3): about 160K parameters each at rank 8.
2. Small subnetworks (PEER-style experts).
3. Latent "program" vectors that condition a shared core, like soft prompts: d parameters each. This is very cheap but weak in expressiveness [S].
4. Fast-weight programs written at test time (Titans, Fast-weight PKM): costs state rather than storage.

**Composition and generalisation.** Sequential application over latent steps gives composition *syntactically*. Whether learned routing yields *systematic* generalisation is the open question, and the evidence is mixed to negative: modular architectures do not specialise automatically [L, 2206.02713, 2302.11529].

**Avoiding destructive interference.**
- In pretraining, sparse routing already localises gradients. The usual failure is router collapse, not forgetting.
- For continual consolidation, use capacity reservation, EWC-style anchoring or freeze-and-append.
- Each of these costs either bytes or plasticity.

**Finite capacity.** Usage-based pruning and merging (task-arithmetic style), with a fixed library size counted in the budget. Consolidation *at inference time* is test-time training. It is allowed if it stays on the GPU, but static benchmarks do not reward it.

**Novel reasoning or repeated workflows?** Playbooks help when test problems decompose into procedures seen in training, which is also when dense models already do well. Their plausible deployment-time win is repeated agentic workflows. That is a system-level benefit the DCI does not measure.

**Falsification tests:**
- **T1:** at equal bytes, a playbook model against a looped MoE and a dense looped core, on held-out primitive combinations.
- **T2:** selective-ablation specificity.
- **T3:** bytes needed to reach a fixed accuracy on a procedure suite, against C0.
- **T4:** whether any gain survives on natural-language data (FineWeb-Edu NLL) rather than only on synthetic tasks.

**Verdict.** Playbooks are a **weak primary architecture** for 27B-class capability under 4 GiB. They target the parameter budget that is least likely to dominate the gap; facts and broad competence dominate it. The neural version is mostly a looped low-rank MoE. **I recommend not building it** unless C1's composition experiments show a specific procedural bottleneck that a library would remove.

---

## 8. Training compute as a separate constraint

### 8.1 What the RTX 5080 can do [D from M]
- Measured B08 throughput: 268,678 tok/s × 0.216 GFLOP/token ≈ **5.8e13 FLOP/s effective in training**.
- That is about **5.0e18 FLOP per day**.
- Utilisation at larger sizes is assumed similar [A].

| Model (unique parameters) | Tokens at 20 per parameter | 6·N·D FLOPs | RTX 5080 time |
|---|---:|---:|---:|
| 50M | 1B | 3.0e17 | about 1.5 h |
| 150M | 3B | 2.7e18 | about 13 h |
| 400M | 8B | 1.9e19 | about 3.8 days |
| 1B | 20B | 1.2e20 | about 24 days |

### 8.2 Implications
- **Decisive questions must be answered at 10–400M parameters.** At that scale, knowledge-capacity and composition probes (synthetic, exact entropy) are *meaningful*. HellaSwag-style benchmarks are not.
- **Toy against language experiments:** synthetic suites isolate the mechanism, and FineWeb-Edu NLL at equal *measured inference bytes* checks that the effect survives on natural text. Both are required.
- **Data.**
  - FineWeb-Edu 10BT is on the NAS at a pinned revision [M]. Only about 5M tokens are packed [M].
  - Packing the full 10BT sample supports models up to about 400M parameters at Chinchilla ratios.
  - The uint16 format caps the vocabulary at 65,536 [M], which is fine.
  - TinyStories should be retired for architecture decisions.
- **Scaling laws.** Fit loss against stored bits (not parameters) at 3–4 sizes. With three seeds per point, report exponent uncertainty by bootstrap. Do not extrapolate beyond about 10× the largest measured size without a confirming point on rented hardware.
- **Distillation accounting.** Use a 2×2 factorial: {C0, candidate} × {no teacher, teacher}. The architecture effect is the candidate's gain *without* a teacher, and the interaction term shows whether the architecture exploits a teacher better.
  - The teacher must be held fixed.
  - Qwen3.8-27B does not fit on 16 GB in BF16 (about 50 GiB) [D].
  - Logit distillation also needs a shared tokenizer, and 248K entries costs 0.5 GiB of embedding at d = 1,024 in int8 [D].
  - At small scale, use a same-tokenizer smaller teacher (for example Qwen3.5-2B or 4B) or sequence-level distillation.
  - Large-teacher distillation belongs to rented-hardware stages.
- **Multiple seeds.** At least 3 for every decision, with paired data order. That is already the pilot plan's design and should be kept.

---

## 9. Ranking and roadmap

### 9.1 Ranking (5 = best)
| Criterion | C0 control | C1 factored | C2 coded | C3 playbooks |
|---|:-:|:-:|:-:|:-:|
| Capability per GiB potential | 3 | 3 | **4** | 2 |
| Potential for broad 27B-class capability | 2 | 2 | **3** | 1 |
| Fundamental novelty | 1 | 2 | 3 (as measurement) / 2 | 2 |
| Mathematical and engineering plausibility | **5** | 3 | 3 | 2 |
| Efficiency of testing | 4 | 3 | **5** | 3 |
| Training requirements (low is good) | 4 | 3 | 3 | 3 |
| Risk (5 = lowest) | **5** | 2 | 3 | 1 |

**C2 is first, C1 second, and C3 is not recommended.** C0 must be built regardless, as the control.

### 9.2 Staged roadmap for the strongest direction ("bytes-first": C2, then C2 + C1)
Each stage is pre-registered and published before execution. Margins are written down before data is seen.

| Stage | Question | Design | Proceed if | Stop if |
|---|---|---|---|---|
| **0. Instruments** (no architecture claim) | Can we measure KBPB, composition and inference bytes reliably? | Synthetic biography suite with exact entropy; a 2/3-hop composition suite; an inference profiler (device-wide plus allocator); a GPT-2-style BF16 baseline. Targets [L, 2404.05405 Results 4–5]: about 2 bits/param at 1,000 exposures and about 1 bit/param at 100 exposures | The baselines reproduce both targets within ±20% in 3 seeds | The baselines cannot reproduce the literature (instrument invalid) |
| **0b. Reference knowledge ladder** (evaluation only; no training; **needs authorisation**, because it runs third-party models) | How much do the benchmarks depend on stored knowledge, and where does a ≤4 GiB model stand today? | Evaluate Qwen3.5 0.8B, 2B, 4B and 9B, plus Qwen3.8-27B (pinned), at the matched 8K budget. Split items into knowledge-loaded (SimpleQA-style, MMLU-Pro subsets) and reasoning-loaded (procedurally generated and math). Include the ≤4 GiB Qwen3.5-4B scoreboard baseline from `EVALS.md` | Produces a score-against-stored-bytes curve per item type. This is C0's first real number | (Decision input, not a gate.) If knowledge-loaded accuracy saturates well below 27B scale, B is less hopeless than §4.4 implies (§11.1) |
| **1. KBPB sweep** (§5, C2.13) | Does any native representation store more knowledge per stored bit? | BF16, int8-QAT, int4-QAT, ternary, VQ-2b and VQ-1.5b at equal stored bits; 10M and 60M; 3 seeds | Best native code ≥ 1.3× int8 KBPB in all seeds, at both sizes and both exposure levels | Nothing beats int8 by 1.3×. **Then stop C2.** Judge objective B's feasibility on stage 0b, not on this result |
| **2. Language transfer** | Does the advantage survive natural text at equal *measured inference bytes*? | FineWeb-Edu (full 10BT pack) at 3 sizes (30M, 100M, 300M stored-equivalent) for C0 against C2; 3 seeds; test NLL plus the fast research tier | C2 beats C0 at every size and the gap does not shrink with scale (bootstrap on the slope) | The gap shrinks with scale, or the decode kernel misses 8 tok/s on an extrapolated reference budget |
| **3. Factored compute** | Does C1 on top of C2 raise reasoning per byte read? | 4-arm design (§5, C1.13) using the C2 representation; bandwidth-matched; composition plus NLL | C1 beats both ablations on held-out multi-hop by the pre-registered margin, in all seeds | No gain over the memory-only or loop-only arm |
| **4. Distillation factorial** | How much is the architecture and how much the teacher? | 2×2 at the best size from stage 3 | Architecture main effect > 0 without a teacher | The gain appears only with a teacher (the architecture adds nothing) |
| **5. Scale confirmation** (rented hardware, needs authorisation) | Do the effects extrapolate? | About 1B and 3B at Chinchilla ratios; DCI subsets at the matched budget | The fitted per-bit scaling exponent beats C0's | The exponent advantage disappears |
| **6. Qualification attempt** | Can a model meet S1–S7 at its decided settings? | Full DCI against the matched-budget reference | Pre-registered milestone (M1, then M2) | |

---

## 10. Final recommendation

1. **Pin the specification first.** Settle S1–S7 before any further architecture runs. The reference-device decision (S1) alone changes which candidates are feasible.
2. **Treat the recurrence line as answered for now.** R02×4 is inconclusive by the project's own draft rules. The published iso-depth law (φ = 0.46) answers "how much is a loop worth" better than more TinyStories runs could. Recurrence should reappear only inside C1, justified by the bandwidth model. Do not run it as a stand-alone track.
3. **Start with stages 0, 0b and 1.**
   - Stage 1 (the KBPB sweep) is cheap, decisive for C2 and attacks the binding constraint.
   - Stage 0b uses no training and measures how knowledge-bound the target really is. Together they show whether objective B is open at all.
4. **Build the C0 control in parallel.** No candidate result means anything without it.
5. **Do not build playbooks as a primary architecture.**
6. **Be explicit in public claims.** On current evidence, the realistic research claim is "maximum capability per GiB, with reasoning parity as the leading objective". 95% DCI parity is a moonshot that stage 0b could show to be infeasible.

---

## 11. Major unknowns: why this recommendation could be wrong

1. **The 27B's used knowledge may be far below its capacity.**
   - Allen-Zhu & Li reach 2 bits per parameter only at about 1,000 exposures per fact, and 1 bit at 100 exposures.
   - Junk-heavy data cuts capacity for useful knowledge by up to 20× [L, Results 4, 5 and 12].
   - Real long-tail facts get few exposures, so the 27B may hold only 10–20 Gbit of benchmark-relevant knowledge [S].
   - If so, objective B is much less hopeless, and C0 with curated data could cover most of MMLU-Pro and GPQA. Note that the same effect also lowers Domovoi's achievable density.
   - **This is the most important unknown.** Stage 1 cannot settle it; stage 0b and diagnostic S9 can.
2. **The capacity figures come from synthetic tuples.** Real knowledge (relational, procedural, fuzzy) may compress differently.
3. **φ was measured at small scale with r ≤ 8 and may change at scale.** Architectural tricks such as hyperconnections raise it (0.65) [L, 2604.21106].
4. **Lookup memory's value per byte compared with dense parameters is unmeasured.** Engram's gains (BBH +5, MATH +2.4) are against an iso-parameter, iso-FLOPs *MoE* [L], which is usually weaker per parameter than a dense model, so they do not settle this. If an equal-byte comparison against a dense model (C1's stage-3 ablation) showed lookup memory beating dense parameters, C1 would rise in rank.
5. **Multi-token prediction acceptance rates.** If self-speculative decoding with looping reaches 3 or more accepted tokens per pass on the reference device, the bandwidth bound (§4.3) relaxes about 3×, which favours C1.
6. **The reference-device decision.** If the owner chooses an RTX 5080 with a cap, compute-for-bytes designs gain about 8× in budget, and C1 may overtake C2.
7. **The fixed-overhead assumptions** (676 MiB) are unmeasured. Each 100 MiB is about 3% of the weight budget.
8. **Literature after my knowledge cutoff (June 2026)** is only partly covered. I verified specific IDs but did not systematically survey the second half of 2026.
9. **Recalled claims.** Some claims are marked [R] and were not checked against an abstract in this session (§3).
10. **Novelty search limits.** It covered arXiv titles and abstracts. Unpublished or industrial work on native-code pretraining may exist.
