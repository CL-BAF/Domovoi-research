# Integration of the complete second Claude roadmap review

25 November 2026 — **planning amendment, not experimental approval.** Original reviewer report: [full 24 November Claude review](2026-11-24-claude-roadmap-review.md). Earlier summary-only [disposition](2026-11-25-CLAUDE-SECOND-REVIEW-DISPOSITION.md). The original 24 November C0/C2 design remains available, marked superseded for scheduling.

## Provenance and preservation

The owner uploaded `2026-11-24-claude-roadmap-review.md`. The uploaded UTF-8 source was **66,728 bytes**, with **SHA-256 `eb6f68a91ad9b40fe28bbe7abc1e4f7cbd0cfb4354509e469c9235358d34ce3d`**, calculated on the source file. It contains **655 newline-terminated lines**. The file was published unmodified at `reviews/2026-11-24-claude-roadmap-review.md` on this review branch. A subsequent read through the GitHub connector matched the uploaded text exactly, including the final newline and UTF-8 byte count; GitHub blob SHA **`5e6a0ab8cfbbc27a4f36a0f64ce7c836686e7fa6`**. This is verification of *transcription*, not an independent replication of the reviewer’s calculations or literature research.

The reviewer explicitly discloses having authored the earlier independent investigation. It is an independent review **relative to Codex's reconciliation**, not an independent second reviewer relative to Claude's own earlier work. Preserve the reviewer's [V] directly verified, [C] calculated, [P] published, [A] assumed, [S] speculative and [U] unverified distinctions when citing its findings.

## Complete-report findings added beyond the pasted summary

| Source section | Material finding / proposed correction | Project action |
|---|---|---|
| B M-1, F.6 | **37.6% of the tiny studied deployment payload is fixed BF16 embeddings/attention/norms**, potentially masking differences between coded MLPs. Add fixed-tensor-only / frozen-MLP controls and `ΔK` per **coded MLP byte**, alongside total-device and entire-model metrics. | Add to the redesigned capacity proposal; the six-run instrument remains a calibration-only experiment. |
| B M-2, F.3, D.8 | Original ternary arm was stored at 2 bits per weight, not approximately 1.6. | Test a specified 5-trits-per-byte scheme or disclose both actual packings, include lookup and group scales. |
| B M-3, F.7, D.3 | Original K metric is sensitive to overconfidence. Predeclare **K\*** with a 16-temperature calibration grid alongside K at T=1; treat D>1 beyond uncertainty as a leakage alarm. Use independently seeded cryptographic PRF labels, never side information passed to inference. | Incorporate into scorer fixture and instrument record, pin analysis and calibration. |
| B M-4, F.8 | Memorised random facts do not establish generalisable representations. | Add structured-world held-out identities and two-hop fact manipulation probes to the later capacity design. Require no material regression before advancing C2. |
| B M-5, G.5 | An original natural-text screen of 20M tokens would give 100M-parameter models only 0.2 tokens/parameter; rankings may reverse. | Add **≥20 training tokens per parameter at 30M/100M** as proposed stability tests, requiring approximately 0.6B/2B tokens respectively (and a prepared 2B-token corpus) with an independently approved compute budget. The shorter screen is explicitly a smoke check, not model ranking. |
| B M-6, G.8 | A competent language model needs fixed and fair post-training (SFT; possibly RL/distillation) to assess IFBench and coding fairly. | Explicit post-training stage added to roadmap; require recipe-matched C0/candidate controls. |
| B M-7/8, H | Quality-first reference must not be cut off by an arbitrary thinking-token limit; absolute context score and 0.80-relative-to-Qwen gates conflict if both are binding. | Require an approved single frozen definition and dual score-vs-budget reporting; include natural delayed evidence as a context gate. |
| B M-10, H | Windows WDDM system-memory fallback could conceal prohibited host offload; official vendor source for some controls was not verified. | On a **specific real device**, verify actual fallback policy availability, dedicated/shared residency and PCIe traffic. Do not assume any setting applies to every driver. Propose a **60-second 8K TTFT ceiling** for owner approval, not as approved policy. |
| B M-11, H | Publisher reports Gemma 3 27B IT LiveCodeBench as **29.7** in Table 6 and **39.0** in Table 18, likely different benchmark windows. | Keep both with table/window attribution; remeasure pinned scores rather than relying on either publisher number. |
| B M-12 / I.3 | Specific protocol gaps: untracked source files; mutable/unpinned analysis/dependencies; GPU measurements via external harnesses; authorless APPROVED; unprotected canonical ref; manually asserted COMPLETED after lost receipt. | Record for a separate implementation/security review, require named human approval, protect ref, pin analysis/runtime, and restrict unsupported crash-recovery successes. Do not claim resolved. |
| C ranks 1–3, E.3 | C0+ mature low-bit baselines likely give the largest near-term efficiency gains; C2 is a narrower native-training scientific measurement; **weight-stationary parallel streams / MTP** may exploit idle arithmetic instead of rereading scarce weight bandwidth. | Preserve novel architecture as distinct goal, add separate candidate research proposal conditional on hardware evidence, assess KV amplification and actual accepted-token/s. |
| G.2, G.9, H | A matched-budget **Gemma 3 27B IT** milestone can precede Qwen3.8-27B moonshot; include small-family ladder, BitNet b1.58 2B4T, quality-first fair references and frozen post-training recipe. | Revised hardware/capability preflight and roadmap, pending user approval. |

### Corrections accepted by reviewer

The original investigation contained **65 distinct arXiv identifiers, not 66**; an earlier C3 memory-label mix-up was corrected; historical TinyStories retained a test split but not a test **score**. The one-seed equal-FLOP checkpoint comparison is tentative due to unmatched cosine schedules, and “4× decode compute” applies only to the recurring block, not embeddings/head. These are reviewer corrections, not fresh local measurements.

### Numerical estimates remain conditional

The reviewer recalculated the original 57-run/30.8-PFLOP arithmetic and proposed a substantially expanded **123-run, 192.2-PFLOP** redesign with a **4.5–9 GPU-hour** conditional estimate and about 1.3M updates. Actual throughput, launch overhead, CUDA-graph capture and codebook assignment on this workload have not been measured. **Do not freeze these numbers or approve an 18-hour campaign without a separate instrument run and a fully republished schedule.** The reviewer’s 4-GiB architecture, physical RTX 2050 memory and parallel compute projections are similarly *scenarios*, not observed target-device qualifications.

## Documentation scope and remaining decisions

This addendum and links amend **research proposals only**. It does not:
- change `experiments/protocol/state.json` or create an APPROVED DRP;
- implement a decoder or experimental harness, fetch new model weights or datasets, or launch training, evaluation, profiling or GPU benchmarks;
- merge any PR, alter branch protection or change implementation code;
- establish that fixed E8 or native coded training is novel or will close the 27B gap.

**Next human review:** settle physical hardware availability, fair reference/reasoning setup, S1–S7 and TTFT, human approval governance, the two proposed no-training evaluation studies and a resource cap for the *separate* six-run BF16/INT8 instrument. The primary study remains **REVISE BEFORE APPROVAL**.
