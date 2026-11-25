# C0/C2 revised first gate — instrument-only proposal

25 November 2026 · **DRAFT / UNAPPROVED / NOT A CAPACITY EXPERIMENT.** No DRP ID, implementation SHA, experiment approval, model download or workload is authorised here. This document replaces the original 57-run proposal as the **next possible experiment for review**, not as a revision to previously collected evidence. [Second-review disposition](../../reviews/2026-11-25-CLAUDE-SECOND-REVIEW-DISPOSITION.md) · [original draft retained](C0-C2-DEPLOYED-BYTE-STUDY.md).

## Decision this small study should answer

**Can we trust the synthetic dataset, the scored-fact estimator, the same-geometry training comparison, the resident-byte counter and the exact native low-bit reference reconstruction?** If not, a large representation-capacity campaign would produce misleading evidence.

The instrument is **not** intended to establish maximum stored knowledge, compare coded versus scalar weights, validate a novel architecture, qualify the 4 GiB goal or claim superior inference throughput.

## Additions required by Claude's full report

The [complete second review](../../reviews/2026-11-24-claude-roadmap-review.md) and [integration ledger](../../reviews/2026-11-25-CLAUDE-FULL-REPORT-INTEGRATION.md) clarify several instrument and later-capacity controls. The six-run I1 remains a **scorer instrument**, not a saturation or codec-advancement test.

**I0 CPU scorer work.** The reference implementation now defines a keyed HMAC-SHA256 generator, the temperature-one K estimator, a frozen 16-point temperature grid, codecs and packed-byte receipts. For independent uniform facts, the implemented leakage alarm uses `S+16` bits for K and `S+20` bits for same-world K*, where S counts every world-dependent deployed bit and four bits pay for temperature selection. Freeze the family of checks and its multiplicity treatment in the execution plan. This alarm has little power in I1 because the fact entropy is smaller than either proposed representation. Independent null and isolation checks are still necessary. [CPU correctness and limits](../../reviews/2026-11-25-CODEX-PR3-REVIEW.md)

**I1 results:** log K and K*, seed-to-seed variance, update wall-time, calibration, instrument reliability and possible leakage. Any time-per-update measurement used as research evidence is included in this **separately approved** DRP, not smuggled in as an ungated correctness test.

**Later C0/C2 capacity study:** add a fixed-tensors-only / frozen-MLP ablation to separate the 197,760 bytes of unchanged BF16 embeddings, attention and norms from information held in coded MLPs. Report incremental retained information ΔK per coded-MLP byte *alongside* full resident-byte density. Add structured-rule generalisation on held-out identities and two-hop fact manipulation to each arm; a random-tuple density gain accompanied by material loss on either does not advance a general architectural claim. Freeze a saturation criterion separate from the 100-exposure learning-speed endpoint. Mandatory arm-symmetric tuning must include every scalar and vector control. Actual native ternary rate should use a documented **1.6-bit effective trit packing** (or report both 1.6 and 2.0 with hardware implications). Separate mathematical vector-code proof from a mere rotation improvement.

## Two separately controlled work packages

### I0 — harness and algebra verification (non-experimental CPU correctness)

Develop, review and test a deterministic data generator, byte accounting, masking, normalized 256-symbol value scoring, per-identity clustering, leakage controls, scalar quantizers and **actual decoded codepoint sets**. Test full-precision CPU reference reconstruction/logit parity with a frozen tolerance. Keep the implementation and verification tests separate from scientific model training and outcome measurement.

Before any C2 capacity study, implement and mathematically classify:

- signed scalar 8/4/2-bit and native ternary controls;
- rotated scalar quantization with its exact rotation, scale and bit accounting;
- two-stage residual codebook with identical and different rotations, fixed and trained books;
- fixed E8 lattice control at an exactly specified effective rate, with all transform, scale and lookup metadata charged;
- **genuinely vector** construction whose decoded codepoints and rate/geometry cannot be explained by a rotated scalar product code.

For the earlier two-stage fixed VQ initialization, explicitly test whether the repeated orthogonal rotation makes its sum algebraically equivalent to a rotated four-level scalar code. Document the exact proof or a counterexample; do not conclude from the algorithm name. Confirm no hidden full-precision decoder state, side information, data leaks or test-dependent codebooks. This verification does not require seven custom GPU kernels and does not establish GPU deployment.

The I0 package may group routine unit tests under the repository's code verification rules. Any benchmark, profiling or experimental comparison intended as research evidence still requires the remote protocol and explicit permission. Pin analysis code and the scorer alongside generator code.

### I1 — six-run BF16 / native INT8 scorer instrument

Only after I0 correctness is independently reviewed and the exact experiment is preregistered and approved:

| Property | Proposed initial definition |
|---|---|
| Arms | BF16 and native INT8 scalar; **no vector-code arms** |
| Seeds | Proposed IDs 1234, 1235 and 1236, paired across arms. Freeze separate derivations for world, initial weights and data order. Six model-seed runs total. |
| Geometry | Same four-layer causal backbone with width 64, four heads, FFN intermediate width 320, 512 reserved symbol slots, tied embedding/output and shared activation precision |
| Fact world | 16,384 trained identities × four attributes = 65,536 independent two-byte facts per seed |
| Training exposure | 20 examples per fact, 4,096 queries per update, for 320 updates per run and 1,920 across the six runs |
| Positions | 65,536 facts × 20 exposures × 7 positions = **9,175,040 processed positions per run** |
| Target positions | Two supervised value bytes per example, thus **2,621,440 scored targets per run** |
| Assignment | Dataset and training-order generation matched across arms within seed; independently derived world keys between seeds |
| Calibration and scoring | In each trained world, calibrate on the 1,024 identities whose ID is divisible by 16. Score the remaining 15,360 identities × four attributes = 61,440 independent facts (983,040 bits). Use the familiar query format without answers for the primary instrument. The novel format is a separate transfer diagnostic, because its marker and ordering are absent during training. |
| Controls | Uniform logits, fixed-seed derangement of labels, independent untrained labels, evaluator-only oracle, deliberately leaked and overconfident wrong predictions in the fixture, tests excluding answer side information. Use teacher forcing only for the second-byte probability and free decoding for accuracy. |
| Evidence | Raw K, K*, whole-model information per bit, per-seed results and paired seed/world dispersion, raw two-byte exact match, invalid-output rate, byte calibration, code/logit parity and byte receipts |
| Representation | 197,760 B shared BF16 payload. MLP payloads give 525,440 B total BF16 and 366,720 B INT8 including 5,120 B of scales, before packaging. Eight aligned `DMC1` MLP headers add 512 B per arm if deployed in that format. Report actual representation including retained copies and indexes. |
| Runtime | Rough common backbone accounting is 1,597,824 FLOPs per processed position, about 87.96 TFLOP for all six runs. This excludes quantization, optimizer, evaluation, and I/O. Wall time and stop cap require hardware-specific approval. |

Freeze the model, native INT8 training rule, activation precision, gradient/scale treatment, strict block-separated attention and loss mask, random-number seeds, runtime, inputs and output artifacts. Evaluation code, metric, transformations and statistics must be pinned alongside training. A single source revision can pin distinct training and scorer files. Keep the root key and evaluator-side labels inaccessible to the predictor. Report the data-cluster uncertainty separately from descriptive variation over three independent pairs. Account for all six runs in one prespecified comparison or register them individually under the protocol.

**Proposed pass/fail.** Oracle and uniform controls must match their analytic values within frozen numerical bounds. Null, permuted and untrained controls must stay within preregistered false-positive limits. Check reproducible entropy, exact byte receipts and decoded logit agreement. Require both arms to clear a preapproved positive-information criterion on scored identities. Select this threshold before results are seen from analytic controls or separately authorized feasibility evidence. The same-world holdout fits only temperature. It cannot select the pass threshold, tune training, or rank codec variants. Publish inconclusive and failed outcomes.

**No capacity conclusion:** 20 exposures samples early memorisation/optimisation, not saturated information density. Ratios with tiny or negative information estimates must not be used to rank representations. Publish failed instruments and close the remote ledger before any rerun.

## Required changes to the later C0/C2 capacity campaign

The historical 57-run design is **not** approved. A later redesigned campaign requires:

1. Mandatory symmetric hyperparameter tuning for all arms on a disjoint, preregistered tuning world, with the same search budget, recorded attempts, compute and selection rules. Tune scalar baselines competitively. Keep final paired worlds sealed through selection and account for all searches and multiplicity.
2. Both fixed E8 and rotated-scalar controls, in addition to different-basis fixed residual VQ and learned VQ. The same-basis two-stage hypercube is a rotated scalar product code. The CPU E8 core is a specified test format. A competitive E8-QAT arm needs its own matched trainer and published codec revision. [E8-QAT](https://arxiv.org/html/2610.27699v1) · [QuIP#](https://arxiv.org/abs/2402.04396)
3. Exposure-to-saturation measurements extending beyond 100 if needed, potentially toward or beyond 1,000 exposures; predefine a stopping/plateau criterion and scaling by fact-world size. Preserve finite-exposure learning rate as a separate endpoint.
4. Equal logical geometry, equal actual representation bytes, and matched **measured or carefully attributed** compute comparisons with clear incompatible-inference caveats.
5. An explicit CPU packed-code/logit correctness stage, then a separate realistic-model GPU decode performance qualification rather than seven premature miniature kernels.
6. Natural-text validation, held-out prompt formats, multiple model sizes and genuine transfer before broad language-model or architectural claims.
7. A newly calculated count of arms/seeds, positions, training FLOPs, optimizer/tuning costs, time and scratch; the reviewer-reported **123 runs / 192 PFLOP** is a conditional scenario, **not** a frozen or affordable plan. Check feasibility from observed throughput first.
8. Freeze the MLP at a world-independent initialization for the fixed-tensor-only control, retaining the same forward path and learning the rest with matched recipe and world. Report `K_full−K_fixed` per coded byte as an ablation contrast alongside full-model density. Keep the structured held-out-identity and two-hop probes out of the random-fact entropy total.

### Decision outcomes

- **Instrument fails:** archive exact fault; repair a demonstrated error under code verification rules; new research executions require a separately reviewed plan.
- **Instrument passes:** publish, then propose the next saturation/representation comparison with its own decision thresholds, stronger baselines and byte accounting.
- **Instrument passes but is too slow:** shrink a preregistered world or change the harness in a new plan; never quietly drop the saturation requirement for the later capacity study.
- **Capacity study eventually shows only fixed E8 or scalar controls win:** discontinue a claim of learned-code superiority; retain the useful negative evidence and strong C0 baseline.
- **Capacity study eventually shows coded weights win only on random facts:** publish narrow finding, require natural-text and downstream capability transfer before promoting C2.

## Approval and publication

Before an I1 run: a reviewer must approve hardware/time cap, the exact six-run schedule, stopping criteria, implementation/config/generator/scorer commits, data hashes, model byte accounting and numerical tolerance. Assign a DRP ID only through the normal PROPOSED→DOCUMENTED→APPROVED lifecycle on the **canonical research ref**. The launcher claims GitHub execution before beginning and blocks until the preceding result is PUBLISHED. No manual bypass, model run, eval or GPU profiling is authorised by creating this draft. Do not perform any training now.
