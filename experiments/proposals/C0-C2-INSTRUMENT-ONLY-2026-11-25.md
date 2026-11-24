# C0/C2 revised first gate — instrument-only proposal

25 November 2026 · **DRAFT / UNAPPROVED / NOT A CAPACITY EXPERIMENT.** No DRP ID, implementation SHA, experiment approval, model download or workload is authorised here. This document replaces the original 57-run proposal as the **next possible experiment for review**, not as a revision to previously collected evidence. [Second-review disposition](../../reviews/2026-11-25-CLAUDE-SECOND-REVIEW-DISPOSITION.md) · [original draft retained](C0-C2-DEPLOYED-BYTE-STUDY.md).

## Decision this small study should answer

**Can we trust the synthetic dataset, the scored-fact estimator, the same-geometry training comparison, the resident-byte counter and the exact native low-bit reference reconstruction?** If not, a large representation-capacity campaign would produce misleading evidence.

The instrument is **not** intended to establish maximum stored knowledge, compare coded versus scalar weights, validate a novel architecture, qualify the 4 GiB goal or claim superior inference throughput.

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
| Seeds | Three paired seeds per arm; six model-seed runs total |
| Geometry | Same four-layer causal backbone with width 64, four heads, FFN intermediate width 320, shared tokenizer and activation precision |
| Fact world | 16,384 trained identities × four attributes = 65,536 independent two-byte facts per seed |
| Training exposure | 20 examples per fact in this **instrument** |
| Positions | 65,536 facts × 20 exposures × 7 positions = **9,175,040 processed positions per run** |
| Target positions | Two supervised value bytes per example, thus **2,621,440 scored targets per run** |
| Assignment | Dataset and training-order generation matched across arms within seed; independently derived world keys between seeds |
| Controls | Uniform probability, permuted-label negative control, forbidden answer-context checks, evaluator-only oracle, untrained identity namespace |
| Evidence | Identity-cluster uncertainty, raw two-byte exact match, invalid-output rate, calibrated value log probabilities, code/logit parity, byte-accounting receipts |
| Runtime | TBD after permitted instrument feasibility checks; **no claimed GPU-hour prediction or cap approved** |

Freeze explicit model/training implementations, exact random-number seeds and derivation, dataset/harness hashes, config files, GPU environment, requested execution command, input digests, checkpoints, evaluation scripts and schedule before requesting approval. Use separate training and scorer source revisions. All six training runs must be accounted for in the single clearly defined approved comparison, or be individually registered per the protocol.

**Correctness pass/fail:** zero leakage on null/permuted/untrained controls within preregistered statistical bounds; oracle at its analytic ceiling; reproducible world entropy; exact byte layout reconstructed by CPU reference; acceptable logit agreement within the frozen numerical tolerance; stable positive information recovery by BF16 and INT8 when the instrument is able to learn. Predefine acceptable learning thresholds using a *disjoint* generator-calibration world and record negative findings rather than silently altering thresholds.

**No capacity conclusion:** 20 exposures samples early memorisation/optimisation, not saturated information density. Ratios with tiny or negative information estimates must not be used to rank representations. Publish failed instruments and close the remote ledger before any rerun.

## Required changes to the later C0/C2 capacity campaign

The historical 57-run design is **not** approved. A later redesigned campaign requires:

1. Mandatory symmetric hyperparameter tuning for all arms on a disjoint, preregistered calibration world, with the same search budget, recorded attempts, compute and selection rules; tune scalar baselines competitively.
2. Both fixed E8 and rotated-scalar controls, in addition to fixed residual VQ and learned VQ; exact, independently audited codepoint and effective-rate distinctions.
3. Exposure-to-saturation measurements extending beyond 100 if needed, potentially toward or beyond 1,000 exposures; predefine a stopping/plateau criterion and scaling by fact-world size. Preserve finite-exposure learning rate as a separate endpoint.
4. Equal logical geometry, equal actual representation bytes, and matched **measured or carefully attributed** compute comparisons with clear incompatible-inference caveats.
5. An explicit CPU packed-code/logit correctness stage, then a separate realistic-model GPU decode performance qualification rather than seven premature miniature kernels.
6. Natural-text validation, held-out prompt formats, multiple model sizes and genuine transfer before broad language-model or architectural claims.
7. A newly calculated count of arms/seeds, positions, training FLOPs, optimizer/tuning costs, time and scratch; the reviewer-reported **123 runs / 192 PFLOP** is a conditional scenario, **not** a frozen or affordable plan. Check feasibility from observed throughput first.

### Decision outcomes

- **Instrument fails:** archive exact fault; repair a demonstrated error under code verification rules; new research executions require a separately reviewed plan.
- **Instrument passes:** publish, then propose the next saturation/representation comparison with its own decision thresholds, stronger baselines and byte accounting.
- **Instrument passes but is too slow:** shrink a preregistered world or change the harness in a new plan; never quietly drop the saturation requirement for the later capacity study.
- **Capacity study eventually shows only fixed E8 or scalar controls win:** discontinue a claim of learned-code superiority; retain the useful negative evidence and strong C0 baseline.
- **Capacity study eventually shows coded weights win only on random facts:** publish narrow finding, require natural-text and downstream capability transfer before promoting C2.

## Approval and publication

Before an I1 run: a reviewer must approve hardware/time cap, the exact six-run schedule, stopping criteria, implementation/config/generator/scorer commits, data hashes, model byte accounting and numerical tolerance. Assign a DRP ID only through the normal PROPOSED→DOCUMENTED→APPROVED lifecycle on the **canonical research ref**. The launcher claims GitHub execution before beginning and blocks until the preceding result is PUBLISHED. No manual bypass, model run, eval or GPU profiling is authorised by creating this draft. Do not perform any training now.
