# Permanent experiment protocol

Commit and push every experiment's documented plan and approval before execution. Commit and push the preceding experiment's outcome and interpretation before starting another. This covers training, architecture experiments, benchmarks, memory and inference profiling, hyperparameter comparisons, model evaluations, experimental optimizations, repeats, and failures. Group routine correctness tests into a verification record. Measurements intended as research evidence must follow this protocol.

The research repository owns the evidence. [Domovoi Labs](https://github.com/CL-BAF/Domovoi-Labs) owns executable code. The supported launchers fetch the GitHub ref `refs/heads/research/real-data-calibration` and use [the canonical execution ledger](../experiments/protocol/state.json). Local files and other branches are drafts until pushed into that history. [Initial gate implementation at `8252fbf361b387dc21d17fd6ff7443d9dc7274ad`](https://github.com/CL-BAF/Domovoi-Labs/commit/8252fbf361b387dc21d17fd6ff7443d9dc7274ad)

## Lifecycle and operator steps

| Phase | Owner and action |
|---|---|
| PROPOSED | Add the next `DRP-NNNN` ledger row with `id` and `state`. Commit and push. Use a distinct ID for a repeat. |
| DOCUMENTED | Add the plan files and update the README and archive index. Set the row to DOCUMENTED. Commit and push. |
| APPROVED | A reviewer inspects the immutable implementation SHA, design, exact command, input digests and decisions. In a separate commit set APPROVED and `plan_commit` to the full DOCUMENTED research SHA. Push before execution. An edited approval returns to DOCUMENTED and gets a new review. |
| RUNNING | Execute the exact command with the leading `--research-id DRP-NNNN` selector. The launcher must receive acknowledgement of its fast-forward claim before work begins. |
| COMPLETED / FAILED / INTERRUPTED | The launcher preserves its local receipt and pushes the execution state. After a crash use the [implementation recovery procedure](https://github.com/CL-BAF/Domovoi-Labs/blob/8252fbf361b387dc21d17fd6ff7443d9dc7274ad/docs/EXPERIMENT_PROTOCOL.md#crash-and-outage-recovery) from the original checkout once both launcher and worker are inactive. |
| ANALYSED | Fast-forward to the terminal ledger commit. Update the original record and README, add the result and evidence, then set ANALYSED with `terminal_commit` pointing to that exact terminal commit and `outcome` equal to its state. Preserve the claim and plan fields recorded by the launcher. Commit and push. |
| PUBLISHED | In a separate commit set PUBLISHED and `analysis_commit` to the full ANALYSED research SHA. Push. Keep the published record, evidence, result and links intact. Add separate linked corrections when needed. |

A single experiment can preregister a comparison with fixed arms and controls. An added arm, changed command, resumed compute after closure or repeat is another experiment. Keep the DML run IDs as artifact IDs and link each one to the protocol record. Make one lifecycle transition per commit. Only the last ledger row can be active, and earlier rows remain immutable. The ledger contains no retroactive approvals for archived experiments.

## Plan files

Store the human-readable plan at `experiments/protocol/records/DRP-NNNN.md` and its structured plan at `experiments/protocol/plans/DRP-NNNN.json`. Add a concise question and standard Markdown link to the record in both the README and archive index. Use a path relative to each document.

Use the following exact JSON field names. Each field in the second column holds a non-empty string.

| Section | Required fields and content |
|---|---|
| Rationale | `question`, `rationale`, `problem`, `implementation_changes`, `design_choice` |
| Hypothesis and comparison | `hypothesis`, `mathematics`, `expected_and_alternative_outcomes`, `baselines` |
| Reproduction | `dataset_tokenizer_config`, `environment`, `evaluation`, `success_and_failure`, `compute_and_memory`, `exact_reproduction_commands` |

Also include `id`, the full `implementation_commit` SHA on GitHub, a `command` object, and an `inputs` object. The command contains `script` (for example `scripts/train.py`) and `argv` (for example `["--config", "configs/baseline_small.yaml"]`). Use exact post-selector arguments. Pin all mutable material inputs in `inputs` as checkout-relative paths with forward slashes mapped to SHA-256 digests. The gate requires explicit config pins for training, resume checks, model inspection and every pilot arm. For a generated campaign config, pin the sources and list resolved configs, training seeds, budgets and their derivation in the record. Pin dataset, tokenizer, checkpoint, harness and driver versions as applicable. Record versions or external artifact digests that cannot be checked as local immutable files and explain how they will be verified.

## Result files

Append the result to the original Markdown record. Add `experiments/protocol/results/DRP-NNNN.json` and update the README with a short result and the original record link. Include the same `id` and terminal `outcome`, an `evidence` array of repository-relative paths, and non-empty strings for the following fields.

| Section | Required fields and content |
|---|---|
| What happened | `execution_history`, `actual_outcomes`, `hypothesis_comparison`, `capability_loss_speed_memory`, `improvements_and_regressions` |
| What the evidence supports | `numerical_analysis`, `observations`, `possible_explanations`, `supporting_evidence`, `competing_hypotheses`, `uncertainty` |
| Failures and next decision | `unexpected_results_and_failures`, `lessons`, `recommended_next_experiment` |

Keep complete execution history, configuration, metrics, checkpoints or digests, stdout and stderr, interventions, errors and retries. Where a measurement is missing or inapplicable, record the reason and its effect on the conclusion. Label observations, calculations and proposed mechanisms separately. For large, sensitive or restricted artifacts, publish a portable receipt with hashes, sizes, access restrictions and reproduction instructions.

## Outages, race conditions and boundaries

Unpushed plans stay local through an outage. Without current GitHub state and an acknowledged claim, the launcher refuses new work. A crash leaves RUNNING until recovery establishes a terminal state. The original receipt identifies the owner, and local operating-system locks prevent recovery while a launcher or worker is active. If that receipt is lost, a repository reviewer must establish that the worker has stopped and publish an audited terminal correction that retains the claim identity. Preserve the existing artifacts.

Fast-forward pushes serialize launchers using the supported code. Review and protect the canonical ref against force-pushes and deletion. Earlier implementation revisions, direct Python APIs and external tools can execute independently. Contributors must apply this protocol to them, and any new maintained research launcher must adopt the shared gate. Test fixtures exercise gate behavior on isolated repositories and make no experimental capability claims.

## Correctness verification · 24 November 2026

The implementation suite ran with `python -m pytest -q` on Windows and PyTorch 2.15.0+cpu. It passed 298 tests with one CUDA-only skip and the existing scheduler-injection warning. The gate tests cover all eight direct CLI denials, approved command and input drift, competing fast-forward claims, publication of prior outcomes, outages, local crash recovery, and a separate spawned campaign worker with a single-use handoff. The ledger integration tests use isolated bare Git repositories and synthetic callbacks. The canonical GitHub ledger was initialized with zero approved experiments. [Enforcement tests](https://github.com/CL-BAF/Domovoi-Labs/blob/8252fbf361b387dc21d17fd6ff7443d9dc7274ad/tests/test_research_protocol.py)

The archive verifier recalculated the calibration and extended comparisons and checked 544 manifest records and 88 public artifact checksums. The archived synthetic harness matches its previous Git blob, and the retired command refused direct execution.
