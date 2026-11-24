# CUDA resume incident · DML-CUDA-001

**Date:** 24 November 2026. **Disposition:** default-CUDA trajectory repeatability failed the calibration gate. Corrected protocol succeeded in the bounded same-device control. The first calibration and failed recovery artifacts are retained.

## Discovery and control

After DML-0002–0004 on [f5fac7a](https://github.com/CL-BAF/Domovoi-Labs/commit/f5fac7a34a211adf93e0dc8ea576c8d973ef87ac), `verify_real_resume.py` used the R02×4 configuration and packed TinyStories input. It configured four updates and one warmup, evaluation every two updates over two sampled batches, checkpoint interval two and log interval one. The model was width 640, 512 positions, vocab 4096, cycle 2×4, AdamW peak LR 0.0006, clip 1, batch 16 with accumulation 2, BF16 autocast and seed 1234. The software stack was Windows 11, Python 3.13.17, RTX 5080 driver 621.01, PyTorch 2.15.0+cu132, CUDA 13.2.

The harness stopped one branch at the first committed checkpoint (step 2), loaded its verified in-memory model and AdamW states, then ran steps 3–4. It compared against a separate uninterrupted four-update execution. Exact tensor restoration and same executable trajectory are distinct tests. The check reopens trusted optimizer state and examines each tensor. The max final parameter delta is the largest absolute elementwise difference among all matching live parameter tensors. The harness rejects error above 5e−5 and requires reconciled metric steps `[1,2,3,4]`.

| Recovery configuration | Checkpoint restore | Reconciled steps | Max final weight delta | Outcome |
|---|---|---|---:|---|
| RCV-DEFAULT · f5fac7a, default CUDA operations | Model and AdamW state matched exact checkpoint values | 1, 2, 3, 4 | 0.001584119163453579 | Failed trajectory gate |
| RCV-DET-MANUAL · f5fac7a, deterministic enabled via setup wrapper | Model and AdamW matched | 1, 2, 3, 4 | 0.0 | Same-device control passed |
| RCV-DET-PINNED · d914684, deterministic config and checksum verification | Model and AdamW matched, resumed sampler at 64 | 1, 2, 3, 4 | 0.0 | Same-device accepted check, final cursor 128 |

[Default artifacts](../experiments/data/archive/cuda-resume-v1/) · [Manual deterministic verification](../experiments/data/archive/cuda-resume-deterministic-v1/verification.json) · [Pinned deterministic verification](../experiments/data/archive/cuda-resume-v2/verification.json)

By update 2 the default interrupted branch's displayed training loss was 7.6884 versus 7.6885 in the independent uninterrupted branch. This difference precedes the restoration boundary, so a restore-only explanation cannot account for the entire divergence. At update 4 the default displayed validation losses were 6.8171 and 6.8172. Selection of nondeterministic CUDA algorithms or reductions is plausible. No kernel trace identifies a specific operation. Exact restored checkpoint values exclude a detectable load corruption on the measured state; they do not prove equivalent paths after resume.

## Correction and reproduction

The controlled wrapper set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA work and called `torch.use_deterministic_algorithms(True)` before invoking the unchanged harness. The matching result justified adopting explicit `training.deterministic_cuda` and making it a resume-checked trajectory field. [d914684](https://github.com/CL-BAF/Domovoi-Labs/commit/d914684dc730cd32132aeff389d6291c447ed7e4) also added per-file hashes and atomic latest-index replacement. All three 200-update arms were rerun under the corrected protocol so that quality and timing across arms share one mode.

From the matching source and data roots, use the original R02×4 calibration YAML for each commit and a new verification output directory:

```powershell
# Default mode at f5fac7a with the V1 configuration
& $CudaPython scripts/verify_real_resume.py --config $V1R02Config --out $NewDefaultRecoveryDir
# Corrected mode at d914684 with the V2 configuration
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
& $CudaPython scripts/verify_real_resume.py --config $V2R02Config --out $NewDeterministicRecoveryDir
```

The first command reproduced the documented failure under the observed conditions. Repetition is permitted only for a separate verification output and can produce a different delta in nondeterministic mode. See [full instructions](../reproduce/README.md). The original failure exception survives as a recorded observation and underlying loss/checkpoint files, not a complete standalone stderr capture. Checksum verification is orthogonal to numerical repeatability. The zero difference is limited to this GPU, driver, OS, PyTorch, seed, input, geometry and short controlled test. No bitwise portability or long-duration fault-coverage claim is made.
