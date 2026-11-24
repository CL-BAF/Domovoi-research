# Training, checkpointing and supervised execution

The executable authority is [implementation d914684](https://github.com/CL-BAF/Domovoi-Labs/tree/d914684dc730cd32132aeff389d6291c447ed7e4). [Reproduction instructions](../reproduce/README.md) give concrete commands and config paths. [Data preparation](../datasets/preparation-and-tokenization.md) describes the stages before training.

## Stage order

`scripts/train.py` validates the configuration and environment, reserves a new experiment or opens an existing ID for resume, and delegates to `Trainer`. Paths, artifact labels and run allocation live in the experiment registry. The model registry resolves the architecture and builds tied parameters before the optimizer is initialized. Training startup verifies tokenizer identity and vocab size, memory maps the train/validation uint16 files, checks actual token counts, and hashes the bytes that are opened. Metadata records commit, cleanliness, environment, config, corpus/version and tokenizer.

For a disjoint run, divide the packed train sequence into stride context+1 positions. A seeded CPU permutation drives segment selection. The checkpoint cursor counts complete selected segments. All paired real-data runs use data seed 1235 derived from model seed 1234 plus one. Their input stream per configuration is identical because the corpus, tokenizer, stride, accumulation and seed are shared. Model initialization does not consume that generator. Historical DML-0001 used random offsets, which can overlap. Validation uses its own fixed generator. Per-run tests run synthetic data to verify sampler position and exact weights through interruption. [Experiment controls](experimental-controls.md)

## Loss and AdamW

At each optimizer update with $K$ equally sized microbatches, backpropagate $L_k/K$ for each $k$ into the same parameter gradients. This is the mean over $KBT$ scored targets. Zero gradients before the accumulation window. Apply global norm clipping after unscaling when FP16 scaling is active. If the total Euclidean norm $G$ exceeds clip $c$, the update sees a gradient rescaled by approximately $c/G$ with a small implementation epsilon. The logged norm is the pre-clip scalar from the last update in that logging window. It is not a window maximum.

For a clipped gradient $g_t$, AdamW tracks $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$ and $v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$. Bias-correct with $1-\beta_1^t$ and $1-\beta_2^t$, then update decayed weights with a separate factor $1-\eta_t\lambda$ and subtract $\eta_t\hat m_t/(\sqrt{\hat v_t}+\epsilon)$. PyTorch provides step counters and default epsilon. Biases and normalization gains are rank-one and assigned to the zero-decay group. Reused matrices update once per unique parameter. The recorded optimizer uses peak LR 0.0006, betas (0.9, 0.95), decay 0.1 on matrices and clip 1.0 in calibration and extended runs. [Optimizer implementation](https://github.com/CL-BAF/Domovoi-Labs/blob/d914684dc730cd32132aeff389d6291c447ed7e4/domovoi/training/optim.py) · [AdamW](https://arxiv.org/abs/1711.05101)

`LambdaLR` uses $\lambda(s)=(s+1)/w$ for scheduler index $s<w$ with warmup $w>0$. Otherwise set $u=\mathrm{clip}[(s-w)/\max(1,N-w),0,1]$ for plan length $N$ and return $\rho+(1-\rho)(1+\cos(\pi u))/2$. The LR used in an update is read before stepping the scheduler. Calibration used \(w=10,N=200\); extended used \(w=200,N=6104\), both with \(\rho=0.1\). The short calibration therefore follows a different LR trajectory. BF16 autocast uses FP32 parameters, gradients and Adam moments with eligible kernels operating at BF16. FP16 uses a scaler where configured; it is not part of the recorded CUDA comparisons.

## Timing, validation and stop policy

The CUDA timing switch synchronizes before and after timed update work. Logs publish sampled windows, elapsed time and step rate separately. Allocator stats report allocated and reserved peaks as distinct process metrics. Validation is periodic and uses fixed random windows for lightweight feedback. Final evaluation is a separate deterministic document-scoring procedure. See [measurement](measurement-methodology.md) and [evaluation](evaluation-methodology.md).

The following limits were active in the extended campaign:

| Condition | Rule and action |
|---|---|
| Nonfinite train loss or gradient | Fail before advancing the affected update; never publish an invalid checkpoint. A nonfinite validation loss fails the affected step before its checkpoint. |
| Sustained instability | Pre-clip norm above 100 for 50 consecutive updates. After warmup, 50-step mean loss greater than 1.5× its running minimum for 200 consecutive updates. The counters are checkpointed. |
| Memory | Stop if tracked CUDA peak allocated bytes exceed 12 GiB. OOM is a failure; the supervisor stops further arms instead of retrying the same failing setup. |
| Time | Shared 1,800 s monotonic campaign deadline. At step 200, project duration from elapsed time and stop if above 3× the arm's calibrated projection. |
| Device and disk | Poll every approximately 2 s. Temperature ≥88 °C, <10 GiB free, monitor failure, or an operator-created stop file request a clean interrupt. Device telemetry includes all active workloads. Checks are sampled and cannot guarantee detection of between-poll transients. |

The supervisor records per-arm commands and exit statuses and stops launching new arms on failure. The live progress path is the console log; structured learning curves use the attempt-aware metric reader. A step-based time limit bounds expected cooperative work. It cannot preempt a hung driver. A raw process kill can leave a stale writer lock and uncommitted temporary files.

## Checkpoint transaction and exact replay

1. A training writer takes an exclusive lock before changing experiment artifacts. It scans step directories and the latest index read-only, rejects inconsistent or stale pointers and checks model/config/data signatures.
2. Saving creates a staged directory, writes alias-aware safetensors plus optimizer, scheduler, scaler and RNG files, verifies the safetensors key set and records hashes when enabled. Publish the checkpoint directory by rename, then atomically replace the latest index. In verified mode an existing committed version is preserved.
3. Resume verifies SHA-256 against recorded files before applying weights and restores state from an available backup in place before recovering its public name. It restores optimizer, schedule, scaler if used, Python/NumPy/Torch and CUDA RNG, sampler cursor, pilot limit state and cumulative training step/token count.
4. Training refuses trajectory-defining changes; an explicit step extension is recorded as a protocol deviation. Logging and evaluation cadence are separate from model trajectory. Reconciled metrics suppress rows from superseded attempt suffixes and preserve the underlying raw log.
5. An interrupt between committed updates can save the latest clean state. An interrupt during optimizer mutation can leave partially advanced state, so no rescue is published from that update. Resume reuses the latest prior committed checkpoint. If the committed index is missing or stale, examine recorded step directories and hashes before deciding a recovery action. The code will not guess.
6. Status begins RUNNING for a fresh ID; successful campaign runs finish COMPLETED, nonfinite or other errors use FAILED, and coordinated stop uses INTERRUPTED. This terminal metadata follows artifact persistence. DML-0001 is historical and its recorded RUNNING status was preserved.

Trainer and CLI share the lock through final metadata publication. Separate legacy readers or administrative tools are outside that guarantee. PyTorch optimizer and RNG deserialization can execute code from untrusted pickle-backed files. Only restore artifacts from a trusted operator and use verified hashes; a hash does not establish the source's trust. Safetensors model reads are separate. Crash-safety does not guarantee storage durability across sudden power loss, and verified checkpoint recovery is scoped to the recorded configurations. [CUDA incident](../findings/CUDA-RESUME-INCIDENT.md) · [Artifact schema and commands](../reproduce/README.md)
