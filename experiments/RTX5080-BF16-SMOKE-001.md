# RTX5080-BF16-SMOKE-001 · Preliminary synthetic training check

Both runs took place on 23 November 2026. This is a commissioning record. It validates GPU execution, finite losses and short-run training-memory behavior at the planned geometry. It provides no evidence of real-corpus learning, steady-state production throughput or 4 GiB inference fit.

## Hypotheses and controls

- All three configurations can execute 24 BF16 updates with the pilot geometry on this host. The CUDA profiling test should exercise synchronization and retain a peak introduced only during the first timed repeat.
- B02 and R02×4 store the same weights by parameter count, so extra executed blocks should raise training cost and saved activation memory.
- B08 and R02×4 apply eight blocks with the same tensor widths. Their modeled dense FLOPs are equal. Tying should reduce persistent parameter, gradient and optimizer state. The direction and size of a runtime improvement depend on the backend and are exploratory.

| Arm | Stored blocks | Applied blocks | Unique parameters | Modeled train GFLOP/token |
|---|---:|---:|---:|---:|
| B02 | 2 | 2 | 10,665,856 | 0.066 |
| B08 | 8 | 8 | 33,812,224 | 0.216 |
| R02×4 (cycle) | 2 | 8 | 10,665,856 | 0.216 |

All arms use width 640, 10 heads, MLP width 1728, vocabulary 4096, learned 512-token positions, a tied head and zero dropout. Microbatch 16, 512 target tokens per sequence and accumulation 2 yield 16,384 tokens per update. Each arm consumes 393,216 target tokens. The sampler visits disjoint 513-token segments, with training seed 1234 and sampling seed 1235. AdamW uses betas (0.9, 0.95), matrix weight decay 0.1, gradient clip 1.0, cosine decay to 0.1 of the 0.0006 LR, and a shortened four-step warmup. One fixed 16-sequence validation batch is scored at step 24. The smoke schedule is shorter than the planned 6,104-update pilot and does not use its calibration thresholds.

Synthetic tokens are uniform uint16 IDs in [0, 4096), generated with NumPy `default_rng(1234)`, using 410,400 train IDs followed by 16,416 validation IDs. A deterministic 4,096-entry BPE stub with zero merges provides the vocabulary and identity check. It was constructed for the test and never fit to a corpus. Data and tokenizer were temporary, with no data/model downloads. Checkpoint writing was disabled by omitting an experiment directory.

## Environment and provenance

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5080, compute capability 12.0, 16,303 MiB device capacity |
| Driver and CPU | NVIDIA 621.01, Windows WDDM, AMD Ryzen 9 9950X3D2 16-Core Processor |
| OS and Python | Windows 11 build 26200, Python 3.13.17 x64 |
| Runtime | PyTorch 2.15.0+cu132, CUDA 13.2, BF16 supported, FP32 parameters and BF16 autocast |
| Dependencies | [Observed pip package freeze](../provenance/RTX5080-BF16-SMOKE-001-pip-freeze.txt) |
| First run source | Local working tree without Git metadata. No exact commit or contemporaneous digest was captured. |
| Repeat source | Same development workspace, fingerprinted during the documentation session. [Per-file SHA-256 manifest](../provenance/RTX5080-BF16-SMOKE-001-source.json) |
| Public implementation reference | [Pinned baseline](https://github.com/CL-BAF/Domovoi-Labs/tree/5e76fc78bbe9a5c12690e84d7c29e558248e6feb). It does not include the recurrent/pilot tree exercised here. |

The repeat verifies the observed behavior on this machine with a fresh process and reversed order. An independent operator and a commit-pinned reproduction are still outstanding. A source fingerprint cannot substitute for a retrievable source commit.

## Measurement method

Run A order was B02 → B08 → R02×4. Run B reversed it, R02×4 → B08 → B02. Both ran sequentially on the same device with PyTorch allocator cache release between arms. The first eight updates of every arm were excluded from the throughput ranges below. Each reported point is one eight-update logging window of 131,072 tokens divided by elapsed wall time. `Trainer` reads scalar losses on the host, which synchronizes parts of the workload. Its periodic logging is not an explicit GPU-event benchmark and can include asynchronous boundary effects. Whole-run rates include initialization and final validation. Wall duration is reported at 0.1-second resolution. Raw window values, final sampled losses, and whole-run rates are available in the [measurement table](data/RTX5080-BF16-SMOKE-001.csv).

The training memory columns use PyTorch's `max_memory_allocated` for live allocator-tracked tensors and `max_memory_reserved` for caching-allocator reservations. The peak is over the smoke training and validation path. It excludes allocations attributable to other processes and covers a different workload from decode. The GPU was driving a display. No run-wide device-usage peak was captured.

| Arm | A post-warmup tok/s, observed range | B post-warmup tok/s, observed range | Peak allocated in both | Peak reserved in both |
|---|---:|---:|---:|---:|
| B02 | 816,718–830,185 | 862,290–862,464 | 911.57 MiB | 1,110 MiB |
| B08 | 264,196–281,148 | 275,946–281,965 | 2,257.50 MiB | 2,450 MiB |
| R02×4 | 296,249–300,760 | 297,888–303,055 | 1,851.17 MiB | 2,050 MiB |

All arms completed, with finite final validation losses near $\log 4096\approx8.318$ nats/token. Run A validation losses were B02 8.3408, B08 8.3389, R02×4 8.3405. Run B matched to the shown precision for B08/R02×4 and B02 was 8.3409. These small differences on random IDs do not measure model quality. The CUDA profiling test passed (`1 passed, 8 deselected`). Its injected first-repeat allocation verified aggregation across repeats. It asserted positive prefill and decode measurements and synchronization calls; no publishable inference tok/s or TTFT benchmark was captured.

## Interpretation

**Direct evidence.** Peak allocated and reserved memory matched across the two executions for each arm to the reported precision. R02×4 used 406.33 MiB less allocated memory than B08, about 18.0% of B08's measured peak. It used 939.60 MiB more than B02 at equal stored parameter count. The recurrent arm had a higher logged warm-window rate than B08 in both short runs. With two windows per arm and a single device, that is a preliminary observation.

**Plausible mechanisms.** B08 has 23,146,368 more unique weights than R02×4. A back-of-the-envelope 16 bytes per weight for FP32 weights, gradients and two AdamW moments gives about 353.2 MiB. This is consistent with much of the direction and scale of the 406.33 MiB observation. Different activation and allocator lifetimes can alter the remainder. The check did not break the peak into components. The B02 versus R02×4 comparison is consistent with more applied blocks requiring additional activations. The two 8-application arms share theoretical FLOPs, so fewer AdamW parameter updates or different memory traffic could contribute to the recurrent arm's throughput. Neither is established without kernel profiling.

**Order effect.** The first arm in each process had a slower opening window. In Run A B02's first window was 12,698 tok/s, and its warm windows were above 816k. In Run B R02×4 opened at 123,049 tok/s, below its roughly 300k warm windows. This is evidence of an opening-window effect, not evidence that a particular CUDA initialization step caused it. It makes the first-run whole-arm B02 rate unsuitable for comparisons. Cache warming, host scheduling, allocator state and GPU power state remain possible explanations.

## Reproduction and next measurements

The commands below preserve the historical invocation from a Domovoi Labs source tree with a separate CUDA environment. The [original synthetic harness](../reproduce/rtx5080_bf16_smoke.py.txt) is retained as source evidence, and the old executable path is retired. A new replay would need a pinned implementation and an approved plan under the [experiment protocol](../docs/EXPERIMENT_PROTOCOL.md).

```powershell
$env:PYTHONDONTWRITEBYTECODE=1
$env:OMP_NUM_THREADS=1
$env:MKL_NUM_THREADS=1
.venv-cuda\Scripts\python.exe -m pytest -q -rs -p no:cacheprovider tests/test_pilot.py -k cuda_profile
.venv-cuda\Scripts\python.exe -u <research-checkout>\reproduce\rtx5080_bf16_smoke.py
.venv-cuda\Scripts\python.exe -u <research-checkout>\reproduce\rtx5080_bf16_smoke.py --reverse
```

Before counting results, publish the implementation revision, freeze the prepared corpus and sampler manifest, then perform the 200-step calibration for every arm. Explicitly warm up and synchronize timed windows or use CUDA events over fixed windows. Repeat in balanced arm orders and also log optimizer and forward/backward times, peak allocated/reserved bytes and separate device-wide peak. Compare loss on the held-out document split at matched training tokens. Future inference work needs exact prompt lengths, output budgets, a position-encoding control and the reference 4 GiB device.
