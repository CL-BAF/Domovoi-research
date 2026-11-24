# Timing, memory and plots

## Training measurements

- Campaign CLI uses a fresh process per arm. The timed update region runs from a CUDA synchronization before batch retrieval to a synchronization after optimizer/scheduler work. It includes model forward, loss, BF16/FP32 ops, backward, clipping and transfer. It excludes periodic verification writes, validation and console I/O. Each logging window reports `step_tokens_per_s` and mean seconds per update. Published median excludes the first 20 updates. The end-to-end `avg_tok_per_s` is tokens divided by Trainer wall time and retains validation and checkpoint costs. Arm wall time is distinct from the shared campaign deadline.
- Learning curves use explicit logged update numbers. Each sampled validation point uses a fixed draw. The full-split points use a common document scorer and matched tokens. Avoid smoothing away divergence or comparing equal steps across different tokens-per-update regimes.
- Pre-clip norm is logged at the last update in each window. The min/max in the archive apply to stored samples rather than every update. Instability monitors see per-step values. GPU device telemetry uses approximately two-second samples from NVIDIA and is not an instantaneous maximum or process-only memory reading.
- PyTorch peak allocated is allocator-tracked live bytes. Peak reserved is the process caching allocator's retained pool including free blocks. These unify training and in-loop validation lifetimes and are separate from device-wide peak. Use MiB = bytes / 2^20 everywhere; field names ending in _mb are historical and emitted values use MiB.
- Checkpoint storage reports file bytes and distinguishes optimizer, weights and state. It is not peak training memory. Optimizer checkpoint size contains two AdamW moments plus small metadata and tensor serialization overhead. A shared-weight run reduces these slots, but measured peak includes activations and implementation-dependent lifetimes.

## Inference measurements

`generate.py` synchronizes timed boundaries around prefill and decode. TTFT includes prefill and first-token sampling. Decode tok/s excludes that first prefill token. GPU smoke uses injected allocator allocations and positive timing assertions; it does not expose an official throughput benchmark. The 8K total-memory target requires explicit device-wide peak sampling, allocator counters, no offload and a real context/quality test. See [memory accounting](../architecture/memory-complexity.md).

## Figure recipe

The [plotting script](../reproduce/plot_results.py) reads the sanitized structured records and produces SVG vectors. Calibration panels separate train and sampled validation windows and show synchronized throughput against allocator peaks. Extended panels show every stored sampled validation point, four exact validation checkpoints at common budgets, and telemetry time series. No interpolation beyond displayed line connections and no extrapolation are used to infer an unmeasured curve. The axis units are specified on every panel. Figures are generated from copied observations; the calculation code independently verifies budgets and medians against source tables.
