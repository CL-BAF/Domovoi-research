# Memory and inference qualification

**Target:** at most 4 GiB total peak inference GPU memory, batch 1, usable 8,192-token context, no CPU or disk offloading, and at least 8 generated tokens/s on the specified reference hardware. Lower memory at comparable capability would be preferable. The RTX 5080 is the training device; the historic 4 GiB project reference was the RTX 2050. A qualifying result still needs a pinned reference device and workload. No current run qualifies.

## Derivation

Using [the architecture notation](mathematical-foundations.md), one biased stored block contains:

| Term | Scalars |
|---|---:|
| Fused QKV | $3d^2+3d$ |
| Attention output projection | $d^2+d$ |
| Two MLP projections | $2df+f+d$ |
| Two affine LayerNorms | $4d$ |
| Sum $P$ | $4d^2+2df+9d+f$ |

Let $P_\mathrm{common}=(V+C)d+2d$ for tied embeddings, position table and final LayerNorm. For an untied head add $Vd$. $P_\mathrm{unique}=P_\mathrm{common}+nP$ and $P_\mathrm{applied}=P_\mathrm{common}+AP$. Applied count is an accounting convention for operator uses, with shared tensors counted again by application. It is neither the number of trainable parameters nor exact FLOPs. At d=640, f=1728, V=4096 and C=512, $P=3,857,728$ and $P_\mathrm{common}=2,950,400$, giving 10,665,856 and 33,812,224 for two and eight stored blocks. DML-0001's ten stored blocks give 41,527,680. [Accounting source](https://github.com/CL-BAF/Domovoi-Labs/blob/d914684dc730cd32132aeff389d6291c447ed7e4/domovoi/profiling/accounting.py)

For L attended keys per query, estimated dense forward FLOPs per token are $A(8d^2+4df+4dL)+2Vd$. Both K/Q dot products and multiplying weights by values contribute $2dL$ per application. Causal full sequences have a logical mean $L=(T+1)/2$; dense implementations may compute masked work. Decode tends toward $L=p+1$. Backward plus forward is approximated at three times forward for matrix products. Optimizer updates, pointwise operations, reduction, launches and memory copies are excluded. For B02 versus R02×4, deeper activation and compute costs remain even with equal storage. B08 and R02×4 have equal modeled matrix-product FLOPs per token.

A logical KV cache for B sequences and S positions is $2BASdb_\mathrm{KV}$ bytes. Every scheduled application uses a different state, so this scales with A, including under tying. Parameter memory is $P_\mathrm{unique}b_w$ bytes. For FP32 training, a first-order persistent subtotal for weights, gradients and two Adam moments is $16P_\mathrm{unique}$ bytes when all gradients and moments exist. This leaves out scalar step state, temporary casts, optimizer intermediates, saved activations and allocator overhead. Peak is an overlap of lifetimes, so subtotals at different stages cannot be added and compared directly to a measured peak. Saved block activations scale with $BAT$ times width and MLP dimensions, plus the attention backend's context-dependent storage. Materialized attention scores can add $BAhTS$ values; fused backends can omit that full stored matrix.

## Cache and generation

The native `ApplicationCache` is list-like and carries the originating model and schedule metadata. Validation checks number of slots, shapes, common prefix length, batch, head geometry, context capacity, device and expected projected dtype. A plain list can be structurally valid without provenance. Each cache slot holds keys and values of shape [B,h,p,q]. In cached forward, projection covers only new positions; K/V concatenate the prefix. Decode remembers one cache pair per executed depth. Fresh prefill K/V can retain the underlying fused QKV tensor because the pair comes from views, giving up to 1.5× logical cache backing in the observed path. A concatenation can have old storage and new allocation live simultaneously. Runtime projections, activations, logits [B,T,V], sorting for sampling, CUDA workspaces and kernels introduce further transient memory. Chunk mask creation can allocate an additional [T,S] Boolean matrix.

Generation enters eval mode for a loaded checkpoint, uses the matching tokenizer, and blocks prompts exceeding C. Greedy argmax is selected by zero temperature. Stochastic sampling applies temperature, optional top-k, and nucleus filtering before multinomial sampling with an optional seeded generator. The prefill supplies the first token. Decode adds one token at a time and returns null decode rate when no decode token is generated. Timing synchronizes CUDA. TTFT measures from model prefill start through sampling the first output; it excludes input encoding and model loading. GPU memory statistics track allocator peaks. The device-wide reading in the profiler is a snapshot, whose timestamp and scope differ from the allocator peak. [Cache implementation](https://github.com/CL-BAF/Domovoi-Labs/blob/d914684dc730cd32132aeff389d6291c447ed7e4/domovoi/models/kv_cache.py) · [Generation](https://github.com/CL-BAF/Domovoi-Labs/blob/d914684dc730cd32132aeff389d6291c447ed7e4/domovoi/inference/generate.py)

## Conditional numerical example

An untrained thought experiment grows the learned position table to C=8192 and uses B=1, d=640, V=4096 and eight or sixteen applications. The implemented models trained at 512 positions. Weight and KV values below assume actual BF16 residency and dtype, which differs from the FP32 checkpoint-load path unless explicitly cast.

| Model and applications | BF16 weights, MiB | BF16 KV at 8,192, MiB | Logical subtotal, MiB |
|---|---:|---:|---:|
| B16, A=16 | 132.73 | 320.00 | 452.73 |
| R02×8, A=16 | 29.72 | 320.00 | 349.72 |

The 103.01 MiB reduction is 22.75% of the B16 logical subtotal. Retained backing storage, allocator allocations/reservations and device-wide use are outside it. Savings in peak training memory for B08 vs R02×4 also come from AdamW and training lifetimes, so they cannot be used to infer the missing inference peak. Longer sequence positions increase KV linearly and logical prefill attention work quadratically. At equal depth and width, tying leaves KV unchanged. Position extrapolation and quality at 8K are separate requirements.

## Qualification plan and remaining conditions

1. Specify reference GPU, OS, exact driver/PyTorch and kernel backend, and any other device activity. Record actual weight and KV dtypes and no offload.
2. Select an 8K positional method and validate positions, masks and offsets across prefill, chunks and single-step decode, comparing both families at common encoding.
3. Reserve enough window for output. Measure cold/warm prefill and steady decode at fixed prompts/lengths, repeated times with GPU synchronization. Use allocator allocated/reserved peaks plus device-wide peak polling with known sampling resolution and disclose possible missed transients.
4. Evaluate long-context quality with held-out document and retrieval workloads, and design equal-budget controls. Confirm ≥8 output tokens/s on the reference device.
5. Separate exact counts from peak forecasts and measured values. Profile 512-token trained checkpoints first for debugging; the CUDA smoke only verified timings/peak aggregation, without a publishable throughput benchmark.

Current alternatives include a static preallocated cache to remove repeated concatenation, grouped-query attention to reduce cache width, compact recurrence across sequence positions, and native low-bit weights. Each changes a different term. No alternative has completed a qualification experiment. [Memory and throughput methodology](../methodology/measurement-methodology.md)
