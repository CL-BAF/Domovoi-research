# Experimental controls and evidence tiers

## Questions and comparisons

The early question is whether tied depth can offer an efficiency–quality trade at a fixed 512-token training workload. The long-term question is whether a capable generator can operate at ≤4 GiB, context 8,192, batch one, no offload and at least 8 output tokens/s. The first question can locate promising candidates; it cannot answer the second.

- **B02 versus R02×4, equal stored parameters.** 10,665,856 unique parameters each means equal stored tensor size at the same dtype. Two versus eight applications changes activation footprint, FLOPs, LR sensitivity and residual initialization. More time is allowed at equal tokens.
- **B08 versus R02×4, equal applied depth.** Both use eight applications, width 640, 10 heads and ffn 1728. B08 uses 33,812,224 unique parameters and R02×4 uses 10,665,856. The modeled matmul FLOPs are equal; optimizer work, activation lifetimes and timing are measured rather than assumed equal.
- **Same stream.** Use one 4,096-entry TinyStories tokenizer, same seeded segment permutation and training draw budget, batch and precision. Compare shared checkpoints at equal cumulative scored target tokens. Every extended arm finished at 100,007,936. At 610, 1,526 and 3,052 updates the budgets are 9,994,240, 25,001,984 and 50,003,968 tokens. Calibration is separate, at 3,276,800.
- **Same environment where possible.** Record commit, dirty flag, CUDA, Python, PyTorch and GPU. V1 CUDA defaults and deterministic V2 differ by protocol. Use the same machine for paired performance observations, report timings and versions for both.
- **No statistical claim across seeds.** These are single-seed learning comparisons with deterministic data order. Document bootstrap intervals quantify conditional validation document variation. Cross-seed uncertainty, initialization and hardware variation need independent repeats. Balanced arm order reduces warmup bias. Random-ID smoke is a functionality control.

## Interpretation boundaries

A shorter calibration and an extended schedule use different warmup lengths and cosine horizons. A crossover in ranking across them is compatible with differing learning speeds, but direct causal attribution needs within-extended matched-step metrics. DML-0001 uses tinyshakespeare, older architecture and different hardware. Its throughput is a historical observation and a poor equal-workload control. TinyStories' narrow, templated stories make a language-model pipeline stress test, with general-domain relevance and near-duplicates unresolved. WikiText and FineWeb were prepared and audited, with no training comparison published.

The pilot decision thresholds have not been frozen for a multi-seed or qualification trial. Read the [versioned pilot plan](https://github.com/CL-BAF/Domovoi-Labs/blob/d914684dc730cd32132aeff389d6291c447ed7e4/docs/DOMOVOI_PILOT_PLAN.md) alongside the [hypothesis ledger](../findings/architecture-hypotheses.md). The archive records completed tests and open proposals. It does not promote any arm to a qualifying inference result.
