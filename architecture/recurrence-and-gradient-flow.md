# Recurrence and gradient flow

## Fixed application schedules

The recurrence in Domovoi is in depth. It is a fixed unroll over the full sequence tensor, with no state transition across tokens other than normal causal attention and its explicit KV cache. There is no adaptive halting or recurrent gate. The `DomovoiRecurrentTransformer` subclass uses the baseline's scheduled loop and changes the indices produced by `recurrence_schedule`.

For $n$ stored blocks and recurrence depth $r$, applications have indices $j=0,\dots,nr-1$. Cycle uses $s_j=j\bmod n$. Sequence uses $s_j=\lfloor j/r\rfloor$. For two blocks repeated four times, cycle is 0,1,0,1,0,1,0,1 and sequence is 0,0,0,0,1,1,1,1. Both spend eight applications and store the same two parameter sets, but the operations generally do not commute. Depth one retrieves the same order as an independent baseline. Cycle is the only recurrence order evaluated in the recorded GPU training. Sequence has CPU correctness evidence. [Recurrent source](https://github.com/CL-BAF/Domovoi-Labs/blob/d914684dc730cd32132aeff389d6291c447ed7e4/domovoi/models/domovoi_recurrent.py)

In code each application reads the same object from the module list for a reused index. It does not materialize untied trainable copies. Block output still depends on that application's input. An early-exit diagnostic exposes intermediate hidden states, applies the final LayerNorm/head, and records loss and residual norms. This offers a probe of intermediates; the output head is trained for final depth. No published run in this archive uses diagnostic results to select a winner.

## Backpropagation

Write $x_{j+1}=F_j(x_j;\theta_{s_j})$ and let $\delta_j=\partial L/\partial x_j$. Reverse-mode automatic differentiation applies
$$
\delta_j=J_{x_j}F_j^\mathsf{T}\delta_{j+1},\qquad
\nabla_{\theta_i}L=\sum_{j:s_j=i}J_{\theta_i}F_j^\mathsf{T}\delta_{j+1}.
$$
The sum is over actual uses of the parameter object, and the upstream gradients already carry all downstream effects. Tying the embedding adds gradient contributions both from token lookup and from the dense head. AdamW translates the final parameter gradient into one state update per stored parameter, so extra use count does not produce extra AdamW slots. Under exact arithmetic, where untied blocks are initialized with copies of the same parameters, their independent gradients sum to the shared gradient. Per-use gradient signs can reinforce or conflict. Products of Jacobians can amplify or attenuate upstream norms. The identity path mitigates an obstruction to flow without imposing a global bound. The residual scaling used at initialization depends on effective depth and could interact with correlation in a repeated layer.

## Evidence

- Depth-one tests compare full baseline state and output, and the audit independently checked baseline equivalence against the pinned original for three configurations.
- Explicit unrolled copies reproduce tied forward outputs at fixed weights. Parameter identity/gradient hooks show reuse at every application. The independent FP64 gradient sum comparison had maximum difference $6.94\times10^{-18}$ over the tested small cases.
- Recent logged training metrics collect the global pre-clip norm at the logged update only. They expose occasional large values, such as the higher early B08 norm, but cannot identify which layer or application produced a learning difference. Training did not store per-application gradients.
- Training state saves activations per use even if weights alias. CUDA peaks confirm the shallow-versus-eight-application memory gap under the tested geometry. They do not measure the allocation attributable to each activation.
- Sharing is an established idea. Fixed sharing provides a controlled starting point. Adaptive depth, attention sparsification, KV sharing across applications, residual gating and bounded recurrent state would be different models with different correctness requirements. [Universal Transformers](https://arxiv.org/abs/1807.03819) · [Layer sharing policies](https://arxiv.org/abs/2104.06022) · [Recurrent depth for language models](https://arxiv.org/abs/2502.05171)

See the [hypothesis ledger](../findings/architecture-hypotheses.md) for how these plausible mechanisms relate to observed loss and throughput.
