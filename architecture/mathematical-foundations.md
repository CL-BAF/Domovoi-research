# Mathematical foundations and implementation map

Source examined: [Domovoi Labs d914684](https://github.com/CL-BAF/Domovoi-Labs/tree/d914684dc730cd32132aeff389d6291c447ed7e4). This reference develops the equations behind the [architecture overview](shared-core-transformer.md). Symbols and equations refer to the implemented dense causal decoder.

## Tensors and operations

Let B be batch size, T new input tokens, p cached prefix length, S=p+T, V vocabulary size, C context capacity, d model width, h attention heads, q=d/h head width, f MLP width, n independent stored blocks and r repetition count. Baselines have r=1. A=nr is the applied depth. Input and target IDs are integer tensors of shape [B,T]. A shifted example pairs inputs at token indices k through k+T−1 with targets k+1 through k+T.

| Component | Implemented operation and configuration | Rationale, alternatives and evidence |
|---|---|---|
| Architecture registry and builders | `domovoi/models/__init__.py` resolves `model.architecture`. Model config validates width/head divisibility, positive sizes, recurrence order/depth and dropout. Builders instantiate baseline or recurrent subclass; the operator inside each block is shared across architectures. | Sharing executable primitives controls implementation differences. An independent recurrent kernel might reduce dispatch overhead but would confound this comparison. Config tests cover opt-ins, backward compatibility and invalid values. |
| Token and position embeddings | `BaselineTransformer` uses `wte` [V,d] and `wpe` [C,d]. Add the positional vector at p+i to each token vector. Broadcasting yields [B,T,d]. `context_length` is a hard table limit and `d_model` controls width. | A learned absolute table keeps the baseline simple. Rotary or relative positions are new variants requiring a controlled migration. Cached decoding checks position offset. |
| Pre-LayerNorm | `Block.ln1` and `ln2` normalize over the last d features, independently per token. Each uses one gain and offset per feature, with the PyTorch default epsilon. `ln_f` normalizes the state before the head. | Pre-normalization gives a direct residual bypass through depth. Post-LN or RMSNorm could change optimization and compute. Shape tests and reference trajectories cover the current choice. |
| Q/K/V and heads | `CausalSelfAttention.qkv` is one affine d→3d. Split its result and reshape/transposes each [B,T,d] into [B,h,T,q]. All h heads are dense and pair queries with their own keys and values. `n_heads` controls h. | Fusing projections avoids three calls. Multi-query or grouped-query attention could reduce K/V width but would change the current baseline. Head shape and cache tests check this layout. |
| Causal attention | PyTorch SDPA scales head dot products by inverse square root of q, applies permitted-position mask, then mixes V. Training attention dropout uses `model.dropout` and is zero in eval. Fresh sequences use the causal flag. Cached singleton decode omits a mask because all supplied keys are legal. Cached chunks use an offset lower triangle. | PyTorch chooses an available backend. Masking and parity tests cover the logic; kernel choice affects time, memory and rounding. Explicit reference attention is useful for validation, and need not be the training implementation. |
| Head merge and attention residual | Transpose attention back to [B,T,h,q], make contiguous, reshape to [B,T,d], apply d→d projection and add to the input. | The projection mixes head information. Residual addition retains an identity contribution in backward. |
| Feed-forward and residual | `MLP.fc` d→f, exact `F.gelu`, `MLP.proj` f→d, then residual addition. The inner width is `ffn_dim`. Linear projections in the standard builders use biases; the vocabulary head does not. | A single GELU MLP provides a conventional control. Gated MLPs would change parameter accounting and initialization. |
| Output and loss | `lm_head` returns [B,T,V]. `tie_embeddings` aliases its weight to the same `Parameter` as `wte`. Training flattens [B,T,V] to [BT,V], flattens shifted targets, and computes mean cross-entropy. | The head still incurs vocabulary projection FLOPs. Untying adds Vd independent parameters and is supported. Tests check tied object identity and untied checkpoint completeness. |
| Application loop | `application_schedule()` returns independent block indices for the baseline and repeated indices for the recurrent model. Hidden states advance after every application. | It isolates depth sharing without a second operator implementation. Tests compare depth one against the baseline and deep recurrence against explicit copies at equal weights. |

## Equations

With residual vector $x$ for one token, LayerNorm computes $\mu=d^{-1}\sum_i x_i$, $\sigma^2=d^{-1}\sum_i(x_i-\mu)^2$, and then $\gamma\odot(x-\mu)/\sqrt{\sigma^2+\epsilon}+\beta$. Statistics are across features, which preserves causal independence across positions. A block at application $j$ applies:
$$
u_j=x_j+\mathrm{Attn}_{s_j}(\mathrm{LN}_{1,s_j}(x_j)),\qquad
x_{j+1}=u_j+W_{2,s_j}\,\mathrm{GELU}(W_{1,s_j}\mathrm{LN}_{2,s_j}(u_j)+b_{1,s_j})+b_{2,s_j}.
$$
Here $s_j$ is the stored block selected by the schedule. GELU uses the Gaussian-CDF form $x\Phi(x)$ through PyTorch's default implementation. For each head, the attention matrix is
$$
A_j=\mathrm{softmax}_{k}\left(Q_jK_j^\mathsf{T}/\sqrt{q}+M\right).
$$
For new query $i$ and key $k$, $M_{ik}=0$ when $k\le p+i$, and $-\infty$ otherwise. Values are mixed by $A_jV_j$. Note that V in a value tensor is overloaded in common notation; the standalone V in dimensions means vocabulary size. The mask, tokenwise MLP, tokenwise normalization and residuals imply by induction that changing tokens after a position cannot change that position's output in exact arithmetic.

The output loss for target $y_{bt}$ with logits $z_{btv}$ is
$$
\ell_{bt}=\log\sum_{v=1}^{V}\exp(z_{btv})-z_{bt,y_{bt}},\qquad
L=(BT)^{-1}\sum_{bt}\ell_{bt}.
$$
Packed training batches have fixed length. For full-split evaluation, padding is ignored and target losses are summed before division by the actual total target count. [Evaluation methodology](../methodology/evaluation-methodology.md)

## Initialization details

Embeddings and linear weights use zero-mean normal initialization with default scale 0.02, and linear biases start at zero. At the end of building a model, the attention and MLP residual-output matrices are reinitialized with standard deviation $0.02/\sqrt{2A}$. A tied vocabulary head and input table refer to a single tensor, so traversal initialization can write that alias more than once. The final initialization still represents one shared parameter. Same seeds pair sampling exactly but can consume different initialization RNG streams when architecture sizes differ. Recurrence and effective-depth-dependent initialization both change the dynamics of a comparison against B02. [Baseline source](https://github.com/CL-BAF/Domovoi-Labs/blob/d914684dc730cd32132aeff389d6291c447ed7e4/domovoi/models/baseline_transformer.py)

For tested mathematical correctness and the evidence boundary, see [recurrence and gradient flow](recurrence-and-gradient-flow.md), [memory accounting](memory-complexity.md), and the [CPU audit record](../experiments/CPU-AUDITS.md).
