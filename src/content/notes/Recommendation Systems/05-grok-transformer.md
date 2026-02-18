---
type: note
course: "[[Recommendation Systems]]"
date: 2026-02-18
---

# Lecture 05 -- The Grok Transformer in X's Recommendation Algorithm

> **Source file:** `phoenix/grok.py`
> **Audience:** MLE intern
> **Prerequisites:** Basic understanding of transformers, matrix multiplication, attention

---

## 0. Why This File Matters

Every tweet you see in your "For You" feed was scored by a transformer. Not GPT, not
BERT -- a custom Grok transformer, adapted from the xAI language model architecture
and re-purposed for recommendation ranking and retrieval. This file (`grok.py`) is
that transformer. It is roughly 590 lines of JAX/Haiku code and implements every
component from scratch.

Your job after this lecture: be able to read any line of `grok.py`, know what it does,
and explain *why* it was designed that way.

---

## 1. The Building Blocks (LEGO Analogy)

Think of the transformer like a LEGO tower. Each layer is a floor, and each floor has
the same two rooms: an *attention room* and a *feed-forward room*. Here is the full
hierarchy:

```
Transformer (lines 504-586)
  |
  +-- DecoderLayer x N (lines 443-497)           <-- one "floor"
        |
        +-- MHABlock (lines 378-411)              <-- attention room
        |     |
        |     +-- MultiHeadAttention (264-363)
        |           |
        |           +-- RotaryEmbedding (205-261) <-- position encoding
        |
        +-- DenseBlock (lines 414-440)            <-- feed-forward room (SwiGLU MLP)
```

Supporting cast:

| Module           | Lines     | Role                            |
|------------------|-----------|---------------------------------|
| `TransformerConfig` | 88-109  | Hyperparameter container        |
| `RMSNorm`        | 162-194   | Normalization (replaces LayerNorm) |
| `Linear`         | 121-159   | Custom linear projection        |
| `RotaryEmbedding`| 205-261   | Rotary position embeddings      |
| `make_recsys_attn_mask` | 39-71 | The recsys-specific mask      |

Let's walk through each one, bottom-up.

---

## 2. TransformerConfig (lines 88-109)

This is just a `@dataclass` that bundles hyperparameters:

```python
@dataclass
class TransformerConfig:
    emb_size: int               # embedding dimension (e.g., 128)
    key_size: int               # key/query head dimension (e.g., 64)
    num_q_heads: int            # number of query heads (e.g., 2)
    num_kv_heads: int           # number of key/value heads (e.g., 2)
    num_layers: int             # how many decoder layers (e.g., 2)
    widening_factor: float = 4.0    # FFN width = widening_factor * emb_size (adjusted)
    attn_output_multiplier: float = 1.0  # scales attention logits before softmax
```

**Key relationship:** `num_q_heads >= num_kv_heads` and `num_q_heads % num_kv_heads == 0`.
When they are equal, you get standard multi-head attention. When `num_q_heads > num_kv_heads`,
you get *Grouped Query Attention* (GQA) -- we will cover this in section 7.

The `make()` method just instantiates a `Transformer` from the config. Nothing fancy.

---

## 3. The Transformer Top Level (lines 504-586)

This is the entry point. It takes embeddings and a padding mask, applies an attention
mask strategy, then runs N decoder layers in sequence.

```
                 embeddings [B, T, D]
                 mask       [B, T]      (True = valid token)
                       |
              +--------+---------+
              |                  |
     candidate_start_offset     None
     is provided?               (standard mode)
              |                  |
     recsys mask           causal mask
     (section 4)           (lower triangle)
              |                  |
              +--------+---------+
                       |
                padding_mask * attn_mask
                       |
              DecoderLayer 0
                       |
              DecoderLayer 1
                       |
                      ...
                       |
              DecoderLayer N-1
                       |
                 output [B, T, D]
```

In pseudocode:

```python
def __call__(self, embeddings, mask, candidate_start_offset=None):
    padding_mask = mask[:, None, None, :]          # [B, 1, 1, T]

    if candidate_start_offset is not None:
        # RECSYS MODE: candidates are isolated from each other
        attn_mask = make_recsys_attn_mask(seq_len, candidate_start_offset)
        mask = padding_mask * attn_mask
    else:
        # STANDARD CAUSAL MODE: each token sees only past tokens
        causal_mask = tril(ones(seq_len, seq_len))  # lower triangle
        mask = padding_mask * causal_mask

    h = embeddings
    for i in range(num_layers):
        h = DecoderLayer(h, mask)

    return h
```

**Two modes:**

| Mode | When used | Mask shape |
|------|-----------|------------|
| Causal | Retrieval user tower (auto-regressive) | Lower triangle |
| RecSys | Ranking with candidates | Causal + candidate isolation |

The causal mode is what you see in any GPT-style model. The recsys mode is the
interesting part -- let's dig in.

---

## 4. `make_recsys_attn_mask` (lines 39-71) -- THE KEY INNOVATION

This is the single most important function for understanding *why* this transformer
works for recommendations. Everything else is standard modern transformer engineering.
This mask is what makes it a *recommendation* transformer.

### The Problem

You have a sequence:

```
[User features] [History item 1] [History item 2] ... [Candidate 1] [Candidate 2] [Candidate 3]
```

You want each candidate scored independently -- candidate 1's score should NOT be
influenced by the fact that candidate 2 happens to be in the same batch. If you used a
normal causal mask, candidate 2 could "see" candidate 1 through attention, creating an
unwanted dependency.

### The Solution: Three Steps

```python
def make_recsys_attn_mask(seq_len, candidate_start_offset):
    # Step 1: Start with a standard causal (lower-triangular) mask
    causal_mask = tril(ones(seq_len, seq_len))

    # Step 2: Zero out the entire candidate-to-candidate block (bottom-right)
    attn_mask[candidate_start_offset:, candidate_start_offset:] = 0

    # Step 3: Restore self-attention on the diagonal for candidates
    for i in range(candidate_start_offset, seq_len):
        attn_mask[i, i] = 1

    return attn_mask
```

### Visual: seq_len=7, candidate_start_offset=4

Tokens: U=user, H1-H3=history, C1-C3=candidates.

```
              Attending TO -->
              U   H1   H2   H3   C1   C2   C3
         U  [ 1    0    0    0    0    0    0 ]   U: sees only itself
        H1  [ 1    1    0    0    0    0    0 ]   H1: sees U + self
   A    H2  [ 1    1    1    0    0    0    0 ]   H2: sees U, H1 + self
   t    H3  [ 1    1    1    1    0    0    0 ]   H3: sees U, H1, H2 + self
   t   ----  ---- ---- ---- ----+---- ---- ----
   e    C1  [ 1    1    1    1  | 1    0    0 ]   C1: user+history + SELF only
   n    C2  [ 1    1    1    1  | 0    1    0 ]   C2: user+history + SELF only
   d    C3  [ 1    1    1    1  | 0    0    1 ]   C3: user+history + SELF only
   s                            |
   FROM                   candidate block:
                          diagonal only (no cross-candidate attention)
```

**Why this matters:**

- Each candidate gets the SAME context (user + history) through attention.
- No candidate can influence another candidate's representation.
- This means you can batch all candidates together for efficiency, but each one is
  scored as if it were the only candidate.
- It is the best of both worlds: batch efficiency + independent scoring.

### The JAX Implementation (actual code)

```python
causal_mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=dtype))

# .at[].set() is JAX's way of doing in-place updates (since JAX arrays are immutable)
attn_mask = causal_mask.at[:, :, candidate_start_offset:, candidate_start_offset:].set(0)

candidate_indices = jnp.arange(candidate_start_offset, seq_len)
attn_mask = attn_mask.at[:, :, candidate_indices, candidate_indices].set(1)
```

Note the `(1, 1, seq_len, seq_len)` shape -- the leading dims are for broadcasting
across batch and head dimensions.

---

## 5. RMSNorm (lines 162-194)

Before we get to the decoder layer, we need to understand the normalization it uses.

### RMSNorm vs LayerNorm

```
LayerNorm:  center by mean, scale by std, apply learned scale + bias
RMSNorm:    scale by root-mean-square only, apply learned scale (NO mean, NO bias)
```

Mathematically:

```
LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta

RMSNorm(x)   = scale * x / sqrt(mean(x^2) + eps)
```

RMSNorm is simpler and faster. The insight from the original paper (Zhang & Sennrich,
2019) is that re-centering by the mean is unnecessary -- the scaling alone is enough.
It is used in LLaMA, Grok, Mistral, and most modern transformers.

### The Grok Implementation

```python
class RMSNorm(hk.RMSNorm):
    def __call__(self, inputs):
        fprop_dtype = inputs.dtype
        # Learned scale parameter, initialized to 0
        scale = hk.get_parameter("scale", (inputs.shape[-1],), init=Constant(0))

        inputs = inputs.astype(jnp.float32)       # compute in float32 for stability
        mean_squared = jnp.mean(jnp.square(inputs), axis=[-1], keepdims=True)
        normed = inputs * jax.lax.rsqrt(mean_squared + self.eps)  # rsqrt = 1/sqrt
        output = scale * normed

        return output.astype(fprop_dtype)          # cast back to original dtype
```

**Two things to notice:**

1. **Scale initialized to zero.** This means at initialization, the norm outputs all
   zeros. Combined with the residual connection, the layer initially acts as an
   identity function. This is a training stability trick -- the network starts simple
   and gradually "grows" each layer's contribution.

2. **Computation in float32.** The square, mean, and rsqrt operations are done in
   float32 even if the model runs in bfloat16. This prevents numerical issues in
   the normalization denominator.

### PyTorch Translation

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(dim))  # note: zeros, not ones

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(mean_sq + self.eps)
        return (self.scale.float() * x_normed).to(dtype)
```

---

## 6. Linear Layer (lines 121-159)

A thin wrapper around `hk.Linear` with two differences from the PyTorch default:

1. **Weights initialized to zero** (`hk.initializers.Constant(0)`) -- same reasoning as
   RMSNorm scale. At init, every linear layer outputs zeros, making residual layers
   start as identity.
2. **dtype preservation** -- computation preserves the input dtype (important for
   mixed-precision training where inputs may be bfloat16).

```python
class Linear(hk.Linear):
    def __call__(self, inputs):
        fprop_dtype = inputs.dtype
        w = hk.get_parameter("w", [input_size, output_size], jnp.float32, init=Constant(0))
        out = jnp.dot(inputs, w.astype(fprop_dtype))
        if self.with_bias:
            b = hk.get_parameter("b", [output_size], jnp.float32, init=Constant(0))
            out = out + b.astype(fprop_dtype)
        return out
```

In PyTorch, you would replicate this with `nn.init.zeros_` on both weight and bias.

---

## 7. MultiHeadAttention (lines 264-363) -- Three Special Features

This is the core attention module. It implements standard scaled dot-product attention
with three additions that make it "modern":

### A. Grouped Query Attention (GQA)

Standard MHA: every head has its own Q, K, V projections.
GQA: multiple query heads share the same K/V heads.

```
Standard MHA (num_q_heads=4, num_kv_heads=4):

  Q1 -> K1, V1     Q2 -> K2, V2     Q3 -> K3, V3     Q4 -> K4, V4
  (each head independent)

GQA (num_q_heads=4, num_kv_heads=2):

  Q1 --|              Q3 --|
       +--> K1, V1         +--> K2, V2
  Q2 --|              Q4 --|
  (query heads share K/V pairs)

MQA (num_q_heads=4, num_kv_heads=1):    <-- extreme case

  Q1 --|
  Q2 --|
       +--> K1, V1
  Q3 --|
  Q4 --|
```

**Why GQA?** The K/V cache in autoregressive inference is proportional to the number of
K/V heads. With GQA you cut memory by the ratio `num_q_heads / num_kv_heads` with
minimal quality loss. The original paper (Ainslie et al., 2023) showed that
GQA with 2x reduction matches full MHA quality on most benchmarks.

In the actual code, the GQA reshape happens at line 334:

```python
# query_heads shape: [B, T, num_q_heads, key_size]
# Reshape to:        [B, T, num_kv_heads, num_q_heads // num_kv_heads, key_size]
query_heads = jnp.reshape(query_heads, (b, t, kv_h, h // kv_h, d))
```

This groups the query heads so each group shares one K/V head for the einsum.

When `num_q_heads == num_kv_heads` (as in the default config with both set to 2), GQA
reduces to standard MHA. The code handles both cases with the same reshape.

---

### B. Rotary Position Embeddings (RoPE) (lines 205-261)

Traditional transformers add position embeddings to the input (`x + pos_embed`).
RoPE instead *rotates* the query and key vectors based on their position in the
sequence. This encodes position information into the *angle* between Q and K vectors.

**Intuition:** Imagine each dimension pair of your Q/K vector as a 2D point on a
circle. RoPE rotates this point by an angle proportional to the token's position.
When you compute Q . K (dot product), the result depends on the *difference* in their
rotation angles -- which is the *relative* position. This is why RoPE naturally gives
you relative position information without explicitly computing it.

The math for one pair of dimensions:

```
x_rotated[2i]   = x[2i]   * cos(pos * freq_i) - x[2i+1] * sin(pos * freq_i)
x_rotated[2i+1] = x[2i+1] * cos(pos * freq_i) + x[2i]   * sin(pos * freq_i)
```

Where `freq_i = 1 / (base_exponent ^ (2i / dim))`.

In the code, this is implemented compactly:

```python
# rotate_half: [x1, x2] -> [-x2, x1]  (90-degree rotation helper)
def rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)

# Apply rotation:
x_rotated = x * cos(phase) + rotate_half(x) * sin(phase)
```

**Where it is applied** (lines 326-328):

```python
rotate = RotaryEmbedding(dim=self.key_size, base_exponent=10000)
key_heads   = rotate(key_heads,   seq_dim=1, offset=0)
query_heads = rotate(query_heads, seq_dim=1, offset=0)
```

Note: RoPE is applied to Q and K only, NOT to V. The value vectors carry raw content;
only Q and K need position awareness for the attention pattern.

**Who uses RoPE?** LLaMA, Mistral, Grok, Gemma, CodeLlama -- essentially every
major open-source LLM since 2023. It replaced learned position embeddings because:
- It works for any sequence length (no max position limit).
- Relative position information emerges naturally.
- No extra parameters to learn.

---

### C. Attention Soft Capping (lines 342-343)

```python
max_attn_val = jnp.array(30.0, dtype=attn_logits.dtype)
attn_logits = max_attn_val * jnp.tanh(attn_logits / max_attn_val)
```

This is a clever trick from Google's Gemma 2 paper. Here is what it does:

```
Without capping:   attn_logits can be arbitrarily large
                   -> softmax collapses to one-hot
                   -> gradient vanishes for non-max entries
                   -> training becomes unstable

With soft capping:  tanh squashes logits / 30 to [-1, 1]
                    multiply by 30 -> logits clamped to [-30, 30]
                    -> softmax stays smooth
                    -> gradients flow to all entries
```

Visualizing the capping function:

```
  attn_logit
  output
    30 |                          _______________
       |                    ___--
       |                 _--
     0 |_______________--
       |             _--
       |         ___-
   -30 |________-
       +---+---+---+---+---+---+----> raw attn_logit
         -100  -50   0   50  100

  Near zero:  approximately identity (tanh(x) ~ x for small x)
  Far from zero:  smoothly saturates at +/- 30
```

The value 30 is deliberately large -- for typical logit values (say -5 to +15), the
capping has almost no effect. It only kicks in for extreme outliers, which is exactly
when you need it.

---

### D. The Full Attention Flow

Putting it all together, here is the complete data flow through `MultiHeadAttention`:

```
Input: x [B, T, D]
   |
   +---> Linear_q --> Q [B, T, num_q_heads, key_size]
   +---> Linear_k --> K [B, T, num_kv_heads, key_size]
   +---> Linear_v --> V [B, T, num_kv_heads, value_size]
   |
   +---> Apply RoPE to Q and K (not V)
   |
   +---> Reshape Q for GQA: [B, T, num_kv_heads, num_q_heads // num_kv_heads, key_size]
   |
   +---> Attention logits: einsum('...thHd,...Thd->...hHtT', Q, K)
   |         (each query group attends to its corresponding K head)
   |
   +---> Scale: logits *= attn_output_multiplier
   |
   +---> Soft cap: logits = 30 * tanh(logits / 30)
   |
   +---> Apply mask: where(mask, logits, -1e30)
   |
   +---> Softmax over last dim (the T dimension of K)
   |
   +---> Weighted sum: einsum('...hHtT,...Thd->...thHd', weights, V)
   |
   +---> Reshape: [B, T, num_q_heads * value_size]
   |
   +---> Linear_out --> [B, T, D]
```

The final output has the same shape as the input. This is important because it
plugs back into the residual connection.

---

## 8. DenseBlock / FFN (lines 414-440) -- SwiGLU

This is NOT a standard two-layer ReLU FFN. It uses a Gated Linear Unit (GLU) variant
with GELU activation, commonly called SwiGLU (though the activation here is GELU
rather than SiLU/Swish -- the gating mechanism is the same).

### Standard FFN vs SwiGLU

```
Standard FFN (what BERT/GPT-2 use):
  output = W2 @ ReLU(W1 @ x + b1) + b2
  Parameters: 2 matrices

SwiGLU / GeGLU (what Grok uses):
  h_v  = W_v @ x               # "value" path -- carries information
  h_w1 = GELU(W1 @ x)          # "gate" path  -- decides what to keep
  output = W_out @ (h_w1 * h_v) # gated combination
  Parameters: 3 matrices
```

ASCII diagram of the data flow:

```
            x [B, T, D]
           / \
          /   \
    Linear_v   Linear_w1
    (no act)   + GELU
         |       |
       h_v     h_w1         h_v = "what information to carry"
         \     /             h_w1 = "how much to let through" (gate)
          \   /
           * *               element-wise multiply (GATING)
            |
        Linear_out
            |
      output [B, T, D]
```

**Why gating works better:** The gate path (`h_w1`) learns to selectively zero out
dimensions. Think of it like a valve -- some features pass through fully, others get
suppressed. This is more expressive than a simple nonlinearity because the
suppression/activation decision is *input-dependent* and *dimension-specific*.

### FFN Size Calculation (lines 32-36)

```python
def ffn_size(emb_size, widening_factor):
    _ffn_size = int(widening_factor * emb_size) * 2 // 3
    _ffn_size = _ffn_size + (8 - _ffn_size) % 8  # round up to multiple of 8
    return _ffn_size
```

Two things here:

1. **The `* 2 // 3` factor.** SwiGLU has 3 weight matrices instead of the standard
   FFN's 2. To keep the total parameter count roughly the same, the hidden dimension
   is reduced by a factor of 2/3. Math: standard FFN has `2 * D * H` parameters.
   SwiGLU has `3 * D * H'` parameters. Setting them equal: `H' = 2H/3`.

2. **Rounding to multiple of 8.** Modern GPUs (NVIDIA A100, H100) and TPUs have
   tensor cores that operate on tiles of 8x8 (or 16x16). If your matrix dimension is
   not a multiple of 8, the hardware pads it anyway -- wasting compute. Rounding up
   ensures you use every FLOP you pay for.

Example: `emb_size=128, widening_factor=4.0`

```
raw     = 4.0 * 128 = 512
adjusted = 512 * 2 // 3 = 341
rounded  = 341 + (8 - 341 % 8) % 8 = 341 + (8 - 5) % 8 = 341 + 3 = 344
```

So the FFN hidden dim is 344, not 512.

---

## 9. DecoderLayer (lines 443-497)

Each decoder layer combines attention and FFN with a specific normalization pattern.

### The Architecture: Pre-Norm + Post-Norm with Residual

```
Input h
  |
  +----> RMSNorm -----> MHABlock(h, mask) -----> RMSNorm ---+
  |                                                         |
  +<------------------------------ + <----------------------+
  |                            (residual add)
  |
  +----> RMSNorm -----> DenseBlock(h) ---------> RMSNorm ---+
  |                                                         |
  +<------------------------------ + <----------------------+
  |                            (residual add)
Output h
```

This is an unusual pattern. Most transformers use either:
- **Pre-norm** (LLaMA, GPT-NeoX): norm *before* the sub-layer, residual around the whole thing.
- **Post-norm** (original transformer): residual first, *then* norm.

Grok uses **both**: norm before the sub-layer AND norm after the sub-layer, then
add the residual. Combined with the zero-initialized scale in RMSNorm, this means:

1. At initialization, the post-norm outputs ~zero (because scale starts at zero).
2. The residual `h += 0` keeps `h` unchanged.
3. As training progresses, the scales grow from zero, and each layer gradually
   contributes more to the output.

This is a training stability technique. It avoids the "signal explosion" problem where
randomly-initialized layers produce wildly different outputs.

### The Code

```python
def __call__(self, inputs, mask, padding_mask):
    h = inputs

    # --- Attention sub-layer ---
    h_attn = MHABlock(...)(layer_norm(h), mask).embeddings   # pre-norm + attention
    h_attn = layer_norm(h_attn)                              # post-norm
    h += h_attn                                              # residual

    # --- FFN sub-layer ---
    h_dense = DenseBlock(...)(layer_norm(h))                  # pre-norm + FFN
    h_dense = layer_norm(h_dense)                             # post-norm
    h += h_dense                                              # residual

    return DecoderOutput(embeddings=h)
```

### PyTorch Translation

```python
def forward(self, h, mask):
    # Attention sub-layer
    h = h + self.post_attn_norm(self.attention(self.pre_attn_norm(h), mask))
    # FFN sub-layer
    h = h + self.post_ffn_norm(self.ffn(self.pre_ffn_norm(h)))
    return h
```

---

## 10. MHABlock (lines 378-411)

This is a thin wrapper around `MultiHeadAttention`. It exists to keep the decoder layer
code clean. All it does is:

1. Validate mask shapes.
2. Call `MultiHeadAttention(query=inputs, key=inputs, value=inputs, mask=mask)`.

Since query, key, and value are all the same tensor (`inputs`), this is
**self-attention**. There is no cross-attention in this architecture -- every token
attends to other tokens in the same sequence.

---

## 11. What Makes This Transformer Special for RecSys

Let's compare this to a standard LLM transformer:

```
+------------------------+--------------------------+----------------------------+
| Feature                | Standard LLM Transformer | Grok RecSys Transformer    |
+------------------------+--------------------------+----------------------------+
| Attention mask         | Causal (triangular)      | Causal + candidate         |
|                        |                          | isolation (section 4)      |
+------------------------+--------------------------+----------------------------+
| FFN activation         | ReLU or GELU             | SwiGLU (gated, section 8)  |
+------------------------+--------------------------+----------------------------+
| Position encoding      | Learned or sinusoidal    | RoPE (rotary, section 7B)  |
+------------------------+--------------------------+----------------------------+
| Normalization          | LayerNorm                | RMSNorm (section 5)        |
+------------------------+--------------------------+----------------------------+
| Attention capping      | None                     | tanh soft cap at +/- 30    |
|                        |                          | (section 7C)               |
+------------------------+--------------------------+----------------------------+
| Query/KV heads         | Equal (standard MHA)     | GQA supported (section 7A) |
+------------------------+--------------------------+----------------------------+
| Weight initialization  | Xavier/Kaiming           | Zeros (section 6)          |
+------------------------+--------------------------+----------------------------+
| Norm placement         | Pre-norm or post-norm    | Both pre-norm AND post-norm|
+------------------------+--------------------------+----------------------------+
```

**The honest summary:** The ONLY truly novel thing for recsys is the attention mask
(`make_recsys_attn_mask`). Everything else -- RoPE, SwiGLU, RMSNorm, GQA, soft
capping -- is standard modern transformer architecture (post-LLaMA era, 2023+). And
that is fine! The insight is that you can take a well-engineered LLM backbone and adapt
it for recommendations with a single, elegant mask change.

---

## 12. Quick Knowledge Check

Test yourself before moving on.

1. **In the recsys attention mask, can history token H2 attend to candidate C1?**

   > Answer: No. The causal mask prevents it -- H2 is at position 2, C1 is at position
   > 4, and tokens can only attend to positions <= their own in the causal region.

2. **Can candidate C2 attend to candidate C1?**

   > Answer: No. The candidate-to-candidate block is zeroed out (except the diagonal).
   > C2 can only see user+history tokens and itself.

3. **What does RoPE encode that learned position embeddings do not?**

   > Answer: Relative position. RoPE encodes position via rotation angles, so the
   > dot product Q . K naturally depends on the *difference* in positions. Learned
   > embeddings encode absolute position only.

4. **Why SwiGLU instead of standard ReLU FFN?**

   > Answer: The gating mechanism is more expressive -- one path controls what
   > information passes through, the other carries the information. This
   > input-dependent gating outperforms a fixed nonlinearity in practice.

5. **Why round FFN size to a multiple of 8?**

   > Answer: GPU/TPU tensor cores operate on 8x8 (or larger) tiles. Non-multiple-of-8
   > dimensions waste hardware cycles on padding.

6. **Why initialize all weights (and RMSNorm scale) to zero?**

   > Answer: With zero weights, each sub-layer initially outputs zeros. Combined with
   > residual connections, each layer starts as identity. The network gradually "grows"
   > each layer's contribution during training, improving stability.

---

## 13. End-to-End Data Flow (Putting It All Together)

Here is what happens when a batch of recommendation data passes through the transformer:

```
Input: embeddings [B=32, T=200, D=128]
       mask [B=32, T=200] (True for real tokens, False for padding)
       candidate_start_offset = 150

Step 1: Build attention mask
  padding_mask:  [32, 1, 1, 200]   (broadcast-ready)
  recsys_mask:   [1, 1, 200, 200]  (causal + candidate isolation)
  final_mask:    [32, 1, 200, 200] (element-wise product)

Step 2: Decoder Layer 0
  2a. Pre-norm h -> RMSNorm
  2b. Attention:
      - Project to Q [32, 200, 2, 64], K [32, 200, 2, 64], V [32, 200, 2, 64]
      - RoPE on Q and K
      - GQA reshape Q -> [32, 200, 2, 1, 64] (num_q_heads//num_kv_heads = 1)
      - Compute QK^T, scale, soft-cap, mask, softmax
      - Weighted sum of V -> [32, 200, 128]
      - Output projection -> [32, 200, 128]
  2c. Post-norm -> RMSNorm
  2d. Residual add: h = h + h_attn

  2e. Pre-norm h -> RMSNorm
  2f. FFN (SwiGLU):
      - Linear_v:  [32, 200, 128] -> [32, 200, 344]
      - Linear_w1: [32, 200, 128] -> [32, 200, 344] -> GELU
      - Gate: h_w1 * h_v -> [32, 200, 344]
      - Linear_out: [32, 200, 344] -> [32, 200, 128]
  2g. Post-norm -> RMSNorm
  2h. Residual add: h = h + h_dense

Step 3: Decoder Layer 1
  (same structure as above)

Output: [32, 200, 128]
  - Positions 0-149: user+history representations
  - Positions 150-199: candidate representations (each independent)
  - Downstream: take candidate positions, project to score for ranking
```

---

## Appendix A: Full Runnable PyTorch Implementation

Below is a complete, self-contained PyTorch implementation of every module in `grok.py`.
You can copy this into a `.py` file and run it directly. The `__main__` block at the
bottom includes a smoke test.

```python
"""
Grok Transformer for Recommendations -- PyTorch Implementation
Translated from phoenix/grok.py (JAX/Haiku) for educational purposes.

Usage:
    python grok_pytorch.py

This will run a smoke test with dummy data in both causal and recsys modes.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility: FFN size calculation
# ---------------------------------------------------------------------------

def ffn_size(emb_size: int, widening_factor: float) -> int:
    """Compute FFN hidden dimension, adjusted for SwiGLU and hardware alignment.

    The 2/3 factor compensates for SwiGLU having 3 matrices instead of 2.
    Rounding to multiple of 8 ensures efficient tensor core utilization.
    """
    _ffn_size = int(widening_factor * emb_size) * 2 // 3
    _ffn_size = _ffn_size + (8 - _ffn_size) % 8  # round up to multiple of 8
    return _ffn_size


# ---------------------------------------------------------------------------
# Utility: Recommendation system attention mask
# ---------------------------------------------------------------------------

def make_recsys_attn_mask(
    seq_len: int,
    candidate_start_offset: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create attention mask for recommendation system inference.

    Produces a [1, 1, seq_len, seq_len] mask where:
    - Positions 0..offset-1 (user+history): standard causal attention
    - Positions offset..end (candidates): attend to user+history and self only,
      NOT to other candidates.

    Args:
        seq_len: Total sequence length
        candidate_start_offset: Where candidates begin in the sequence
        dtype: Desired mask dtype
        device: Desired mask device

    Returns:
        Mask tensor of shape [1, 1, seq_len, seq_len], 1=attend, 0=block
    """
    # Step 1: Standard causal (lower-triangular) mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=dtype, device=device))

    # Step 2: Zero out candidate-to-candidate block (bottom-right)
    causal_mask[candidate_start_offset:, candidate_start_offset:] = 0

    # Step 3: Restore self-attention for candidates (diagonal of candidate block)
    diag_indices = torch.arange(candidate_start_offset, seq_len, device=device)
    causal_mask[diag_indices, diag_indices] = 1

    # Add batch and head dimensions for broadcasting
    return causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm does not center by the mean or learn a bias.
    It normalizes by the root-mean-square of the input and applies a learned scale.

    Scale is initialized to zero so that the sub-layer starts as identity
    (when combined with a residual connection).
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(mean_sq + self.eps)
        return (self.scale.float() * x_normed).to(orig_dtype)


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE).

    Encodes position by rotating pairs of dimensions in the query/key vectors.
    The dot product Q . K then naturally depends on relative position.

    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, base_exponent: int = 10000):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        self.dim = dim
        self.base_exponent = base_exponent

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply rotary embeddings to input tensor.

        Args:
            x: Input tensor of shape [B, T, num_heads, dim]
            offset: Position offset (useful for incremental decoding)

        Returns:
            Rotated tensor of same shape as input
        """
        _, seq_len, _, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Compute inverse frequencies: 1 / (base ^ (2i / dim)) for i in [0, dim/2)
        exponents = torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (self.base_exponent ** (exponents / self.dim))

        # Compute position indices
        t = torch.arange(seq_len, dtype=torch.float32, device=device) + offset

        # Compute phase angles: [T, dim/2]
        phase = torch.einsum("t,d->td", t, inv_freq)

        # Duplicate for both sin and cos: [T, dim]
        phase = torch.cat([phase, phase], dim=-1)

        # Reshape for broadcasting: [1, T, 1, dim]
        phase = phase.unsqueeze(0).unsqueeze(2)

        # Apply rotation
        x_float = x.float()
        x_rotated = x_float * phase.cos() + self._rotate_half(x_float) * phase.sin()

        return x_rotated.to(dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Split x into two halves and rotate: [x1, x2] -> [-x2, x1]."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


# ---------------------------------------------------------------------------
# Multi-Head Attention with GQA, RoPE, and Soft Capping
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Multi-head attention with Grouped Query Attention, RoPE, and soft capping.

    Supports:
    - Standard MHA (when num_q_heads == num_kv_heads)
    - Grouped Query Attention (when num_q_heads > num_kv_heads)
    - Rotary position embeddings on Q and K
    - Soft capping of attention logits at +/- 30
    """

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        key_size: int,
        model_size: int,
        value_size: Optional[int] = None,
        attn_output_multiplier: float = 1.0,
        max_attn_val: float = 30.0,
    ):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0, (
            f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.key_size = key_size
        self.value_size = value_size or key_size
        self.model_size = model_size
        self.attn_output_multiplier = attn_output_multiplier
        self.max_attn_val = max_attn_val
        self.heads_per_group = num_q_heads // num_kv_heads

        # Projection layers (no bias, weights initialized to zero per Grok convention)
        self.q_proj = nn.Linear(model_size, num_q_heads * key_size, bias=False)
        self.k_proj = nn.Linear(model_size, num_kv_heads * key_size, bias=False)
        self.v_proj = nn.Linear(model_size, num_kv_heads * self.value_size, bias=False)
        self.out_proj = nn.Linear(num_q_heads * self.value_size, model_size, bias=False)

        # RoPE for query and key
        self.rope = RotaryEmbedding(dim=key_size)

        # Initialize all weights to zero (Grok convention for training stability)
        self._init_weights()

    def _init_weights(self):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.zeros_(module.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [B, T, D]
            mask: Attention mask [B, 1, T, T], 1=attend, 0=block

        Returns:
            Output embeddings [B, T, D]
        """
        B, T, _ = x.shape

        # --- Project to Q, K, V ---
        q = self.q_proj(x).view(B, T, self.num_q_heads, self.key_size)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.key_size)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.value_size)

        # --- Apply RoPE to Q and K (not V) ---
        q = self.rope(q)
        k = self.rope(k)

        # --- Reshape Q for GQA ---
        # From [B, T, num_q_heads, key_size]
        # To   [B, T, num_kv_heads, heads_per_group, key_size]
        q = q.view(B, T, self.num_kv_heads, self.heads_per_group, self.key_size)

        # --- Compute attention logits ---
        # Q: [B, T_q, kv_h, H_per_group, d]  -> "b t h g d"
        # K: [B, T_k, kv_h, d]               -> "b s h d"
        # Result: [B, kv_h, H_per_group, T_q, T_k] -> "b h g t s"
        attn_logits = torch.einsum("bthgd,bshd->bhgts", q, k).float()

        # --- Scale ---
        attn_logits = attn_logits * self.attn_output_multiplier

        # --- Soft capping: 30 * tanh(logits / 30) ---
        attn_logits = self.max_attn_val * torch.tanh(attn_logits / self.max_attn_val)

        # --- Apply mask ---
        # mask shape: [B, 1, T, T] -> expand to [B, 1, 1, T, T] for GQA dims
        expanded_mask = mask.unsqueeze(2)  # [B, 1, 1, T, T]
        attn_logits = attn_logits.masked_fill(expanded_mask == 0, -1e30)

        # --- Softmax ---
        attn_weights = F.softmax(attn_logits, dim=-1).to(x.dtype)

        # --- Weighted sum of values ---
        # weights: [B, kv_h, H_per_group, T_q, T_k]
        # V:       [B, T_k, kv_h, v_size]
        # Result:  [B, T_q, kv_h, H_per_group, v_size]
        attn_output = torch.einsum("bhgts,bshd->bthgd", attn_weights, v)

        # --- Reshape back to [B, T, num_q_heads * value_size] ---
        attn_output = attn_output.reshape(B, T, self.num_q_heads * self.value_size)

        # --- Output projection ---
        return self.out_proj(attn_output)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network (DenseBlock)
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """Gated Linear Unit FFN with GELU activation (GeGLU variant).

    Uses three linear projections:
    - linear_v:  "value" path (carries information)
    - linear_w1: "gate" path (decides what to let through, with GELU)
    - linear_out: output projection

    The gate and value are combined via element-wise multiplication.
    """

    def __init__(self, model_size: int, widening_factor: float = 4.0):
        super().__init__()
        hidden_size = ffn_size(model_size, widening_factor)

        self.linear_v = nn.Linear(model_size, hidden_size, bias=False)
        self.linear_w1 = nn.Linear(model_size, hidden_size, bias=False)
        self.linear_out = nn.Linear(hidden_size, model_size, bias=False)

        # Zero initialization
        self._init_weights()

    def _init_weights(self):
        for module in [self.linear_v, self.linear_w1, self.linear_out]:
            nn.init.zeros_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, D]

        Returns:
            Output tensor [B, T, D]
        """
        h_v = self.linear_v(x)                     # value path
        h_w1 = F.gelu(self.linear_w1(x))           # gate path with GELU
        return self.linear_out(h_w1 * h_v)          # gated combination


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """Single transformer decoder layer with pre-norm + post-norm pattern.

    Architecture:
        h -> pre_attn_norm -> MHA -> post_attn_norm -> + residual
        h -> pre_ffn_norm  -> FFN -> post_ffn_norm  -> + residual
    """

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        key_size: int,
        model_size: int,
        widening_factor: float = 4.0,
        attn_output_multiplier: float = 1.0,
    ):
        super().__init__()

        # Attention sub-layer
        self.pre_attn_norm = RMSNorm(model_size)
        self.attention = MultiHeadAttention(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            key_size=key_size,
            model_size=model_size,
            attn_output_multiplier=attn_output_multiplier,
        )
        self.post_attn_norm = RMSNorm(model_size)

        # FFN sub-layer
        self.pre_ffn_norm = RMSNorm(model_size)
        self.ffn = SwiGLUFFN(model_size, widening_factor)
        self.post_ffn_norm = RMSNorm(model_size)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Input embeddings [B, T, D]
            mask: Attention mask [B, 1, T, T]

        Returns:
            Output embeddings [B, T, D]
        """
        # Attention sub-layer: pre-norm -> attention -> post-norm -> residual
        h_attn = self.attention(self.pre_attn_norm(h), mask)
        h_attn = self.post_attn_norm(h_attn)
        h = h + h_attn

        # FFN sub-layer: pre-norm -> FFN -> post-norm -> residual
        h_dense = self.ffn(self.pre_ffn_norm(h))
        h_dense = self.post_ffn_norm(h_dense)
        h = h + h_dense

        return h


# ---------------------------------------------------------------------------
# Transformer Config
# ---------------------------------------------------------------------------

@dataclass
class TransformerConfig:
    """Hyperparameters for the Grok RecSys Transformer."""
    emb_size: int = 128
    key_size: int = 64
    num_q_heads: int = 2
    num_kv_heads: int = 2
    num_layers: int = 2
    widening_factor: float = 4.0
    attn_output_multiplier: float = 1.0


# ---------------------------------------------------------------------------
# Transformer (top level)
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    """Grok Transformer for recommendation ranking and retrieval.

    Supports two attention modes:
    - Causal: standard auto-regressive masking (for retrieval user tower)
    - RecSys: causal + candidate isolation (for ranking with batched candidates)

    The mode is selected via the candidate_start_offset argument to forward().
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList([
            DecoderLayer(
                num_q_heads=config.num_q_heads,
                num_kv_heads=config.num_kv_heads,
                key_size=config.key_size,
                model_size=config.emb_size,
                widening_factor=config.widening_factor,
                attn_output_multiplier=config.attn_output_multiplier,
            )
            for _ in range(config.num_layers)
        ])

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        candidate_start_offset: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: Input embeddings [B, T, D]
            mask: Padding mask [B, T], True/1 for valid positions, False/0 for padding
            candidate_start_offset: If provided, activates recsys attention mask.
                Positions >= offset are candidates that cannot attend to each other.

        Returns:
            Output embeddings [B, T, D]
        """
        B, T, D = embeddings.shape
        device = embeddings.device
        dtype = embeddings.dtype

        # Expand padding mask: [B, T] -> [B, 1, 1, T]
        padding_mask = mask.unsqueeze(1).unsqueeze(2).to(dtype)

        if candidate_start_offset is not None:
            # RecSys mode: causal + candidate isolation
            attn_mask = make_recsys_attn_mask(T, candidate_start_offset, dtype, device)
            full_mask = padding_mask * attn_mask  # [B, 1, T, T]
        else:
            # Standard causal mode
            causal_mask = torch.tril(
                torch.ones(1, 1, T, T, dtype=dtype, device=device)
            )
            full_mask = padding_mask * causal_mask  # [B, 1, T, T]

        h = embeddings
        for layer in self.layers:
            h = layer(h, full_mask)

        return h


# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    config = TransformerConfig(
        emb_size=128,
        key_size=64,
        num_q_heads=2,
        num_kv_heads=2,
        num_layers=2,
        widening_factor=4.0,
        attn_output_multiplier=1.0,
    )

    model = Transformer(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"FFN hidden size:  {ffn_size(config.emb_size, config.widening_factor)}")

    # --- Test 1: Causal mode ---
    B, T, D = 4, 20, 128
    x = torch.randn(B, T, D)
    mask = torch.ones(B, T)
    mask[:, -3:] = 0  # last 3 tokens are padding

    out_causal = model(x, mask)
    print(f"
Causal mode:  input {x.shape} -> output {out_causal.shape}")
    assert out_causal.shape == (B, T, D), "Shape mismatch in causal mode"

    # --- Test 2: RecSys mode ---
    candidate_offset = 15
    out_recsys = model(x, mask, candidate_start_offset=candidate_offset)
    print(f"RecSys mode:  input {x.shape} -> output {out_recsys.shape}")
    assert out_recsys.shape == (B, T, D), "Shape mismatch in recsys mode"

    # --- Test 3: Verify recsys mask structure ---
    recsys_mask = make_recsys_attn_mask(7, 4)
    print(f"
RecSys mask (seq_len=7, offset=4):")
    print(recsys_mask.squeeze().int())

    # --- Test 4: Verify candidate isolation ---
    # Candidates should not attend to each other (off-diagonal in candidate block = 0)
    candidate_block = recsys_mask[0, 0, 4:, 4:]
    off_diagonal = candidate_block - torch.diag(torch.diag(candidate_block))
    assert off_diagonal.sum() == 0, "Candidates should not attend to each other!"
    print("Candidate isolation verified: no cross-candidate attention.")

    # --- Test 5: Verify candidates see user+history ---
    user_history_block = recsys_mask[0, 0, 4:, :4]
    assert user_history_block.sum() == 3 * 4, "Each candidate should see all user+history"
    print("User+history access verified: all candidates see full context.")

    print("
All tests passed.")
```

---

## Appendix B: Notation Reference

| Symbol | Meaning | Typical Value |
|--------|---------|---------------|
| B | Batch size | 32-256 |
| T | Sequence length | 50-500 |
| D | Embedding dimension (`emb_size`) | 128-1024 |
| H | Number of query heads (`num_q_heads`) | 2-16 |
| kv_h | Number of KV heads (`num_kv_heads`) | 1-16 |
| d_k | Key/query head dimension (`key_size`) | 64-128 |
| d_v | Value head dimension (defaults to `key_size`) | 64-128 |
| N | Number of layers (`num_layers`) | 2-12 |
| F | FFN hidden dimension (computed by `ffn_size`) | varies |

---

## Appendix C: Further Reading

- **RoPE:** Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021). https://arxiv.org/abs/2104.09864
- **GQA:** Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023). https://arxiv.org/abs/2305.13245
- **SwiGLU:** Shazeer, "GLU Variants Improve Transformer" (2020). https://arxiv.org/abs/2002.05202
- **RMSNorm:** Zhang and Sennrich, "Root Mean Square Layer Normalization" (2019). https://arxiv.org/abs/1910.07467
- **Soft Capping:** Gemma 2 Technical Report (2024). https://arxiv.org/abs/2408.00118
- **Original Transformer:** Vaswani et al., "Attention Is All You Need" (2017). https://arxiv.org/abs/1706.03762

---

*End of Lecture 05. Next up: how the transformer integrates with the full ranking pipeline.*
