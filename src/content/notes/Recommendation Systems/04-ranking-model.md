---
type: note
course: "[[Recommendation Systems]]"
date: 2026-02-18
---

# Lecture 4: The Ranking Model -- PhoenixModel

> **Prerequisites**: Lecture 3 (Retrieval Model). You should understand two-tower
> architecture, hash embeddings, `block_user_reduce`, and `block_history_reduce`.
>
> **Source file**: `phoenix/recsys_model.py` (PhoenixModel, lines 245-474)
> **Scoring logic**: `phoenix/runners.py` (lines 202-387)

---

## 1. What Ranking Does vs. Retrieval

Let me set the scene with the full pipeline so you can see where ranking fits:

```
  All posts on X (~hundreds of millions)
         |
         | Fan-out / heuristic filters
         v
  Candidate pool (~millions)
         |
         | RETRIEVAL (Lecture 3): two-tower + ANN search
         v
  ~1,000 candidates (cheap similarity score each)
         |
         | RANKING (this lecture): transformer + candidate isolation  <-- YOU ARE HERE
         v
  ~1,000 candidates, now with 19 action probabilities each
         |
         | Blending / policy layer (business rules, diversity, etc.)
         v
  Your timeline (20-50 posts)
```

The key differences in a sentence:

- **Retrieval** is a *cheap filter*. It computes one dot-product similarity score
  per candidate. It can do millions per second because user and candidate towers
  are independent.

- **Ranking** is an *expensive evaluator*. It runs every candidate through a
  shared transformer alongside the user's history, producing **19 separate
  action probability predictions** per candidate. It can only handle ~1,000
  candidates because the computation is quadratic in sequence length.

Think of it this way: retrieval asks "is this post vaguely relevant?" and ranking
asks "will this user favorite it? reply to it? repost it? block the author?"

---

## 2. PhoenixModel Architecture Overview

Here is the full data flow from raw features to ranked output:

```
  +-----------+    +-----------------+    +------------------+
  | User      |    | History (S=128) |    | Candidates (C=32)|
  | Hashes    |    | Post hashes     |    | Post hashes      |
  |           |    | Author hashes   |    | Author hashes    |
  |           |    | Actions (19-d)  |    | Product surface   |
  |           |    | Product surface |    | (NO actions!)     |
  +-----------+    +-----------------+    +------------------+
       |                  |                       |
       v                  v                       v
  block_user_reduce  block_history_reduce   block_candidate_reduce
    [B, 1, D]          [B, S, D]              [B, C, D]
       |                  |                       |
       +------------------+-----------------------+
       |
       v
  Concatenate along sequence axis: [B, 1+S+C, D]
  =  [B, 161, D]  (with default config: 1 + 128 + 32)
       |
       v
  +-----------------------------------------------------+
  | Transformer (causal attention + candidate isolation) |
  | candidate_start_offset = 1 + S = 129                |
  +-----------------------------------------------------+
       |
       v
  Full output: [B, 161, D]
       |
       | Extract ONLY positions >= candidate_start_offset
       v
  Candidate outputs: [B, C, D] = [B, 32, D]
       |
       v
  LayerNorm
       |
       v
  Linear(D -> 19)   (the "unembedding" matrix, no bias)
       |
       v
  Raw logits: [B, 32, 19]
       |
       v
  Sigmoid (in runners.py, NOT in the model itself)
       |
       v
  Probabilities: [B, 32, 19]
       |
       v
  Sort by p_favorite (index 0) -> ranked_indices
```

Notice something important: the model itself (`recsys_model.py`) returns raw
logits. The sigmoid and sorting happen in `runners.py`. This is a common pattern
-- keep the model pure and put post-processing in the serving layer.

---

## 3. PhoenixModelConfig (lines 245-281)

Let's look at the config that defines the model's shape:

```python
@dataclass
class PhoenixModelConfig:
    model: TransformerConfig       # The underlying transformer config
    emb_size: int                  # Embedding dimension D (e.g., 64)
    num_actions: int               # 19 action types to predict

    history_seq_len: int = 128     # S: how many history posts to consider
    candidate_seq_len: int = 32    # C: how many candidates to rank at once

    name: Optional[str] = None
    fprop_dtype: Any = jnp.bfloat16   # Forward pass precision

    hash_config: HashConfig = None    # How many hashes per entity type
    product_surface_vocab_size: int = 16  # "home timeline", "search", etc.
```

**What is `product_surface`?** It tells the model WHERE the user saw a post.
Did they see it on the home timeline? In search results? On a profile page?
There are 16 possible surfaces, each gets its own learned embedding.

**What is `HashConfig`?**

```python
@dataclass
class HashConfig:
    num_user_hashes: int = 2       # 2 independent hash functions for users
    num_item_hashes: int = 2       # 2 for posts
    num_author_hashes: int = 2     # 2 for authors
```

Each entity (user, post, author) is hashed with multiple independent hash
functions. This is the "hashing trick" -- instead of maintaining a 1:1 mapping
of entity-to-embedding, we hash entities into a fixed-size table. Two hashes
reduce collision effects.

**PyTorch mental model:**

```python
# In PyTorch, this config would roughly map to:
class PhoenixModelConfig:
    d_model: int = 64          # emb_size
    num_actions: int = 19
    history_len: int = 128     # S
    num_candidates: int = 32   # C
    num_user_hashes: int = 2
    num_item_hashes: int = 2
    num_author_hashes: int = 2
    product_surface_vocab: int = 16
```

---

## 4. build_inputs: Constructing the Transformer's Input Sequence (lines 365-437)

This is where the magic happens. We take raw features and construct one big
sequence that the transformer will process. Let me walk you through it
step-by-step.

### Step 1: Product surface embeddings (lines 384-395)

```python
# History product surfaces: [B, S] -> [B, S, D]
history_product_surface_embeddings = self._single_hot_to_embeddings(
    batch.history_product_surface,
    config.product_surface_vocab_size,   # 16
    config.emb_size,                     # D
    "product_surface_embedding_table",   # <-- SAME table name!
)

# Candidate product surfaces: [B, C] -> [B, C, D]
candidate_product_surface_embeddings = self._single_hot_to_embeddings(
    batch.candidate_product_surface,
    config.product_surface_vocab_size,   # 16
    config.emb_size,                     # D
    "product_surface_embedding_table",   # <-- SAME table name!
)
```

Notice: both history and candidates share the **same** embedding table for
product surface. This is parameter sharing -- the model learns one set of
surface embeddings that work for both contexts.

Under the hood, `_single_hot_to_embeddings` does `one_hot(input, 16) @ table`,
which is equivalent to `nn.Embedding(16, D)` in PyTorch.

### Step 2: History action embeddings (line 397)

```python
history_actions_embeddings = self._get_action_embeddings(batch.history_actions)
```

This is the same signed embedding trick from Lecture 3:

```
actions_signed = 2 * actions - 1     # Map {0,1} -> {-1,+1}
action_emb = actions_signed @ action_projection   # [B, S, 19] @ [19, D] -> [B, S, D]
```

Why signed? Because 0 (did NOT do action) carries information too. If you just
used raw {0,1}, then "no action" would contribute nothing (0 * weight = 0).
With {-1,+1}, "did not favorite" actively pushes the embedding in the opposite
direction from "did favorite."

**Important**: only HISTORY posts have actions. Candidates do not -- the user
hasn't seen them yet, so there is no engagement data.

### Step 3: block_user_reduce -> [B, 1, D] (lines 399-405)

```python
user_embeddings, user_padding_mask = block_user_reduce(
    batch.user_hashes,                  # [B, 2]
    recsys_embeddings.user_embeddings,  # [B, 2, D]
    hash_config.num_user_hashes,        # 2
    config.emb_size,                    # D
)
```

This takes 2 user hash embeddings, flattens them to `[B, 1, 2*D]`, and
projects down to `[B, 1, D]` with a learned linear layer.

```
  [hash_emb_1 | hash_emb_2]     flatten       proj_mat_1
     [D]          [D]        -->  [2*D]   -->    [D]
```

### Step 4: block_history_reduce -> [B, S, D] (lines 407-416)

```python
history_embeddings, history_padding_mask = block_history_reduce(
    batch.history_post_hashes,                    # [B, S, 2]
    recsys_embeddings.history_post_embeddings,    # [B, S, 2, D]
    recsys_embeddings.history_author_embeddings,  # [B, S, 2, D]
    history_product_surface_embeddings,           # [B, S, D]
    history_actions_embeddings,                   # [B, S, D]
    hash_config.num_item_hashes,                  # 2
    hash_config.num_author_hashes,                # 2
)
```

This concatenates FOUR things per history position, then projects:

```
  For each of the S=128 history positions:

  [post_hash1 | post_hash2 | author_hash1 | author_hash2 | actions | product_surface]
      [D]         [D]           [D]            [D]           [D]         [D]
                                  |
                          flatten to [6*D]
                                  |
                          proj_mat_3: [6*D, D]
                                  |
                               [D]
```

Four ingredients: post identity + author identity + what the user did + where
they saw it.

### Step 5: block_candidate_reduce -> [B, C, D] (lines 418-426)

```python
candidate_embeddings, candidate_padding_mask = block_candidate_reduce(
    batch.candidate_post_hashes,                    # [B, C, 2]
    recsys_embeddings.candidate_post_embeddings,    # [B, C, 2, D]
    recsys_embeddings.candidate_author_embeddings,  # [B, C, 2, D]
    candidate_product_surface_embeddings,           # [B, C, D]
    hash_config.num_item_hashes,                    # 2
    hash_config.num_author_hashes,                  # 2
)
```

Only THREE ingredients for candidates (no actions!):

```
  For each of the C=32 candidate positions:

  [post_hash1 | post_hash2 | author_hash1 | author_hash2 | product_surface]
      [D]         [D]           [D]            [D]              [D]
                                  |
                          flatten to [5*D]
                                  |
                          proj_mat_2: [5*D, D]
                                  |
                               [D]
```

**Why no actions for candidates?** Because the user hasn't interacted with
these posts yet. That is literally why we are ranking them -- to predict
what actions the user WILL take.

### Step 6: Concatenate everything -> [B, 1+S+C, D] (lines 428-433)

```python
embeddings = jnp.concatenate(
    [user_embeddings, history_embeddings, candidate_embeddings], axis=1
)
padding_mask = jnp.concatenate(
    [user_padding_mask, history_padding_mask, candidate_padding_mask], axis=1
)
```

The resulting sequence looks like this:

```
Position:  0      1    2    ...   128     129    130   ...   160
Content: [USER | H_1 | H_2 | ... | H_128 | C_1 | C_2 | ... | C_32]
          ^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^
          1 tok        128 tokens (S)              32 tokens (C)
```

### Step 7: Record where candidates start (line 435)

```python
candidate_start_offset = user_padding_mask.shape[1] + history_padding_mask.shape[1]
# = 1 + 128 = 129
```

This integer is critical. It tells the transformer where to apply candidate
isolation (more on this soon) and tells the forward pass where to slice out
candidate outputs.

---

## 5. The Forward Pass: `__call__` (lines 439-474)

Now let's see how `build_inputs` feeds into the full forward pass:

```python
def __call__(self, batch, recsys_embeddings):
    # STEP 1: Build the input sequence
    embeddings, padding_mask, candidate_start_offset = self.build_inputs(
        batch, recsys_embeddings
    )
    # embeddings:              [B, 161, D]
    # padding_mask:            [B, 161]
    # candidate_start_offset:  129

    # STEP 2: Run through transformer (with candidate isolation!)
    model_output = self.model(
        embeddings,
        padding_mask,
        candidate_start_offset=candidate_start_offset,  # <-- THIS IS THE KEY
    )
    # model_output.embeddings: [B, 161, D]

    # STEP 3: Layer normalization on all outputs
    out_embeddings = layer_norm(model_output.embeddings)

    # STEP 4: Extract ONLY the candidate positions
    candidate_embeddings = out_embeddings[:, candidate_start_offset:, :]
    # candidate_embeddings: [B, 32, D]

    # STEP 5: Project to action logits via unembedding matrix
    unembeddings = self._get_unembedding()   # [D, 19]
    logits = jnp.dot(candidate_embeddings.astype(unembeddings.dtype), unembeddings)
    # logits: [B, 32, 19]

    return RecsysModelOutput(logits=logits)
```

**Why extract ONLY candidate outputs?** The user and history positions produce
transformer outputs too, but we don't need them. We only want to predict
engagement for the candidates. The user/history tokens exist purely to provide
context.

Think of it like a reading comprehension test: the "passage" (user + history) is
there for context, but we only grade the "answers" (candidate predictions).

---

## 6. Candidate Isolation: The Key Innovation

This is the single most important architectural decision in the ranking model.
Let me explain it carefully.

### The Attention Mask

In a standard causal (autoregressive) transformer, each position can attend to
itself and all prior positions. The mask looks like a lower-triangular matrix.

But for ranking, we need something different. Here is the attention mask for a
small example with 1 user token, 3 history tokens, and 3 candidates
(`candidate_start_offset = 4`):

```
  Query      Key -->
   |      U   H1   H2   H3   C1   C2   C3
   v    +----+----+----+----+----+----+----+
   U    | 1  | 0  | 0  | 0  | 0  | 0  | 0  |   U sees only itself
   H1   | 1  | 1  | 0  | 0  | 0  | 0  | 0  |   H1 sees U, itself
   H2   | 1  | 1  | 1  | 0  | 0  | 0  | 0  |   H2 sees U, H1, itself
   H3   | 1  | 1  | 1  | 1  | 0  | 0  | 0  |   H3 sees U, H1-H3
        +----+----+----+----+----+----+----+
   C1   | 1  | 1  | 1  | 1  | 1  | 0  | 0  |   C1 sees U+history + itself ONLY
   C2   | 1  | 1  | 1  | 1  | 0  | 1  | 0  |   C2 sees U+history + itself ONLY
   C3   | 1  | 1  | 1  | 1  | 0  | 0  | 1  |   C3 sees U+history + itself ONLY
        +----+----+----+----+----+----+----+

   Legend:  1 = can attend    0 = masked (cannot attend)
```

Look at the candidate rows (C1, C2, C3). Each candidate can see:
- All user + history tokens (the "context") -- columns U through H3
- Its own position (the diagonal) -- so it knows its own content
- **NOT** any other candidate -- the off-diagonal candidate positions are masked

### Why This Matters

**Without candidate isolation** (standard causal mask):

```
   C2 sees: [U, H1, H2, H3, C1, C2]

   The output for C2 depends on C1!
```

This means the score for candidate C2 changes depending on which other
candidates happen to be in the same batch. That is a problem because:

1. **Non-determinism**: The same candidate gets different scores depending on
   its batch-mates.
2. **No caching**: You cannot cache a candidate's score because it depends on
   the full candidate set.
3. **Ordering effects**: Changing the order of candidates changes scores.

**With candidate isolation**:

```
   C2 sees: [U, H1, H2, H3, C2]

   The output for C2 depends ONLY on user context + C2 itself.
```

This gives us:

1. **Determinism**: Same user + same candidate = same score, always.
2. **Cacheability**: Compute score once, reuse it. In production, if the user
   context hasn't changed, previously scored candidates keep their scores.
3. **Parallelism**: We batch 32 candidates for GPU efficiency, but the scores
   are independent -- we could score them one at a time and get the same result.

### How to Build This Mask

The mask has two regions:

```
  Region 1 (rows 0..offset-1): Standard causal (lower-triangular)
  Region 2 (rows offset..end):  Context columns are 1, candidate columns are
                                 diagonal-only (identity matrix)
```

In code, the logic is:

```python
def make_recsys_attn_mask(seq_len, candidate_start_offset):
    """
    Build the candidate-isolation attention mask.

    seq_len: total sequence length (1 + S + C)
    candidate_start_offset: where candidates begin (1 + S)
    """
    # Start with standard causal mask for the context region
    mask = torch.tril(torch.ones(seq_len, seq_len))

    # For candidate rows: zero out attention to OTHER candidates
    num_candidates = seq_len - candidate_start_offset
    for i in range(num_candidates):
        row = candidate_start_offset + i
        # Zero out all candidate columns
        mask[row, candidate_start_offset:] = 0
        # But allow self-attention (diagonal)
        mask[row, row] = 1

    return mask  # [seq_len, seq_len]
```

Or more efficiently (vectorized):

```python
def make_recsys_attn_mask(seq_len, candidate_start_offset):
    # Causal mask for context portion
    causal = torch.tril(torch.ones(seq_len, seq_len))

    # Candidate-to-candidate block: identity matrix only
    C = seq_len - candidate_start_offset
    cand_block = torch.eye(C)  # [C, C]

    # Replace the candidate-to-candidate sub-matrix
    causal[candidate_start_offset:, candidate_start_offset:] = cand_block

    return causal
```

---

## 7. The Unembedding Matrix (lines 353-363)

After the transformer produces contextualized candidate embeddings `[B, C, D]`,
we need to convert each D-dimensional vector into 19 action predictions.

```python
def _get_unembedding(self):
    unembed_mat = hk.get_parameter(
        "unembeddings",
        [config.emb_size, config.num_actions],   # [D, 19]
        dtype=jnp.float32,
        init=embed_init,
    )
    return unembed_mat
```

This is simply a matrix multiply: `[B, C, D] @ [D, 19] -> [B, C, 19]`.

In PyTorch terms, this is `nn.Linear(D, 19, bias=False)`.

**Why no bias?** The layer norm right before the unembedding already has a bias
term (the beta parameter). Adding another bias would be redundant.

**Why "unembedding"?** The name comes from language models. In an LLM, the
embedding matrix maps tokens -> vectors, and the "unembedding" matrix maps
vectors -> token logits. Here, instead of mapping to vocabulary logits, we map
to action logits. Same idea, different output space.

---

## 8. The 19 Actions (from runners.py lines 202-222)

Here are all 19 actions the model predicts, organized by category:

```
  INDEX   ACTION                   CATEGORY      DESCRIPTION
  -----   ------                   --------      -----------
    0      favorite_score          Positive      User likes/hearts the post
    1      reply_score             Positive      User replies to the post
    2      repost_score            Positive      User reposts (retweets)
    3      photo_expand_score      Positive      User taps to expand a photo
    4      click_score             Positive      User clicks on the post
    5      profile_click_score     Positive      User clicks the author's profile
    6      vqv_score               Positive      "Valuable Qualified View" -- meaningful view
    7      share_score             Positive      User shares the post (generic)
    8      share_via_dm_score      Positive      User shares via direct message
    9      share_via_copy_link     Positive      User copies the link
   10      dwell_score             Positive      User dwells (reads for a while)
   11      quote_score             Positive      User quote-tweets
   12      quoted_click_score      Positive      User clicks on a quoted tweet
   13      follow_author_score     Positive      User follows the author
  -----   ------                   --------      -----------
   14      not_interested_score    Negative      User marks "not interested"
   15      block_author_score      Negative      User blocks the author
   16      mute_author_score       Negative      User mutes the author
   17      report_score            Negative      User reports the post
  -----   ------                   --------      -----------
   18      dwell_time              Continuous    Predicted dwell time (seconds)
```

A few things to notice:

- **Index 0 (favorite) is the primary ranking signal.** In `runners.py`, the
  candidates are sorted by `probs[:, :, 0]` -- the favorite probability.

- **Positive actions (0-13)** are things the platform wants to encourage.

- **Negative actions (14-17)** are things the platform wants to avoid. These
  get SUBTRACTED in the final blending score (not shown in this codebase, but
  in the policy layer).

- **dwell_time (18)** is special -- it is continuous, not binary. The sigmoid
  output here represents a normalized time value, not a probability.

---

## 9. How Logits Become Probabilities (runners.py lines 341-347)

The model returns raw logits. The serving layer converts them:

```python
def hk_rank_candidates(batch, recsys_embeddings):
    output = hk_forward(batch, recsys_embeddings)
    logits = output.logits                        # [B, C, 19]

    probs = jax.nn.sigmoid(logits)                # [B, C, 19]  -- element-wise sigmoid

    primary_scores = probs[:, :, 0]               # [B, C]  -- favorite probability
    ranked_indices = jnp.argsort(-primary_scores, axis=-1)  # sort descending

    return RankingOutput(
        scores=probs,
        ranked_indices=ranked_indices,
        p_favorite_score=probs[:, :, 0],
        p_reply_score=probs[:, :, 1],
        # ... etc for all 19 actions ...
    )
```

**Why sigmoid and not softmax?**

This is an important design choice. Let me draw out the difference:

```
  SOFTMAX (what we do NOT use):
  +-----------+-----------+-----------+
  | favorite  |   reply   |  repost   |   ...must sum to 1
  |   0.6     |    0.3    |   0.1     |   "pick ONE action"
  +-----------+-----------+-----------+

  SIGMOID (what we DO use):
  +-----------+-----------+-----------+
  | favorite  |   reply   |  repost   |   ...each independent
  |   0.8     |    0.7    |   0.2     |   "which actions will happen?"
  +-----------+-----------+-----------+
```

With **softmax**, the probabilities must sum to 1. This implies the actions are
mutually exclusive -- you can only do ONE thing. But that is wrong! A user can
favorite AND reply AND repost the same tweet. These are independent actions.

With **sigmoid**, each action gets its own independent probability in [0, 1].
A post can have P(favorite)=0.9 and P(reply)=0.8 simultaneously. This correctly
models the real world where multiple engagements happen on the same post.

---

## 10. Comparison: Retrieval vs. Ranking

This table summarizes every architectural difference:

```
+---------------------------+----------------------------------+----------------------------------+
|         Aspect            |          RETRIEVAL               |           RANKING                |
|                           |    (PhoenixRetrievalModel)       |       (PhoenixModel)             |
+---------------------------+----------------------------------+----------------------------------+
| Purpose                   | Narrow millions -> ~1000         | Score ~1000 -> ranked list       |
+---------------------------+----------------------------------+----------------------------------+
| Architecture              | Two-tower (user + candidate)     | Single sequence (all together)   |
+---------------------------+----------------------------------+----------------------------------+
| User representation       | Separate transformer             | Same transformer, positions      |
|                           | -> mean pool -> L2 normalize     | 0..S in the shared sequence      |
+---------------------------+----------------------------------+----------------------------------+
| Candidate representation  | CandidateTower MLP               | block_candidate_reduce           |
|                           | (separate from user tower)       | (embedded IN the sequence)       |
+---------------------------+----------------------------------+----------------------------------+
| User x Candidate          | NONE during encoding.            | DEEP interaction via             |
| interaction               | Only dot product at retrieval.   | cross-attention in transformer.  |
+---------------------------+----------------------------------+----------------------------------+
| Output per candidate      | 1 similarity score               | 19 action probabilities          |
+---------------------------+----------------------------------+----------------------------------+
| Key architectural trick   | Two-tower separation for         | Candidate isolation mask for     |
|                           | independent encoding + ANN       | deterministic, cacheable scores  |
+---------------------------+----------------------------------+----------------------------------+
| candidate_start_offset    | None (no shared sequence)        | 1 + S = 129                     |
+---------------------------+----------------------------------+----------------------------------+
| L2 normalization          | Yes (both towers)                | No                               |
+---------------------------+----------------------------------+----------------------------------+
| Product surface used?     | No                               | Yes (for both history + cands)   |
+---------------------------+----------------------------------+----------------------------------+
| Candidate actions?        | N/A                              | No (haven't been seen yet)       |
+---------------------------+----------------------------------+----------------------------------+
| Scalability               | O(1) per candidate (dot product) | O(S+C) per candidate (attention) |
+---------------------------+----------------------------------+----------------------------------+
```

---

## 11. Key Difference: block_candidate_reduce vs. CandidateTower

These two components serve the same conceptual purpose -- "encode a candidate
post" -- but they are architecturally different for good reasons.

### CandidateTower (retrieval, from recsys_retrieval_model.py)

```
  Input: [post_hash_embs | author_hash_embs]  (concatenated along hash dim)
         Shape: [B, C, num_hashes, D]
            |
            v
         Flatten: [B, C, (num_item_hashes + num_author_hashes) * D]
            |
            v
         Linear -> [B, C, 2*D]     (expand)
            |
            v
         SiLU activation
            |
            v
         Linear -> [B, C, D]       (compress)
            |
            v
         L2 normalize              (unit sphere)
            |
            v
         Output: [B, C, D]  (normalized)
```

**Key properties:**
- Expand-then-compress MLP (bottleneck architecture)
- L2 normalized output (for dot-product similarity)
- Only 2 ingredients: post hashes + author hashes
- No product surface
- No activation data (same as ranking, but notable)

### block_candidate_reduce (ranking, from recsys_model.py)

```
  Input: [post_hash1 | post_hash2 | author_hash1 | author_hash2 | product_surface]
         Shape after flatten: [B, C, (2+2)*D + D] = [B, C, 5*D]
            |
            v
         Single Linear -> [B, C, D]   (just project down, no expansion)
            |
            v
         Output: [B, C, D]  (NOT normalized)
```

**Key properties:**
- Simple flatten + linear (no MLP, no activation function)
- NOT normalized
- 3 ingredients: post hashes + author hashes + product surface
- Structurally identical to `block_history_reduce` but without actions

### Why the difference?

**Retrieval needs normalization** because similarity is computed via dot product
between user and candidate vectors. For dot product to measure cosine
similarity, both vectors must be on the unit sphere.

**Ranking does NOT need normalization** because the candidate embedding is just
one input to the transformer. The transformer will learn to use whatever scale
is useful. Normalizing would throw away magnitude information.

**Retrieval uses an MLP** because it must encode all candidate information into
a single vector that works in isolation (no user context available). It needs
more capacity to do this well.

**Ranking uses a simple linear** because the heavy lifting happens in the
transformer, which has full cross-attention between candidates and user context.
The input projection just needs to get the features into the right dimension.

---

## 12. Deep Dive: The Reduce Functions

Let me show you the pattern shared by all three reduce functions so you really
internalize it.

### Pattern: Flatten Multiple Hash Embeddings + Project

```
  Given: N hash embeddings of dim D each, plus optional extra features

  Step 1: Flatten hash embeddings
          [B, ?, num_hashes, D] -> [B, ?, num_hashes * D]

  Step 2: Concatenate all features along last dimension
          [B, ?, num_hashes * D + extra_dims] = [B, ?, total_dim]

  Step 3: Linear projection
          [B, ?, total_dim] @ [total_dim, D] -> [B, ?, D]

  Step 4: Padding mask (hash == 0 means invalid)
          padding_mask = (first_hash != 0)
```

Here is a side-by-side summary:

```
  +---------------------+-------------+-------------------------------------------+
  | Function            | Output      | Ingredients (concatenated before proj)     |
  +---------------------+-------------+-------------------------------------------+
  | block_user_reduce   | [B, 1, D]   | user_hash1, user_hash2                    |
  |                     |             | = 2 * D dims                              |
  +---------------------+-------------+-------------------------------------------+
  | block_history_reduce| [B, S, D]   | post_hash1, post_hash2,                   |
  |                     |             | author_hash1, author_hash2,               |
  |                     |             | actions_emb, product_surface_emb          |
  |                     |             | = 4*D + D + D = 6*D dims                 |
  +---------------------+-------------+-------------------------------------------+
  | block_candidate_    | [B, C, D]   | post_hash1, post_hash2,                   |
  | reduce              |             | author_hash1, author_hash2,               |
  |                     |             | product_surface_emb                       |
  |                     |             | = 4*D + D = 5*D dims                     |
  +---------------------+-------------+-------------------------------------------+
```

---

## 13. The Action Embedding Trick (lines 293-321)

This deserves its own section because it is reused from retrieval and is a
clever technique.

```python
def _get_action_embeddings(self, actions):
    # actions: [B, S, 19]  -- binary multi-hot vector

    # Step 1: Signed encoding
    actions_signed = (2 * actions - 1).astype(jnp.float32)
    # Now: favorited=1 -> +1, not_favorited=0 -> -1

    # Step 2: Project to embedding dimension
    # action_projection: [19, D]
    action_emb = jnp.dot(actions_signed, action_projection)
    # action_emb: [B, S, D]

    # Step 3: Zero out positions with NO actions at all
    valid_mask = jnp.any(actions, axis=-1, keepdims=True)
    action_emb = action_emb * valid_mask

    return action_emb
```

**Why zero out positions with no actions?** If a history slot is padding (no
real post there), its action vector is all zeros, which becomes all -1 after
signing. That would create a non-zero embedding for a non-existent interaction.
The `valid_mask` catches this and zeros it out.

In PyTorch:

```python
class ActionEmbedding(nn.Module):
    def __init__(self, num_actions, d_model):
        super().__init__()
        self.projection = nn.Linear(num_actions, d_model, bias=False)

    def forward(self, actions):
        # actions: [B, S, num_actions], binary
        signed = 2.0 * actions - 1.0          # {0,1} -> {-1,+1}
        emb = self.projection(signed)         # [B, S, D]
        valid = actions.any(dim=-1, keepdim=True).float()
        return emb * valid
```

---

## 14. Quick Check

Test your understanding before moving on. Try to answer each question before
looking at the answer.

**Q1: What is `candidate_start_offset` for 1 user token + 128 history tokens +
32 candidates?**

<details>
<summary>Answer</summary>

`candidate_start_offset = 1 + 128 = 129`

It is the index of the first candidate in the sequence. The user occupies
position 0, history occupies positions 1 through 128, so candidates start at
129.
</details>

**Q2: Can candidate #5 see candidate #10's embedding during attention?**

<details>
<summary>Answer</summary>

No. The candidate isolation mask ensures each candidate can only attend to the
user + history context and its own position. Candidate #5 cannot see any other
candidate's embedding.
</details>

**Q3: Why do we extract only candidate outputs after the transformer, not
user/history outputs?**

<details>
<summary>Answer</summary>

Because we want to predict engagement actions for the candidates, not for
history posts. The user and history tokens exist solely to provide context. Their
transformer outputs are computed (the transformer processes the whole sequence)
but we discard them -- we only need the candidate representations at
`positions >= candidate_start_offset`.
</details>

**Q4: Why sigmoid instead of softmax for the 19 action predictions?**

<details>
<summary>Answer</summary>

Because the 19 actions are independent, not mutually exclusive. A user can
favorite AND reply AND repost the same post simultaneously. Softmax would force
the probabilities to sum to 1, implying the user can only do one thing. Sigmoid
gives each action its own independent probability in [0, 1].
</details>

**Q5: `block_candidate_reduce` concatenates 5*D features. `block_history_reduce`
concatenates 6*D features. What is the extra D in history?**

<details>
<summary>Answer</summary>

The action embeddings. History posts have associated user engagement actions
(favorite, reply, etc.) that are converted to a D-dimensional embedding via
`_get_action_embeddings`. Candidates do not have actions because the user hasn't
seen them yet.
</details>

**Q6: Why does `block_candidate_reduce` use a simple linear projection while
`CandidateTower` in retrieval uses a 2-layer MLP with SiLU activation?**

<details>
<summary>Answer</summary>

Two reasons: (1) In retrieval, the candidate representation must stand alone --
it will never interact with user features until the final dot product. It needs
more capacity to encode everything into one useful vector. In ranking, the
transformer provides deep cross-attention between candidate and user, so a simple
projection suffices. (2) In retrieval, the output must be L2-normalized for
dot-product similarity. The MLP gives the model more flexibility before that
information-destroying normalization step.
</details>

---

## 15. End-to-End Data Flow Recap

Let me trace one candidate through the entire system to make it concrete.

```
  Say we are ranking candidate C_7 for user U.

  1. INPUT FEATURES for C_7:
     - candidate_post_hashes[7]:    [hash_a, hash_b]       (2 ints)
     - candidate_author_hashes[7]:  [hash_c, hash_d]       (2 ints)
     - candidate_product_surface[7]: 3                      (1 int, e.g. "home")

  2. EMBEDDING LOOKUP (done before model, passed via RecsysEmbeddings):
     - post_emb_a, post_emb_b:     each [D]                (from hash embedding tables)
     - author_emb_c, author_emb_d: each [D]                (from hash embedding tables)
     - surface_emb_3:              [D]                      (from product_surface_embedding_table)

  3. block_candidate_reduce:
     concat = [post_emb_a | post_emb_b | author_emb_c | author_emb_d | surface_emb_3]
            = [5*D]
     c7_embedding = concat @ proj_mat_2    -> [D]

  4. INPUT SEQUENCE (position 129 + 7 - 1 = 135):
     [..., U, H_1, ..., H_128, C_1, ..., C_6, **C_7**, C_8, ..., C_32]
                                                  ^
                                              position 135

  5. TRANSFORMER with candidate isolation:
     C_7 attends to: [U, H_1, H_2, ..., H_128, C_7]   (129 + 1 = 130 tokens)
     C_7 does NOT attend to: [C_1, ..., C_6, C_8, ..., C_32]

  6. OUTPUT EXTRACTION:
     c7_output = layer_norm(transformer_output[:, 135, :])    -> [D]

  7. UNEMBEDDING:
     c7_logits = c7_output @ unembedding_matrix               -> [19]

  8. SIGMOID (in runners.py):
     c7_probs = sigmoid(c7_logits)                            -> [19]

  9. RESULT for C_7:
     p_favorite    = 0.73
     p_reply       = 0.12
     p_repost      = 0.31
     p_block       = 0.002
     ... (19 total)
```

---

## 16. PyTorch Appendix: Complete Runnable Implementation

Below is a self-contained PyTorch translation of the PhoenixModel ranking system.
You can copy this into a `.py` file and run it directly.

I have added extensive comments mapping each section back to the original JAX/Haiku
code. The architecture is faithful to the original; the main differences are
framework-specific (Haiku uses functional parameters while PyTorch uses `nn.Module`
state).

```python
"""
PyTorch translation of PhoenixModel (ranking model).

Maps to: phoenix/recsys_model.py (PhoenixModel, lines 245-474)
         phoenix/runners.py (scoring logic, lines 341-387)

Run this file directly for a demonstration:
    python phoenix_ranking_pytorch.py
"""

import math
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class HashConfig:
    """Maps to recsys_model.py HashConfig (lines 32-38)."""
    num_user_hashes: int = 2
    num_item_hashes: int = 2
    num_author_hashes: int = 2


@dataclass
class PhoenixRankingConfig:
    """
    Maps to recsys_model.py PhoenixModelConfig (lines 245-281).

    We fold TransformerConfig fields directly into this config for simplicity.
    """
    # Embedding dimensions
    d_model: int = 64                  # emb_size in original
    num_actions: int = 19              # Number of action types to predict

    # Sequence lengths
    history_seq_len: int = 128         # S
    candidate_seq_len: int = 32        # C

    # Hash config
    hash_config: HashConfig = None     # type: ignore

    # Product surface
    product_surface_vocab_size: int = 16

    # Transformer config
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 256                  # Feed-forward intermediate dim
    dropout: float = 0.0

    def __post_init__(self):
        if self.hash_config is None:
            self.hash_config = HashConfig()

    @property
    def total_seq_len(self) -> int:
        """1 (user) + S (history) + C (candidates)."""
        return 1 + self.history_seq_len + self.candidate_seq_len

    @property
    def candidate_start_offset(self) -> int:
        """Position where candidates begin in the sequence."""
        return 1 + self.history_seq_len


# ============================================================================
# Data containers
# ============================================================================

class RecsysBatch(NamedTuple):
    """
    Maps to recsys_model.py RecsysBatch (lines 62-77).

    Contains raw feature data (hashes, actions, product surfaces).
    Embeddings are looked up separately.
    """
    user_hashes: torch.Tensor               # [B, num_user_hashes]
    history_post_hashes: torch.Tensor       # [B, S, num_item_hashes]
    history_author_hashes: torch.Tensor     # [B, S, num_author_hashes]
    history_actions: torch.Tensor           # [B, S, num_actions]  float {0, 1}
    history_product_surface: torch.Tensor   # [B, S]  int indices
    candidate_post_hashes: torch.Tensor     # [B, C, num_item_hashes]
    candidate_author_hashes: torch.Tensor   # [B, C, num_author_hashes]
    candidate_product_surface: torch.Tensor # [B, C]  int indices


class RecsysEmbeddings(NamedTuple):
    """
    Maps to recsys_model.py RecsysEmbeddings (lines 42-54).

    Pre-looked-up embeddings from hash tables.
    """
    user_embeddings: torch.Tensor              # [B, num_user_hashes, D]
    history_post_embeddings: torch.Tensor      # [B, S, num_item_hashes, D]
    candidate_post_embeddings: torch.Tensor    # [B, C, num_item_hashes, D]
    history_author_embeddings: torch.Tensor    # [B, S, num_author_hashes, D]
    candidate_author_embeddings: torch.Tensor  # [B, C, num_author_hashes, D]


class RankingOutput(NamedTuple):
    """
    Maps to runners.py RankingOutput (lines 225-254).
    """
    logits: torch.Tensor          # [B, C, num_actions]  raw logits
    probs: torch.Tensor           # [B, C, num_actions]  sigmoid(logits)
    ranked_indices: torch.Tensor  # [B, C]  indices sorted by p_favorite descending


# ============================================================================
# Attention mask construction
# ============================================================================

def make_recsys_attn_mask(
    seq_len: int,
    candidate_start_offset: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Build the candidate-isolation attention mask.

    This implements the key innovation of the ranking model: candidates can
    attend to the full user+history context and to themselves, but NOT to
    other candidates.

    Maps to the mask logic inside the Grok Transformer when
    candidate_start_offset is provided.

    Args:
        seq_len: Total sequence length (1 + S + C).
        candidate_start_offset: Index where candidates begin (1 + S).
        device: Device for the output tensor.

    Returns:
        mask: [seq_len, seq_len] float tensor.
              1.0 = can attend, 0.0 = cannot attend.

    Example with seq_len=7, offset=4 (1 user + 3 history + 3 candidates):

         U  H1  H2  H3  C1  C2  C3
    U  [ 1   0   0   0   0   0   0 ]
   H1  [ 1   1   0   0   0   0   0 ]
   H2  [ 1   1   1   0   0   0   0 ]
   H3  [ 1   1   1   1   0   0   0 ]
   C1  [ 1   1   1   1   1   0   0 ]
   C2  [ 1   1   1   1   0   1   0 ]
   C3  [ 1   1   1   1   0   0   1 ]
    """
    # Start with standard causal (lower-triangular) mask
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))

    # Number of candidate positions
    C = seq_len - candidate_start_offset

    if C > 0:
        # Replace the candidate-to-candidate block with identity matrix.
        # This ensures each candidate can only see itself, not other candidates.
        candidate_block = torch.eye(C, device=device)
        mask[candidate_start_offset:, candidate_start_offset:] = candidate_block

    return mask


# ============================================================================
# Building blocks: reduce functions
# ============================================================================

class BlockUserReduce(nn.Module):
    """
    Maps to recsys_model.py block_user_reduce (lines 79-119).

    Combines multiple user hash embeddings into a single [B, 1, D] vector.
    """

    def __init__(self, num_user_hashes: int, d_model: int):
        super().__init__()
        self.num_user_hashes = num_user_hashes
        self.d_model = d_model
        # proj_mat_1 in original: [num_user_hashes * D, D]
        self.projection = nn.Linear(num_user_hashes * d_model, d_model, bias=False)

    def forward(
        self,
        user_hashes: torch.Tensor,       # [B, num_user_hashes]
        user_embeddings: torch.Tensor,   # [B, num_user_hashes, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            user_embedding: [B, 1, D]
            user_padding_mask: [B, 1] bool
        """
        B = user_embeddings.shape[0]

        # Flatten hash embeddings: [B, num_user_hashes, D] -> [B, 1, num_user_hashes * D]
        user_flat = user_embeddings.reshape(B, 1, self.num_user_hashes * self.d_model)

        # Project: [B, 1, num_user_hashes * D] -> [B, 1, D]
        user_embedding = self.projection(user_flat)

        # Padding mask: hash 0 is reserved for padding
        user_padding_mask = (user_hashes[:, 0] != 0).reshape(B, 1)

        return user_embedding, user_padding_mask


class BlockHistoryReduce(nn.Module):
    """
    Maps to recsys_model.py block_history_reduce (lines 122-182).

    Combines history embeddings (post hashes + author hashes + actions + product
    surface) into a single [B, S, D] representation.
    """

    def __init__(
        self,
        num_item_hashes: int,
        num_author_hashes: int,
        d_model: int,
    ):
        super().__init__()
        self.num_item_hashes = num_item_hashes
        self.num_author_hashes = num_author_hashes
        self.d_model = d_model

        # Input dimension: item_hashes*D + author_hashes*D + actions_D + surface_D
        input_dim = (num_item_hashes + num_author_hashes + 1 + 1) * d_model
        # proj_mat_3 in original: [input_dim, D]
        self.projection = nn.Linear(input_dim, d_model, bias=False)

    def forward(
        self,
        history_post_hashes: torch.Tensor,               # [B, S, num_item_hashes]
        history_post_embeddings: torch.Tensor,            # [B, S, num_item_hashes, D]
        history_author_embeddings: torch.Tensor,          # [B, S, num_author_hashes, D]
        history_product_surface_embeddings: torch.Tensor, # [B, S, D]
        history_actions_embeddings: torch.Tensor,         # [B, S, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            history_embeddings: [B, S, D]
            history_padding_mask: [B, S] bool
        """
        B, S, _, D = history_post_embeddings.shape

        # Flatten hash embeddings along the hash dimension
        post_flat = history_post_embeddings.reshape(B, S, self.num_item_hashes * D)
        author_flat = history_author_embeddings.reshape(B, S, self.num_author_hashes * D)

        # Concatenate all features: [B, S, (2+2)*D + D + D] = [B, S, 6*D]
        combined = torch.cat(
            [post_flat, author_flat, history_actions_embeddings, history_product_surface_embeddings],
            dim=-1,
        )

        # Project down: [B, S, 6*D] -> [B, S, D]
        history_embeddings = self.projection(combined)

        # Padding mask: first hash == 0 means this position is padding
        history_padding_mask = (history_post_hashes[:, :, 0] != 0)

        return history_embeddings, history_padding_mask


class BlockCandidateReduce(nn.Module):
    """
    Maps to recsys_model.py block_candidate_reduce (lines 185-242).

    Combines candidate embeddings (post hashes + author hashes + product surface)
    into [B, C, D]. NOTE: no actions -- candidates haven't been seen yet.
    """

    def __init__(
        self,
        num_item_hashes: int,
        num_author_hashes: int,
        d_model: int,
    ):
        super().__init__()
        self.num_item_hashes = num_item_hashes
        self.num_author_hashes = num_author_hashes
        self.d_model = d_model

        # Input dimension: item_hashes*D + author_hashes*D + surface_D
        # Note: one fewer D than history (no actions)
        input_dim = (num_item_hashes + num_author_hashes + 1) * d_model
        # proj_mat_2 in original: [input_dim, D]
        self.projection = nn.Linear(input_dim, d_model, bias=False)

    def forward(
        self,
        candidate_post_hashes: torch.Tensor,               # [B, C, num_item_hashes]
        candidate_post_embeddings: torch.Tensor,            # [B, C, num_item_hashes, D]
        candidate_author_embeddings: torch.Tensor,          # [B, C, num_author_hashes, D]
        candidate_product_surface_embeddings: torch.Tensor, # [B, C, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            candidate_embeddings: [B, C, D]
            candidate_padding_mask: [B, C] bool
        """
        B, C, _, D = candidate_post_embeddings.shape

        # Flatten hash embeddings
        post_flat = candidate_post_embeddings.reshape(B, C, self.num_item_hashes * D)
        author_flat = candidate_author_embeddings.reshape(B, C, self.num_author_hashes * D)

        # Concatenate: [B, C, (2+2)*D + D] = [B, C, 5*D]
        combined = torch.cat(
            [post_flat, author_flat, candidate_product_surface_embeddings],
            dim=-1,
        )

        # Project down: [B, C, 5*D] -> [B, C, D]
        candidate_embeddings = self.projection(combined)

        # Padding mask
        candidate_padding_mask = (candidate_post_hashes[:, :, 0] != 0)

        return candidate_embeddings, candidate_padding_mask


# ============================================================================
# Action embedding
# ============================================================================

class ActionEmbedding(nn.Module):
    """
    Maps to recsys_model.py _get_action_embeddings (lines 293-321).

    Converts binary multi-hot action vectors to dense embeddings using the
    signed encoding trick: {0, 1} -> {-1, +1}.
    """

    def __init__(self, num_actions: int, d_model: int):
        super().__init__()
        # action_projection in original: [num_actions, D]
        self.projection = nn.Linear(num_actions, d_model, bias=False)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: [B, S, num_actions]  binary float tensor {0.0, 1.0}

        Returns:
            action_embeddings: [B, S, D]
        """
        # Signed encoding: 0 -> -1, 1 -> +1
        actions_signed = 2.0 * actions - 1.0

        # Project: [B, S, num_actions] -> [B, S, D]
        action_emb = self.projection(actions_signed)

        # Zero out positions with no actions at all (padding positions)
        valid_mask = actions.any(dim=-1, keepdim=True).float()
        action_emb = action_emb * valid_mask

        return action_emb


# ============================================================================
# Transformer components
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Standard multi-head attention with support for custom attention masks."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,           # [B, T, D]
        attn_mask: torch.Tensor,    # [T, T] or [B, T, T]
        padding_mask: torch.Tensor, # [B, T] bool
    ) -> torch.Tensor:
        B, T, D = x.shape

        # Project to Q, K, V
        q = self.W_q(x).reshape(B, T, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, T, d_k]
        k = self.W_k(x).reshape(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).reshape(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, T, T]

        # Apply the candidate-isolation attention mask
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        elif attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)               # [B, 1, T, T]

        scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        # Apply padding mask: mask out attention TO padded positions
        padding_mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        scores = scores.masked_fill(~padding_mask_expanded, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # Handle all-masked rows
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        out = torch.matmul(attn_weights, v)  # [B, H, T, d_k]
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.W_o(out)

        return out


class TransformerBlock(nn.Module):
    """Single transformer block: attention + feed-forward with residual + LayerNorm."""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-norm attention
        normed = self.ln1(x)
        x = x + self.attn(normed, attn_mask, padding_mask)
        # Pre-norm feed-forward
        normed = self.ln2(x)
        x = x + self.ff(normed)
        return x


class Transformer(nn.Module):
    """
    Simple transformer encoder with configurable attention masking.

    This maps to the Grok Transformer used in the original code.
    The key feature is support for candidate_start_offset, which triggers
    the candidate isolation attention mask.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,            # [B, T, D]
        padding_mask: torch.Tensor,  # [B, T] bool
        attn_mask: torch.Tensor,     # [T, T]
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask, padding_mask)
        return x


# ============================================================================
# PhoenixRankingModel -- the main model
# ============================================================================

class PhoenixRankingModel(nn.Module):
    """
    Maps to recsys_model.py PhoenixModel (lines 284-474).

    This is the full ranking model that:
    1. Builds input embeddings from user, history, and candidate features
    2. Runs them through a transformer with candidate isolation
    3. Extracts candidate outputs and projects to action logits
    """

    def __init__(self, config: PhoenixRankingConfig):
        super().__init__()
        self.config = config
        D = config.d_model
        hc = config.hash_config

        # --- Reduce blocks ---
        self.block_user_reduce = BlockUserReduce(hc.num_user_hashes, D)
        self.block_history_reduce = BlockHistoryReduce(hc.num_item_hashes, hc.num_author_hashes, D)
        self.block_candidate_reduce = BlockCandidateReduce(hc.num_item_hashes, hc.num_author_hashes, D)

        # --- Action embedding ---
        self.action_embedding = ActionEmbedding(config.num_actions, D)

        # --- Product surface embedding (shared between history and candidates) ---
        # Maps to _single_hot_to_embeddings with "product_surface_embedding_table"
        self.product_surface_embedding = nn.Embedding(
            config.product_surface_vocab_size, D
        )

        # --- Transformer ---
        self.transformer = Transformer(
            d_model=D,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
        )

        # --- Output layer norm ---
        # Maps to layer_norm(model_output.embeddings) at line 466
        self.output_ln = nn.LayerNorm(D)

        # --- Unembedding matrix ---
        # Maps to _get_unembedding (lines 353-363): [D, num_actions]
        # In PyTorch, nn.Linear(D, num_actions, bias=False) stores weight as [num_actions, D]
        # and computes x @ W^T, which gives the same result as x @ [D, num_actions].
        self.unembedding = nn.Linear(D, config.num_actions, bias=False)

        # Pre-compute and register the attention mask as a buffer (not a parameter)
        attn_mask = make_recsys_attn_mask(
            config.total_seq_len,
            config.candidate_start_offset,
        )
        self.register_buffer("attn_mask", attn_mask)

    def build_inputs(
        self,
        batch: RecsysBatch,
        recsys_embeddings: RecsysEmbeddings,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Maps to recsys_model.py build_inputs (lines 365-437).

        Constructs the full input sequence for the transformer.

        Returns:
            embeddings:              [B, 1+S+C, D]
            padding_mask:            [B, 1+S+C]  bool
            candidate_start_offset:  int
        """
        config = self.config

        # --- Step 1: Product surface embeddings (lines 384-395) ---
        # Shared embedding table for both history and candidates
        history_surface_emb = self.product_surface_embedding(
            batch.history_product_surface      # [B, S] -> [B, S, D]
        )
        candidate_surface_emb = self.product_surface_embedding(
            batch.candidate_product_surface    # [B, C] -> [B, C, D]
        )

        # --- Step 2: History action embeddings (line 397) ---
        history_action_emb = self.action_embedding(
            batch.history_actions              # [B, S, 19] -> [B, S, D]
        )

        # --- Step 3: block_user_reduce (lines 399-405) ---
        user_emb, user_mask = self.block_user_reduce(
            batch.user_hashes,                 # [B, num_user_hashes]
            recsys_embeddings.user_embeddings, # [B, num_user_hashes, D]
        )
        # user_emb: [B, 1, D], user_mask: [B, 1]

        # --- Step 4: block_history_reduce (lines 407-416) ---
        history_emb, history_mask = self.block_history_reduce(
            batch.history_post_hashes,
            recsys_embeddings.history_post_embeddings,
            recsys_embeddings.history_author_embeddings,
            history_surface_emb,
            history_action_emb,
        )
        # history_emb: [B, S, D], history_mask: [B, S]

        # --- Step 5: block_candidate_reduce (lines 418-426) ---
        candidate_emb, candidate_mask = self.block_candidate_reduce(
            batch.candidate_post_hashes,
            recsys_embeddings.candidate_post_embeddings,
            recsys_embeddings.candidate_author_embeddings,
            candidate_surface_emb,
        )
        # candidate_emb: [B, C, D], candidate_mask: [B, C]

        # --- Step 6: Concatenate (lines 428-433) ---
        embeddings = torch.cat([user_emb, history_emb, candidate_emb], dim=1)
        # embeddings: [B, 1 + S + C, D]

        padding_mask = torch.cat([user_mask, history_mask, candidate_mask], dim=1)
        # padding_mask: [B, 1 + S + C]

        # --- Step 7: Compute offset (line 435) ---
        candidate_start_offset = user_mask.shape[1] + history_mask.shape[1]
        # = 1 + S

        return embeddings, padding_mask, candidate_start_offset

    def forward(
        self,
        batch: RecsysBatch,
        recsys_embeddings: RecsysEmbeddings,
    ) -> RankingOutput:
        """
        Maps to recsys_model.py __call__ (lines 439-474) plus
        runners.py scoring logic (lines 341-371).

        Full forward pass: build inputs -> transformer -> extract candidates ->
        unembedding -> sigmoid -> sort.

        Args:
            batch: RecsysBatch with raw features.
            recsys_embeddings: Pre-looked-up hash embeddings.

        Returns:
            RankingOutput with logits, probabilities, and ranked indices.
        """
        # Step 1: Build input sequence
        embeddings, padding_mask, offset = self.build_inputs(batch, recsys_embeddings)
        # embeddings: [B, 161, D]
        # padding_mask: [B, 161]
        # offset: 129

        # Step 2: Transformer with candidate isolation mask
        transformer_out = self.transformer(
            embeddings,
            padding_mask,
            self.attn_mask,  # [161, 161] with candidate isolation
        )
        # transformer_out: [B, 161, D]

        # Step 3: Layer norm (line 466)
        normed = self.output_ln(transformer_out)

        # Step 4: Extract ONLY candidate outputs (line 468)
        candidate_out = normed[:, offset:, :]
        # candidate_out: [B, C, D] = [B, 32, D]

        # Step 5: Unembedding -> logits (lines 470-472)
        logits = self.unembedding(candidate_out)
        # logits: [B, C, 19]

        # Step 6: Sigmoid -> probabilities (runners.py line 343)
        probs = torch.sigmoid(logits)
        # probs: [B, C, 19]

        # Step 7: Rank by favorite probability (runners.py lines 345-347)
        primary_scores = probs[:, :, 0]   # [B, C]
        ranked_indices = torch.argsort(primary_scores, dim=-1, descending=True)
        # ranked_indices: [B, C]

        return RankingOutput(
            logits=logits,
            probs=probs,
            ranked_indices=ranked_indices,
        )


# ============================================================================
# Action names (from runners.py lines 202-222)
# ============================================================================

ACTIONS: List[str] = [
    "favorite_score",          # 0  -- PRIMARY ranking signal
    "reply_score",             # 1
    "repost_score",            # 2
    "photo_expand_score",      # 3
    "click_score",             # 4
    "profile_click_score",     # 5
    "vqv_score",               # 6
    "share_score",             # 7
    "share_via_dm_score",      # 8
    "share_via_copy_link_score",  # 9
    "dwell_score",             # 10
    "quote_score",             # 11
    "quoted_click_score",      # 12
    "follow_author_score",     # 13
    "not_interested_score",    # 14 (negative)
    "block_author_score",      # 15 (negative)
    "mute_author_score",       # 16 (negative)
    "report_score",            # 17 (negative)
    "dwell_time",              # 18 (continuous)
]


# ============================================================================
# Helper: create random example data
# ============================================================================

def create_example_data(
    config: PhoenixRankingConfig,
    batch_size: int = 4,
    num_user_embeddings: int = 100_000,
    num_post_embeddings: int = 100_000,
    num_author_embeddings: int = 100_000,
) -> Tuple[RecsysBatch, RecsysEmbeddings]:
    """
    Maps to runners.py create_example_batch (lines 389-487).

    Creates random batch data for testing. In production, these would come
    from feature stores and embedding lookup services.
    """
    hc = config.hash_config
    S = config.history_seq_len
    C = config.candidate_seq_len
    D = config.d_model
    B = batch_size

    torch.manual_seed(42)

    # --- Hashes (integer IDs into embedding tables) ---
    user_hashes = torch.randint(1, num_user_embeddings, (B, hc.num_user_hashes))
    history_post_hashes = torch.randint(1, num_post_embeddings, (B, S, hc.num_item_hashes))
    history_author_hashes = torch.randint(1, num_author_embeddings, (B, S, hc.num_author_hashes))
    candidate_post_hashes = torch.randint(1, num_post_embeddings, (B, C, hc.num_item_hashes))
    candidate_author_hashes = torch.randint(1, num_author_embeddings, (B, C, hc.num_author_hashes))

    # Mark some history positions as padding (hash = 0)
    for b in range(B):
        valid_len = torch.randint(S // 2, S + 1, (1,)).item()
        history_post_hashes[b, valid_len:, :] = 0
        history_author_hashes[b, valid_len:, :] = 0

    # --- Actions (binary multi-hot) ---
    history_actions = (torch.rand(B, S, config.num_actions) > 0.7).float()

    # --- Product surfaces (categorical) ---
    history_product_surface = torch.randint(0, config.product_surface_vocab_size, (B, S))
    candidate_product_surface = torch.randint(0, config.product_surface_vocab_size, (B, C))

    batch = RecsysBatch(
        user_hashes=user_hashes,
        history_post_hashes=history_post_hashes,
        history_author_hashes=history_author_hashes,
        history_actions=history_actions,
        history_product_surface=history_product_surface,
        candidate_post_hashes=candidate_post_hashes,
        candidate_author_hashes=candidate_author_hashes,
        candidate_product_surface=candidate_product_surface,
    )

    # --- Pre-looked-up embeddings (in production, these come from embedding tables) ---
    embeddings = RecsysEmbeddings(
        user_embeddings=torch.randn(B, hc.num_user_hashes, D),
        history_post_embeddings=torch.randn(B, S, hc.num_item_hashes, D),
        candidate_post_embeddings=torch.randn(B, C, hc.num_item_hashes, D),
        history_author_embeddings=torch.randn(B, S, hc.num_author_hashes, D),
        candidate_author_embeddings=torch.randn(B, C, hc.num_author_hashes, D),
    )

    return batch, embeddings


# ============================================================================
# Demo: run the model end-to-end
# ============================================================================

def main():
    """Run a complete forward pass and print results."""

    print("=" * 70)
    print("PhoenixRankingModel -- PyTorch Demo")
    print("=" * 70)

    # 1. Create config
    config = PhoenixRankingConfig(
        d_model=64,
        num_actions=19,
        history_seq_len=128,
        candidate_seq_len=32,
        num_heads=4,
        num_layers=4,
        ff_dim=256,
    )
    print(f"
Config:")
    print(f"  d_model            = {config.d_model}")
    print(f"  num_actions        = {config.num_actions}")
    print(f"  history_seq_len    = {config.history_seq_len}")
    print(f"  candidate_seq_len  = {config.candidate_seq_len}")
    print(f"  total_seq_len      = {config.total_seq_len}")
    print(f"  candidate_offset   = {config.candidate_start_offset}")

    # 2. Create model
    model = PhoenixRankingModel(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"
Model parameters: {num_params:,}")

    # 3. Create example data
    batch_size = 4
    batch, embeddings = create_example_data(config, batch_size=batch_size)
    print(f"
Batch size: {batch_size}")
    print(f"Input shapes:")
    print(f"  user_hashes:             {list(batch.user_hashes.shape)}")
    print(f"  history_post_hashes:     {list(batch.history_post_hashes.shape)}")
    print(f"  history_actions:         {list(batch.history_actions.shape)}")
    print(f"  candidate_post_hashes:   {list(batch.candidate_post_hashes.shape)}")
    print(f"  user_embeddings:         {list(embeddings.user_embeddings.shape)}")
    print(f"  history_post_embeddings: {list(embeddings.history_post_embeddings.shape)}")
    print(f"  candidate_post_embeddings: {list(embeddings.candidate_post_embeddings.shape)}")

    # 4. Forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch, embeddings)

    print(f"
Output shapes:")
    print(f"  logits:          {list(output.logits.shape)}")
    print(f"  probs:           {list(output.probs.shape)}")
    print(f"  ranked_indices:  {list(output.ranked_indices.shape)}")

    # 5. Show predictions for first user, top-3 candidates
    print(f"
--- Predictions for User 0 ---")
    top3 = output.ranked_indices[0, :3]
    for rank, cand_idx in enumerate(top3):
        cand_idx = cand_idx.item()
        print(f"
  Rank {rank+1}: Candidate {cand_idx}")
        for action_idx, action_name in enumerate(ACTIONS):
            prob = output.probs[0, cand_idx, action_idx].item()
            bar = "#" * int(prob * 40)
            print(f"    {action_name:30s}  {prob:.4f}  |{bar}")

    # 6. Demonstrate candidate isolation
    print(f"
--- Candidate Isolation Verification ---")
    print(f"Attention mask shape: {list(model.attn_mask.shape)}")
    offset = config.candidate_start_offset
    # Check that candidate 0 cannot see candidate 1
    c0_sees_c1 = model.attn_mask[offset, offset + 1].item()
    c0_sees_self = model.attn_mask[offset, offset].item()
    c0_sees_last_hist = model.attn_mask[offset, offset - 1].item()
    print(f"  Candidate 0 sees candidate 1?     {c0_sees_c1} (should be 0.0)")
    print(f"  Candidate 0 sees itself?           {c0_sees_self} (should be 1.0)")
    print(f"  Candidate 0 sees last history?     {c0_sees_last_hist} (should be 1.0)")

    # 7. Visualize a small attention mask
    print(f"
--- Small Attention Mask Example (1U + 3H + 3C) ---")
    small_mask = make_recsys_attn_mask(seq_len=7, candidate_start_offset=4)
    labels = ["U ", "H1", "H2", "H3", "C1", "C2", "C3"]
    print(f"        {' '.join(labels)}")
    for i, label in enumerate(labels):
        row = "  ".join(str(int(small_mask[i, j].item())) for j in range(7))
        print(f"  {label}  [ {row} ]")

    print(f"
Done. The model successfully ranked {config.candidate_seq_len} "
          f"candidates with {config.num_actions} action predictions each.")


if __name__ == "__main__":
    main()
```

---

## 17. Summary

Here is what you should take away from this lecture:

1. **Ranking is the expensive, high-fidelity scorer** that follows retrieval in
   the pipeline. It takes ~1,000 candidates and predicts 19 independent action
   probabilities for each.

2. **The input sequence is `[User | History | Candidates]`**, with dimensions
   `[B, 1+128+32, D]`. Everything goes through one shared transformer.

3. **Candidate isolation** is the key innovation. Each candidate can attend to
   user+history and itself, but NOT to other candidates. This makes scores
   deterministic and cacheable.

4. **The output is sliced, not pooled.** We extract only the candidate positions
   from the transformer output, apply layer norm, then project to 19 logits
   with a bias-free linear layer.

5. **Sigmoid, not softmax.** Actions are independent (you can favorite AND
   reply), so each gets its own probability via sigmoid.

6. **block_candidate_reduce is simpler than CandidateTower** because the
   transformer provides deep user-candidate interaction. Retrieval's
   CandidateTower needs more capacity because it operates in isolation.

**Next up**: Lecture 5 will cover the blending/policy layer -- how the 19
probabilities get combined into a single final score for timeline ordering,
including how negative signals (block, mute, report) are penalized.

---

*File: `phoenix/recsys_model.py` (PhoenixModel, lines 245-474)*
*File: `phoenix/runners.py` (scoring logic, lines 202-387)*
