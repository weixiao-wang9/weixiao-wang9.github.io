---
type: note
course: "[[Recommendation Systems]]"
date: 2026-02-18
---

# Lecture 3: The Two-Tower Retrieval Model

**Source file:** `phoenix/recsys_retrieval_model.py`
**Prerequisites:** Lecture 2 (hash embeddings, `block_user_reduce`, `block_history_reduce`)

---

## 1. Why Two Stages?

Before we look at any code, let's talk about why this architecture exists at all.

X has hundreds of millions of posts created every day. When you open the app,
the system needs to decide which posts to show you -- and it has to do that in
under ~200 milliseconds. Here is the problem:

- The **ranking model** (Lecture 4) is a full transformer. It is accurate but
  expensive. It can maybe score ~1,000 candidates per request.
- There are ~500,000,000 candidate posts in the corpus.
- 500M / 1,000 = you would need 500,000x the compute to rank everything.

The solution used at every major recommender system (YouTube, TikTok, Netflix,
Instagram, Pinterest -- everyone) is a **two-stage funnel**:

```
         500,000,000 posts in corpus
                    |
                    v
    +----- RETRIEVAL STAGE -----+
    |  Cheap model + ANN index  |
    |  Narrow to ~1,000 posts   |
    +---------------------------+
                    |
                    v
              ~1,000 posts
                    |
                    v
    +------ RANKING STAGE ------+
    |  Expensive transformer    |
    |  Score all ~1,000         |
    +---------------------------+
                    |
                    v
            Final feed (~50 posts)
```

**Retrieval** is the bouncer at the door: fast, approximate, casts a wide net.
**Ranking** is the sommelier: careful, precise, considers every detail.

This lecture is about the bouncer.

---

## 2. Two-Tower Architecture: The Big Picture

The retrieval model is called a "two-tower" model because it has two separate
neural networks (towers) that never talk to each other during inference:

```
    User Tower                         Candidate Tower
    (transformer)                      (2-layer MLP)
         |                                  |
         v                                  v
  [user embedding]                  [post embedding]
    shape: [B, D]                    shape: [B, C, D]
         |                                  |
         v                                  v
   L2 normalize                       L2 normalize
         |                                  |
         +----------> dot product <---------+
                          |
                          v
                   similarity scores
                     shape: [B, C]
```

The key insight is that **both towers produce vectors of the same
dimensionality D**. Once you have two unit-length vectors, their dot product
is the cosine similarity -- a number between -1 and +1 that tells you how
"aligned" they are.

### Why separate towers?

Because of a serving trick that makes retrieval feasible at scale:

1. **Offline (batch job, runs daily/hourly):** Run the candidate tower on
   every post in the corpus. Store the resulting D-dimensional vectors in
   an approximate nearest neighbor (ANN) index like FAISS.

2. **Online (per user request, ~10ms budget):** Run the user tower once to
   get a single D-dimensional vector. Query the ANN index to find the
   ~1,000 closest candidate vectors.

If the two towers shared parameters or cross-attended to each other, you
could not pre-compute candidate embeddings. That is the entire reason for
the architectural separation.

---

## 3. CandidateTower (lines 46-99)

Let's start with the simpler tower. Open `recsys_retrieval_model.py` and
look at the `CandidateTower` class.

### 3.1 What It Does

The candidate tower takes concatenated post + author hash embeddings and
projects them to a normalized D-dimensional vector:

```
Input:  [B, C, num_hashes, D]   (post + author hash embeddings concatenated)
           |
           v
  Reshape: [B, C, num_hashes * D]          # flatten hash dimension
           |
           v
  Linear:  [num_hashes*D] --> [2*D]        # EXPAND (projection 1)
           |
           v
  SiLU activation                          # nonlinearity
           |
           v
  Linear:  [2*D] --> [D]                   # COMPRESS (projection 2)
           |
           v
  L2 normalize                             # onto the unit sphere
           |
           v
Output: [B, C, D]                          # one unit vector per candidate
```

### 3.2 The JAX Code, Annotated

Here is the core of the forward pass (lines 68-99), with my annotations:

```python
# --- Step 1: Flatten hash embeddings ---
# From [B, C, num_hashes, D] to [B, C, num_hashes * D]
if len(post_author_embedding.shape) == 4:
    B, C, _, _ = post_author_embedding.shape
    post_author_embedding = jnp.reshape(post_author_embedding, (B, C, -1))

# --- Step 2: Expand projection ---
# proj_1 has shape [num_hashes*D, 2*D]
hidden = jnp.dot(post_author_embedding, proj_1)   # [B, C, 2*D]
hidden = jax.nn.silu(hidden)                       # nonlinearity

# --- Step 3: Compress projection ---
# proj_2 has shape [2*D, D]
candidate_embeddings = jnp.dot(hidden, proj_2)     # [B, C, D]

# --- Step 4: L2 normalize ---
candidate_norm_sq = jnp.sum(candidate_embeddings**2, axis=-1, keepdims=True)
candidate_norm = jnp.sqrt(jnp.maximum(candidate_norm_sq, EPS))
candidate_representation = candidate_embeddings / candidate_norm
```

### 3.3 The Expand-Then-Compress Pattern

You might ask: why two linear layers with a wider hidden size? Why not just
one linear layer `[num_hashes*D, D]`?

One linear layer can only learn **linear combinations** of the input features.
No matter how big you make the weight matrix, `y = Wx` is still a hyperplane.

Two linear layers with a nonlinear activation in between can learn
**arbitrary nonlinear functions** (by the universal approximation theorem).
The network can now do things like:

- "If post hash 1 is strong AND author hash 2 is strong, boost the embedding
  in direction X"
- "Suppress noisy hash collisions by learning to cancel them out"

The `2*D` width is a design choice: it is wide enough to give the hidden
layer some room to represent combinations, but narrow enough to keep the
candidate tower cheap. Remember, this tower runs on **every post in the
corpus** during the offline batch job.

### 3.4 SiLU Activation

The codebase uses SiLU (Sigmoid Linear Unit), also known as "Swish":

```
SiLU(x) = x * sigmoid(x)
```

Compare with ReLU:
```
ReLU(x) = max(0, x)       # hard cutoff at 0
SiLU(x) = x * sigmoid(x)  # smooth curve, slightly negative dip
```

```
   output
    ^
    |       SiLU          ReLU
    |      ./            /
    |    ./            /
    |  ./            /
    |./            /
----+-------.-------> input
    |  `--'
```

SiLU is smoother than ReLU, which helps gradient flow (no dead neurons).
It also allows slightly negative outputs, which gives the network more
expressiveness. It has become the standard activation in modern architectures
(LLaMA, Grok, etc.).

### 3.5 L2 Normalization: Why the Unit Sphere?

After the MLP, the code manually normalizes each vector to unit length:

```python
norm = sqrt(sum(x^2) + eps)       # eps = 1e-12 prevents division by zero
x_normalized = x / norm           # now ||x|| = 1.0
```

This is critical for three reasons:

1. **Dot product becomes cosine similarity.** When both vectors have length 1,
   `dot(u, v) = cos(angle between u and v)`. The score is purely about
   direction, not magnitude.

2. **Scores are bounded in [-1, +1].** No need for extra calibration. A score
   of 0.95 always means "very similar" regardless of the embedding dimension.

3. **ANN indexes require it.** FAISS, ScaNN, and other ANN libraries use
   inner product search. If your vectors are not normalized, you get weird
   results where a "long" but misaligned vector scores higher than a "short"
   but perfectly aligned one.

```
   Before normalization:           After normalization:
                                          *
   *                                    / | \
   |  long vector,                     /  |  \
   |  maybe high dot product          *   |   *
   |  even if misaligned            /  \  |  /  \
   *---*----> target               *----*-+-*----*
                                   All on unit sphere.
                                   Dot product = cosine.
```

---

## 4. User Tower: `build_user_representation` (lines 206-276)

The user tower is where the real intelligence lives. It takes everything we
know about a user -- who they are, what they did, where they did it -- and
compresses it into a single D-dimensional vector.

### 4.1 The Four Ingredients

The user tower combines four types of input. Let's walk through each one.

#### Ingredient A: Product Surface Embeddings (lines 227-232)

```python
history_product_surface_embeddings = self._single_hot_to_embeddings(
    batch.history_product_surface,       # [B, S] integer indices
    config.product_surface_vocab_size,   # 16 possible surfaces
    config.emb_size,                     # D
    "product_surface_embedding_table",
)
# Output: [B, S, D]
```

"Product surface" means **where on the app** the user took an action:
- Home timeline
- Search results
- Notifications tab
- Profile page
- ...up to 16 different surfaces

Each surface gets its own learned D-dimensional embedding. The lookup is
implemented as `one_hot -> matmul` which is mathematically identical to
`nn.Embedding` in PyTorch -- the one-hot matmul pattern is just more natural
in JAX's functional style.

Why does this matter? A "like" on a post you found via search means something
different from a "like" on a post that appeared in your home feed. The surface
embedding lets the model distinguish these contexts.

#### Ingredient B: Action Embeddings (line 234)

```python
history_actions_embeddings = self._get_action_embeddings(
    batch.history_actions  # [B, S, num_actions] multi-hot binary
)
# Output: [B, S, D]
```

This is one of the cleverest parts of the model. Let's look at the signed
embedding trick inside `_get_action_embeddings` (lines 161-184):

```python
# actions: [B, S, num_actions], values are 0 or 1
# Example for one post: [1, 0, 1, 0, 0, 1, 0, 0]
# Meaning: liked=1, retweeted=0, replied=1, bookmarked=0, ...

# Step 1: Convert 0/1 to -1/+1
actions_signed = (2 * actions - 1)
# Result:          [1, -1, 1, -1, -1, 1, -1, -1]

# Step 2: Multiply by learned projection [num_actions, D]
action_emb = jnp.dot(actions_signed, action_projection)

# Step 3: Zero out padding positions
valid_mask = jnp.any(actions, axis=-1, keepdims=True)
action_emb = action_emb * valid_mask
```

**Why -1 instead of 0?**

With 0/1 encoding, a "didn't engage" action contributes **nothing** to the
embedding. The model only sees evidence of what you DID do.

With -1/+1 encoding, a "didn't engage" action actively **pushes** the
embedding in the **opposite** direction. The model sees both:
- "I liked this post" (+1 in the "liked" direction)
- "I did NOT retweet this post" (-1 in the "retweeted" direction)

This is much more informative. Not retweeting a post you liked tells the
model something meaningful about your preferences.

```
   0/1 encoding:                    -1/+1 encoding:

   liked:      ---->  (contributes) liked:      ---->  (contributes)
   retweeted:  (nothing)            retweeted:  <----  (pushes opposite!)
   replied:    ---->  (contributes) replied:    ---->  (contributes)
   bookmarked: (nothing)            bookmarked: <----  (pushes opposite!)
```

The `valid_mask` handles padding: if ALL actions are 0 (meaning this history
position is empty/padding), the mask zeros out the entire embedding. This
prevents padding tokens from contaminating the representation.

#### Ingredient C: User Identity (lines 236-242)

```python
user_embeddings, user_padding_mask = block_user_reduce(
    batch.user_hashes,                    # [B, num_user_hashes]
    recsys_embeddings.user_embeddings,    # [B, num_user_hashes, D]
    hash_config.num_user_hashes,          # 2
    config.emb_size,                      # D
    1.0,
)
# Output: user_embeddings [B, 1, D], user_padding_mask [B, 1]
```

This comes from Lecture 2. The user's identity is represented by multiple
hash embeddings (to handle collisions), which are flattened and projected
down to a single D-dimensional vector. Think of it as a learned user ID
embedding that captures "who you are" independent of your recent behavior.

Output shape is `[B, 1, D]` -- one embedding per user, ready to be
concatenated as the first token in the sequence.

#### Ingredient D: History Posts + Authors (lines 244-253)

```python
history_embeddings, history_padding_mask = block_history_reduce(
    batch.history_post_hashes,
    recsys_embeddings.history_post_embeddings,
    recsys_embeddings.history_author_embeddings,
    history_product_surface_embeddings,       # from ingredient A
    history_actions_embeddings,               # from ingredient B
    hash_config.num_item_hashes,              # 2
    hash_config.num_author_hashes,            # 2
    1.0,
)
# Output: history_embeddings [B, S, D], history_padding_mask [B, S]
# where S = 128 (history_seq_len)
```

This also comes from Lecture 2. For each of the 128 history positions, it
concatenates the post hashes, author hashes, action embedding, and surface
embedding, then projects them down to D dimensions.

Each history position becomes one rich token that encodes:
- **What** post it was (post hashes)
- **Who** wrote it (author hashes)
- **What you did** with it (action embedding)
- **Where you saw it** (surface embedding)

### 4.2 Building the Sequence (lines 255-256)

Now we concatenate user identity + history into one sequence:

```python
embeddings = jnp.concatenate([user_embeddings, history_embeddings], axis=1)
# user_embeddings:    [B,   1, D]    "who am I"
# history_embeddings: [B, 128, D]    "what I did recently"
# Result:             [B, 129, D]

padding_mask = jnp.concatenate([user_padding_mask, history_padding_mask], axis=1)
# Result:             [B, 129]
```

The sequence looks like this:

```
Position:  0       1        2        3               128
Token:   [USER | post_1 | post_2 | post_3 | ... | post_128]
Mask:    [  1  |   1    |   1    |   1    | ... |    0     ]
                                                  (padding)
```

Position 0 is always the user identity. Positions 1-128 are the user's
recent interaction history, ordered chronologically. Positions with no
real data are marked as padding in the mask.

### 4.3 Transformer Encoding (lines 258-264)

```python
model_output = self.model(
    embeddings.astype(self.fprop_dtype),    # [B, 129, D]
    padding_mask,                           # [B, 129]
    candidate_start_offset=None,            # <-- important!
)
user_outputs = model_output.embeddings     # [B, 129, D]
```

The transformer processes the entire sequence with self-attention. Each
token can attend to every other (non-padding) token.

**Why `candidate_start_offset=None`?** In the ranking model (Lecture 4),
this parameter isolates candidates so they cannot attend to each other.
In the retrieval model, there are no candidates in the sequence -- it is
purely user context -- so we pass `None` to disable that masking. Every
token can see every other token.

What does the transformer learn here? Cross-interaction patterns like:
- "User who liked sports posts yesterday is now liking cooking posts
  -> maybe they are into sports cooking shows"
- "User clicked 5 posts from the same author in a row
  -> strong author affinity signal"
- "User identity embedding modulates which history signals matter"

### 4.4 Mean Pooling + L2 Normalization (lines 266-276)

Now we need to collapse the sequence `[B, 129, D]` into a single vector
`[B, D]`. The model uses **masked mean pooling**:

```python
# Step 1: Zero out padding positions
mask_float = padding_mask.astype(jnp.float32)[:, :, None]  # [B, 129, 1]
user_embeddings_masked = user_outputs * mask_float          # [B, 129, D]

# Step 2: Sum non-padding outputs
user_embedding_sum = jnp.sum(user_embeddings_masked, axis=1)  # [B, D]

# Step 3: Divide by count of non-padding tokens
mask_sum = jnp.sum(mask_float, axis=1)                        # [B, 1]
user_representation = user_embedding_sum / jnp.maximum(mask_sum, 1.0)

# Step 4: L2 normalize to unit sphere
user_norm_sq = jnp.sum(user_representation**2, axis=-1, keepdims=True)
user_norm = jnp.sqrt(jnp.maximum(user_norm_sq, EPS))
user_representation = user_representation / user_norm
# Final output: [B, D], unit-length vectors
```

Let me draw this step by step:

```
Transformer outputs: [tok0, tok1, tok2, tok3, PAD,  PAD ]
                       |     |     |     |     |     |
Mask:                [ 1,    1,    1,    1,    0,    0  ]
                       |     |     |     |     |     |
After masking:       [tok0, tok1, tok2, tok3, ZERO, ZERO]
                       \     |     |     /
                        \    |     |    /
Sum:                  tok0 + tok1 + tok2 + tok3
                               |
Divide by 4:              mean pooling
                               |
L2 normalize:          unit sphere projection
                               |
                               v
                     user_representation [B, D]
```

**Why mean pooling?** Alternatives include:
- **CLS token pooling** (use only position 0): loses information from history
- **Max pooling**: biased toward extreme values, less stable
- **Attention pooling**: more parameters, slower, minimal accuracy gain here

Mean pooling is simple, stable, and works well in practice. The `max(mask_sum, 1.0)`
guard prevents division by zero when a user has no valid tokens (edge case
during padding).

---

## 5. Why Asymmetric Towers?

Let's step back and appreciate the asymmetry:

```
+-------------------+------------------------+
|                   | User Tower   | Cand. Tower  |
+-------------------+------------------------+
| Architecture      | Transformer  | 2-layer MLP  |
| Depth             | Many layers  | 2 layers     |
| Runs when?        | Per request  | Offline batch |
| Runs how often?   | ~1B/day      | ~500M/day    |
| Input             | User + 128   | 1 post       |
|                   | history items|              |
| Parameters        | ~100M+       | ~few M       |
| Latency budget    | ~50ms        | ~0.1ms/post  |
+-------------------+------------------------+
```

The user tower can afford to be a transformer because it runs **once per
request**. Even if it takes 50ms, that is fine.

The candidate tower must be an MLP because it runs on **every post in the
corpus**. If you have 500M posts and each takes 0.1ms, that is already
50,000 seconds = ~14 hours. You parallelize across GPUs, but the point
stands: the candidate tower must be as cheap as possible.

This is the fundamental trade-off of two-tower retrieval: **you buy serving
efficiency at the cost of interaction**. The two towers cannot cross-attend
to each other, which means the model cannot learn patterns like "this specific
user would love this specific word in this specific post." That level of
nuance is left for the ranking stage.

---

## 6. Retrieval: `_retrieve_top_k` (lines 346-372)

After both towers produce their embeddings, retrieval is embarrassingly
simple -- it is just a matrix multiply:

```python
def _retrieve_top_k(self, user_representation, corpus_embeddings, top_k, corpus_mask):
    # user_representation: [B, D]     -- one vector per user
    # corpus_embeddings:   [N, D]     -- one vector per post in corpus

    # Step 1: Compute ALL similarity scores
    scores = jnp.matmul(user_representation, corpus_embeddings.T)  # [B, N]
    # Each entry scores[b, n] = dot(user_b, post_n) = cosine similarity

    # Step 2: Mask out invalid corpus entries
    if corpus_mask is not None:
        scores = jnp.where(corpus_mask[None, :], scores, -INF)
        # -INF ensures invalid posts never appear in top-k

    # Step 3: Find the top K
    top_k_scores, top_k_indices = jax.lax.top_k(scores, top_k)
    # top_k_scores:  [B, K]  -- similarity scores of the best matches
    # top_k_indices: [B, K]  -- their positions in the corpus

    return top_k_indices, top_k_scores
```

That is it. **Retrieval is just a matrix multiply followed by argmax.**

The entire intelligence of the system lives in the embeddings. The dot
product is just the readout mechanism.

### Production vs. Training

In the code above, the retrieval is **brute-force**: compute dot products
against every post in the corpus. This is used during training (where you
want exact gradients) and evaluation.

In production, you would never do a brute-force matmul against 500M vectors.
Instead:

1. Pre-compute all candidate embeddings with the candidate tower (offline).
2. Build an ANN (Approximate Nearest Neighbor) index, e.g. using FAISS with
   HNSW or IVF-PQ.
3. At serving time, compute the user embedding, query the ANN index, get
   back ~1,000 approximate nearest neighbors in ~5ms.

The ANN index returns *approximate* results (might miss some true top-k),
but it is 10,000x faster than brute-force.

---

## 7. RetrievalOutput (lines 38-43)

The output of the full model is a clean named tuple:

```python
class RetrievalOutput(NamedTuple):
    user_representation: jax.Array   # [B, D]  -- the user embedding
    top_k_indices: jax.Array         # [B, K]  -- which posts to retrieve
    top_k_scores: jax.Array          # [B, K]  -- their similarity scores
```

The `user_representation` is returned so it can be cached, logged, or passed
to downstream systems. The indices and scores are what actually feed into
the next stage (ranking).

---

## 8. Full Data Flow Summary

Let me draw the complete data flow from raw features to retrieved posts:

```
                       RAW FEATURES
    +-----------------------------------------------------+
    | user_hashes          [B, 2]                          |
    | history_post_hashes  [B, 128, 2]                     |
    | history_author_hashes[B, 128, 2]                     |
    | history_actions      [B, 128, num_actions]           |
    | history_product_surface [B, 128]                     |
    +-----------------------------------------------------+
                            |
                            v
                     EMBEDDING LOOKUP
    +-----------------------------------------------------+
    | user_embeddings            [B, 2, D]                 |
    | history_post_embeddings    [B, 128, 2, D]            |
    | history_author_embeddings  [B, 128, 2, D]            |
    | product_surface_embeddings [B, 128, D]               |
    | action_embeddings          [B, 128, D]     (signed!) |
    +-----------------------------------------------------+
                            |
                            v
                       REDUCTION
    +-----------------------------------------------------+
    | user_embedding     [B,   1, D]  (block_user_reduce)  |
    | history_embeddings [B, 128, D]  (block_history_reduce)|
    +-----------------------------------------------------+
                            |
                            v
                      CONCATENATE
    +-----------------------------------------------------+
    | sequence:  [B, 129, D]                               |
    | mask:      [B, 129]                                  |
    +-----------------------------------------------------+
                            |
                            v
                      TRANSFORMER
    +-----------------------------------------------------+
    | output:    [B, 129, D]                               |
    +-----------------------------------------------------+
                            |
                            v
                MEAN POOLING + L2 NORM
    +-----------------------------------------------------+
    | user_representation: [B, D]  (unit vector)           |
    +-----------------------------------------------------+
                            |
                            v
              DOT PRODUCT WITH CORPUS
    +-----------------------------------------------------+
    | scores: [B, N]                                       |
    | top_k:  [B, K] indices + scores                      |
    +-----------------------------------------------------+
                            |
                            v
                    RetrievalOutput
```

---

## 9. Key Takeaways

1. **Asymmetric towers enable serving efficiency.** The expensive transformer
   runs once per user. The cheap MLP runs once per post, offline. This is
   not optional -- it is the entire reason two-tower models exist.

2. **Signed action embeddings (-1/+1) are more informative than 0/1.** Not
   engaging is a signal, not an absence of signal.

3. **Mean pooling + L2 normalization: sequence to fixed vector.** The
   transformer outputs a variable-length sequence. Mean pooling collapses
   it to a single vector. L2 normalization puts it on the unit sphere so
   dot products become cosine similarities.

4. **Retrieval is just matmul.** All the intelligence is in the embeddings.
   The actual retrieval step is a matrix multiply and an argmax. In
   production, even that is replaced by an ANN index.

5. **The separation is the limitation.** Because the two towers never
   interact, the model cannot capture fine-grained user-post interactions.
   That is what the ranking model (Lecture 4) is for.

---

## 10. Quick Check Questions

Test your understanding before moving on.

**Q1: Why can't we just use the ranking transformer directly on millions of posts?**

The ranking transformer processes each candidate with full cross-attention
to the user context. At ~1ms per candidate, scoring 500M posts would take
~500,000 seconds (~6 days) per request. The two-tower model decouples the
towers, enabling ANN search, which takes ~5ms regardless of corpus size.

**Q2: Why is the user tower a transformer but the candidate tower just an MLP?**

The user tower runs once per request, so it can afford to be expensive. It
also needs to capture **sequential patterns** in the user's 128-item history
(e.g., "liked 5 sports posts then 1 cooking post"). The candidate tower runs
on every post in the corpus (offline), so it must be cheap. It only needs
to encode **one post at a time** -- no sequential reasoning needed.

**Q3: What does L2 normalization buy us?**

Three things: (1) dot product becomes cosine similarity (direction-only,
ignoring magnitude), (2) scores are bounded in [-1, +1] for easy
thresholding and calibration, (3) ANN indexes like FAISS require normalized
vectors for correct inner-product search.

**Q4: If the user only interacted with 50 posts but the sequence length is
128, what happens to the other 78 positions?**

They are padding. The `padding_mask` marks them as 0 (invalid). The
transformer ignores them via masked attention. During mean pooling, they are
explicitly zeroed out and excluded from the denominator (`mask_sum` counts
only the 51 valid positions: 1 user + 50 history). So the output is the mean
of the 51 real tokens only.

---

## Appendix A: Complete PyTorch Translation

Below is a fully runnable PyTorch implementation of the two-tower retrieval
model. This matches the JAX code semantically, translated to idiomatic
PyTorch. You can copy this into a `.py` file and run it.

```python
"""
Two-Tower Retrieval Model -- PyTorch Translation
Matches: phoenix/recsys_retrieval_model.py

Run with: python appendix_two_tower.py
Requires: pip install torch
"""

import math
from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EPS = 1e-12
INF = 1e12


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------
class RetrievalOutput(NamedTuple):
    user_representation: torch.Tensor   # [B, D]
    top_k_indices: torch.Tensor         # [B, K]
    top_k_scores: torch.Tensor          # [B, K]


# ---------------------------------------------------------------------------
# CandidateTower
# ---------------------------------------------------------------------------
class CandidateTower(nn.Module):
    """2-layer MLP that projects post+author hash embeddings to a unit vector.

    Matches CandidateTower in recsys_retrieval_model.py (lines 46-99).

    Architecture:
        [B, C, num_hashes*D] -> Linear(2D) -> SiLU -> Linear(D) -> L2Norm
    """

    def __init__(self, num_hashes: int, emb_size: int):
        super().__init__()
        self.num_hashes = num_hashes  # total hashes (post + author combined)
        self.emb_size = emb_size

        input_dim = num_hashes * emb_size
        # Expand then compress: input_dim -> 2*D -> D
        self.proj_1 = nn.Linear(input_dim, emb_size * 2, bias=False)  # [num_hashes*D, 2*D]
        self.proj_2 = nn.Linear(emb_size * 2, emb_size, bias=False)   # [2*D, D]

        # Match JAX VarianceScaling(1.0, mode="fan_out")
        nn.init.kaiming_normal_(self.proj_1.weight, mode="fan_out", nonlinearity="linear")
        nn.init.kaiming_normal_(self.proj_2.weight, mode="fan_out", nonlinearity="linear")

    def forward(self, post_author_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            post_author_embedding: [B, C, num_hashes, D] or [B, num_hashes, D]

        Returns:
            Normalized candidate embeddings: [B, C, D] or [B, D]
        """
        # --- Step 1: Flatten hash dimension ---
        if post_author_embedding.dim() == 4:
            B, C, H, D = post_author_embedding.shape        # [B, C, num_hashes, D]
            x = post_author_embedding.reshape(B, C, H * D)  # [B, C, num_hashes*D]
        else:
            B, H, D = post_author_embedding.shape            # [B, num_hashes, D]
            x = post_author_embedding.reshape(B, H * D)      # [B, num_hashes*D]

        # --- Step 2: Expand + SiLU ---
        hidden = self.proj_1(x)          # [B, C, 2*D] or [B, 2*D]
        hidden = F.silu(hidden)          # smooth nonlinearity

        # --- Step 3: Compress ---
        candidate = self.proj_2(hidden)  # [B, C, D] or [B, D]

        # --- Step 4: L2 normalize ---
        norm = torch.sqrt(torch.sum(candidate ** 2, dim=-1, keepdim=True).clamp(min=EPS))
        candidate = candidate / norm     # unit length

        return candidate                 # [B, C, D] or [B, D]


# ---------------------------------------------------------------------------
# ActionEmbedding
# ---------------------------------------------------------------------------
class ActionEmbedding(nn.Module):
    """Converts multi-hot action vectors to embeddings using the signed trick.

    Matches _get_action_embeddings in recsys_retrieval_model.py (lines 161-184).

    Key insight: maps 0 -> -1, 1 -> +1 so that "did not engage" actively
    contributes a negative signal rather than being ignored.
    """

    def __init__(self, num_actions: int, emb_size: int):
        super().__init__()
        # Learned projection: each action direction is a D-dim vector
        self.action_projection = nn.Parameter(
            torch.empty(num_actions, emb_size)               # [num_actions, D]
        )
        nn.init.kaiming_normal_(self.action_projection, mode="fan_out", nonlinearity="linear")

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: [B, S, num_actions] binary multi-hot (0 or 1)

        Returns:
            action_embeddings: [B, S, D]
        """
        # --- Step 1: Signed encoding ---
        # 0 -> -1, 1 -> +1
        actions_signed = (2.0 * actions - 1.0).float()       # [B, S, num_actions]

        # --- Step 2: Project to embedding space ---
        action_emb = torch.matmul(
            actions_signed, self.action_projection            # [B, S, D]
        )

        # --- Step 3: Zero out padding positions ---
        # If ALL actions are 0 for a position, it is padding
        valid_mask = actions.any(dim=-1, keepdim=True).float()  # [B, S, 1]
        action_emb = action_emb * valid_mask                    # [B, S, D]

        return action_emb


# ---------------------------------------------------------------------------
# SurfaceEmbedding
# ---------------------------------------------------------------------------
class SurfaceEmbedding(nn.Module):
    """Embeds product surface IDs (where on the app the action happened).

    Matches _single_hot_to_embeddings for product surfaces (lines 186-204).
    Uses nn.Embedding which is equivalent to one_hot -> matmul.
    """

    def __init__(self, num_surfaces: int, emb_size: int):
        super().__init__()
        self.embedding = nn.Embedding(num_surfaces, emb_size)  # [16, D]
        nn.init.kaiming_normal_(self.embedding.weight, mode="fan_out", nonlinearity="linear")

    def forward(self, surface_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            surface_ids: [B, S] integer indices in [0, num_surfaces)

        Returns:
            surface_embeddings: [B, S, D]
        """
        return self.embedding(surface_ids)                     # [B, S, D]


# ---------------------------------------------------------------------------
# Simplified Transformer Encoder (stand-in for Grok transformer)
# ---------------------------------------------------------------------------
class SimpleTransformerEncoder(nn.Module):
    """A standard transformer encoder, standing in for the Grok transformer.

    In the real codebase, this is the full Grok transformer from grok.py.
    For this appendix, we use PyTorch's built-in TransformerEncoder.
    """

    def __init__(self, emb_size: int, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=emb_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,              # [B, T, D] format
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        embeddings: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings:   [B, T, D]
            padding_mask: [B, T] -- True = valid, False = padding

        Returns:
            outputs: [B, T, D]
        """
        # PyTorch TransformerEncoder expects True = IGNORE, so we invert
        src_key_padding_mask = ~padding_mask.bool()            # [B, T]
        outputs = self.encoder(
            embeddings,
            src_key_padding_mask=src_key_padding_mask,
        )
        return outputs                                         # [B, T, D]


# ---------------------------------------------------------------------------
# UserTower
# ---------------------------------------------------------------------------
class UserTower(nn.Module):
    """Encodes user identity + history into a single normalized embedding.

    Matches build_user_representation in recsys_retrieval_model.py (lines 206-276).

    Pipeline:
        1. Embed surfaces, actions, user identity, history
        2. Concatenate user + history -> sequence [B, 1+S, D]
        3. Transformer encoding
        4. Masked mean pooling -> [B, D]
        5. L2 normalize -> unit sphere
    """

    def __init__(
        self,
        emb_size: int,
        num_actions: int,
        num_surfaces: int,
        num_user_hashes: int,
        num_item_hashes: int,
        num_author_hashes: int,
        transformer: nn.Module,
    ):
        super().__init__()
        self.emb_size = emb_size

        # Ingredient A: Surface embeddings
        self.surface_embedding = SurfaceEmbedding(num_surfaces, emb_size)

        # Ingredient B: Action embeddings (signed trick)
        self.action_embedding = ActionEmbedding(num_actions, emb_size)

        # Ingredient C: User identity projection
        # Flatten num_user_hashes*D -> D
        self.user_proj = nn.Linear(
            num_user_hashes * emb_size, emb_size, bias=False   # [num_user_hashes*D, D]
        )
        nn.init.kaiming_normal_(self.user_proj.weight, mode="fan_out", nonlinearity="linear")

        # Ingredient D: History reduction projection
        # post_hashes + author_hashes + actions + surface -> D
        history_input_dim = (
            num_item_hashes * emb_size     # post hash embeddings
            + num_author_hashes * emb_size # author hash embeddings
            + emb_size                     # action embeddings
            + emb_size                     # surface embeddings
        )
        self.history_proj = nn.Linear(
            history_input_dim, emb_size, bias=False            # [history_input_dim, D]
        )
        nn.init.kaiming_normal_(self.history_proj.weight, mode="fan_out", nonlinearity="linear")

        # Transformer
        self.transformer = transformer

    def forward(
        self,
        user_embeddings: torch.Tensor,
        user_hashes: torch.Tensor,
        history_post_embeddings: torch.Tensor,
        history_author_embeddings: torch.Tensor,
        history_actions: torch.Tensor,
        history_product_surface: torch.Tensor,
        history_post_hashes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            user_embeddings:           [B, num_user_hashes, D]
            user_hashes:               [B, num_user_hashes]
            history_post_embeddings:   [B, S, num_item_hashes, D]
            history_author_embeddings: [B, S, num_author_hashes, D]
            history_actions:           [B, S, num_actions]
            history_product_surface:   [B, S]
            history_post_hashes:       [B, S, num_item_hashes]

        Returns:
            user_representation: [B, D]  (unit vector)
            user_norm:           [B, 1]  (pre-normalization magnitude)
        """
        B, S = history_post_hashes.shape[:2]
        D = self.emb_size

        # --- Ingredient A: Surface embeddings ---
        surface_emb = self.surface_embedding(history_product_surface)   # [B, S, D]

        # --- Ingredient B: Action embeddings (signed trick) ---
        action_emb = self.action_embedding(history_actions)             # [B, S, D]

        # --- Ingredient C: User identity ---
        user_flat = user_embeddings.reshape(B, 1, -1)                  # [B, 1, num_user_hashes*D]
        user_emb = self.user_proj(user_flat)                           # [B, 1, D]
        user_padding = (user_hashes[:, 0] != 0).reshape(B, 1)         # [B, 1]

        # --- Ingredient D: History posts + authors ---
        _, _, num_item_h, _ = history_post_embeddings.shape
        _, _, num_auth_h, _ = history_author_embeddings.shape

        post_flat = history_post_embeddings.reshape(B, S, num_item_H := num_item_h * D)
        auth_flat = history_author_embeddings.reshape(B, S, num_auth_H := num_auth_h * D)
        # Concatenate all history features for each position
        history_concat = torch.cat(
            [post_flat, auth_flat, action_emb, surface_emb], dim=-1    # [B, S, history_input_dim]
        )
        history_emb = self.history_proj(history_concat)                # [B, S, D]
        history_padding = (history_post_hashes[:, :, 0] != 0)         # [B, S]

        # --- Build sequence ---
        embeddings = torch.cat([user_emb, history_emb], dim=1)        # [B, 1+S, D]
        padding_mask = torch.cat([user_padding, history_padding], dim=1)  # [B, 1+S]

        # --- Transformer ---
        outputs = self.transformer(embeddings, padding_mask)           # [B, 1+S, D]

        # --- Masked mean pooling ---
        mask_float = padding_mask.float().unsqueeze(-1)                # [B, 1+S, 1]
        outputs_masked = outputs * mask_float                          # [B, 1+S, D]
        emb_sum = outputs_masked.sum(dim=1)                            # [B, D]
        mask_sum = mask_float.sum(dim=1).clamp(min=1.0)                # [B, 1]
        user_repr = emb_sum / mask_sum                                 # [B, D]

        # --- L2 normalize ---
        user_norm = torch.sqrt(
            (user_repr ** 2).sum(dim=-1, keepdim=True).clamp(min=EPS)  # [B, 1]
        )
        user_repr = user_repr / user_norm                              # [B, D] unit vector

        return user_repr, user_norm


# ---------------------------------------------------------------------------
# TwoTowerRetrieval
# ---------------------------------------------------------------------------
class TwoTowerRetrieval(nn.Module):
    """Complete two-tower retrieval model.

    Matches PhoenixRetrievalModel in recsys_retrieval_model.py.

    Usage:
        model = TwoTowerRetrieval(...)

        # Offline: encode all posts
        corpus_emb = model.encode_candidates(post_author_embeddings)

        # Online: encode user + retrieve
        output = model.retrieve(user_features, corpus_emb, top_k=1000)
    """

    def __init__(
        self,
        emb_size: int = 64,
        num_actions: int = 8,
        num_surfaces: int = 16,
        num_user_hashes: int = 2,
        num_item_hashes: int = 2,
        num_author_hashes: int = 2,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.num_user_hashes = num_user_hashes
        self.num_item_hashes = num_item_hashes
        self.num_author_hashes = num_author_hashes

        total_candidate_hashes = num_item_hashes + num_author_hashes

        # Candidate tower: cheap 2-layer MLP
        self.candidate_tower = CandidateTower(
            num_hashes=total_candidate_hashes,
            emb_size=emb_size,
        )

        # User tower: transformer-based
        transformer = SimpleTransformerEncoder(
            emb_size=emb_size,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
        )
        self.user_tower = UserTower(
            emb_size=emb_size,
            num_actions=num_actions,
            num_surfaces=num_surfaces,
            num_user_hashes=num_user_hashes,
            num_item_hashes=num_item_hashes,
            num_author_hashes=num_author_hashes,
            transformer=transformer,
        )

    def encode_candidates(
        self, post_author_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Encode candidates (offline batch job).

        Args:
            post_author_embeddings: [B, C, num_hashes, D] or [N, num_hashes, D]

        Returns:
            Normalized candidate embeddings: [B, C, D] or [N, D]
        """
        return self.candidate_tower(post_author_embeddings)

    def encode_user(
        self,
        user_embeddings: torch.Tensor,
        user_hashes: torch.Tensor,
        history_post_embeddings: torch.Tensor,
        history_author_embeddings: torch.Tensor,
        history_actions: torch.Tensor,
        history_product_surface: torch.Tensor,
        history_post_hashes: torch.Tensor,
    ) -> torch.Tensor:
        """Encode user (online, once per request).

        Returns:
            user_representation: [B, D] unit vector
        """
        user_repr, _ = self.user_tower(
            user_embeddings=user_embeddings,
            user_hashes=user_hashes,
            history_post_embeddings=history_post_embeddings,
            history_author_embeddings=history_author_embeddings,
            history_actions=history_actions,
            history_product_surface=history_product_surface,
            history_post_hashes=history_post_hashes,
        )
        return user_repr

    @staticmethod
    def retrieve_top_k(
        user_representation: torch.Tensor,
        corpus_embeddings: torch.Tensor,
        top_k: int,
        corpus_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Brute-force retrieval via dot product (training/eval).

        In production, replace this with a FAISS ANN index query.

        Args:
            user_representation: [B, D]
            corpus_embeddings:   [N, D]
            top_k:               number of candidates to retrieve
            corpus_mask:         [N] optional, True = valid

        Returns:
            top_k_indices: [B, K]
            top_k_scores:  [B, K]
        """
        # All-pairs similarity: [B, D] x [D, N] = [B, N]
        scores = torch.matmul(user_representation, corpus_embeddings.T)

        # Mask invalid corpus entries
        if corpus_mask is not None:
            scores = scores.masked_fill(~corpus_mask.unsqueeze(0), -INF)

        # Top-K selection
        top_k_scores, top_k_indices = torch.topk(scores, k=top_k, dim=-1)

        return top_k_indices, top_k_scores

    def forward(
        self,
        user_embeddings: torch.Tensor,
        user_hashes: torch.Tensor,
        history_post_embeddings: torch.Tensor,
        history_author_embeddings: torch.Tensor,
        history_actions: torch.Tensor,
        history_product_surface: torch.Tensor,
        history_post_hashes: torch.Tensor,
        corpus_embeddings: torch.Tensor,
        top_k: int = 100,
        corpus_mask: Optional[torch.Tensor] = None,
    ) -> RetrievalOutput:
        """Full forward pass: encode user, then retrieve from corpus.

        Args:
            (user features -- see encode_user)
            corpus_embeddings: [N, D] pre-computed candidate embeddings
            top_k:             number to retrieve
            corpus_mask:       [N] optional validity mask

        Returns:
            RetrievalOutput with user_representation, top_k_indices, top_k_scores
        """
        user_repr = self.encode_user(
            user_embeddings=user_embeddings,
            user_hashes=user_hashes,
            history_post_embeddings=history_post_embeddings,
            history_author_embeddings=history_author_embeddings,
            history_actions=history_actions,
            history_product_surface=history_product_surface,
            history_post_hashes=history_post_hashes,
        )

        top_k_indices, top_k_scores = self.retrieve_top_k(
            user_repr, corpus_embeddings, top_k, corpus_mask
        )

        return RetrievalOutput(
            user_representation=user_repr,
            top_k_indices=top_k_indices,
            top_k_scores=top_k_scores,
        )


# ---------------------------------------------------------------------------
# Demo: run with synthetic data
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(42)

    # --- Hyperparameters ---
    B = 4                  # batch size (4 users)
    S = 128                # history sequence length
    D = 64                 # embedding dimension
    N = 10_000             # corpus size (10K posts)
    K = 100                # retrieve top 100
    num_actions = 8        # like, retweet, reply, bookmark, ...
    num_surfaces = 16      # home, search, notifications, ...
    num_user_hashes = 2
    num_item_hashes = 2
    num_author_hashes = 2

    print("=" * 60)
    print("Two-Tower Retrieval Model -- PyTorch Demo")
    print("=" * 60)

    # --- Build model ---
    model = TwoTowerRetrieval(
        emb_size=D,
        num_actions=num_actions,
        num_surfaces=num_surfaces,
        num_user_hashes=num_user_hashes,
        num_item_hashes=num_item_hashes,
        num_author_hashes=num_author_hashes,
        transformer_heads=4,
        transformer_layers=2,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"
Total parameters: {total_params:,}")
    print(f"  Candidate tower: {sum(p.numel() for p in model.candidate_tower.parameters()):,}")
    print(f"  User tower:      {sum(p.numel() for p in model.user_tower.parameters()):,}")

    # --- Synthetic user data ---
    user_hashes = torch.randint(1, 1000, (B, num_user_hashes))                   # [B, 2]
    user_embeddings = torch.randn(B, num_user_hashes, D)                         # [B, 2, D]

    history_post_hashes = torch.randint(0, 1000, (B, S, num_item_hashes))        # [B, 128, 2]
    history_post_embeddings = torch.randn(B, S, num_item_hashes, D)              # [B, 128, 2, D]
    history_author_embeddings = torch.randn(B, S, num_author_hashes, D)          # [B, 128, 2, D]
    history_actions = torch.randint(0, 2, (B, S, num_actions)).float()           # [B, 128, 8]
    history_product_surface = torch.randint(0, num_surfaces, (B, S))             # [B, 128]

    # Simulate padding: last 78 positions have no data
    history_post_hashes[:, 50:, :] = 0
    history_actions[:, 50:, :] = 0.0

    # --- Synthetic corpus (offline pre-computation) ---
    # In production: these come from running candidate tower on all posts
    corpus_post_author_emb = torch.randn(N, num_item_hashes + num_author_hashes, D)

    print("
--- Offline: Encoding corpus ---")
    with torch.no_grad():
        corpus_embeddings = model.encode_candidates(corpus_post_author_emb)     # [N, D]
    print(f"Corpus embeddings shape: {corpus_embeddings.shape}")
    print(f"Corpus embedding norms (should be ~1.0): "
          f"{corpus_embeddings.norm(dim=-1).mean():.4f}")

    # --- Online: Encode user + retrieve ---
    print("
--- Online: Encoding users + retrieving ---")
    with torch.no_grad():
        output = model(
            user_embeddings=user_embeddings,
            user_hashes=user_hashes,
            history_post_embeddings=history_post_embeddings,
            history_author_embeddings=history_author_embeddings,
            history_actions=history_actions,
            history_product_surface=history_product_surface,
            history_post_hashes=history_post_hashes,
            corpus_embeddings=corpus_embeddings,
            top_k=K,
        )

    print(f"User representation shape: {output.user_representation.shape}")      # [4, 64]
    print(f"User repr norms (should be 1.0): "
          f"{output.user_representation.norm(dim=-1).tolist()}")
    print(f"Top-K indices shape: {output.top_k_indices.shape}")                  # [4, 100]
    print(f"Top-K scores shape:  {output.top_k_scores.shape}")                   # [4, 100]
    print(f"Score range: [{output.top_k_scores.min():.4f}, "
          f"{output.top_k_scores.max():.4f}]")
    print(f"Top 5 posts for user 0: {output.top_k_indices[0, :5].tolist()}")
    print(f"Top 5 scores for user 0: "
          f"{[f'{s:.4f}' for s in output.top_k_scores[0, :5].tolist()]}")

    print("
Done.")


if __name__ == "__main__":
    main()
```

---

## Appendix B: JAX-to-PyTorch Cheat Sheet

| JAX / Haiku | PyTorch | Notes |
|---|---|---|
| `hk.get_parameter("name", [M, N])` | `nn.Linear(M, N, bias=False)` | Haiku uses raw params; PyTorch wraps in modules |
| `jnp.dot(x, W)` | `F.linear(x, W.T)` or `nn.Linear` | PyTorch Linear stores W transposed |
| `jax.nn.silu(x)` | `F.silu(x)` | Same function: `x * sigmoid(x)` |
| `jax.nn.one_hot(idx, N)` | `F.one_hot(idx, N).float()` | PyTorch returns LongTensor by default |
| `jnp.sum(x**2, axis=-1)` | `(x**2).sum(dim=-1)` | Identical semantics |
| `jax.lax.top_k(x, k)` | `torch.topk(x, k)` | Returns `(values, indices)` in both |
| `jnp.where(mask, x, -INF)` | `x.masked_fill(~mask, -INF)` | Note the mask inversion |
| `jnp.concatenate([a, b], axis=1)` | `torch.cat([a, b], dim=1)` | Same semantics |
| `hk.initializers.VarianceScaling(1.0, "fan_out")` | `nn.init.kaiming_normal_(w, mode="fan_out")` | Approximately equivalent |

---

## Appendix C: Further Reading

- **FAISS** (Facebook AI Similarity Search): The most common ANN library for
  production retrieval. Supports IVF, PQ, and HNSW indexes.
- **ScaNN** (Google): Another ANN library, optimized for TPU-style hardware.
- **"Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations"
  (Yi et al., 2019)**: The foundational two-tower paper from YouTube.
- **"Deep Neural Networks for YouTube Recommendations" (Covington et al., 2016)**:
  The original two-stage retrieval + ranking paper.

---

*Next up: Lecture 4 -- The Ranking Model (full transformer with candidate isolation)*
