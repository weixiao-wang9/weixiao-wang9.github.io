---
type: note
course: "[[Recommendation Systems]]"
date: 2026-02-18
---

# Lecture 02 -- Hash Embeddings in X's Recommendation Algorithm

> **Source files:**
> `/phoenix/recsys_model.py` and `/phoenix/runners.py`
>
> **Prerequisites:** Basic understanding of embeddings, PyTorch `nn.Embedding`, and matrix
> multiplication. Familiarity with recommendation systems is helpful but not required.

---

## Table of Contents

1. [The Problem: Billion Entities](#1-the-problem-billion-entities)
2. [HashConfig](#2-hashconfig)
3. [The Collision Problem and Multi-Hash Solution](#3-the-collision-problem-and-multi-hash-solution)
4. [Data Containers: RecsysBatch and RecsysEmbeddings](#4-data-containers)
5. [Where Hash Lookups Actually Happen](#5-where-hash-lookups-actually-happen)
6. [block_user_reduce](#6-block_user_reduce)
7. [block_history_reduce](#7-block_history_reduce)
8. [block_candidate_reduce](#8-block_candidate_reduce)
9. [Why Flatten + Linear, Not Average](#9-why-flatten--linear-not-average)
10. [Quick Quiz](#10-quick-quiz)
11. [PyTorch Appendix: Full Runnable Code](#11-pytorch-appendix)

---

## 1. The Problem: Billion Entities

X has billions of users and billions of posts. The recommendation system needs to learn a
dense vector (an "embedding") for every single one of them so it can figure out who likes
what.

Let's do the back-of-the-envelope math on a naive approach:

```
nn.Embedding(2_000_000_000, 64)

  = 2 billion rows * 64 floats/row * 4 bytes/float
  = 512,000,000,000 bytes
  = 512 GB

...for ONE embedding table. And we need several.
```

That does not fit on a single GPU (typically 40-80 GB), and it would be wildly expensive
even in CPU memory. We need a trick.

**The trick: hash-based embeddings with multiple hash functions.**

Instead of giving every entity its own row, we hash entities into a much smaller table
(say, 100K rows). Multiple entities will share rows -- that is called a **collision** --
but we use multiple independent hash functions to make those collisions nearly harmless.

```
  Naive approach:                Hash approach:
  +-----------------------+      +------------------+
  | Row 0: user_0 embed   |      | Row 0: shared    |   <-- 100K rows, not 2B
  | Row 1: user_1 embed   |      | Row 1: shared    |
  | Row 2: user_2 embed   |      | ...              |
  | ...                    |      | Row 99999: shared|
  | Row 1,999,999,999     |      +------------------+
  +-----------------------+
       512 GB                        ~25 MB
```

The trade-off: collisions degrade quality. The solution: use N hash functions into N
separate tables, then combine the results. The chance of two entities colliding in ALL N
tables simultaneously is astronomically low.

---

## 2. HashConfig

**File:** `recsys_model.py`, lines 32-38

```python
@dataclass
class HashConfig:
    """Configuration for hash-based embeddings."""

    num_user_hashes: int = 2
    num_item_hashes: int = 2
    num_author_hashes: int = 2
```

Three entity types, each hashed independently:

| Entity Type | What It Represents       | Default Hash Count |
|-------------|--------------------------|-------------------|
| User        | The person viewing       | 2                 |
| Item (Post) | A tweet / post           | 2                 |
| Author      | The person who wrote it  | 2                 |

Each entity type gets its own set of embedding tables. With `num_user_hashes=2`, there
are 2 separate user embedding tables, and every user gets looked up in both of them.

Think of it this way: if you have a library with 2 card catalogs organized differently,
you look up each book in both catalogs and combine the results for a richer description.

---

## 3. The Collision Problem and Multi-Hash Solution

### The problem with one hash function

Suppose we hash all 2 billion users into 100K buckets using a single hash function.
On average, each bucket holds 20,000 users. Alice and Bob land in the same bucket.
The model literally cannot tell them apart -- they get the same embedding.

```
  hash("alice") = 42          hash("bob") = 42
         |                          |
         +--- both map to Row 42 ---+
         |                          |
       COLLISION: identical embeddings, different people
```

### Multiple hashes to the rescue

With 2 hash functions and 2 tables, Alice and Bob would need to collide in BOTH tables
to be truly indistinguishable. With independent hash functions, that probability is:

```
  P(collision in table 1) * P(collision in table 2)
  = (1/100,000) * (1/100,000)
  = 1 in 10,000,000,000

  (vs. 1 in 100,000 with a single hash)
```

Each entity gets N independent lookups from N separate tables, and the results are
combined by a learned projection.

### Analogy

Think of it like identifying someone:

| Number of Hashes | Analogy                              | Collision Risk     |
|-------------------|--------------------------------------|--------------------|
| 1                 | Last name only                       | Very common        |
| 2                 | Last name + birthday                 | Rare               |
| 3                 | Last name + birthday + zip code      | Very rare          |
| 4                 | Last name + birthday + zip + SSN     | Basically unique   |

The default of 2 hashes per entity is a practical sweet spot: it catches the vast majority
of collisions while keeping the number of embedding lookups (and hence memory + compute)
reasonable.

---

## 4. Data Containers

There are three key data structures that flow through the system. Understanding them is
essential before reading any of the reduce functions.

### Analogy: The Restaurant

Think of inference as a restaurant order:

| Data Structure       | Restaurant Analogy                                    |
|----------------------|-------------------------------------------------------|
| `RecsysBatch`        | The order slip -- small integers telling the kitchen what to fetch |
| `RecsysEmbeddings`   | The ingredients pulled from the pantry -- big float tensors       |
| `RecsysModelOutput`  | The finished dish -- logits/scores for each candidate             |

### Dimension Key

Before we look at the shapes, let's define our dimension letters:

```
B = batch size           (e.g., 128 requests processed together)
S = history seq length   (e.g., 128 past posts the user interacted with)
C = candidate count      (e.g., 32 candidate posts to rank)
D = embedding dimension  (e.g., 64 or 128)
```

### RecsysEmbeddings (lines 42-53)

This holds the **pre-looked-up** float embeddings. By the time the model sees these,
the hash lookups have already happened somewhere upstream (on parameter servers).

```python
@dataclass
class RecsysEmbeddings:
    user_embeddings: ArrayLike              # [B, num_user_hashes, D]
    history_post_embeddings: ArrayLike      # [B, S, num_item_hashes, D]
    candidate_post_embeddings: ArrayLike    # [B, C, num_item_hashes, D]
    history_author_embeddings: ArrayLike    # [B, S, num_author_hashes, D]
    candidate_author_embeddings: ArrayLike  # [B, C, num_author_hashes, D]
```

Notice the pattern: there is always a `num_*_hashes` dimension. That is the dimension
that will be flattened and projected away by the reduce functions we will see shortly.

```
  user_embeddings shape:  [B, num_user_hashes, D]
                           |        |           |
                     batch size   2 lookups   64-dim vector per lookup
```

### RecsysBatch (lines 62-77)

This holds the **raw feature data** -- integer hashes, action vectors, and surface IDs.
These are small and cheap to transmit.

```python
class RecsysBatch(NamedTuple):
    user_hashes: ArrayLike                  # [B, num_user_hashes]
    history_post_hashes: ArrayLike          # [B, S, num_item_hashes]
    history_author_hashes: ArrayLike        # [B, S, num_author_hashes]
    history_actions: ArrayLike              # [B, S, num_actions]
    history_product_surface: ArrayLike      # [B, S]
    candidate_post_hashes: ArrayLike        # [B, C, num_item_hashes]
    candidate_author_hashes: ArrayLike      # [B, C, num_author_hashes]
    candidate_product_surface: ArrayLike    # [B, C]
```

**Why separate the two?** Because in production, `RecsysBatch` lives on the serving
tier (small integers from the request), while `RecsysEmbeddings` lives on parameter
servers (big float tensors). Keeping them separate lets you optimize the network hop:
send small hash integers to the parameter server, get back big embeddings. The model
itself never touches the raw embedding tables.

### RecsysModelOutput (lines 56-59)

```python
class RecsysModelOutput(NamedTuple):
    logits: jax.Array   # [B, C, num_actions]
```

The final output: for each candidate in the batch, a score for each engagement type
(favorite, reply, repost, etc.).

---

## 5. Where Hash Lookups Actually Happen

**File:** `runners.py`, lines 389-487, function `create_example_batch()`

This function is a test/example utility, but it perfectly illustrates the two-step
process that happens in production:

### Step 1: Generate hash indices (lines 421-458)

```python
# User hashes: random integers in [1, num_user_embeddings)
user_hashes = rng.integers(
    1, num_user_embeddings,
    size=(batch_size, num_user_hashes)
).astype(np.int32)

# Post hashes: same idea, for each history position
history_post_hashes = rng.integers(
    1, num_post_embeddings,
    size=(batch_size, history_len, num_item_hashes)
).astype(np.int32)
```

In production, these are not random -- they come from actual hash functions applied to
user IDs, post IDs, and author IDs. But the shape and dtype are identical.

**Critical detail:** hash values start at 1, not 0. Hash value 0 is reserved for
padding/invalid entries. This is how the model knows which positions in a sequence are
real and which are empty (see the padding masks in the reduce functions).

### Step 2: Look up embeddings from tables (lines 471-485)

```python
embeddings = RecsysEmbeddings(
    user_embeddings=rng.normal(
        size=(batch_size, num_user_hashes, emb_size)
    ).astype(np.float32),
    history_post_embeddings=rng.normal(
        size=(batch_size, history_len, num_item_hashes, emb_size)
    ).astype(np.float32),
    # ... etc.
)
```

Again, in production these come from actual embedding table lookups on parameter servers.
The test function uses random normal vectors.

### The full data flow

```
  +-------------------+      +---------------------+     +------------------+
  | Raw entity IDs    | ---> | Hash functions       | --> | Integer hashes   |
  | (user_id=12345)   |      | h1(x), h2(x), ...   |     | [42, 8891]       |
  +-------------------+      +---------------------+     +------------------+
                                                                  |
                                                                  v
                                                          +------------------+
                                                          | Embedding tables  |
                                                          | (param servers)   |
                                                          | table1[42] -> D  |
                                                          | table2[8891]-> D |
                                                          +------------------+
                                                                  |
                                                                  v
                                                          +------------------+
                                                          | RecsysEmbeddings |
                                                          | [B, 2, D]        |
                                                          +------------------+
                                                                  |
                                                                  v
                                                          +------------------+
                                                          | block_*_reduce   |
                                                          | flatten + linear |
                                                          | -> [B, 1, D]     |
                                                          +------------------+
```

---

## 6. block_user_reduce

**File:** `recsys_model.py`, lines 79-119

This function takes multiple user hash embeddings and combines them into a single user
representation.

### Signature

```python
def block_user_reduce(
    user_hashes: jnp.ndarray,       # [B, num_user_hashes]
    user_embeddings: jnp.ndarray,   # [B, num_user_hashes, D]
    num_user_hashes: int,
    emb_size: int,
    embed_init_scale: float = 1.0,
) -> Tuple[jax.Array, jax.Array]:
    # Returns: (user_embedding [B, 1, D], user_padding_mask [B, 1])
```

### Step-by-step walkthrough

**1. Reshape -- flatten the hash dimension (line 102)**

```python
user_embedding = user_embeddings.reshape((B, 1, num_user_hashes * D))
```

```
  Before:  [B, num_user_hashes, D]     e.g. [128, 2, 64]
  After:   [B, 1, num_user_hashes * D]      [128, 1, 128]

  The "1" in the middle is the sequence-length dimension.
  A user is treated as a sequence of length 1 (one "token").
```

Why the `1`? Because later, the user embedding will be concatenated with history
(length S) and candidates (length C) along the sequence dimension. Having the user
as a length-1 sequence makes that concatenation clean.

**2. Linear projection (lines 104-114)**

```python
proj_mat_1 = hk.get_parameter(
    "proj_mat_1",
    [num_user_hashes * D, D],    # e.g. [128, 64]
    ...
)
user_embedding = jnp.dot(user_embedding, proj_mat_1)  # [B, 1, D]
```

This is just a learned linear layer: `Linear(num_user_hashes * D, D)`.

```
  [B, 1, 128] @ [128, 64] -> [B, 1, 64]
```

**3. Padding mask (line 117)**

```python
user_padding_mask = (user_hashes[:, 0] != 0).reshape(B, 1)
```

Checks the first hash value. If it is 0, this user slot is padding (invalid).
The mask is True for valid users, False for padding.

### ASCII summary

```
  user_embeddings:  [B, 2, 64]
        |
        | reshape
        v
  [B, 1, 128]      (flatten 2 hash embeddings into one long vector)
        |
        | Linear(128, 64)
        v
  [B, 1, 64]       (projected back to embedding dim)

  + padding_mask:   [B, 1]  (True where user_hashes[:,0] != 0)
```

### PyTorch translation

```python
# JAX/Haiku version (from codebase):
user_embedding = user_embeddings.reshape((B, 1, num_user_hashes * D))
user_embedding = jnp.dot(user_embedding, proj_mat_1)

# PyTorch equivalent:
user_embedding = user_embeddings.reshape(B, 1, num_user_hashes * D)
user_embedding = self.user_proj(user_embedding)  # nn.Linear(num_user_hashes * D, D)
```

---

## 7. block_history_reduce

**File:** `recsys_model.py`, lines 122-182

For each post in the user's history, this function combines **four** types of information
into a single embedding:

1. Post hash embeddings (who is this post?)
2. Author hash embeddings (who wrote it?)
3. Action embeddings (what did the user do -- like, retweet, etc.?)
4. Product surface embeddings (where did the user see it -- home timeline, search, etc.?)

### Signature

```python
def block_history_reduce(
    history_post_hashes: jnp.ndarray,              # [B, S, num_item_hashes]
    history_post_embeddings: jnp.ndarray,           # [B, S, num_item_hashes, D]
    history_author_embeddings: jnp.ndarray,         # [B, S, num_author_hashes, D]
    history_product_surface_embeddings: jnp.ndarray, # [B, S, D]
    history_actions_embeddings: jnp.ndarray,        # [B, S, D]
    num_item_hashes: int,
    num_author_hashes: int,
    embed_init_scale: float = 1.0,
) -> Tuple[jax.Array, jax.Array]:
    # Returns: (history_embedding [B, S, D], history_padding_mask [B, S])
```

### Step-by-step walkthrough

**1. Flatten hash dimensions (lines 151-154)**

```python
history_post_embeddings_reshaped = history_post_embeddings.reshape(
    (B, S, num_item_hashes * D)
)
history_author_embeddings_reshaped = history_author_embeddings.reshape(
    (B, S, num_author_hashes * D)
)
```

```
  post embeddings:   [B, S, 2, 64] -> [B, S, 128]
  author embeddings: [B, S, 2, 64] -> [B, S, 128]
```

**2. Concatenate all four ingredients (lines 156-164)**

```python
post_author_embedding = jnp.concatenate(
    [
        history_post_embeddings_reshaped,       # [B, S, num_item_hashes * D]
        history_author_embeddings_reshaped,      # [B, S, num_author_hashes * D]
        history_actions_embeddings,              # [B, S, D]
        history_product_surface_embeddings,      # [B, S, D]
    ],
    axis=-1,
)
```

```
  Concatenated dim = num_item_hashes*D + num_author_hashes*D + D + D
                   = 2*64 + 2*64 + 64 + 64
                   = 128 + 128 + 64 + 64
                   = 384
```

**3. Linear projection (lines 166-178)**

```python
proj_mat_3 = hk.get_parameter("proj_mat_3", [384, D], ...)
history_embedding = jnp.dot(post_author_embedding, proj_mat_3)  # [B, S, D]
```

```
  [B, S, 384] @ [384, 64] -> [B, S, 64]
```

**4. Padding mask (line 180)**

```python
history_padding_mask = (history_post_hashes[:, :, 0] != 0).reshape(B, S)
```

For each position in the history, check if the first post hash is non-zero.

### ASCII summary

```
  post_emb:      [B, S, 2, 64] --flatten--> [B, S, 128] --+
  author_emb:    [B, S, 2, 64] --flatten--> [B, S, 128] --+-- concat --> [B, S, 384]
  actions_emb:   [B, S, 64]    --------------------------------+              |
  surface_emb:   [B, S, 64]    --------------------------------+              |
                                                                         Linear(384, 64)
                                                                               |
                                                                               v
                                                                         [B, S, 64]
  + padding_mask: [B, S]  (True where history_post_hashes[:,:,0] != 0)
```

### PyTorch translation

```python
# JAX/Haiku version:
post_flat = history_post_embeddings.reshape((B, S, num_item_hashes * D))
auth_flat = history_author_embeddings.reshape((B, S, num_author_hashes * D))
combined = jnp.concatenate([post_flat, auth_flat, actions_emb, surface_emb], axis=-1)
history_embedding = jnp.dot(combined, proj_mat_3)

# PyTorch equivalent:
post_flat = history_post_embeddings.reshape(B, S, num_item_hashes * D)
auth_flat = history_author_embeddings.reshape(B, S, num_author_hashes * D)
combined = torch.cat([post_flat, auth_flat, actions_emb, surface_emb], dim=-1)
history_embedding = self.history_proj(combined)  # nn.Linear(384, D)
```

---

## 8. block_candidate_reduce

**File:** `recsys_model.py`, lines 185-242

This is structurally almost identical to `block_history_reduce`, but with one key
difference: **no action embeddings**. Candidates are posts we have not interacted with
yet, so there is no action history to include. Only three ingredients:

1. Post hash embeddings
2. Author hash embeddings
3. Product surface embeddings

### Signature

```python
def block_candidate_reduce(
    candidate_post_hashes: jnp.ndarray,              # [B, C, num_item_hashes]
    candidate_post_embeddings: jnp.ndarray,           # [B, C, num_item_hashes, D]
    candidate_author_embeddings: jnp.ndarray,         # [B, C, num_author_hashes, D]
    candidate_product_surface_embeddings: jnp.ndarray, # [B, C, D]
    num_item_hashes: int,
    num_author_hashes: int,
    embed_init_scale: float = 1.0,
) -> Tuple[jax.Array, jax.Array]:
    # Returns: (candidate_embedding [B, C, D], candidate_padding_mask [B, C])
```

### The three ingredients

```python
post_author_embedding = jnp.concatenate(
    [
        candidate_post_embeddings_reshaped,          # [B, C, num_item_hashes * D]
        candidate_author_embeddings_reshaped,        # [B, C, num_author_hashes * D]
        candidate_product_surface_embeddings,        # [B, C, D]
    ],
    axis=-1,
)
```

```
  Concatenated dim = num_item_hashes*D + num_author_hashes*D + D
                   = 128 + 128 + 64
                   = 320

  (Compare to history's 384 -- the missing 64 is the action embedding.)
```

### ASCII summary

```
  post_emb:      [B, C, 2, 64] --flatten--> [B, C, 128] --+
  author_emb:    [B, C, 2, 64] --flatten--> [B, C, 128] --+-- concat --> [B, C, 320]
  surface_emb:   [B, C, 64]    --------------------------------+              |
                                                                         Linear(320, 64)
                                                                               |
                                                                               v
                                                                         [B, C, 64]
  + padding_mask: [B, C]  (True where candidate_post_hashes[:,:,0] != 0)
```

### Side-by-side: history vs. candidate reduce

```
  block_history_reduce:          block_candidate_reduce:
  +---------------------------+  +---------------------------+
  | post hashes  (flattened)  |  | post hashes  (flattened)  |
  | author hashes (flattened) |  | author hashes (flattened) |
  | action embeddings         |  | (no actions -- not yet    |
  | surface embeddings        |  |  interacted with)         |
  +---------------------------+  | surface embeddings        |
  | concat dim: 384           |  +---------------------------+
  | Linear(384, D)            |  | concat dim: 320           |
  +---------------------------+  | Linear(320, D)            |
                                 +---------------------------+
```

---

## 9. Why Flatten + Linear, Not Average

You might wonder: why not just average the N hash embeddings? That would also reduce
`[B, N, D]` to `[B, D]` and is simpler.

```python
# Simpler alternative (NOT what the code does):
user_embedding = user_embeddings.mean(dim=1)  # average over hash dim
```

Three reasons the codebase uses `flatten + linear` instead:

### 1. Strictly more expressive

Averaging forces equal weight on each hash table. But maybe hash table 1 learned better
representations than hash table 2. The linear projection can learn arbitrary weights:

```
  Average:    embed = 0.5 * table1[h1] + 0.5 * table2[h2]    (fixed weights)

  Linear:     embed = W @ [table1[h1]; table2[h2]]            (learned weights)

  The linear layer subsumes averaging as a special case
  (when W happens to be [0.5*I | 0.5*I]).
```

### 2. Cross-hash interactions

With concatenation, the linear layer can learn to combine specific dimensions from
different tables. Dimension 3 of table 1 can interact with dimension 7 of table 2.
Averaging cannot do this -- it operates element-wise.

### 3. Feature mixing

In `block_history_reduce`, the linear layer projects the concatenation of post hashes,
author hashes, actions, AND surfaces. This means the model can learn interactions across
all these feature types in a single matrix multiply, which is both powerful and efficient.

---

## 10. Quick Quiz

Test your understanding before moving on.

**Q1:** If `num_item_hashes=4` and `emb_size=64`, what is the flattened dimension for
post embeddings in `block_history_reduce`?

<details>
<summary>Answer</summary>

`4 * 64 = 256`. The reshape goes from `[B, S, 4, 64]` to `[B, S, 256]`.

</details>

**Q2:** Why is hash value 0 reserved?

<details>
<summary>Answer</summary>

Hash value 0 means "padding / invalid." The reduce functions use `hashes[:, :, 0] != 0`
to build padding masks. This tells the transformer which positions in the sequence are
real data and which are empty padding. If we allowed hash value 0 for real entities, we
could not distinguish real data from padding.

</details>

**Q3:** Why are `RecsysBatch` and `RecsysEmbeddings` separate data structures?

<details>
<summary>Answer</summary>

`RecsysBatch` contains small integer hashes that come from the serving request -- cheap
to transmit. `RecsysEmbeddings` contains large float tensors looked up from parameter
servers -- expensive to store. Separating them lets the system optimize the data flow:
send small hashes over the network to the parameter server, get back the big embeddings,
then feed both into the model. The model itself never touches the raw embedding tables.

</details>

**Q4:** What is the total input dimension for `block_candidate_reduce`'s linear layer when
`num_item_hashes=2`, `num_author_hashes=2`, and `emb_size=64`?

<details>
<summary>Answer</summary>

`num_item_hashes * D + num_author_hashes * D + D = 2*64 + 2*64 + 64 = 320`.
No action embedding for candidates (they have not been interacted with yet).

</details>

**Q5:** If you increased `num_user_hashes` from 2 to 4, what changes in `block_user_reduce`?

<details>
<summary>Answer</summary>

The flattened dimension doubles from `2*D` to `4*D`, so the projection matrix
`proj_mat_1` grows from `[2*D, D]` to `[4*D, D]`. The user gets looked up in 4 tables
instead of 2, giving better collision resistance but more parameters and more embedding
lookups.

</details>

---

## 11. PyTorch Appendix

Below is a complete, runnable PyTorch implementation of the hash embedding system. You
can copy-paste this into a `.py` file and run it directly.

```python
"""
PyTorch translation of X's hash embedding system.

Covers:
  - MultiHashEmbedding: the hash-based lookup tables
  - BlockUserReduce: combines multiple user hash embeddings -> single user token
  - BlockHistoryReduce: combines post/author/action/surface -> history tokens
  - BlockCandidateReduce: combines post/author/surface -> candidate tokens

Run this file directly to see a full forward pass with random data.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HashConfig:
    """Mirror of recsys_model.py:32-38"""
    num_user_hashes: int = 2
    num_item_hashes: int = 2
    num_author_hashes: int = 2


# ---------------------------------------------------------------------------
# MultiHashEmbedding -- the hash-based embedding lookup
# ---------------------------------------------------------------------------

class MultiHashEmbedding(nn.Module):
    """
    Multiple independent embedding tables, one per hash function.

    In production, the hash functions + lookups happen on parameter servers.
    This module simulates both the tables and the lookups for local testing.

    Args:
        num_hashes: Number of independent hash functions / tables (e.g. 2)
        table_size: Number of rows per table (e.g. 100_000)
        emb_dim: Embedding dimension D (e.g. 64)
    """

    def __init__(self, num_hashes: int, table_size: int, emb_dim: int):
        super().__init__()
        self.num_hashes = num_hashes
        self.table_size = table_size
        self.emb_dim = emb_dim

        # One nn.Embedding per hash function
        self.tables = nn.ModuleList([
            nn.Embedding(table_size, emb_dim, padding_idx=0)
            for _ in range(num_hashes)
        ])

    def forward(self, hashes: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings from each table.

        Args:
            hashes: [..., num_hashes] integer hash indices.
                    0 = padding/invalid.

        Returns:
            embeddings: [..., num_hashes, D]
        """
        # hashes[..., i] indexes into self.tables[i]
        parts = []
        for i in range(self.num_hashes):
            emb = self.tables[i](hashes[..., i])  # [..., D]
            parts.append(emb)
        # Stack along the hash dimension
        return torch.stack(parts, dim=-2)  # [..., num_hashes, D]


# ---------------------------------------------------------------------------
# BlockUserReduce
# ---------------------------------------------------------------------------

class BlockUserReduce(nn.Module):
    """
    Combine multiple user hash embeddings into a single user representation.

    Mirror of recsys_model.py:79-119 (block_user_reduce).

    Input:  user_embeddings [B, num_user_hashes, D]
    Output: user_embedding  [B, 1, D], user_padding_mask [B, 1]
    """

    def __init__(self, num_user_hashes: int, emb_dim: int):
        super().__init__()
        self.num_user_hashes = num_user_hashes
        self.emb_dim = emb_dim
        # Linear(num_user_hashes * D, D) -- corresponds to proj_mat_1
        self.proj = nn.Linear(num_user_hashes * emb_dim, emb_dim, bias=False)

    def forward(
        self,
        user_hashes: torch.Tensor,       # [B, num_user_hashes]
        user_embeddings: torch.Tensor,    # [B, num_user_hashes, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = user_embeddings.shape[0]
        D = self.emb_dim

        # Step 1: Flatten hash embeddings into one long vector
        # [B, num_user_hashes, D] -> [B, 1, num_user_hashes * D]
        user_embedding = user_embeddings.reshape(B, 1, self.num_user_hashes * D)

        # Step 2: Project back to D
        # [B, 1, num_user_hashes * D] -> [B, 1, D]
        user_embedding = self.proj(user_embedding)

        # Step 3: Padding mask (hash 0 = padding)
        # [B, 1]
        user_padding_mask = (user_hashes[:, 0] != 0).reshape(B, 1)

        return user_embedding, user_padding_mask


# ---------------------------------------------------------------------------
# BlockHistoryReduce
# ---------------------------------------------------------------------------

class BlockHistoryReduce(nn.Module):
    """
    Combine history post/author/action/surface embeddings into history tokens.

    Mirror of recsys_model.py:122-182 (block_history_reduce).

    Inputs:
        history_post_embeddings:    [B, S, num_item_hashes, D]
        history_author_embeddings:  [B, S, num_author_hashes, D]
        history_actions_embeddings: [B, S, D]
        history_surface_embeddings: [B, S, D]

    Output: history_embedding [B, S, D], history_padding_mask [B, S]
    """

    def __init__(self, num_item_hashes: int, num_author_hashes: int, emb_dim: int):
        super().__init__()
        self.num_item_hashes = num_item_hashes
        self.num_author_hashes = num_author_hashes
        self.emb_dim = emb_dim

        # Input dim = num_item_hashes*D + num_author_hashes*D + D + D
        input_dim = (num_item_hashes + num_author_hashes + 2) * emb_dim
        # Corresponds to proj_mat_3
        self.proj = nn.Linear(input_dim, emb_dim, bias=False)

    def forward(
        self,
        history_post_hashes: torch.Tensor,         # [B, S, num_item_hashes]
        history_post_embeddings: torch.Tensor,      # [B, S, num_item_hashes, D]
        history_author_embeddings: torch.Tensor,    # [B, S, num_author_hashes, D]
        history_actions_embeddings: torch.Tensor,   # [B, S, D]
        history_surface_embeddings: torch.Tensor,   # [B, S, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, _, D = history_post_embeddings.shape

        # Step 1: Flatten hash dimensions
        # [B, S, num_item_hashes, D] -> [B, S, num_item_hashes * D]
        post_flat = history_post_embeddings.reshape(B, S, self.num_item_hashes * D)

        # [B, S, num_author_hashes, D] -> [B, S, num_author_hashes * D]
        author_flat = history_author_embeddings.reshape(B, S, self.num_author_hashes * D)

        # Step 2: Concatenate all four ingredients along the last dimension
        # [B, S, num_item_hashes*D + num_author_hashes*D + D + D]
        combined = torch.cat([
            post_flat,                    # [B, S, num_item_hashes * D]
            author_flat,                  # [B, S, num_author_hashes * D]
            history_actions_embeddings,   # [B, S, D]
            history_surface_embeddings,   # [B, S, D]
        ], dim=-1)

        # Step 3: Project to D
        # [B, S, input_dim] -> [B, S, D]
        history_embedding = self.proj(combined)

        # Step 4: Padding mask
        # [B, S]
        history_padding_mask = (history_post_hashes[:, :, 0] != 0)

        return history_embedding, history_padding_mask


# ---------------------------------------------------------------------------
# BlockCandidateReduce
# ---------------------------------------------------------------------------

class BlockCandidateReduce(nn.Module):
    """
    Combine candidate post/author/surface embeddings into candidate tokens.

    Mirror of recsys_model.py:185-242 (block_candidate_reduce).

    Same as BlockHistoryReduce but WITHOUT action embeddings (candidates
    have not been interacted with yet).

    Inputs:
        candidate_post_embeddings:    [B, C, num_item_hashes, D]
        candidate_author_embeddings:  [B, C, num_author_hashes, D]
        candidate_surface_embeddings: [B, C, D]

    Output: candidate_embedding [B, C, D], candidate_padding_mask [B, C]
    """

    def __init__(self, num_item_hashes: int, num_author_hashes: int, emb_dim: int):
        super().__init__()
        self.num_item_hashes = num_item_hashes
        self.num_author_hashes = num_author_hashes
        self.emb_dim = emb_dim

        # Input dim = num_item_hashes*D + num_author_hashes*D + D  (no actions)
        input_dim = (num_item_hashes + num_author_hashes + 1) * emb_dim
        # Corresponds to proj_mat_2
        self.proj = nn.Linear(input_dim, emb_dim, bias=False)

    def forward(
        self,
        candidate_post_hashes: torch.Tensor,         # [B, C, num_item_hashes]
        candidate_post_embeddings: torch.Tensor,      # [B, C, num_item_hashes, D]
        candidate_author_embeddings: torch.Tensor,    # [B, C, num_author_hashes, D]
        candidate_surface_embeddings: torch.Tensor,   # [B, C, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, _, D = candidate_post_embeddings.shape

        # Step 1: Flatten hash dimensions
        post_flat = candidate_post_embeddings.reshape(B, C, self.num_item_hashes * D)
        author_flat = candidate_author_embeddings.reshape(B, C, self.num_author_hashes * D)

        # Step 2: Concatenate three ingredients (no actions for candidates)
        # [B, C, num_item_hashes*D + num_author_hashes*D + D]
        combined = torch.cat([
            post_flat,                      # [B, C, num_item_hashes * D]
            author_flat,                    # [B, C, num_author_hashes * D]
            candidate_surface_embeddings,   # [B, C, D]
        ], dim=-1)

        # Step 3: Project to D
        # [B, C, input_dim] -> [B, C, D]
        candidate_embedding = self.proj(combined)

        # Step 4: Padding mask
        candidate_padding_mask = (candidate_post_hashes[:, :, 0] != 0)

        return candidate_embedding, candidate_padding_mask


# ---------------------------------------------------------------------------
# Full demo: wire everything together and run a forward pass
# ---------------------------------------------------------------------------

def main():
    """
    Demonstrates the full hash embedding pipeline with random data.
    Mirrors the flow in runners.py:389-487 (create_example_batch).
    """
    torch.manual_seed(42)

    # --- Configuration ---
    B = 4             # batch size
    S = 8             # history sequence length
    C = 4             # number of candidates
    D = 64            # embedding dimension
    num_actions = 19  # number of action types (matches ACTIONS list in runners.py)
    table_size = 100_000
    product_surface_vocab_size = 16

    config = HashConfig(
        num_user_hashes=2,
        num_item_hashes=2,
        num_author_hashes=2,
    )

    # --- Build modules ---
    user_hash_emb = MultiHashEmbedding(config.num_user_hashes, table_size, D)
    post_hash_emb = MultiHashEmbedding(config.num_item_hashes, table_size, D)
    author_hash_emb = MultiHashEmbedding(config.num_author_hashes, table_size, D)

    action_proj = nn.Linear(num_actions, D, bias=False)
    surface_emb_table = nn.Embedding(product_surface_vocab_size, D)

    user_reduce = BlockUserReduce(config.num_user_hashes, D)
    history_reduce = BlockHistoryReduce(config.num_item_hashes, config.num_author_hashes, D)
    candidate_reduce = BlockCandidateReduce(config.num_item_hashes, config.num_author_hashes, D)

    # --- Step 1: Create fake hashes (simulates serving request) ---
    # Hash value 0 = padding. Real hashes start at 1.
    user_hashes = torch.randint(1, table_size, (B, config.num_user_hashes))

    history_post_hashes = torch.randint(1, table_size, (B, S, config.num_item_hashes))
    history_post_hashes[:, S // 2 :, :] = 0  # last half is padding

    history_author_hashes = torch.randint(1, table_size, (B, S, config.num_author_hashes))
    history_author_hashes[:, S // 2 :, :] = 0

    history_actions = (torch.rand(B, S, num_actions) > 0.7).float()
    history_surface = torch.randint(0, product_surface_vocab_size, (B, S))

    candidate_post_hashes = torch.randint(1, table_size, (B, C, config.num_item_hashes))
    candidate_author_hashes = torch.randint(1, table_size, (B, C, config.num_author_hashes))
    candidate_surface = torch.randint(0, product_surface_vocab_size, (B, C))

    # --- Step 2: Look up embeddings from hash tables ---
    user_embs = user_hash_emb(user_hashes)                          # [B, 2, D]
    hist_post_embs = post_hash_emb(history_post_hashes)             # [B, S, 2, D]
    hist_author_embs = author_hash_emb(history_author_hashes)       # [B, S, 2, D]
    cand_post_embs = post_hash_emb(candidate_post_hashes)           # [B, C, 2, D]
    cand_author_embs = author_hash_emb(candidate_author_hashes)     # [B, C, 2, D]

    hist_action_embs = action_proj(history_actions)                  # [B, S, D]
    hist_surface_embs = surface_emb_table(history_surface)           # [B, S, D]
    cand_surface_embs = surface_emb_table(candidate_surface)         # [B, C, D]

    # --- Step 3: Reduce ---
    user_token, user_mask = user_reduce(user_hashes, user_embs)
    # user_token: [B, 1, D], user_mask: [B, 1]

    hist_tokens, hist_mask = history_reduce(
        history_post_hashes, hist_post_embs, hist_author_embs,
        hist_action_embs, hist_surface_embs,
    )
    # hist_tokens: [B, S, D], hist_mask: [B, S]

    cand_tokens, cand_mask = candidate_reduce(
        candidate_post_hashes, cand_post_embs, cand_author_embs,
        cand_surface_embs,
    )
    # cand_tokens: [B, C, D], cand_mask: [B, C]

    # --- Step 4: Concatenate into transformer input sequence ---
    # [user_token, history_tokens, candidate_tokens]
    # Shape: [B, 1 + S + C, D]
    sequence = torch.cat([user_token, hist_tokens, cand_tokens], dim=1)
    mask = torch.cat([user_mask, hist_mask, cand_mask], dim=1)

    # --- Report shapes ---
    print("=== Hash Embedding Pipeline - Shape Report ===")
    print()
    print(f"Configuration:")
    print(f"  B={B}, S={S}, C={C}, D={D}")
    print(f"  num_user_hashes={config.num_user_hashes}")
    print(f"  num_item_hashes={config.num_item_hashes}")
    print(f"  num_author_hashes={config.num_author_hashes}")
    print()
    print(f"Raw hashes (RecsysBatch-like):")
    print(f"  user_hashes:           {list(user_hashes.shape)}")
    print(f"  history_post_hashes:   {list(history_post_hashes.shape)}")
    print(f"  history_author_hashes: {list(history_author_hashes.shape)}")
    print(f"  candidate_post_hashes: {list(candidate_post_hashes.shape)}")
    print()
    print(f"Looked-up embeddings (RecsysEmbeddings-like):")
    print(f"  user_embs:        {list(user_embs.shape)}")
    print(f"  hist_post_embs:   {list(hist_post_embs.shape)}")
    print(f"  hist_author_embs: {list(hist_author_embs.shape)}")
    print(f"  cand_post_embs:   {list(cand_post_embs.shape)}")
    print()
    print(f"After reduce (ready for transformer):")
    print(f"  user_token:  {list(user_token.shape)}  mask: {list(user_mask.shape)}")
    print(f"  hist_tokens: {list(hist_tokens.shape)}  mask: {list(hist_mask.shape)}")
    print(f"  cand_tokens: {list(cand_tokens.shape)}  mask: {list(cand_mask.shape)}")
    print()
    print(f"Final concatenated sequence:")
    print(f"  sequence: {list(sequence.shape)}  (= [B, 1+S+C, D])")
    print(f"  mask:     {list(mask.shape)}")
    print()
    print(f"Padding mask sample (batch 0): {mask[0].int().tolist()}")
    print(f"  (1 = real token, 0 = padding)")
    print()

    # --- Count parameters ---
    total_params = 0
    for name, module in [
        ("user_hash_emb", user_hash_emb),
        ("post_hash_emb", post_hash_emb),
        ("author_hash_emb", author_hash_emb),
        ("action_proj", action_proj),
        ("surface_emb_table", surface_emb_table),
        ("user_reduce", user_reduce),
        ("history_reduce", history_reduce),
        ("candidate_reduce", candidate_reduce),
    ]:
        n = sum(p.numel() for p in module.parameters())
        total_params += n
        print(f"  {name:25s}  {n:>12,} params")

    print(f"  {'TOTAL':25s}  {total_params:>12,} params")
    print()
    print("Done. All shapes match the JAX codebase.")


if __name__ == "__main__":
    main()
```

### Expected output

When you run the above, you should see something like:

```
=== Hash Embedding Pipeline - Shape Report ===

Configuration:
  B=4, S=8, C=4, D=64
  num_user_hashes=2
  num_item_hashes=2
  num_author_hashes=2

Raw hashes (RecsysBatch-like):
  user_hashes:           [4, 2]
  history_post_hashes:   [4, 8, 2]
  history_author_hashes: [4, 8, 2]
  candidate_post_hashes: [4, 4, 2]

Looked-up embeddings (RecsysEmbeddings-like):
  user_embs:        [4, 2, 64]
  hist_post_embs:   [4, 8, 2, 64]
  hist_author_embs: [4, 8, 2, 64]
  cand_post_embs:   [4, 4, 2, 64]

After reduce (ready for transformer):
  user_token:  [4, 1, 64]  mask: [4, 1]
  hist_tokens: [4, 8, 64]  mask: [4, 8]
  cand_tokens: [4, 4, 64]  mask: [4, 4]

Final concatenated sequence:
  sequence: [4, 13, 64]  (= [B, 1+S+C, D])
  mask:     [4, 13]
```

---

## Summary: The Big Picture

Here is the entire flow from raw entity IDs to transformer input, in one diagram:

```
  RAW ENTITY IDS                    HASH FUNCTIONS              EMBEDDING TABLES
  ================                  ==============              ================
  user_id = 12345                   h1(12345) = 42    --------> table_1[42]   = vec_a  (D=64)
                                    h2(12345) = 8891  --------> table_2[8891] = vec_b  (D=64)
                                                                     |
                                                                     v
                                                            [vec_a, vec_b]  (shape: [2, 64])
                                                                     |
                                                                     | flatten
                                                                     v
                                                            [vec_a ; vec_b] (shape: [1, 128])
                                                                     |
                                                                     | Linear(128, 64)
                                                                     v
                                                            user_token      (shape: [1, 64])


  Same for every post and author in history and candidates.
  Then concatenate:

  [user_token | history_token_1 | ... | history_token_S | cand_token_1 | ... | cand_token_C]
        1     +         S              +              C           = 1 + S + C tokens

  Feed this sequence into the transformer. Done.
```

Key takeaways:

1. **Hash embeddings trade memory for accuracy** -- billions of entities mapped into much
   smaller tables via hash functions.
2. **Multiple hashes per entity** make collisions nearly impossible and give the model
   multiple "views" of each entity.
3. **Flatten + Linear** is strictly more expressive than averaging -- the model learns
   which hash tables are more informative.
4. **RecsysBatch vs. RecsysEmbeddings** separates the cheap request data from the
   expensive embedding lookups, enabling distributed parameter serving.
5. **Hash value 0 is sacred** -- it is the sentinel for padding, and everything flows
   from that convention.

---

*Next lecture: Lecture 03 -- The Transformer Backbone and Candidate Ranking.*
