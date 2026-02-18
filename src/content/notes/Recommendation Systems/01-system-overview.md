---
type: note
course: "[[Recommendation Systems]]"
date: 2026-02-18
---

# Lecture 1: System Overview

> **Series**: X Recommendation Algorithm Deep Dive
> **Audience**: MLE Intern
> **Prerequisites**: Basic ML knowledge, Python

---

## What Does the "For You" Feed Do?

When you open X (formerly Twitter) and tap the "For You" tab, the system needs to answer one question:

**"Out of hundreds of millions of posts, which ~50 should this user see right now?"**

This happens in under 2 seconds, for hundreds of millions of users, constantly. The algorithm we're studying is the engine behind this.

---

## The 4 Components

The codebase has 4 major components. Think of them as a restaurant:

```
x-algorithm/
+-- candidate-pipeline/   # The recipe framework (reusable patterns)
+-- home-mixer/           # The head chef (orchestrates everything)
+-- thunder/              # The pantry (stores fresh ingredients)
+-- phoenix/              # The secret sauce (ML models) <-- YOUR FOCUS
```

| Component | Language | What It Does | Restaurant Analogy |
|-----------|----------|--------------|-------------------|
| **candidate-pipeline** | Rust | Reusable pipeline framework with traits (Source, Filter, Scorer, etc.) | The recipe book |
| **home-mixer** | Rust | Orchestration layer: assembles the feed using the pipeline framework | The head chef |
| **thunder** | Rust | In-memory post store for real-time posts from followed accounts | The fresh ingredients pantry |
| **phoenix** | Python/JAX | ML models for retrieval and ranking | The secret sauce |

### What You (MLE Intern) Care About

```
candidate-pipeline/  --> Skip (Rust infrastructure)
home-mixer/          --> Understand conceptually (Lecture 6)
thunder/             --> Understand conceptually (Lecture 6)
phoenix/             --> READ EVERY LINE (Lectures 2-5)
```

---

## The End-to-End Flow

Here's what happens when you open the "For You" tab:

```
User opens "For You"
        |
        v
+------ HOME MIXER (gRPC Server) ------+
|                                       |
|  1. QUERY HYDRATION                   |
|     +-- Fetch user engagement history |
|     +-- Fetch following list          |
|                                       |
|  2. CANDIDATE SOURCING (parallel)     |
|     +-- Thunder: recent posts from    |
|     |   accounts you follow           |
|     +-- Phoenix Retrieval: posts from |
|         accounts you DON'T follow     |
|         (ML two-tower model)          |
|                                       |
|  3. CANDIDATE HYDRATION (parallel)    |
|     +-- Fetch post text, media        |
|     +-- Fetch author info             |
|     +-- Fetch video duration          |
|     +-- Check subscription status     |
|                                       |
|  4. PRE-SCORING FILTERS (sequential)  |
|     +-- Remove duplicates             |
|     +-- Remove old posts              |
|     +-- Remove your own posts         |
|     +-- Remove blocked/muted authors  |
|     +-- Remove already-seen posts     |
|     +-- Remove muted keywords         |
|     +-- ...6 more filters             |
|                                       |
|  5. SCORING (sequential)              |
|     +-- Phoenix Scorer: ML predicts   |
|     |   19 engagement probabilities   |
|     +-- Weighted Scorer: combine      |
|     |   19 probs into 1 score         |
|     +-- Author Diversity: penalize    |
|     |   repeated authors              |
|     +-- OON Scorer: boost in-network  |
|                                       |
|  6. SELECTION                         |
|     +-- Sort by score, pick top K     |
|                                       |
|  7. POST-SELECTION FILTERS            |
|     +-- Visibility filtering          |
|     +-- Conversation dedup            |
|                                       |
|  8. SIDE EFFECTS (async)              |
|     +-- Cache results                 |
|                                       |
+---------------------------------------+
        |
        v
   Ranked Feed Response
```

---

## The ML Pipeline (Your Focus)

Within Phoenix, there are two ML stages:

### Stage 1: Retrieval (Two-Tower Model)

**Problem**: Hundreds of millions of posts exist. Can't score them all.
**Solution**: Use a cheap model to narrow down to ~1000 candidates.

```
User Tower (Transformer)          Candidate Tower (MLP)
        |                                 |
        v                                 v
  [user embedding]              [post embeddings] (pre-computed)
        |                                 |
        +---> dot product similarity <----+
                     |
                     v
              Top 1000 posts
```

### Stage 2: Ranking (Transformer with Candidate Isolation)

**Problem**: Need to carefully score ~1000 candidates.
**Solution**: Use a powerful transformer that predicts 19 engagement types.

```
[User + History + Candidates] --> Transformer --> 19 probabilities per candidate
                                                    |
                                                    v
                                              P(like) = 0.85
                                              P(reply) = 0.12
                                              P(repost) = 0.34
                                              P(block) = 0.001
                                              ...15 more actions
```

**Key Design Insight**: The model predicts 19 different actions (like, reply, repost, click, block, mute, report, etc.) rather than a single "relevance" score. These are combined with configurable weights downstream. This means X can tune the feed behavior (e.g., prioritize replies over likes) **without retraining the model**.

---

## The Phoenix Directory (10 Python Files)

```
phoenix/
+-- grok.py                        # Transformer architecture (from xAI's Grok-1)
+-- recsys_model.py                # Ranking model + data structures
+-- recsys_retrieval_model.py      # Two-tower retrieval model
+-- runners.py                     # Training/inference runners
+-- run_ranker.py                  # Demo: run ranking inference
+-- run_retrieval.py               # Demo: run retrieval inference
+-- test_recsys_model.py           # Tests for ranking
+-- test_recsys_retrieval_model.py # Tests for retrieval
+-- pyproject.toml                 # Python dependencies
+-- README.md                      # Documentation
```

**Reading order for this lecture series:**
1. `recsys_model.py` (data structures, hash embeddings) --> Lecture 2
2. `recsys_retrieval_model.py` (retrieval) --> Lecture 3
3. `recsys_model.py` (ranking model) --> Lecture 4
4. `grok.py` (transformer internals) --> Lecture 5
5. `runners.py` (inference pipeline) --> Lecture 6

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **Phoenix** | Python + JAX + Haiku | ML models, TPU training |
| **Home Mixer** | Rust + Tonic (gRPC) | High-throughput serving |
| **Thunder** | Rust + Kafka | Sub-ms in-memory lookups |
| **Candidate Pipeline** | Rust | Composable, type-safe framework |
| **Communication** | gRPC + Protobuf | Cross-service RPC |

### JAX vs PyTorch (For Your Reference)

This codebase uses JAX/Haiku. You're used to PyTorch. The core difference:

```
PyTorch: model = MyModel()     # model carries weights inside
         output = model(x)     # call it like an object

JAX:     forward_fn = hk.transform(forward)    # wrap pure function
         params = forward_fn.init(rng, x)       # extract weights
         output = forward_fn.apply(params, x)   # pass weights explicitly
```

The math is identical. Throughout this series, we'll provide PyTorch translations for every piece of JAX code.

---

## The 19 Predicted Actions

The ML model predicts probability of 19 user engagement types:

```
# From runners.py:202-222

POSITIVE ENGAGEMENT:
  1. favorite_score         # Will they like it?
  2. reply_score            # Will they reply?
  3. repost_score           # Will they repost?
  4. photo_expand_score     # Will they expand the photo?
  5. click_score            # Will they click the link?
  6. profile_click_score    # Will they visit the author's profile?
  7. vqv_score              # Will they watch the video (quality view)?
  8. share_score            # Will they share it?
  9. share_via_dm_score     # Will they share via DM?
  10. share_via_copy_link_score  # Will they copy the link?
  11. dwell_score           # Will they spend time reading?
  12. quote_score           # Will they quote-tweet?
  13. quoted_click_score    # Will they click a quoted tweet?
  14. follow_author_score   # Will they follow the author?

NEGATIVE ENGAGEMENT:
  15. not_interested_score  # Will they mark "not interested"?
  16. block_author_score    # Will they block the author?
  17. mute_author_score     # Will they mute the author?
  18. report_score          # Will they report the post?

CONTINUOUS:
  19. dwell_time            # How long will they spend on it?
```

These are combined into a single score by the Weighted Scorer (Lecture 6).

---

## Quick Check

Before moving to Lecture 2, make sure you can answer:

1. What are the two ML stages in the recommendation pipeline?
2. Why does the model predict 19 actions instead of 1 relevance score?
3. Which directory should you focus on as an MLE intern?
4. What's the difference between Thunder (in-network) and Phoenix Retrieval (out-of-network)?

---

**Next: [Lecture 2 - Hash Embeddings](02-hash-embeddings.md)**
