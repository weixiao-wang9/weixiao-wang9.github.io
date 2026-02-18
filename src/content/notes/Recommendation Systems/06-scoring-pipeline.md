---
type: note
course: "[[Recommendation Systems]]"
date: 2026-02-18
---

# Lecture 6: Scoring Pipeline

> **Series**: X Recommendation Algorithm Deep Dive
> **Audience**: MLE Intern
> **Prerequisites**: Lectures 1-5

---

## Overview

So far we've studied the ML models in isolation:
- **Lecture 3**: Two-tower retrieval (narrow 100M posts → 1000 candidates)
- **Lecture 4**: Ranking transformer (score 1000 candidates with 19 action probabilities)

Now we connect the pieces. This lecture covers the **full inference pipeline** that executes when a user opens their "For You" feed.

**Key insight**: The ML models are just one component. The real system is a pipeline of sources, filters, scorers, and selectors orchestrated by Rust code.

---

## The Pipeline Framework

X uses a composable pipeline framework (`candidate-pipeline/`) with 5 trait types:

```rust
// Simplified Rust pseudocode

trait Source {
    fn get_candidates(&self, ctx: &Context) -> Vec<Candidate>;
}

trait Hydrator {
    fn enrich(&self, candidates: Vec<Candidate>) -> Vec<Candidate>;
}

trait Filter {
    fn filter(&self, candidates: Vec<Candidate>) -> Vec<Candidate>;
}

trait Scorer {
    fn score(&self, candidates: Vec<Candidate>) -> Vec<ScoredCandidate>;
}

trait Selector {
    fn select(&self, candidates: Vec<ScoredCandidate>) -> Vec<ScoredCandidate>;
}
```

Think of it as LEGO blocks:
- **Sources**: Generate raw candidates (where do posts come from?)
- **Hydrators**: Add metadata (fetch text, author info, media)
- **Filters**: Remove bad candidates (dedup, blocked authors, etc.)
- **Scorers**: Rank what remains (ML models + heuristics)
- **Selectors**: Pick top-K for the final feed

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          HOME MIXER (Rust gRPC Server)                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  REQUEST: user_id = 12345, request_time = 2024-01-15T10:30:00Z │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              v                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    QUERY HYDRATION                              │   │
│  │  + Fetch user engagement history (last 1000 interactions)      │   │
│  │  + Fetch following list (who does user follow?)                │   │
│  │  + Fetch blocked/muted authors                                 │   │
│  │  + Fetch user preferences (language, timezone, etc.)           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              v                                          │
│  ╔═══════════════════════════════════════════════════════════════════╗  │
│  ║                      CANDIDATE SOURCING (parallel)               ║  │
│  ╠═══════════════════════════════════════════════════════════════════╣  │
│  ║                                                                   ║  │
│  ║  ┌──────────────────┐              ┌──────────────────────┐      ║  │
│  ║  │   THUNDER SOURCE │              │ PHOENIX RETRIEVAL    │      ║  │
│  ║  │                  │              │     SOURCE           │      ║  │
│  ║  │  "In-network"    │              │  "Out-of-network"    │      ║  │
│  ║  │  posts from      │              │  posts from accounts │      ║  │
│  ║  │  followed users  │              │  you DON'T follow    │      ║  │
│  ║  │                  │              │                      │      ║  │
│  ║  │  ~500 posts      │              │  ~500 posts          │      ║  │
│  ║  │  (very fresh)    │              │  (ML similarity)     │      ║  │
│  ║  └──────────────────┘              └──────────────────────┘      ║  │
│  ║                                                                   ║  │
│  ╚═══════════════════════════════════════════════════════════════════╝  │
│                              │                                          │
│                              v                                          │
│                    1000 raw candidates                                 │
│                              │                                          │
│                              v                                          │
│  ╔═══════════════════════════════════════════════════════════════════╗  │
│  ║                  CANDIDATE HYDRATION (parallel)                  ║  │
│  ╠═══════════════════════════════════════════════════════════════════╣  │
│  ║                                                                   ║  │
│  ║  For each candidate, fetch:                                       ║  │
│  ║  + Post text, media URLs, video duration                         ║  │
│  ║  + Author info (name, verified status, follower count)          ║  │
│  ║  + Engagement counts (likes, replies, reposts)                  ║  │
│  ║  + Subscription status (is this premium content?)               ║  │
│  ║  + Tweet type (original, reply, quote, retweet)                 ║  │
│  ║                                                                   ║  │
│  ╚═══════════════════════════════════════════════════════════════════╝  │
│                              │                                          │
│                              v                                          │
│  ╔═══════════════════════════════════════════════════════════════════╗  │
│  ║              PRE-SCORING FILTERS (sequential, 10 total)          ║  │
│  ╠═══════════════════════════════════════════════════════════════════╣  │
│  ║                                                                   ║  │
│  ║  1. Remove duplicates (same tweet_id)                           ║  │
│  ║  2. Remove tweets older than X days                             ║  │
│  ║  3. Remove user's own tweets                                    ║  │
│  ║  4. Remove from blocked authors                                ║  │
│  ║  5. Remove from muted authors                                  ║  │
│  ║  6. Remove already-seen tweets (in session)                     ║  │
│  ║  7. Remove muted keywords                                       ║  │
│  ║  8. Remove sensitive content (if user setting)                  ║  │
│  ║  9. Remove NSFW content (if user setting)                       ║  │
│  ║  10. Cap tweets per author (max 2-3 per author)                 ║  │
│  ║                                                                   ║  │
│  ║                    ~500 candidates remaining                     ║  │
│  ║                                                                   ║  │
│  ╚═══════════════════════════════════════════════════════════════════╝  │
│                              │                                          │
│                              v                                          │
│  ╔═══════════════════════════════════════════════════════════════════╗  │
│  ║                    SCORING (sequential, 4 scorers)               ║  │
│  ╠═══════════════════════════════════════════════════════════════════╣  │
│  ║                                                                   ║  │
│  ║  ┌────────────────────────────────────────────────────────┐     ║  │
│  ║  │  1. PHOENIX SCORER (ML Model)                          │     ║  │
│  ║  │                                                         │     ║  │
│  ║  │  Input: user + history + candidates (batch)            │     ║  │
│  ║  │  Output: 19 probabilities per candidate                │     ║  │
│  ║  │                                                         │     ║  │
│  ║  │  Candidate 1: P(like)=0.85, P(reply)=0.12, ...          │     ║  │
│  ║  │  Candidate 2: P(like)=0.23, P(reply)=0.03, ...          │     ║  │
│  ║  │  ...                                                     │     ║  │
│  ║  │                                                         │     ║  │
│  ║  │  (See Lecture 4 for model details)                     │     ║  │
│  ║  └────────────────────────────────────────────────────────┘     ║  │
│  ║                              │                                  ║  │
│  ║                              v                                  ║  │
│  ║  ┌────────────────────────────────────────────────────────┐     ║  │
│  ║  │  2. WEIGHTED SCORER (Combine 19 probs → 1 score)       │     ║  │
│  ║  │                                                         │     ║  │
│  ║  │  score = w1*P(like) + w2*P(reply) + w3*P(repost) + ... │     ║  │
│  ║  │                                                         │     ║  │
│  ║  │  Weights are configurable without retraining!          │     ║  │
│  ║  │                                                         │     ║  │
│  ║  │  Example weights:                                       ║  │
│  ║  │    like: 1.0      reply: 3.0      repost: 1.5         ║  │
│  ║  │    click: 0.5     follow: 5.0    block: -10.0         ║  │
│  ║  └────────────────────────────────────────────────────────┘     ║  │
│  ║                              │                                  ║  │
│  ║                              v                                  ║  │
│  ║  ┌────────────────────────────────────────────────────────┐     ║  │
│  ║  │  3. AUTHOR DIVERSITY SCORER (Penalize repetition)      │     ║  │
│  ║  │                                                         │     ║  │
│  ║  │  For each author:                                      │     ║  │
│  ║  │    1st tweet: score *= 1.0  (no penalty)               │     ║  │
│  ║  │    2nd tweet: score *= 0.7  (30% penalty)              │     ║  │
│  ║  │    3rd tweet: score *= 0.4  (60% penalty)              │     ║  │
│  ║  │                                                         │     ║  │
│  ║  │  Goal: Prevent feed domination by single author        │     ║  │
│  ║  └────────────────────────────────────────────────────────┘     ║  │
│  ║                              │                                  ║  │
│  ║                              v                                  ║  │
│  ║  ┌────────────────────────────────────────────────────────┐     ║  │
│  ║  │  4. OON SCORER (Boost "Out-of-Network" content)        │     ║  │
│  ║  │                                                         │     ║  │
│  ║  │  If author NOT in user's following list:               │     ║  │
│  ║  │    score *= 1.1  (10% boost)                           │     ║  │
│  ║  │                                                         │     ║  │
│  ║  │  Goal: Help users discover new accounts                │     ║  │
│  ║  └────────────────────────────────────────────────────────┘     ║  │
│  ║                                                                   ║  │
│  ╚═══════════════════════════════════════════════════════════════════╝  │
│                              │                                          │
│                              v                                          │
│  ╔═══════════════════════════════════════════════════════════════════╗  │
│  ║                    SELECTION (Sort + Top-K)                      ║  │
│  ╠═══════════════════════════════════════════════════════════════════╣  │
│  ║                                                                   ║  │
│  ║  1. Sort all candidates by final score (descending)              ║  │
│  ║  2. Select top K (e.g., 50-100 tweets)                           ║  │
│  ║  3. Apply final insertion constraint (e.g., promote ads)          ║  │
│  ║                                                                   ║  │
│  ╚═══════════════════════════════════════════════════════════════════╝  │
│                              │                                          │
│                              v                                          │
│  ╔═══════════════════════════════════════════════════════════════════╗  │
│  ║              POST-SELECTION FILTERS (2 filters)                  ║  │
│  ╠═══════════════════════════════════════════════════════════════════╣  │
│  ║                                                                   ║  │
│  ║  1. Visibility filtering (remove based on country restrictions)   ║  │
│  ║  2. Conversation dedup (if multiple from same thread, pick one)    ║  │
│  ║                                                                   ║  │
│  ╚═══════════════════════════════════════════════════════════════════╝  │
│                              │                                          │
│                              v                                          │
│                    Final ~50 ranked tweets                             │
│                              │                                          │
│                              v                                          │
│  ╔═══════════════════════════════════════════════════════════════════╗  │
│  ║              SIDE EFFECTS (async, non-blocking)                  ║  │
│  ╠═══════════════════════════════════════════════════════════════════╣  │
│  ║                                                                   ║  │
│  ║  + Cache results (for quick retry)                                ║  │
│  ║  + Log impressions (for training data)                            ║  │
│  ║  + Update metrics (latency, hit rates)                            ║  │
│  ║                                                                   ║  │
│  ╚═══════════════════════════════════════════════════════════════════╝  │
│                              │                                          │
│                              v                                          │
│                      RETURN TO USER                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Thunder: The In-Network Source

Thunder is X's in-memory post store optimized for real-time delivery.

### What It Stores

```
thunder/posts/post_store.rs

Post {
    tweet_id: u64,
    author_id: u64,
    text: String,
    created_at: DateTime<Utc>,
    media_urls: Vec<String>,
    language: String,
    ... (metadata)
}
```

### How It Works

```
When you tweet:
        |
        v
[Your tweet] --> Kafka Topic --> Thunder (in-memory store)
                                        |
                                        v
                                Available for retrieval
                                by your followers
```

**Key characteristics:**
- **Sub-millisecond latency**: Everything is in memory (no disk I/O)
- **Time-windowed**: Only stores recent posts (e.g., last 7 days)
- **Follower-indexed**: Can quickly answer "what's new from people I follow?"
- **Rust + Kafka**: High throughput, low garbage collection pauses

### Why Thunder + Phoenix Retrieval?

```
┌─────────────────────────────────────────────────────────────┐
│                     CANDIDATE SOURCES                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  THUNDER (In-Network)        PHOENIX RETRIEVAL (Out-of-Network)  │
│  ┌─────────────────┐         ┌──────────────────────┐        │
│  │ "Who you follow"│         │  "Who you DON'T      │        │
│  │                 │         │   follow"            │        │
│  │                 │         │                      │        │
│  │ Fresh content   │         │  Discovery content  │        │
│  │ from known      │         │  from unknown        │        │
│  │ sources         │         │  sources             │        │
│  │                 │         │                      │        │
│  │ ~500 posts      │         │  ~500 posts          │        │
│  └─────────────────┘         └──────────────────────┘        │
│                                                             │
│  Together: ~1000 candidates for ranking                    │
└─────────────────────────────────────────────────────────────┘
```

**Design insight**: Thunder ensures you never miss recent posts from people you follow. Phoenix Retrieval helps you discover new accounts you'll like.

---

## The 4 Scorers (Deep Dive)

### Scorer 1: Phoenix Scorer (ML Model)

This is the ranking transformer from Lecture 4.

```python
# Pseudocode (see Lecture 4 for full implementation)

def phoenix_scorer(user, history, candidates):
    # Build input sequence
    inputs = [user_embedding] + history_embeddings + candidate_embeddings

    # Run transformer
    outputs = transformer(inputs, candidate_start_offset=1 + len(history))

    # Extract candidate outputs and apply unembedding
    candidate_outputs = outputs[candidate_start_offset:]
    logits = candidate_outputs @ unembedding_matrix  # [num_candidates, 19]

    # Convert to probabilities
    probs = sigmoid(logits)

    return probs  # Shape: [num_candidates, 19]
```

**Output**: 19 probabilities per candidate (like, reply, repost, block, etc.)

---

### Scorer 2: Weighted Scorer

Combines 19 probabilities into a single score using configurable weights.

```python
# From home-mixer/scorers/weighted_scorer.rs

WEIGHTS = {
    "favorite": 1.0,
    "reply": 3.0,
    "repost": 1.5,
    "photo_expand": 0.8,
    "click": 0.5,
    "profile_click": 2.0,
    "vqv": 1.2,
    "share": 1.0,
    "share_via_dm": 0.8,
    "share_via_copy_link": 0.5,
    "dwell": 0.3,
    "quote": 1.5,
    "quoted_click": 0.7,
    "follow_author": 5.0,
    "not_interested": -5.0,
    "block_author": -20.0,
    "mute_author": -10.0,
    "report": -15.0,
    "dwell_time": 0.01,  # Continuous
}

def weighted_scorer(probs):
    """
    probs: [num_candidates, 19] array of probabilities

    Returns: [num_candidates] array of scores
    """
    score = 0.0
    for i, (action, weight) in enumerate(WEIGHTS.items()):
        score += weight * probs[:, i]

    return score
```

**Key insight**: X can change feed behavior by adjusting weights **without retraining the model**.

Example:
```python
# Want to prioritize conversation? Increase reply weight.
WEIGHTS["reply"] = 5.0  # Was 3.0

# Want to reduce sensitive content? Increase report weight.
WEIGHTS["report"] = -30.0  # Was -15.0

# Want to promote long-term engagement? Increase follow weight.
WEIGHTS["follow_author"] = 10.0  # Was 5.0
```

---

### Scorer 3: Author Diversity Scorer

Prevents a single author from dominating the feed.

```python
# From home-mixer/scorers/diversity_scorer.rs

def author_diversity_scorer(candidates, scores):
    """
    candidates: list of Candidate with author_id field
    scores: list of scores from Weighted Scorer

    Returns: adjusted scores with diversity penalty
    """
    author_counts = {}
    adjusted_scores = []

    for candidate, score in zip(candidates, scores):
        author_id = candidate.author_id
        count = author_counts.get(author_id, 0)

        # Penalty schedule
        if count == 0:
            multiplier = 1.0
        elif count == 1:
            multiplier = 0.7
        elif count == 2:
            multiplier = 0.4
        else:
            multiplier = 0.1

        adjusted_scores.append(score * multiplier)
        author_counts[author_id] = count + 1

    return adjusted_scores
```

**Why this matters**: Without diversity, a user who follows Elon Musk might see 20 of his tweets in a row because he scores highly. The diversity scorer ensures variety.

---

### Scorer 4: OON (Out-of-Network) Scorer

Boosts content from accounts the user doesn't follow.

```python
# From home-mixer/scorers/oon_scorer.rs

def oon_scorer(candidates, scores, following_set):
    """
    candidates: list of Candidate with author_id field
    scores: list of scores from Diversity Scorer
    following_set: set of author_ids that user follows

    Returns: adjusted scores with OON boost
    """
    adjusted_scores = []

    for candidate, score in zip(candidates, scores):
        author_id = candidate.author_id

        if author_id not in following_set:
            # 10% boost for out-of-network content
            adjusted_scores.append(score * 1.1)
        else:
            adjusted_scores.append(score)

    return adjusted_scores
```

**Why this matters**: Helps users discover new accounts. Without OON boost, the feed might show too much content from already-followed accounts (Thunder source).

---

## Complete Request Lifecycle (Timeline)

Here's what happens when you tap "For You":

```
T+0ms:     User sends gRPC request to Home Mixer
T+5ms:     Query hydration (fetch user metadata from cache)
T+20ms:    Candidate sourcing (Thunder + Phoenix Retrieval, parallel)
           - Thunder: scan in-memory store for followed accounts (~5ms)
           - Phoenix Retrieval: matmul user_embedding @ corpus_embeddings (~15ms)
T+40ms:    Candidate hydration (fetch metadata for 1000 candidates, parallel)
T+80ms:    Pre-scoring filters (remove ~50% of candidates)
T+150ms:   Phoenix Scorer (batch inference on GPU/TPU)
T+160ms:   Weighted Scorer (simple arithmetic)
T+165ms:   Author Diversity Scorer (count authors, apply penalties)
T+170ms:   OON Scorer (check following set, apply boosts)
T+175ms:   Selection (sort + top-K)
T+180ms:   Post-selection filters (visibility + conversation dedup)
T+185ms:   Return response to user
           Total latency: ~185ms (well under 2 second SLA)
```

---

## Performance Characteristics

| Component | Latency | Throughput | Scalability |
|-----------|---------|------------|-------------|
| **Query Hydration** | ~5ms | 100K QPS | Horizontal (cache sharding) |
| **Thunder** | ~5ms | 200K QPS | Horizontal (Kafka partitions) |
| **Phoenix Retrieval** | ~15ms | 50K QPS | Vertical (larger GPU) or ANN index |
| **Candidate Hydration** | ~40ms | 100K QPS | Horizontal (microservice calls) |
| **Filters** | ~40ms | 100K QPS | Horizontal (stateless) |
| **Phoenix Scorer** | ~70ms | 20K QPS | Vertical (larger GPU) or horizontal (model sharding) |
| **Weighted/Diversity/OON** | ~15ms | 100K QPS | Horizontal (stateless) |

**Bottleneck**: Phoenix Scorer (ML inference). Mitigated by:
- Using smaller batch sizes for inference
- Running on dedicated GPU/TPU pools
- Caching model outputs for repeat requests

---

## Summary: The ML vs. Infrastructure Split

As an MLE intern, you should understand:

```
┌─────────────────────────────────────────────────────────────┐
│                   WHAT YOU OWN (ML)                        │
├─────────────────────────────────────────────────────────────┤
│  ✓ Phoenix Retrieval model architecture                    │
│  ✓ Phoenix Ranking model architecture                      │
│  ✓ Training data pipelines                                 │
│  ✓ Model evaluation and A/B testing                        │
│  ✓ Feature engineering (embeddings, actions)               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                WHAT ENGINEERS OWN (Infra)                  │
├─────────────────────────────────────────────────────────────┤
│  ✓ Thunder in-memory store                                 │
│  ✓ gRPC server (Home Mixer)                                │
│  ✓ Pipeline orchestration (sources, filters, scorers)      │
│  ✓ Caching strategies                                      │
│  ✓ Monitoring and alerting                                 │
└─────────────────────────────────────────────────────────────┘
```

**Your job**: Make the ML models as accurate as possible.
**Their job**: Run your models fast and reliably at scale.

---

## Quick Check

Before moving to Lecture 7, make sure you can answer:

1. What are the 5 trait types in the pipeline framework?
2. What's the difference between Thunder and Phoenix Retrieval as sources?
3. Why predict 19 actions instead of 1 relevance score?
4. What do the 4 scorers do (Phoenix, Weighted, Diversity, OON)?
5. What's the typical end-to-end latency for generating a feed?

---

**Next: [Lecture 7 - Replication Guide](07-replication-guide.md)**
