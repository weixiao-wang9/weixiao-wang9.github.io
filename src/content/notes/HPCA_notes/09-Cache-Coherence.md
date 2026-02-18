---
type: source
course: "[[HPCA (High Performance Computer Architecture)]]"
lessons: LessonCacheCoherence
prerequisites: "[[08-Advanced-Caches]]"
---

# Cache Coherence

> **Prerequisites**: [[08-Advanced-Caches]]
> **Learning Goals**: Understand the cache coherence problem in multicore processors, the two fundamental approaches (update vs. invalidate), major coherence protocols (MSI, MOSI, MOESI), and directory-based coherence for scalable systems.

---

## The Cache Coherence Problem

In a multicore system, each core has its own private cache. Multiple cores can cache the same memory location, causing their views to diverge.

**Incoherent**: Each cache copy behaves as an independent copy instead of as the same shared memory location.

**Programmer expectation**: shared memory — any write by one core should be visible to all cores.

---

## Coherence Requirements

Three requirements for a coherent memory system:

1. A core reading a memory location receives the value written by the **last valid write**
2. If core A writes and then core B reads the **same location**, B should see A's value
3. All cores **agree on the order** of writes to any given memory location

---

## Approaches to Coherence (Non-Solutions First)

| Approach | Problem |
|----------|---------|
| No caches | Correct, but terrible performance |
| All cores share one L1 cache | Correct, but terrible performance |
| Private write-through caches (no protocol) | **Not coherent** |

---

## Maintaining Coherence Property 2: Update vs. Invalidate

| Strategy | Mechanism | When Better |
|----------|-----------|-------------|
| **Write-Update** | On a write, broadcast the new value to all caches holding that block | One core writes, many cores read frequently |
| **Write-Invalidate** | On a write, invalidate all other copies | Burst writes to one address; writes to different words in same block; thread migration |

> **All modern processors use write-invalidate** — it handles thread migration better.

---

## Maintaining Coherence Property 3: Ordering

| Mechanism | Description |
|-----------|-------------|
| **Snooping** | All writes go on a shared bus; all cores monitor (snoop) the bus |
| **Directory** | Each block has a directory entry tracking its state; no broadcast needed |

---

## Write-Update Optimization: Dirty Bit

Problem: Broadcasting all writes to memory creates a bandwidth bottleneck.

**Solution**: Delay writes to memory using a **dirty bit** per cache block.
- When a core writes → broadcast to update other caches, set dirty bit
- Write to main memory deferred until the dirty block is **evicted**

**Benefits**:
- Greatly reduces writes to memory
- Greatly reduces reads from memory (dirty copy is served from cache)

---

## Write-Invalidate Snooping

- A write causes all other copies to be **invalidated**
- The writing cache becomes the **only valid copy**
- Other cores that read next will **miss** and request the data
- A **shared bit** indicates whether multiple caches have clean copies

**Disadvantage**: Every reader gets a miss when a core writes.
**Advantage**: Multiple consecutive writes to the same block are fast — no need to broadcast after the first write (no other valid copies).

---

## MSI Protocol

An invalidation-based snooping protocol with 3 states:

| State | Meaning |
|-------|---------|
| **I** (Invalid) | This cache does not have a valid copy |
| **S** (Shared) | This cache has a clean (read-only) copy; other caches may also |
| **M** (Modified) | This cache has the only valid (dirty) copy |

### MSI State Transitions (Summary)

| Current State | Event | Action | Next State |
|--------------|-------|--------|------------|
| I | Local read | Put Read on bus; get data | S |
| I | Local write | Put Write+Invalidate on bus; get data | M |
| S | Local read | — | S |
| S | Local write | Put Invalidation on bus | M |
| S | Snoop write on bus | Invalidate | I |
| M | Local read/write | — | M |
| M | Snoop read on bus | Write back; supply data | S |
| M | Snoop write on bus | Write back; supply data | I |

---

## Cache-to-Cache Transfers

When cache C1 has a block in the **M** state and C2 requests it:

| Method | Description | Cost |
|--------|-------------|------|
| **Abort and Retry** | C1 aborts C2's request; C2 retries after writeback | 2× memory latency |
| **Intervention** | C1 tells memory it will respond directly to C2 | 1× memory latency (better) |

> **Modern processors use Intervention.**

**Intervention** requires an extra signal on the bus; hardware is more complex but faster.

---

## MOSI Protocol

**Problem with MSI**: Going from Shared → Modified requires passing through Invalid (wasteful).

**O (Owner) state**: A core modified the data and shared it — it is responsible for:
1. Responding to read requests from other cores
2. Writing back to memory when the block is evicted

| State | Meaning |
|-------|---------|
| **M** | Core has modified; only valid copy |
| **O** | Core has modified; has shared with ≥1 other core (owner responsible for writeback) |
| **S** | ≥1 core has clean copy |
| **I** | Invalid |

---

## MOESI Protocol

**Problem with MOSI**: Going from S → M still requires passing through I.

**E (Exclusive) state**: Core is the **only** core with a clean copy.

When a block is in the **E** state:
- No other core has a copy
- This core can move directly to **M** on a write (no bus transaction needed!)

| State | Meaning |
|-------|---------|
| **M** | Modified; only valid copy |
| **O** | Modified; shared; owner does writeback |
| **E** | Exclusive clean copy; can write silently |
| **S** | Shared clean copy |
| **I** | Invalid |

---

## Directory-Based Coherence

**Snooping limitation**: Requires a **shared bus** — only scales to ~16 processors.

**Directory approach**: Each memory block has a directory entry; no broadcast needed.

### Directory Structure

- **Distributed** across all cores — each core has a slice of the directory
- Each slice manages a set of memory blocks

### Directory Entry Fields

| Field | Meaning |
|-------|---------|
| **Dirty bit** | Is any cache's copy dirty? |
| **Present bits** (1 per cache) | Is this block in a valid state in each cache? |

For an 8-core system: 8 present bits per directory entry.

**Communication**: After a request, the directory sends a command to the relevant caches; caches send an **acknowledgement** back.

---

## Cache Misses with Coherence: 4 Cs

The classic "3 Cs" become 4:

| Miss Type | Cause |
|-----------|-------|
| **Compulsory** | First access to a block |
| **Capacity** | Cache too small |
| **Conflict** | Limited associativity |
| **Coherence** | Another core invalidated/updated the block |

### Two Types of Coherence Misses

| Type | Description |
|------|-------------|
| **True Sharing** | Different cores genuinely access the **same data** (expected coherence cost) |
| **False Sharing** | Different cores access **different data** that happens to be in the **same cache block** — coherence treats them as the same |

> **False sharing** can be reduced by padding data structures so independently-accessed fields land in different cache blocks.

---

## Summary

**Key Takeaways**:
- Coherence = every core sees a consistent view of memory
- Write-invalidate dominates modern processors (better for thread migration, burst writes)
- Snooping: simple, scales to ~16 cores; requires shared bus
- Directory: scales to many cores; no bus needed; more latency per operation
- MSI → add O state for owner writeback avoidance (MOSI) → add E state for silent upgrades (MOESI)
- Coherence misses: true sharing (unavoidable) and false sharing (avoidable by padding)

**Common Exam Topics**:
- Draw state transitions for MSI/MOESI given a sequence of read/write operations from multiple cores
- Identify true vs. false sharing scenarios
- Compare snooping vs. directory for scalability

**See Also**: [[08-Advanced-Caches]]
