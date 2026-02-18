---
type: source
course: "[[HPCA (High Performance Computer Architecture)]]"
lessons: LessonAdvancedCaches
prerequisites: "[[01-Introduction-and-Metrics]]"
---

# Advanced Caches

> **Prerequisites**: [[01-Introduction-and-Metrics]]
> **Learning Goals**: Understand the AMAT equation and the three levers to improve it: hit time, miss rate, and miss penalty. Learn specific hardware and compiler techniques for each lever.

---

## The AMAT Framework

$$\text{AMAT} = \text{Hit Time} + \text{Miss Rate} \times \text{Miss Penalty}$$

Three ways to improve AMAT:

| Method | Lever |
|--------|-------|
| Reduce Hit Time | Pipelined caches, VIPT caches, way prediction, fast replacement policy |
| Reduce Miss Rate | Larger cache, higher associativity, larger blocks, prefetching, loop interchange |
| Reduce Miss Penalty | Non-blocking caches, MSHR, cache hierarchies |

---

## Reducing Hit Time

### 1. Pipelined Caches

**Pipelining the cache** = overlapping one cache hit with the next.

$$\text{Hit Time} = \text{Actual Hit} + \text{Wait Time}$$

Three pipeline stages for a cache:
1. Read the index → find the set
2. Determine hit/miss; begin data read
3. Finish data read

### 2. Overlap Cache Hit with TLB Hit (VIPT Cache)

A **PIPT cache** (Physically Indexed, Physically Tagged) uses the physical address → must wait for TLB translation before looking up the cache.

A **VIPT cache** (Virtually Indexed, Physically Tagged) uses the virtual address for indexing but physical address for tag comparison → TLB translation and cache lookup happen **in parallel**.

**VIPT Design**:
- VA page offset bits → used as cache index (no translation needed for index)
- Physical frame number from TLB → used as tag for verification

**Constraint for no aliasing**:
$$\text{Cache Size} \leq \text{Associativity} \times \text{Page Size}$$

> If all index bits come from the page offset portion of the address (bits below the page boundary), there is no aliasing.

**Virtually Addressed (VIVT) Cache** — full virtual address used:
- TLB and cache can work in parallel
- **Problem 1**: Must flush cache on every context switch (virtual addresses are process-specific)
- **Problem 2**: **Aliasing** — two virtual addresses can map to the same physical location, causing incorrect execution

### 3. Way Prediction

The processor **predicts which way** of a set-associative cache will be a hit.
- Correct prediction → fast (like direct-mapped)
- Incorrect prediction → fall back to normal set-associative lookup

**Way Prediction Performance**:

| Cache Type | Way Prediction Benefit |
|-----------|----------------------|
| Fully associative | Yes |
| 8-way set associative | Yes |
| 2-way set associative | Yes |
| Direct mapped | No (already one way) |

### 4. Replacement Policy and Hit Time

| Policy | Hit Time | Miss Rate |
|--------|----------|-----------|
| **Random** | Fast (no bookkeeping) | Worse |
| **LRU** | Slower (update all counters on hit) | Best |
| **NMRU** (Not-Most-Recently-Used) | Fast (only track MRU) | Good |
| **PLRU** (Pseudo-LRU) | Medium | Good (compromise) |

**NMRU**: Only track the most recently used block; replace any other block.

**PLRU**: Set LRU bit to 1 on access; evict block with LRU bit = 0. When all bits are 1, reset all to 0 and set the evicted block to 1.

---

## Reducing Miss Rate

### The 3 Cs of Cache Misses

| Miss Type | Cause | Fix |
|-----------|-------|-----|
| **Compulsory** | First access to a block | Prefetching, larger blocks |
| **Capacity** | Cache is full | Larger cache |
| **Conflict** | Limited associativity | Higher associativity, larger blocks |

- **Larger cache** → reduces capacity misses
- **Higher associativity** → reduces conflict misses (and some capacity misses)

### Larger Cache Blocks

Larger blocks bring in more data per miss (exploits **spatial locality**):
- **Good spatial locality** → miss rate improves
- **Poor spatial locality** → miss rate degrades (wasted bandwidth)
- Larger blocks can reduce compulsory, capacity, and conflict misses

### Prefetching

**Prefetching** = loading a block into cache before it is needed.

- **Good guess** → eliminates a future miss
- **Bad guess** → wastes bandwidth, pollutes cache

**Compiler prefetching**: Insert prefetch instructions; difficult to determine optimal prefetch distance.

**Hardware prefetching**: Hardware guesses what will be needed soon:

| Type | Strategy |
|------|---------|
| **Stream buffer** | Prefetch the next sequential block |
| **Stride prefetch** | Prefetch the block at a fixed distance `d` |
| **Correlating prefetcher** | If block A is fetched, prefetch B (learned correlation) |

### Loop Interchange

**Loop interchange**: swap inner and outer loops to access arrays in **row-major** (sequential) order rather than column-major.
- Improves spatial locality → reduces miss rate

---

## Reducing Miss Penalty

### Non-Blocking Cache

**Blocking cache**: Only one outstanding miss at a time — no progress while waiting.

**Non-blocking cache** allows:
- **Hit-under-miss**: execute cache hits while waiting for a miss
- **Miss-under-miss**: issue additional memory requests while waiting for a miss

> Non-blocking caches can reduce the effective miss penalty by **~50%** (memory-level parallelism).

### Miss Status Handling Registers (MSHR)

MSHRs track outstanding misses:

| Scenario | Action |
|----------|--------|
| New miss (not yet requested) | Allocate an MSHR; record waiting instruction |
| Half-miss (already requested) | Add instruction to existing MSHR |
| Data arrives from memory | Use MSHR to notify waiting instructions; free MSHR |

**Typical MSHR count**: 16–32 registers.

---

## Cache Hierarchies

Multiple cache levels reduce AMAT by catching misses at a closer level:

$$\text{AMAT} = T_{L1} + m_{L1} \cdot (T_{L2} + m_{L2} \cdot (T_{L3} + m_{L3} \cdot T_{mem}))$$

| Property | Relationship |
|----------|-------------|
| Capacity | L1 < L2 < L3 |
| Latency | L1 < L2 < L3 |
| Miss rate | L1 > L2 > L3 (typically) |

The final level before main memory is the **Last Level Cache (LLC)**.

### Local vs. Global Hit Rate

| Metric | Formula |
|--------|---------|
| **Local Hit Rate** | Hits in this cache / Accesses to this cache |
| **Global Miss Rate** | Misses in this cache / Total memory accesses |
| **Global Hit Rate** | 1 − Global Miss Rate |
| **MPKI** | Misses per 1,000 instructions |

---

## Inclusion Property

Three possibilities for the same block in L1 and L2:

| Policy | Rule |
|--------|------|
| **Default (neither)** | Block may or may not be in L2 |
| **Inclusion** | Block in L1 **must** also be in L2 |
| **Exclusion** | Block in L1 **cannot** also be in L2 |

**Enforcing inclusion**: requires an **inclusion bit** per block to track whether the block is in another cache level.

---

## Summary

**Key Takeaways**:
- AMAT = Hit Time + Miss Rate × Miss Penalty — optimize all three
- VIPT caches: index with virtual, tag with physical → parallel TLB + cache; no flush needed if size ≤ associativity × page size
- Way prediction: direct-mapped speed with set-associative miss rate
- 3 Cs: compulsory (prefetch), capacity (larger cache), conflict (higher associativity)
- Prefetching: hardware stream buffers and stride prefetchers are common
- Non-blocking + MSHR: overlap multiple outstanding misses
- Cache hierarchy: AMAT is recursive through all levels to main memory

**Common Exam Topics**:
- Compute AMAT for a multi-level hierarchy
- Determine if VIPT cache has aliasing given sizes
- Compare replacement policies

**See Also**: [[01-Introduction-and-Metrics]], [[09-Cache-Coherence]]
**Next**: [[09-Cache-Coherence]]
