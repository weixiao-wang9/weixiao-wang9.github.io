---
type: source
course: "[[HPCA (High Performance Computer Architecture)]]"
lessons: Study Guide
prerequisites: "All Modules"
---


# HPCA Study Guide — CS 6290 Georgia Tech
*Based on all 10 Problem Sets + Solutions*

---

## How to Use This Guide

Each section follows this structure:
1. **Core Concepts** — what you must know cold
2. **Key Formulas** — the equations that appear on every problem
3. **Problem Patterns** — the types of questions asked, with worked strategy
4. **Practice Questions** — new questions modeled after the problem sets
5. **Common Traps** — mistakes the solutions explicitly call out

---

## Module 1: Metrics and Evaluation

### Core Concepts
- **Iron Law**: `Time = (Instructions/Program) × (Cycles/Instruction) × (Time/Cycle)`
- **Amdahl's Law**: Overall speedup is limited by the fraction that *cannot* be enhanced
- **CPI** (Cycles Per Instruction): weighted average across instruction types
- The "unenhanced" portion is the ceiling on speedup — no matter how fast the rest gets

### Key Formulas

**Amdahl's Law (single enhancement):**
```
S = 1 / [(1 - F_e) + F_e/S_e]
```
- `F_e` = fraction of time the enhancement applies
- `S_e` = speedup of the enhanced portion

**Generalized Amdahl's Law (multiple non-overlapping enhancements):**
```
S = 1 / [(1 - Σf_i) + Σ(f_i / S_i)]
```

**Iron Law speedup ratio (when instruction count doesn't change):**
```
S = (CPI_old × CycleTime_old) / (CPI_new × CycleTime_new)
```

**Weighted CPI:**
```
CPI = Σ (fraction_i × CPI_i)
```

**Cache miss penalty CPI adjustment:**
```
CPI_new = CPI_base + (miss_rate × miss_penalty × fraction_memory_instructions)
```

### Problem Patterns

**Pattern 1 — Single enhancement (P1, P3):**
- Identify `F_e` and `S_e`, plug into Amdahl
- Example: 40% of time sped up 10x → S = 1/(0.6 + 0.04) = **1.56**

**Pattern 2 — Compare two design choices (P2, P4):**
- Compute CPI or execution time for each design separately, then compare
- For P4 "Triple RISC": use generalized Amdahl with 1/3 at 3x, 1/3 at 2x, 1/3 at 1x
- For "RISCily Fast": use Iron Law, compute new CPI by adding miss penalty × rate

**Pattern 3 — Four-part Amdahl (P5):**
- Treat parts with S=1 as "unenhanced residual" in denominator
- Key insight: a 20x speedup on 23% matters less than a 1.6x speedup on 48%

**Pattern 4 — Nested enhancements (P6 — L1 + L2 cache):**
- Do NOT multiply individual speedups — that is wrong
- Use generalized Amdahl: L1 covers 80%×30% = 24%, L2 covers ½×20%×30% = 3%
- Both applied simultaneously in the formula

**Pattern 5 — Solve for unknown fraction (P9):**
- Set up Amdahl equal to target speedup, solve for `F_c` algebraically

**Pattern 6 — Fraction of enhanced time with no enhancement active (P10):**
```
F_NE_enhanced = (1 - Σf_i) / S_total
```
This equals the unenhanced time divided by the new total time.

**Pattern 7 — Best single / best pair enhancement (P11):**
- Compute S for each individually, pick highest
- For pairs: compute all three combinations; note that the best pair usually includes the two individually best enhancements

### Practice Questions

**P-1.1:** An enhancement speeds up 60% of execution time by 8x. What is the overall speedup?

**P-1.2:** A processor has 30% FP instructions (CPI=6) and 70% integer (CPI=1.2). Two options:
- Option A: Reduce FP CPI to 3
- Option B: Reduce integer CPI to 1.0
Which option gives better overall CPI?

**P-1.3:** Three enhancements affect 15%, 25%, and 40% of execution time with speedups 20, 8, and 4 respectively. Assume no overlap. What is the overall speedup?

**P-1.4:** Given the same three enhancements above, what fraction of the reduced execution time has no enhancement active?

**P-1.5 (challenge):** Enhancement A affects 30% (speedup=50) and Enhancement B has speedup=10. What fraction of time must Enhancement B be used to achieve an overall speedup of 5?

### Common Traps
- **Overlap trap (P8):** If the problem doesn't say enhancements affect *different* parts, you cannot use generalized Amdahl's Law — the answer may be indeterminate
- **L1+L2 trap (P6):** Never multiply speedups sequentially. After L1 is applied, the remaining miss time is no longer 20% of original — it's larger as a fraction of new time
- **Clock frequency vs CPI:** When clock frequency changes, use Iron Law — don't just compare CPI alone (P4 "RISCily Fast")
- **Amdahl's ceiling:** Speedup can never exceed `1/(1 - F_e)` regardless of how large `S_e` gets

---

## Module 2: Pipelining and Hazards

### Core Concepts
- **5 stages**: IF → ID → EX → MEM → WB
- **Hazard types**: Structural, Data (RAW/WAW/WAR), Control
- **Dependency vs Hazard**: Dependency is a property of the code; hazard is a runtime problem caused by pipelining
- **True dependency**: RAW — actual data flows from producer to consumer
- **Name dependencies**: WAR and WAW — only about register names, not actual data
- **Forwarding**: Passes results directly to later stages, avoiding stalls for most RAW hazards
- **Load-use hazard**: Even with forwarding, a load followed immediately by a use of the loaded value requires 1 stall cycle

### Key Formulas

**Pipelining speedup (ideal):**
```
S = CPI_unpipelined × T_stage_old / (CPI_pipelined × T_cycle_new)
```

**CPI with stalls:**
```
CPI_actual = CPI_ideal + stalls_per_instruction
```

**Pipeline clock cycle time:**
```
T_cycle = max(stage_latencies) + latch_overhead
```

**Unpipelined execution time per instruction:**
```
T_avg = T_cycle × weighted_average_CPI
```

### Problem Patterns

**Pattern 1 — Identify dependencies (P1, P3):**
- Go through each instruction pair; for every register that is written by instruction A and read/written by instruction B (B after A), note the dependency type
- RAW: A writes, B reads same register
- WAR: A reads, B writes same register (B would overwrite before A reads, if reordered)
- WAW: A writes, B writes same register

**Pattern 2 — Pipeline timing diagrams (P4, P5):**
- Without forwarding: consumer must wait until WB stage of producer (can read in same cycle as write)
- With forwarding: result available at end of EX, can be forwarded to start of next EX
- Mark stall cycles as "S" in the pipeline diagram
- Count total cycles from IF of first instruction to end of EX of last instruction

**Pattern 3 — Delay slots (P5A):**
- The instruction in the delay slot always executes after the branch
- Move an independent instruction into the slot (e.g., the SD whose address is now -4(R2) because R2 was already incremented)

**Pattern 4 — Clock cycle time and 3-stage / 6-stage pipelines (P6):**
- T_cycle = max(combined_stage_latency) + latch_overhead per intermediate latch
- To minimize clock cycle time: balance stages as evenly as possible
- To split for 6 stages: split the bottleneck stage (longest latency)

**Pattern 5 — Out-of-order execution table (P12):**
- Instructions stall in DI if operand not ready (no forwarding from WB)
- WB stage can only handle one instruction at a time
- WAW hazard: name-dependent instruction must not complete WB before the earlier instruction

### Practice Questions

**P-2.1:** For the following code, identify all RAW, WAW, and WAR dependencies:
```
I1: ADD R1, R2, R3
I2: SUB R4, R1, R5
I3: MUL R2, R4, R6
I4: DIV R1, R2, R3
```

**P-2.2:** A 5-stage pipeline has stage latencies IF=150ps, ID=80ps, EX=200ps, MEM=220ps, WB=70ps. Each latch adds 15ps. (a) What is the clock cycle time? (b) If CPI=1 unpipelined (single-cycle) and CPI=1.3 pipelined, what is the speedup?

**P-2.3:** For the following loop with forwarding, show the pipeline timing and count the cycles for one iteration:
```
Loop: LD R1, 0(R2)
      ADD R3, R1, R4
      SW 0(R2), R3
      ADDI R2, R2, 4
      BNE R2, R5, Loop
```

**P-2.4:** What is the difference between a structural hazard and a data hazard? Give one example of each.

### Common Traps
- **Dependency ≠ hazard**: A WAR dependency between I1 and I5 may not be a hazard if they are far enough apart in the pipeline
- **Same-cycle WB/ID**: If the problem says "registers can be read and written in the same cycle during WB," then an instruction in ID that needs a value can get it from WB without stalling
- **Delay slot**: The instruction in the delay slot always executes — even if the branch is not taken

---

## Module 3: Branch Prediction

### Core Concepts
- **Always-taken predictor**: predict every branch as taken; mispredicts on not-taken branches and on the last iteration of loops
- **1-bit predictor**: remembers last outcome; poor for loops (mispredicts twice per loop — exit and first re-entry)
- **2-bit saturating counter**: states 0(SN), 1(WN), 2(WT), 3(ST); changes prediction only after two consecutive wrong predictions; much better for loops
- **Local history predictor**: per-branch history register selects which 2-bit counter to use; captures patterns within a single branch's behavior
- **Aliasing**: two different branches hashing to the same predictor entry corrupt each other's state

### Key Formulas

**Misprediction count for always-taken:**
- Always-taken branches: 0 mispredictions per execution
- Always-not-taken branches: N mispredictions (one per execution)
- Loop-exit branches: 1 misprediction per loop execution (the exit)
- Unconditional branches (BEQ R0,R0,...): always taken, 0 mispredictions

**CPI with branch penalty (from ROB PS Problem 1):**
```
CPI = CPI_base + f_branch × [f_taken × (f_miss×2 + (1-f_miss)×1)
      + (1-f_taken) × (f_miss×2 + (1-f_miss)×0)]
```

### Problem Patterns

**Pattern 1 — Count mispredictions for always-taken (P1):**
- Trace the actual execution of the nested loop
- For each branch instruction, determine how many times it is taken vs not-taken
- Always-taken mispredicts when branch is actually NOT taken

**Pattern 2 — 1-bit predictor with 8 entries (P2):**
- Map each branch address to a table entry using bits [4:2] of the PC (after stripping 2 LSBs)
- Track aliasing: if two branches share an entry, they interfere
- Follow state transitions: 0→taken prediction wrong → state flips

**Pattern 3 — Local history predictor (P3):**
- Each entry has a 2-bit history register and 4 2-bit saturating counters (one per history pattern)
- On each branch: use current history bits to index counter → predict → update counter → shift history

**Pattern 4 — Branch resolution location and data hazards (P3A):**
- Resolving branches in ID reduces control hazard stalls (only 1 cycle penalty instead of 2)
- But ID runs before EX, so it needs register values earlier → creates more data hazard stalls for branches that read registers being computed by recent instructions

### Practice Questions

**P-3.1:** For a loop that executes 100 times, how many mispredictions occur with: (a) always-taken, (b) always-not-taken, (c) 1-bit predictor initialized to "not taken"?

**P-3.2:** Two branches at addresses 0x200 and 0x220 share the same entry in a 4-entry 1-bit predictor (initialized to 0). Branch at 0x200 is always taken. Branch at 0x220 alternates T/NT/T/NT... How many mispredictions occur in 8 executions of each?

**P-3.3:** Why does increasing branch predictor size beyond a point yield diminishing returns?

### Common Traps
- **Address indexing**: strip the 2 LSBs first, then take the N least-significant bits of the result to index into an N-entry table
- **Unconditional branches (BEQ R0,R0)**: always predicted taken = always correct with always-taken predictor
- **1-bit aliasing**: even if the predictor has enough entries, aliasing can occur if two branches happen to hash to the same entry

---

## Module 4: ILP and Register Renaming

### Core Concepts
- **ILP** (Instruction-Level Parallelism): number of instructions that can execute simultaneously
- **True ILP limit**: only RAW dependencies are true; WAR/WAW can be eliminated by renaming
- **Register renaming**: assigns physical registers to eliminate false (WAR/WAW) dependencies
- **Practical ILP limits**: branch mispredictions, cache misses, limited rename registers, limited issue width, load/store ordering

### Key Formulas

**Minimum cycles with N-way issue:**
```
Cycles = max(critical_path_length, ceil(num_instructions / N))
```
Where critical path = longest chain of RAW-dependent instructions

### Problem Patterns

**Pattern 1 — Register renaming (P1):**
- Scan instructions in order
- When an instruction WRITES a register: assign a new physical location (L4, L5, ...)
- When an instruction READS a register: use whatever location that register currently maps to
- Example: `MUL R1, R2, R3` — R1 gets new name L4; reads use L1 (R1), L2 (R2), L3 (R3)

**Pattern 2 — Cycle count with 2-way issue (P2):**
- Draw dependency graph
- Group independent instructions together for parallel issue
- Count cycles along the critical path

**Pattern 3 — Cycle count with 4-way issue (P3):**
- Same as above, but up to 4 instructions per cycle
- More parallelism available; the critical path (RAW chains) still limits

### Practice Questions

**P-4.1:** Use register renaming to eliminate false dependencies. Assume 8 physical locations L1-L8; initially R1=L1, R2=L2, R3=L3:
```
MUL R1, R2, R3
ADD R2, R1, R3
SUB R1, R2, R3
MUL R3, R1, R2
```

**P-4.2:** How many cycles does the following take with 3-way issue (all instructions latency=1)?
```
ADD R1, R2, R3
MUL R4, R1, R5
SUB R6, R4, R7
ADD R8, R2, R9
LD  R10, (R11)
MUL R12, R10, R4
```

**P-4.3:** List three factors that prevent a real processor from achieving the theoretical maximum ILP shown in benchmarks.

### Common Traps
- **Renaming only eliminates WAR and WAW**: RAW dependencies remain and constrain execution order
- **Critical path determines minimum cycles**: even with infinite issue width, you can't execute faster than the longest dependency chain

---

## Module 5: Tomasulo's Algorithm and Out-of-Order Execution

### Core Concepts
- **Reservation stations (RS)**: hold instructions waiting for operands; tag unavailable operands with the RS/ROB entry that will produce them
- **Common Data Bus (CDB)**: broadcasts result + tag; all RS entries watching for that tag latch the value
- **Issue**: send instruction to RS (in-order); read available operands; tag unavailable ones
- **Execute**: when all operands ready, send to functional unit
- **Write result**: broadcast on CDB; free RS
- **RAW resolution**: when the producing instruction writes its result to CDB, all waiting RS entries automatically receive the value

### Key Rules
- Issue is in-order; execution and write-result can be out-of-order
- An instruction cannot begin executing until all its operands are available
- The result is written in the LAST cycle of execution (not the cycle after)
- A dependent instruction can begin the cycle AFTER the result is written

### Problem Patterns

**Pattern 1 — Fill in Tomasulo scheduling table (Instruction Scheduling PS P1, P4):**
1. Issue cycle: first cycle where an RS and (for MUL/DIV) the execution unit is free
2. First execution cycle: cycle after all operands are available (either from register file or CDB write)
3. Write result cycle: first execution cycle + latency - 1 (result written in last exec cycle)

**Pattern 2 — Tomasulo with ROB (Interrupts PS P4):**
- Same as above, but RS is freed when result is written (not at commit)
- Commit happens in-order; an instruction commits the cycle after it writes its result (earliest)
- ROB entry is freed at commit

### Practice Questions

**P-5.1:** Schedule the following using Tomasulo's algorithm. 1 MUL unit (latency 4, 2 RS), 1 ADD unit (latency 2, 3 RS). Issue starts at cycle 1:
```
I1: MUL F1, F2, F3
I2: ADD F4, F1, F5
I3: MUL F6, F4, F2
I4: ADD F1, F6, F3
```

**P-5.2:** Explain the role of the Register Alias Table (RAT) in Tomasulo's algorithm. What does it store, and when is it updated?

**P-5.3:** Why can Tomasulo's algorithm eliminate WAR and WAW hazards but not RAW hazards?

### Common Traps
- **Latency timing**: if latency is 4 and first execution is cycle 3, write result is cycle 6 (cycles 3,4,5,6)
- **Structural hazard**: if the execution unit is already busy, the new instruction must wait even if all operands are ready
- **Reservation station capacity**: if all RS are full, no new instruction can issue — even if operands are ready

---

## Module 6: ReOrder Buffer (ROB) and Exceptions

### Core Concepts
- **ROB purpose**: enables in-order commit of out-of-order execution; supports precise exceptions; supports branch misprediction recovery
- **ROB fields per entry**: instruction, destination register, value, ready bit, exception bit
- **Commit**: instructions commit in-order from the head of the ROB when the ready bit is set
- **Exception handling with ROB**: set exception bit when exception detected; wait until that instruction reaches the head and would normally commit; then flush the ROB, RS, and reset the RAT

### ROB Cleanup on Exception
- **(a) Reservation stations**: flush all instructions (they are speculative / after the excepting instruction)
- **(b) ROB**: flush all entries after the excepting instruction; the excepting instruction's ROB entry can be used to restore state or jumped past
- **(c) RAT**: reset all entries to point to the register file (architectural state is now correct after in-order commit up to the exception)

### Key Formulas

**CPI with branch penalties:**
```
CPI = CPI_base + f_branch × [f_taken × (f_miss×P_T + (1-f_miss)×P_T_correct)
      + (1-f_taken) × (f_miss×P_NT + (1-f_miss)×0)]
```
Where P_T = penalty for taken-predicted branch, P_NT = penalty for not-taken misprediction

### Problem Patterns

**Pattern 1 — CPI calculation with branch predictor (P1, P2):**
- Substitute all given fractions and penalties into the CPI formula
- P2: also account for frequency scaling (compare time = CPI × cycle_time)

**Pattern 2 — Exception handling questions (P3, P5):**
- P3: Without ROB, you cannot get correct register values because some instructions before I5 may not have committed yet, and some after may have already written to registers
- P5: The ROB allows in-order commitment up to (but not including) the faulting instruction, giving the correct architectural state

**Pattern 3 — 2-bit branch prediction explanation (P4 Q1):**
- The 2-bit counter starts in state "strongly not taken" (00); requires two wrong predictions before changing the prediction
- Two (or more) branches share the global/local history table

### Practice Questions

**P-6.1:** A 5-stage pipeline has 20% branch instructions, 70% taken. The predictor is correct 90% of the time. Misprediction penalty is 3 cycles. Base CPI = 1.0. What is the actual CPI?

**P-6.2:** Explain why a processor without an ROB cannot guarantee a precise exception when instructions execute out of order.

**P-6.3:** After an exception is triggered and caught, what is the correct state in the register file? How does the ROB ensure this?

---

## Module 7: Advanced Caches

### Core Concepts
- **VIPT cache** (Virtually Indexed, Physically Tagged): uses virtual address bits for index, physical tag for comparison
  - Aliasing occurs when two virtual addresses map to the same index but different physical pages
  - Safe if index bits come only from the page offset (bits below the page boundary)
  - Max page size for no aliasing: `Cache_size / Associativity`
- **Cache geometry**: `Sets = Cache_size / (Block_size × Associativity)`
- **Tag bits** = Physical_address_bits − Index_bits − Block_offset_bits
- **PIPT cache** (Physically Indexed, Physically Tagged): no aliasing, but requires TLB before cache lookup
- **LRU replacement**: LRU counter of 0 = least recently used (evict this one)
- **Write-back + dirty bit**: dirty bit set on writes; eviction of dirty block requires write-back to memory
- **Prefetching**: issue cache request N iterations ahead to hide miss latency
  - Optimal prefetch distance ≈ miss_latency / iteration_time instructions ahead

### Key Formulas
```
Index bits = log2(Sets)
Block offset bits = log2(Block_size)
Tag bits (PIPT) = Physical_address_bits − Index_bits − Block_offset_bits
Total tag storage = tag_bits × num_lines
```

**For VIPT aliasing limit:**
```
Max page size = Cache_size / Associativity
(The index bits must not extend above the page offset bits)
```

### Problem Patterns

**Pattern 1 — Cache geometry (P1-P4):**
- Compute sets, tag bits, and overhead bits systematically

**Pattern 2 — Cache state table (P5-P14):**
For each access:
1. Decode address: determine which set and what tag
2. Check if any valid line in that set has matching tag → **hit** or **miss**
3. On hit: update LRU; if write, set dirty bit
4. On miss: choose victim (LRU=0); if victim is dirty → **write-back**; load new block; set tag, valid, LRU

**Pattern 3 — Prefetching analysis (P15-P21):**
- P15: Count how many unique cache blocks are accessed → compulsory misses drive total count
- P16: Analyze which cache change (size, associativity, block size) eliminates the most misses for this access pattern
- P17-P21: Track when prefetch completes vs when load executes; prefetch at offset K means block arrives (49 cycles after prefetch) = K/4 iterations later

### Practice Questions

**P-7.1:** A 4KB 4-way set-associative cache has 64-byte blocks. Physical address = 32 bits. (a) How many sets? (b) How many tag bits? (c) What is the max page size for VIPT with no aliasing?

**P-7.2:** A 2-set, 2-way PIPT write-back LRU 32-byte-block cache with 10-bit physical addresses. Initial state: Set 0: [V=1,D=0,Tag=1F,LRU=1] [V=1,D=1,Tag=2F,LRU=0]. Set 1: [V=0] [V=0]. Perform access: WR address 0x1E0. Show state after access. Is there a write-back?

**P-7.3:** The following array traversal runs on a 256-byte direct-mapped cache with 16-byte blocks. The array has 64 elements (4 bytes each). How many cache misses occur on the first pass?

### Common Traps
- **LRU=0 means LEAST recently used** (the one to evict), not most recently used
- **Write-allocate**: on a write miss, the block is first brought into the cache, then written
- **Dirty bit is set only on writes**, not on reads
- **Prefetch at offset 0 is useless**: it fetches the same block that the immediately following load needs — only 1 cycle of latency hiding

---

## Module 8: Memory Hierarchy (Virtual Memory, TLB, Cache Systems)

### Core Concepts
- **TLB** (Translation Lookaside Buffer): caches recent virtual-to-physical translations
- **TLB miss penalty**: extra memory accesses to walk the page table
- **Effective CPI with cache:**
```
CPI_eff = CPI_base + f_mem × miss_rate × miss_penalty_cycles
```
- **Miss penalty** for write-back cache:
  - On a clean eviction: just fetch new block from memory
  - On a dirty eviction: write back dirty block, then fetch new block
- **AMAT** (Average Memory Access Time):
```
AMAT = Hit_time + Miss_rate × Miss_penalty
```

### Key Formula — CPI with dirty write-back:
```
Miss_penalty = (block_size / bus_width) × transfer_cycles + memory_latency
Dirty_writeback_overhead = 0.5 × dirty_fraction × miss_rate × writeback_cycles
CPI_eff = CPI_base + miss_rate × (miss_penalty + dirty_fraction × writeback_penalty)
```

### Problem Patterns

**Pattern 1 — Compare three cache configurations (P1):**
- Compute effective CPI for each: `CPI_eff = CPI × (cycle_length_factor) + miss_penalty_stalls`
- Miss penalty stalls = `f_mem × miss_rate × penalty_cycles`
- Add TLB penalty: `+ f_mem × TLB_miss_rate × TLB_penalty`

**Pattern 2 — Optimal block size (P2, P3):**
- AMAT = 1 + miss_rate × miss_penalty(block_size)
- Miss penalty increases with block size (more transfers); miss rate decreases
- Find the minimum AMAT

### Practice Questions

**P-8.1:** CPI_base = 2.0, 25% of instructions are memory ops, cache miss rate = 3%, miss penalty = 50 cycles. What is effective CPI?

**P-8.2:** A system has a TLB with 1% miss rate and 15-cycle penalty. Cache is physically addressed, so TLB must be checked on every memory access. How does this affect CPI from P-8.1?

---

## Module 9: Cache Coherence (MESI Protocol)

### Core Concepts
- **MESI states**: Modified, Exclusive, Shared, Invalid
- **M (Modified)**: only copy; cache has latest value; memory is stale; must write-back before any other cache can access
- **E (Exclusive)**: only copy; cache matches memory; can be silently upgraded to M on write
- **S (Shared)**: multiple caches may have this block; must invalidate all others before writing
- **I (Invalid)**: cache does not have valid copy
- **Bus snooping**: every cache watches all bus transactions and updates its state accordingly

### State Transition Rules

| Action | Current State | New State | Bus Transaction |
|--------|--------------|-----------|-----------------|
| Read miss | I | S or E | BusRd |
| Write miss | I | M | BusRdX |
| Write hit | S | M | BusUpgr (invalidates others) |
| Write hit | E | M | Silent (no bus) |
| Write hit | M | M | Silent (no bus) |
| Read hit | S/E/M | unchanged | none |
| Snoop BusRd | M | S | write-back to memory |
| Snoop BusRdX | M/S/E | I | write-back if M |

**Key**: Data source on miss:
- If another cache has it in M state → that cache supplies data (and transitions to S or I)
- If all other caches have I → data comes from memory

### Problem Patterns

**Pattern 1 — Single access trace (Multiprocessing PS P1-P10):**
Step by step:
1. Decode address to determine which cache set/tag is accessed
2. Check the requesting processor's cache: hit or miss?
3. If miss: broadcast on bus; check other caches for M/E/S copies
4. Determine write-back needed (if any M state elsewhere)
5. Determine data source
6. Determine final state in requestor's cache

**Pattern 2 — False sharing (P11-P13):**
- When two cores write to different words in the same cache block, the block constantly ping-pongs between M states on different cores
- Solution: pad data so each core's data occupies a separate cache block

**Pattern 3 — Compiler optimization vs memory consistency (P14-P16):**
- Loop-invariant code motion can move a variable read outside a loop
- If another thread modifies that variable, the hoisted read sees a stale value
- This is a classic memory consistency problem requiring `volatile` or memory barriers

### Practice Questions

**P-9.1:** With the same initial MESI state table from the PS (P0:I/M/I/M, P1:S/E/I/E, P2:S/M/E/S, P3:S/E/I/I for sets 0-3), what happens when P0 reads address 0x01xx (set 1)?

**P-9.2:** Explain what "false sharing" is and how it causes performance degradation even when two threads never access the same data.

**P-9.3:** Why does the MSI protocol need fewer states than MESI, and what performance does MESI gain by adding the E state?

### Common Traps
- **Address decoding**: with 4 blocks per cache and 64-byte blocks, the set is bits [7:6] of the address; the tag is bits [15:8]
- **Each problem is independent**: the PS explicitly says each question starts from the same initial state, not the state left by the previous question
- **M state always requires write-back** when another processor requests the block (whether for read or write)
- **E state → M on write**: no bus transaction needed (already the only copy)

---

## Module 10: Interrupts and Exceptions

### Core Concepts
- **Precise exception**: at the point of exception, all instructions before the faulting instruction have completed, and no instructions after have modified architectural state
- **Problem without ROB**: out-of-order execution means later instructions may have already written to registers while earlier ones haven't — impossible to reconstruct correct state
- **Solution**: ROB maintains in-order commit; architectural registers only updated at commit

### Problem Patterns

**Pattern 1 — Correct register values at exception (P1):**
- Trace execution in program order (not execution order)
- The "correct" values are those that would result from executing I1 through I4 sequentially (everything before I5)
- I5 causes exception → I5 and after produce no architectural changes

**Pattern 2 — Fill Tomasulo table with ROB (P4):**
- Same scheduling rules as Tomasulo
- Commit column: instruction commits in-order, one per cycle, starting the cycle after write-result
- Must wait for all previous instructions to commit first

**Pattern 3 — Exception clean-up steps (P5):**
- Reservation stations: flush all (no longer valid)
- ROB: flush all entries after the faulting instruction's entry (they are speculative); faulting instruction's entry is used to restore and then freed
- RAT: reset all entries to point to the register file (which now has the correct architectural state after all prior commits)

### Practice Questions

**P-10.1:** Explain in one sentence why it is impossible to get precise exception behavior in an OOO processor without a ROB.

**P-10.2:** In a processor with a ROB, what is the difference between "write result" and "commit"? Why is this distinction important for exceptions?

---

## Quick Reference: Formula Sheet

```
Iron Law:           Time = IC × CPI × T_cycle
Amdahl's:           S = 1 / [(1-Fe) + Fe/Se]
Generalized:        S = 1 / [(1-Σfi) + Σ(fi/Si)]
CPI weighted:       CPI = Σ(f_i × CPI_i)
CPI miss penalty:   CPI_eff = CPI_base + f_mem × miss_rate × penalty
CPI branch penalty: CPI_eff = CPI_base + f_branch × penalty_contribution
Pipeline speedup:   S = T_old / T_new = (CPI_old × T_old) / (CPI_new × T_new)
Pipeline T_cycle:   max(stage_latency) + latch_overhead
Cache sets:         Sets = Size / (Block_size × Ways)
Tag bits:           PA_bits − log2(Sets) − log2(Block_size)
VIPT max page:      Cache_size / Associativity
AMAT:               Hit_time + Miss_rate × Miss_penalty
Prefetch distance:  miss_latency / cycles_per_iteration  (in iterations)
```

---

## Study Strategy

1. **Amdahl's Law problems** → practice until the formula is automatic; the tricky part is always identifying what `f_i` actually is (fraction of *original* execution time, not instruction count)

2. **Pipeline timing diagrams** → draw these out physically; count stall cycles explicitly; the answer is always "count the columns"

3. **Branch prediction** → trace the actual execution by hand; track predictor state after each branch

4. **Tomasulo / ROB scheduling** → fill in tables cycle by cycle; always ask "is the execution unit free?" and "are all operands available?"

5. **MESI coherence** → memorize the state transition table; for each access, work through all five questions (hit/miss? write-back? bus broadcast? other caches? data source?)

6. **Cache arithmetic** → always start with: offset bits → index bits → tag bits; decode the address and compare with stored tags
