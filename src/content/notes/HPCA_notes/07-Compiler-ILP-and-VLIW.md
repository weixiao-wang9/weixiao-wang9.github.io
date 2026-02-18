---
type: source
course: "[[HPCA (High Performance Computer Architecture)]]"
lessons: Lesson 10 (Compiler ILP), Lesson 11 (VLIW)
prerequisites: "[[04-ILP-and-Register-Renaming]]"
---

# Compiler ILP and VLIW

> **Prerequisites**: [[04-ILP-and-Register-Renaming]]
> **Learning Goals**: Understand how compilers extract ILP through scheduling and loop transformations, and how VLIW architecture shifts scheduling responsibility to the compiler.

---

## Compiler ILP Goals

Two basic goals:
1. **Improve instruction scheduling** — reduce stalls by placing independent instructions between dependent ones
2. **Reduce the number of instructions** — eliminate overhead

---

## Tree Height Reduction

**Tree height** = the length of the longest dependency chain in a computation.

**Tree Height Reduction**: Re-group calculations to shorten dependency chains.

- Only works for **associative** operations (e.g., addition, multiplication)
- Example: `((A+B)+C)+D` has depth 3; `(A+B)+(C+D)` has depth 2 (both paths can execute in parallel)

---

## Techniques to Expose Independent Instructions

### 1. Instruction Scheduling

When there is a dependency between two instructions, the processor would normally stall. The compiler can **move an independent instruction** into the stall slot.

**Modifications that may be needed**:
- **Address offset changes**: when an instruction moves relative to an address load
- **Destination register changes**: when a move causes a register to be written earlier than expected (may need to use a different destination register)

### 2. Scheduling and If Conversion

**If conversion** (predication) helps instruction scheduling in two ways:
1. **Reduces branches** — both sides execute using predication
2. **More scheduling flexibility** — without branches, instructions can be moved more freely, reducing stalls

> A **loop** cannot use if-conversion, but can be improved with loop unrolling.

### 3. Loop Unrolling

**Loop unrolling**: expand a loop so each iteration does the work of N original iterations.

```
Original (N=1):   loop body once per iteration
Unrolled (N=4):   loop body 4x per iteration, loop runs ¼ as many times
```

**Benefits**:
- Reduces **loop overhead** (branch, counter update) → fewer total instructions
- Gives the scheduler **more instructions to work with** → better CPI

**Downsides**:
- **Code bloat** — program size increases
- If iteration count is unknown or not a multiple of N → requires extra handling (not covered in this course)

### 4. Function Call Inlining

**Inlining**: replace a function call with the function body directly at the call site.

**Benefits**:
- Eliminates function call + return overhead
- Allows the scheduler to see across the call boundary → better scheduling

**Downside**:
- **Code bloat** — especially if large functions are inlined repeatedly
- Best applied to **small functions**

---

## Other Compiler IPC Techniques (not covered in detail)

- **Software Pipelining** — interleave iterations of a loop at the instruction level
- **Trace Scheduling** — schedule across multiple basic blocks based on the most likely execution path

---

## VLIW Architecture

### Processors that Can Issue > 1 Instruction/Cycle

| Processor Type | Scheduling | Cost | Notes |
|---------------|-----------|------|-------|
| **OOO Superscalar** | Hardware (RS, RAT) | Very high | Looks at many instructions; compiler helps |
| **In-Order Superscalar** | Partial hardware | Medium | Fewer instructions visible; needs compiler help |
| **VLIW** | Compiler only | Low | Executes 1 large instruction per cycle |

---

## VLIW: How It Works

A **VLIW processor** executes one very long instruction per cycle. Each VLIW instruction contains multiple operation slots (e.g., one ALU op, one FP op, one load/store op).

| Step | Superscalar | VLIW |
|------|-------------|------|
| 1 | Hardware fetches multiple instructions | Compiler packs independent ops into one word |
| 2 | Hardware checks dependencies at runtime | Compiler checks dependencies at compile time |
| 3 | Hardware schedules for parallel execution | Hardware just executes the next large instruction word |

> If there are dependencies, the compiler places them in **separate instruction words**. This can cause **code bloat** (many NOPs in unused operation slots).

---

## VLIW: The Good and the Bad

### Advantages
- **Compiler does work once** at compile time; program runs many times → amortized cost
- **Simpler hardware** than superscalar (no RS, no ROB needed)
- **Energy efficient**
- Works well on **"regular code"** — loops, array processing, DSP workloads

### Disadvantages
- **Latencies are not always the same** across executions → compiler schedule may be pessimistic
- **Many applications are irregular** — hard to schedule statically
- **Code bloat** — NOPs in unused slots (partially mitigated by VLIW instruction compaction)

---

## VLIW Instruction Features

| Feature | Purpose |
|---------|---------|
| Standard ISA opcodes | All usual operations |
| Full predication support | Compiler can eliminate branches |
| Many registers | Needed due to scheduling optimizations (loop unrolling exposes more live values) |
| Branch hints | Compiler tells hardware its branch predictions |
| **Instruction compaction** | Replace NOP slots with "stop" markers → reduces code bloat |

---

## VLIW Examples

| Processor | Notes |
|-----------|-------|
| **Itanium (Intel IA-64)** | Too complicated; poor performance on irregular code |
| **DSP Processors** | Excellent performance, energy efficient — regular workloads |

---

## Summary

**Key Takeaways**:
- Compiler ILP extracts parallelism at compile time vs. hardware at runtime
- Tree height reduction: restructure associative operations to shorten dependency chains
- Instruction scheduling: fill stall slots with independent instructions
- Loop unrolling: fewer branches, more scheduling flexibility, code bloat tradeoff
- Function inlining: eliminates call overhead, enables cross-function scheduling
- VLIW: compiler does all scheduling; simple hardware; great for regular code, poor for irregular
- VLIW compaction (stops) reduces the NOP problem

**See Also**: [[04-ILP-and-Register-Renaming]], [[03-Branch-Prediction]]
**Next**: [[08-Advanced-Caches]]
