---
type: source
course: "[[HPCA (High Performance Computer Architecture)]]"
lessons: Lesson 3
prerequisites: "[[01-Introduction-and-Metrics]]"
---

# Pipelining and Hazards

> **Prerequisites**: [[01-Introduction-and-Metrics]]
> **Learning Goals**: Understand how pipelining improves throughput, what types of hazards arise, and how they are detected and resolved.

---

## Pipelining in a Processor

**Pipelining** overlaps the execution of multiple instructions to improve throughput.

### 5-Stage Basic Pipeline

| Stage | Description |
|-------|-------------|
| **Fetch** | Retrieve instruction from memory |
| **Read/Decode** | Decode instruction and read registers |
| **ALU** | Execute arithmetic/logic operation |
| **Memory Access** | Load or store data in memory |
| **Write** | Write result back to registers |

> Pipelining the instructions takes the **same total time per instruction**, but **throughput is improved**.

---

## Pipelining CPI

In an ideal pipeline with no hazards, CPI = 1.

When an instruction must **wait** at a pipeline stage:
- All instructions **ahead** of it proceed normally
- All instructions **behind** it are **stalled**
- This creates a "bubble" that propagates through the pipeline

> **As the number of stalls (delays) increases, CPI increases.**

### Overall CPI Formula

$$\text{Overall CPI} = \text{Ideal CPI} + \text{Stall cycles per instruction}$$

For branch mispredictions:
$$\text{Overall CPI} = \text{CPI}_\text{program} + \frac{\text{Mispredictions}}{\text{Instructions}} \times \text{Penalty}$$

---

## Pipeline Stalls and Flushes

### Pipeline Flush
- Caused by **branch mispredictions**
- When an incorrect branch is taken, instructions that were fetched incorrectly must be **flushed** (replaced with NOPs)
- Correct instructions are then fetched from the correct address

---

## Data Dependencies

A **data dependence** occurs when an instruction needs data produced by an earlier instruction.

### Types of Data Dependencies

| Type | Also Called | True Dependency? | Causes Hazard? |
|------|-------------|-----------------|----------------|
| **RAW** (Read After Write) | Flow / True Dependence | Yes | Yes |
| **WAW** (Write After Write) | Output Dependence | No (False) | Sometimes |
| **WAR** (Write After Read) | Anti-Dependence | No (False) | Sometimes |
| **RAR** (Read After Read) | — | Not a dependence | Never |

> - **RAW** = true dependency: program semantics require this ordering
> - **WAW and WAR** = **false (name) dependencies**: caused by register reuse, not by actual data flow — can be eliminated by register renaming

---

## Control Dependencies

A **control dependence** arises when an instruction's execution depends on the outcome of a branch.

- ~**20% of all instructions** are branches or jumps
- ~**50% of branch/jump instructions** are taken

---

## Dependencies vs. Hazards

| Concept | Caused By | Always a Problem? |
|---------|-----------|-------------------|
| **Dependency** | The program | No — many dependencies don't cause hazards |
| **Hazard** | Program + Pipeline | Yes — hazards cause incorrect execution |

Not every dependency becomes a hazard. A RAW dependency only causes a hazard if the dependent instruction reaches its read stage before the producing instruction has finished its write stage.

---

## Handling Hazards

### Step 1: Detect
Identify dependencies that will actually cause a hazard in this pipeline.

### Step 2: Remove
Three strategies:

| Method | Used For | How It Works |
|--------|----------|--------------|
| **Flush** | Control dependencies | Remove wrong instructions; fetch correct ones |
| **Stall** | Data dependencies | Hold dependent instruction until data is ready |
| **Forward (Fix)** | Data dependencies | Bypass: route the computed value directly to where it's needed without waiting for register writeback |

> **Forwarding** (Option 3) does not always work — e.g., a load followed immediately by a dependent instruction still requires a 1-cycle stall (load-use hazard).

---

## How Many Pipeline Stages?

When adding more stages:
1. More potential hazards are introduced
2. Penalty per hazard **increases** (more stages to flush/stall)
3. Less work per stage → **shorter cycle time** possible

The Iron Law says: **balance CPI and cycle time**.

| Optimization Goal | Ideal Pipeline Depth |
|-------------------|----------------------|
| Performance only | 30–40 stages |
| Performance + Power | 10–15 stages |

---

## Summary

**Key Takeaways**:
- Pipelining improves throughput but introduces hazards
- RAW is the only true dependency — it must be respected
- WAW and WAR are false dependencies — they can be eliminated (register renaming)
- Hazards are handled by: flush (control), stall (data), or forward (data)
- Deeper pipelines have lower cycle time but higher hazard penalties — balance is key

**Common Patterns**:
- Stall = insert bubble = CPI penalty
- Forward = short-circuit result = avoids stall when possible
- Flush = pipeline penalty for mispredicted branches

**See Also**: [[03-Branch-Prediction]], [[04-ILP-and-Register-Renaming]]
**Next**: [[03-Branch-Prediction]]
