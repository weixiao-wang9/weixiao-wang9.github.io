---
type: source
course: "[[HPCA (High Performance Computer Architecture)]]"
lessons: Lesson 6
prerequisites: "[[02-Pipelining-and-Hazards]]"
---

# ILP and Register Renaming

> **Prerequisites**: [[02-Pipelining-and-Hazards]]
> **Learning Goals**: Understand ILP as an upper bound on processor parallelism, how register renaming removes false dependencies, and how ILP relates to actual IPC.

---

## Instruction-Level Parallelism (ILP)

**ILP** = the maximum number of instructions that can be executed per cycle in an ideal processor.

- ILP is a property of the **program**, not the processor
- Computed assuming: perfect processor, infinite execution units, perfect branch prediction, no structural limits

$$\text{ILP} = \frac{\text{# Instructions}}{\text{# Cycles required (ideal)}}$$

> **ILP ≥ IPC** always — real processor constraints reduce IPC below the ILP upper bound.

---

## The Execute Stage and Forwarding

- **Forwarding** can supply data in the **next** cycle after production, not the same cycle
- For a RAW dependency: the reading instruction must be delayed by at least 1 cycle to allow the write to complete

---

## Types of Dependencies Recap

| Type | True/False | Impact on CPI |
|------|-----------|---------------|
| **RAW** | True | Directly limits CPI |
| **WAW** | False (Name) | Can cause OOO issues |
| **WAR** | False (Name) | Can cause stalls |

- **CPI is primarily determined by RAW dependencies**
- **WAW** can cause incorrect results in out-of-order execution if not handled
- **WAR and WAW** are called *false* or *name* dependencies because they arise from reusing the same register name, not from a genuine data flow

---

## Removing False Dependencies

### Why False Dependencies Exist
Two instructions use the **same register** to hold different, unrelated values. The processor sees a "conflict" even though there is no actual data flow between them.

### Method: Register Renaming

**Architectural registers**: registers visible to the programmer (e.g., R0–R31)
**Physical registers**: all storage locations available in hardware (more than architectural)

The processor **maps architectural registers to physical registers** dynamically, giving each instruction write a fresh physical register. This eliminates WAW and WAR conflicts.

---

## Register Allocation Table (RAT)

The **RAT** tracks which physical register currently holds the value for each architectural register.

### How RAT Works
- Each entry: `architectural register → physical register (or "register file")`
- When an instruction **writes** to an architectural register:
  - Allocate a new physical register
  - Update the RAT entry for that architectural register
- When an instruction **reads** a register:
  - Look up the RAT → get the physical register holding the current value

### Effect of Renaming
- **WAR eliminated**: reader uses the old physical register; writer gets a new one
- **WAW eliminated**: each write goes to a different physical register
- **RAW preserved**: the RAT correctly chains reads to the latest write

> **Renaming improves CPI and IPC** by removing artificial stalls caused by false dependencies.

---

## Steps to Compute ILP

1. **Rename registers** as they are written, tracking which physical registers are free
2. **"Execute" the program** assuming infinite resources and no false dependencies
3. Determine the earliest cycle each instruction can execute given only true (RAW) dependencies
4. **ILP = Instructions / Cycles**

---

## ILP with Control Dependencies

- Structural dependencies do not apply to ILP (perfect processor has unlimited units)
- For control dependencies: assume **perfect branch prediction** — all branches are resolved instantly with no mispredictions

---

## ILP vs. IPC

| Metric | Processor Model | Branch Prediction | Issue Width |
|--------|----------------|-------------------|-------------|
| **ILP** | Ideal, infinite | Perfect | Infinite |
| **IPC** | Real, limited | Real predictor | Finite N |

ILP is always ≥ IPC. The gap between them comes from:
- Finite issue width
- In-order constraints
- Imperfect branch prediction
- Structural limitations

### In-Order vs. Out-of-Order

| Processor Type | Limiting Factor |
|----------------|-----------------|
| In-order, narrow issue | Issue width (narrow issue is more limiting than in-order) |
| In-order, wide issue | In-order constraint (more limiting than width) |

> If a processor can issue **4+ instructions per cycle** (wide-issue), it should also be **out-of-order** to fully exploit the width. It also needs register renaming to eliminate false dependencies.

---

## Summary

**Key Takeaways**:
- ILP is the program's inherent parallelism — the ideal upper bound
- RAW = true dependency; limits ILP/IPC regardless of renaming
- WAW/WAR = false dependencies; can be **eliminated** with register renaming
- The RAT maps architectural → physical registers, enabling renaming
- Wide-issue processors need OOO + renaming to be effective

**Common Patterns**:
- Compute ILP: rename registers, identify earliest execution cycle per instruction
- RAT table exercise: trace through Issue/Rename/Read for each instruction

**See Also**: [[02-Pipelining-and-Hazards]], [[05-Tomasulo-and-OOO-Execution]]
**Next**: [[05-Tomasulo-and-OOO-Execution]]
