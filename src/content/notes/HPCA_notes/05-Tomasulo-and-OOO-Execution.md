---
type: source
course: "[[HPCA (High Performance Computer Architecture)]]"
lessons: Lesson 7 (Instruction Scheduling)
prerequisites: "[[04-ILP-and-Register-Renaming]]"
---

# Tomasulo's Algorithm and OOO Execution

> **Prerequisites**: [[04-ILP-and-Register-Renaming]]
> **Learning Goals**: Understand how Tomasulo's Algorithm implements hardware register renaming and out-of-order execution via Reservation Stations, and the three-phase Issue/Dispatch/Broadcast cycle.

---

## Improving IPC: The Full Picture

To achieve high IPC, a processor must address all dependency types:

| Dependency Type | Solution |
|----------------|---------|
| Control | Branch Prediction |
| WAR / WAW (false) | Register Renaming |
| RAW (true) | Out-of-Order Execution |
| Structural | Wider Issue (more execution units) |

> ILP should be at least **4 instructions per cycle** to justify wide-issue hardware.

---

## Tomasulo's Algorithm

Originally designed for floating-point units; modern processors extend it to all instructions.

### Modern Extensions over Original Tomasulo
1. All instructions use the algorithm (not just FP)
2. Hundreds of instructions considered simultaneously (not just a few)
3. Supports **exception handling** (original did not)

---

## Data Flow Paths

### Data Manipulation Path
```
Instruction Queue → Reservation Stations (RS)
                  → Execution Units (Adder / Multiplier)
                  → Broadcast on Bus
```

### Load/Store Path
```
Instruction Queue → Address Adder (PC + offset)
                  → Load/Store Buffer
                  → Memory
                  → Broadcast on Bus
```

---

## Key Terminology

| Term | Meaning |
|------|---------|
| **Issue** | Instruction exits IQ; goes to RS or address adder |
| **Dispatch** | Instruction exits RS; goes to execution unit (Adder or Multiplier) |
| **Broadcast / Write Result** | Result exits execution unit; placed on shared bus |

---

## Phase 1: Issue

Steps when issuing an instruction:

1. Take next instruction from IQ (in **program order**)
2. Look up source registers in the **RAT**
   - If RAT has an entry → use the RS/tag it points to
   - If RAT has no entry → read from register file
3. Find a **free RS** of the correct type (adder RS or multiplier RS)
   - If no free RS → stall and wait
4. Place instruction (with operand values or tags) in the RS
5. **Tag the destination register** in the RAT → points to this RS entry

---

## Phase 2: Dispatch

Steps when an instruction is ready to execute:

1. RS monitors broadcast bus — **match** broadcast tags to pending operand tags
2. When a match is found, capture the value into the RS entry
3. When an RS has **all inputs ready** → eligible to dispatch
4. Select which eligible RS to dispatch (strategies):
   - **Oldest first**
   - **Most dependencies first** (most waiting instructions) — hard to implement
   - **Random**
5. Send instruction to the Adder or Multiplier
6. **Free the RS** once dispatched

---

## Phase 3: Broadcast

Steps when execution completes:

1. Put the **tag** and **result** on the bus
2. Write result to the **register file**
3. Update the **RAT** — clear the entry (empty RAT entry = value in register file)
4. Free the RS (clear valid bit)

### Multiple Broadcasts Ready?
Options:
1. **Separate bus per arithmetic unit** (more hardware)
2. **Prioritize the slower unit** — multiply/divide takes more cycles, so it likely has more downstream dependencies waiting

---

## Stale Results

A result is **stale** if the RAT no longer points to its RS (a newer write has been issued to the same architectural register).

- Still broadcast and captured by waiting RS entries that tagged it
- **RAT is NOT updated** for stale results (future instructions won't use this value)

> **⚠ Exam note**: Stale results are broadcast but don't update the RAT.

---

## Cycle-Level Parallelism

In each clock cycle, the processor can simultaneously:
- **Issue** one new instruction
- **Dispatch** one instruction per execution unit
- **Broadcast (Write)** one result per execution unit

### Timing Rules
- Issue and dispatch in the same cycle? Technically possible but generally avoided (keeps cycles short)
- Capturing operands and dispatching: **not** in the same cycle
- Broadcasting and issuing to the same RS: possible but requires extra logic

---

## Memory Dependencies in Tomasulo

Load/Store instructions can have data dependencies through memory:

| Dependency | Scenario |
|-----------|---------|
| **RAW** | Store to address A, then Load from A |
| **WAR** | Load from A, then Store to A |
| **WAW** | Two stores to the same address A |

### Tomasulo's Solution
Tomasulo's original algorithm handles memory by doing **loads and stores in-order**. Modern processors identify and reorder memory operations more aggressively (covered in [[06-ROB-and-Memory-Ordering]]).

---

## Summary

**Key Takeaways**:
- Tomasulo uses RS as temporary storage + renaming medium
- Issue is in-order; execution (dispatch) and broadcast are out-of-order
- RAT tracks which RS (or register file) holds the current value for each architectural register
- Stale results: broadcast and captured, but RAT not updated
- Multiple broadcasts: prioritize slower units (multiply/divide)

**Common Exam Task**: Given an instruction sequence, trace the RS state, RAT state, and cycle numbers for Issue/Dispatch/Broadcast of each instruction.

**See Also**: [[04-ILP-and-Register-Renaming]], [[06-ROB-and-Memory-Ordering]]
**Next**: [[06-ROB-and-Memory-Ordering]]
