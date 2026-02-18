---
type: source
course: "[[HPCA (High Performance Computer Architecture)]]"
lessons: Lesson 7b (ReOrder Buffer), Lesson 8 (Memory Ordering)
prerequisites: "[[05-Tomasulo-and-OOO-Execution]]"
---

# ReOrder Buffer and Memory Ordering

> **Prerequisites**: [[05-Tomasulo-and-OOO-Execution]]
> **Learning Goals**: Understand how the ROB enables correct OOO execution through in-order commit, how exceptions and branch mispredictions are handled, and how the Load/Store Queue manages memory dependencies.

---

## The Problem with Tomasulo's Algorithm

**Major drawback**: Tomasulo's algorithm writes results directly to registers out-of-order. This causes two problems:

1. **OOO Exceptions**: Instructions after an exception-causing instruction may already have written results → incorrect state
2. **Phantom Exceptions**: After a branch misprediction, instructions from the wrong path may have executed and caused exceptions → these exceptions should never have occurred

> **Exception handling should not occur until the processor is certain the exception is not a phantom.**

---

## Correct OOO Execution with ROB

The solution:
- **Execute** out-of-order ✓
- **Broadcast** out-of-order ✓
- **Write to registers** in-order via **Commit** ✓

The **ReOrder Buffer (ROB)** holds results until it is safe to commit them in program order.

> A ReOrder Buffer **remembers program order** and keeps results until it is safe to write them to architectural registers.

---

## ROB Structure

### Fields per ROB Entry

| Field | Purpose |
|-------|---------|
| **Value** | The result data held temporarily |
| **Done bit** | Whether this instruction has finished executing |
| **Destination register** | Which architectural register will receive this value |

### ROB Pointers

| Pointer | Points To |
|---------|----------|
| **Issue pointer** (head) | Next empty slot — where the next issued instruction is placed |
| **Commit pointer** (tail) | Oldest in-flight instruction — next to be committed |

Entries between Issue and Commit are currently in-flight (executing or waiting).

---

## ROB Phase: Issue

Steps when issuing with ROB:

1. Take instruction from IQ
2. Get a **free RS** of the correct type
3. Get the **next ROB entry** (at Issue pointer)
4. Look up **source registers** in RAT
5. Update RAT for the **destination register** → point to the ROB entry (not the RS)

> Key difference from vanilla Tomasulo: RAT points to ROB entries, not RS entries.

---

## ROB Phase: Dispatch

Steps for dispatch with ROB:

1. Find waiting RS entries where broadcast tags match
2. Only dispatch instructions where **all inputs are ready**
3. Pick one instruction per functional unit
4. **Free the RS** immediately (broadcast uses ROB tag, not RS tag → RS freed sooner)

---

## ROB Phase: Broadcast

Steps for broadcast with ROB:

1. Capture result in waiting RS entries (tag matching)
2. Write result to the **ROB entry** (not directly to register file)

---

## ROB Phase: Commit

Steps for commit:

1. Look at the **oldest instruction** in the ROB (at Commit pointer)
2. Wait for its **Done bit** to be set
3. Copy result from ROB → **register file** (this is the only place register writes happen)
4. Update RAT: if RAT entry still points to this ROB entry → clear it (value now in register file)
5. Free the ROB entry; advance Commit pointer

> **Commit is always in-order.** This guarantees architectural state is always correct.

### RAT Updates on Commit
- If RAT still points to this ROB entry → update RAT to point to the register file
- If RAT has been updated to a newer ROB entry (a later write to the same register) → leave RAT alone

---

## Branch Misprediction Recovery

Branches are not committed until resolved. All instructions after an unresolved branch are also not committed.

**Recovery steps**:
1. Move Issue pointer back to Commit pointer position (flush ROB entries after the branch)
2. Reset RAT entries to point to the correct registers
3. Fetch correct instructions from the correct address

> **Committed instructions cannot be "uncommitted"** — they are permanent.

---

## ROB and Exception Handling

Two exception scenarios solved by ROB:

| Problem | ROB Solution |
|---------|-------------|
| OOO instructions with exceptions | ROB hasn't committed yet → flush everything, load exception handler |
| Phantom exceptions (wrong-path) | After branch flush, wrong-path instructions (and their exceptions) are discarded |

**Rule**: Treat an exception like any other result → delay handling until **Commit**.

---

## Unified Reservation Stations

**Unified RS**: One large array of RS shared across all functional units (adder + multiplier).
- **Advantage**: RS can be used by whichever unit needs it — no waiting for a specific RS type
- **Disadvantage**: More complex hardware (must route each RS entry to the correct unit)

---

## Superscalar

A superscalar processor can handle **multiple instructions per cycle** in every stage:
- Fetch > 1 instruction/cycle
- Decode > 1 instruction/cycle
- Issue > 1 instruction/cycle
- Dispatch > 1 instruction/cycle
- Broadcast > 1 instruction/cycle
- Commit > 1 instruction/cycle

> **Weakest link**: the processor is limited by the **slowest stage**. If all stages handle 4 instructions/cycle but Decode only handles 2, the whole processor is limited to 2/cycle.

---

## What is Really Out of Order?

| Stage | Order |
|-------|-------|
| Fetch | In-order |
| Decode | In-order |
| Issue | In-order |
| Execute | **Out-of-order** |
| Write Result | **Out-of-order** |
| Commit | In-order |

---

## Memory Ordering: Load/Store Queue (LSQ)

### When Do Memory Writes Happen?
Memory writes occur at **Commit**. Writing before commit would make the write irrevocable — it could not be undone on an exception or misprediction.

Memory reads (loads) should complete **as early as possible** for performance.

### LSQ Fields

| Field | Purpose |
|-------|---------|
| Load/Store bit | Which type of operation |
| Address | Target memory address |
| Data value | Value to be stored (for stores) |
| Completion bit | Whether address/value is computed |

> **Loads and stores are placed in the LSQ in program order.**

---

## Store-to-Load Forwarding

When a load is issued, its address is compared against all **previous stores** in the LSQ.
- If a match is found → use the value from the LSQ (no need to go to memory)
- This is **Store-to-Load Forwarding**

### What if the Store Doesn't Have its Address Yet?

| Strategy | Description | Performance |
|----------|-------------|-------------|
| 1. Wait for the store | In-order: stall load until all prior stores have addresses | Slow |
| 2. Wait for all prior stores' addresses | Check for address match; only proceed if no conflict | Medium |
| 3. Speculative load (aggressive) | Load from memory immediately; recover if a prior store had same address | Best performance, requires recovery |

> **Modern processors use Strategy 3** (speculative load) because it yields the best performance.

**Recovery from stale load**: When the store gets its address and discovers a matching load already executed with stale data → re-execute the load (and all dependent instructions).

---

## OOO vs. In-Order Load/Store

| Mode | Rule | Performance |
|------|------|------------|
| **OOO Load/Store** | Load goes to memory as soon as address is ready | Best, but may need recovery |
| **In-Order Load/Store** | Load cannot execute until all prior stores have addresses | Safe, but slow |

---

## LSQ, ROB, RS Interaction

| Instruction Type | Resources Required at Issue |
|-----------------|--------------------------|
| Load/Store | ROB entry + LSQ entry |
| Non-Load/Store | ROB entry + RS entry |

### Executing a Load/Store
1. Compute address
2. Produce the value (load from memory or from LSQ forwarding)
3. Loads: write result → broadcast

### Committing a Load/Store
1. Free ROB entry
2. Free LSQ entry
3. For stores: **write value to memory** (at commit, not at execute)

---

## Summary

**Key Takeaways**:
- ROB enables OOO execution with in-order commit → correct architectural state
- Commit is always in-order; register writes only happen at commit
- Branch misprediction: flush ROB; exception: handle at commit
- Phantom exceptions: safely discarded when branch misprediction flushes the ROB
- Memory writes only happen at commit; loads are as early as possible
- Store-to-load forwarding: LSQ provides value before memory is written
- Modern processors use speculative loads with recovery

**Common Exam Task**: Trace ROB/RS/RAT state through Issue/Dispatch/Broadcast/Commit for a sequence of instructions.

**See Also**: [[05-Tomasulo-and-OOO-Execution]], [[08-Advanced-Caches]]
**Next**: [[07-Compiler-ILP-and-VLIW]]
