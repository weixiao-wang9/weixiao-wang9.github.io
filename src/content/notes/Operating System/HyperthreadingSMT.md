---
date: 2025-12-14
tags:
  - atomic
  - concept
  - hardware
source: "[[OS CPU Scheduling]]"
---

# Hyperthreading (SMT)

## Relationships
- **Parent**: [[Multiprocessor Scheduling]]
- **Solves Problem**: [[Memory Stall]] (Hides memory latency)
- **Contrast**: [[Multicore]] (Logical cores vs. Physical cores)

## 1. Definition
**Hyperthreading (Simultaneous Multithreading / SMT)** is a hardware technique that allows a single physical CPU core to act like two separate **logical cores**.
* **Goal:** To keep the hardware execution units busy by running two threads simultaneously.

## 2. Context: Why & How
### The Problem: Memory Stall
CPUs are much faster than Memory (RAM).
* When a thread needs to fetch data from memory, the CPU often has to sit idle (**Stall**) for hundreds of cycles. This is a waste of resources.

### The Solution: Interleaving
The physical core has **two sets of architectural state** (Registers, PC) but only **one set of execution units**.
* **Mechanism:** When Thread T1 stalls (waiting for memory), the hardware instantly switches to Thread T2 to utilize the idle execution units.

## 3. Scheduling Strategy (Co-scheduling)
The OS Scheduler must decide which two threads to pair on a single SMT core.

### Thread Types
* **CPU-bound:** Low CPI (Cycles Per Instruction). Always ready to run.
* **Memory-bound:** High CPI. Frequently waits for memory.

### Co-scheduling Matrix
| Scenario | Result | Verdict |
| :--- | :--- | :--- |
| **CPU + CPU** | Both fight for the single pipeline/ALU. Performance drops (~50% each). | ❌ **Bad** |
| **Mem + Mem** | Both frequently stall. CPU remains underutilized. | ❌ **Bad** |
| **CPU + Mem** | When Mem thread stalls, CPU thread runs. High utilization. | ✅ **Best** |

### Implementation
The Scheduler uses **Hardware Performance Counters** to make decisions:
* **Metrics:** Monitor IPC (Instructions Per Cycle) or LLC (Last Level Cache) misses.
* **Logic:** Try to mix high-CPI threads with low-CPI threads.

## Reference
Source: [[OS CPU Scheduling]]