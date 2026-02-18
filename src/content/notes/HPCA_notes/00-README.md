---
type: meta
course: "[[HPCA (High Performance Computer Architecture)]]"
date: 2026-02-17
---

# HPCA Study Guide

## Course Overview

High Performance Computer Architecture covers the principles and techniques used to design fast, efficient processors. Topics progress from fundamentals through advanced parallelism and memory systems.

---

## Learning Path

### Prerequisites
- Basic digital logic and computer organization
- Understanding of assembly language / instruction sets
- Familiarity with binary arithmetic

### Recommended Study Order

1. [[01-Introduction-and-Metrics]] — Why architecture matters; how to measure performance
2. [[02-Pipelining-and-Hazards]] — The core pipeline mechanism and its problems
3. [[03-Branch-Prediction]] — Resolving control hazards efficiently
4. [[04-ILP-and-Register-Renaming]] — Exploiting instruction-level parallelism; removing false dependencies
5. [[05-Tomasulo-and-OOO-Execution]] — Hardware out-of-order scheduling
6. [[06-ROB-and-Memory-Ordering]] — Correct OOO execution; memory dependency handling
7. [[07-Compiler-ILP-and-VLIW]] — Compiler-driven ILP; VLIW architecture
8. [[08-Advanced-Caches]] — Cache optimization techniques; hierarchy design
9. [[09-Cache-Coherence]] — Multi-core coherence protocols

---

## Quick Reference

### Key Formulas

| Formula | Meaning |
|---------|---------|
| `CPU Time = #Instructions × CPI × Clock Cycle Time` | Iron Law of Performance |
| `Speedup = Latency(Y) / Latency(X)` | Performance comparison |
| `Speedup = 1 / ((1 - Frac) + Frac/SpeedupEnh)` | Amdahl's Law |
| `P = ½ C × V² × freq × alpha` | Dynamic power |
| `CPI = 1 + (Mispredictions/Inst) × (Penalty/Misprediction)` | Branch penalty CPI |
| `AMAT = Hit Time + Miss Rate × Miss Penalty` | Average Memory Access Time |

### Key Concepts at a Glance

- **Moore's Law** — Transistors double every 18–24 months
- **Memory Wall** — Memory latency improves only ~1.1× every 2 years vs faster CPUs
- **ILP** — Instruction-Level Parallelism (ideal); ILP ≥ IPC always
- **RAW** — True data dependency (Read After Write)
- **WAR / WAW** — False (name) dependencies, removable by register renaming
- **Tomasulo's Algorithm** — Hardware OOO via Reservation Stations + RAT
- **ROB** — ReOrder Buffer enables in-order commit for correct OOO execution
- **AMAT** — Average Memory Access Time; optimized via hit time, miss rate, miss penalty

---

## File Descriptions

### [[01-Introduction-and-Metrics]]
**Topics**: Computer architecture goals, Moore's Law, Memory Wall, power consumption (dynamic/static), fabrication costs, performance metrics, benchmarks, Iron Law, Amdahl's Law
**Key Learning Goals**: Understand why architecture matters, how performance is measured and compared

### [[02-Pipelining-and-Hazards]]
**Topics**: 5-stage pipeline, pipelining CPI, pipeline stalls & flushes, control/data dependencies, hazard types (RAW/WAW/WAR), hazard handling strategies, pipeline depth trade-offs
**Key Learning Goals**: Understand how pipelining improves throughput and what hazards cost performance

### [[03-Branch-Prediction]]
**Topics**: Branch behavior, BTB, BHT, 1-bit/2-bit predictors, history-based predictors, PShare/GShare, tournament/hierarchical predictors, Return Address Stack
**Key Learning Goals**: Understand how branch mispredictions hurt performance and how predictors minimize this

### [[04-ILP-and-Register-Renaming]]
**Topics**: ILP definition, register renaming, RAT (Register Allocation Table), architectural vs physical registers, predication for branches
**Key Learning Goals**: Understand ILP as an ideal upper bound, and how register renaming removes false dependencies

### [[05-Tomasulo-and-OOO-Execution]]
**Topics**: Tomasulo's Algorithm, Reservation Stations (RS), Issue/Dispatch/Broadcast pipeline, stale results, load/store dependencies
**Key Learning Goals**: Understand the hardware mechanism for out-of-order instruction scheduling

### [[06-ROB-and-Memory-Ordering]]
**Topics**: ReOrder Buffer (ROB), exception/misprediction handling in OOO, commit semantics, superscalar, Load/Store Queue (LSQ), store-to-load forwarding
**Key Learning Goals**: Understand how the ROB enables correct OOO execution and how memory ordering is maintained

### [[07-Compiler-ILP-and-VLIW]]
**Topics**: Tree height reduction, instruction scheduling, loop unrolling, function inlining, VLIW architecture, superscalar vs VLIW comparison
**Key Learning Goals**: Understand how compilers extract ILP and the trade-offs of VLIW design

### [[08-Advanced-Caches]]
**Topics**: AMAT optimization, pipelined caches, VIPT caches, way prediction, replacement policies (LRU/NMRU/PLRU), 3 Cs of misses, prefetching, non-blocking caches, MSHR, cache hierarchies, inclusion/exclusion
**Key Learning Goals**: Understand techniques to reduce hit time, miss rate, and miss penalty

### [[09-Cache-Coherence]]
**Topics**: Coherence problem in multicore, write-update vs write-invalidate, snooping, MSI/MOSI/MOESI protocols, directory-based coherence, coherence misses (true/false sharing)
**Key Learning Goals**: Understand why multicore systems need coherence protocols and how major protocols work

---

## Study Tips

- The Iron Law and Amdahl's Law appear on every exam — know them cold
- Pipeline hazard questions require tracing instruction timing cycle by cycle
- Tomasulo's Algorithm questions require filling in RS/RAT tables step by step
- ROB + commit = correct OOO; remember "phantom exceptions"
- AMAT questions: always expand the full hierarchy equation
- Cache coherence: draw state machine transitions for MSI/MOESI
