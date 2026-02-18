---
type: source
course: "[[HPCA (High Performance Computer Architecture)]]"
lessons: Lesson 1, Lesson 2
prerequisites: none
---

# Introduction and Metrics

> **Learning Goals**: Understand what computer architecture is, why it matters, and how to rigorously measure and compare performance using standard tools and laws.

---

## What is Computer Architecture?

**Computer Architecture** = Designing a computer that is well suited for its purpose.

### Why Do We Need It?
1. Improve **performance**
2. Improve **capabilities** of the computer

### Designing for the Future
Computer architects must be aware of trends and design for the future. Because product development takes years, if you design for *today's* technology, the product will be obsolete by launch.

---

## Moore's Law

> Every **18–24 months**, techniques allow **twice as many transistors** on the same chip area.

By extension:
1. Processor speed doubles every 18–24 months
2. Energy per operation is halved every 18–24 months
3. Memory capacity doubles every 18–24 months

### Moore's Law in Practice
With Moore's Law, a designer can either:
1. **Reduce cost** — make the same processor in a smaller area (cheaper)
2. **Improve performance** — use the same area to make a better processor

---

## The Memory Wall

While processor speed and memory capacity have followed Moore's Law, **memory latency** has only improved ~**1.1× every 2 years**.

- Old bottleneck: processor speed
- **New bottleneck: memory latency** → called the **Memory Wall**

**Mitigation**: Caches are used to bridge the gap between fast processors and slow memory.

---

## Power Consumption

Processor performance is usually discussed in terms of speed, but **power** and **fabrication cost** are equally important design constraints.

### Dynamic Power (Active Power)

$$P = \frac{1}{2} C \cdot V^2 \cdot f \cdot \alpha$$

| Symbol | Meaning |
|--------|---------|
| C | Capacitance |
| V | Power supply voltage |
| f | Clock frequency |
| α | Activity factor |

### Static Power (Leakage Power)
- Power consumed when the chip is **idle**
- As voltage decreases → leakage **increases**
- There is an **optimal voltage** that minimizes total power (static + dynamic trade-off)

> **Key insight**: Static power prevents lowering the voltage indefinitely.

---

## Fabrication Cost

**Fabrication Yield** = (# working chips) / (# chips on wafer)

- Larger die → higher % of defective parts → lower yield → higher cost
- Smaller feature sizes (via Moore's Law) reduce die area, improving yield and reducing cost

---

## Performance Metrics

### Speed Dimensions
- **Latency** — time from start to finish of one task
- **Throughput** — tasks completed per unit time

> Note: Throughput ≠ 1/Latency in general (e.g., pipelined systems)

### Speedup
"X is N times faster than Y":

$$\text{Speedup} = N = \frac{\text{Speed}(X)}{\text{Speed}(Y)} = \frac{\text{Throughput}(X)}{\text{Throughput}(Y)} = \frac{\text{Latency}(Y)}{\text{Latency}(X)}$$

- **Speedup < 1**: performance got *worse*
- **Speedup > 1**: performance got *better*

Performance ∝ 1 / Latency

---

## Benchmarks

A **benchmark** is a standard suite of programs representing common tasks, used to compare processor performance fairly.

### Types of Benchmarks

| Type | Realism | Ease of Setup | Use Case |
|------|---------|---------------|----------|
| Real Applications | Highest | Hardest | Real machine comparisons |
| Kernels | High | Hard | Prototypes |
| Synthetic | Medium | Easy | Design studies |
| Peak Performance | — | Easy | Marketing |

### Summarizing Benchmark Results
- Use **average execution time** to summarize performance
- When comparing **speedups**, use the **geometric mean** (NOT arithmetic mean)

$$\text{Geometric Mean} = \left(\prod_{i=1}^{n} \text{term}_i\right)^{1/n}$$

---

## Iron Law of Performance

$$\text{CPU Time} = \frac{\text{Instructions}}{\text{Program}} \times \frac{\text{Cycles}}{\text{Instruction}} \times \frac{\text{Time}}{\text{Cycle}}$$

All three factors matter:

| Factor | Influenced By |
|--------|---------------|
| # Instructions | Algorithm, compiler, instruction set |
| CPI (Cycles per Instruction) | Instruction set, processor design |
| Clock Cycle Time | Processor design, circuit design, transistor physics |

> Computer architects control the **instruction set** and **processor design**.

### Iron Law for Unequal Instruction Times

$$\text{CPU Time} = \left[\sum_i \frac{\text{Inst}_i}{\text{Program}} \times \frac{\text{Cycles}}{\text{Inst}_i}\right] \times \frac{\text{Time}}{\text{Cycle}}$$

---

## Amdahl's Law

Used when only a **fraction** of the system is improved:

$$\text{Speedup} = \frac{1}{(1 - F) + \frac{F}{S_E}}$$

- **F** = fraction of original execution time affected by the enhancement
- **S_E** = speedup of the enhanced portion

### Implications

**Make the Common Case Fast**: Small improvements to a large fraction of execution time yield more benefit than large improvements to a small fraction.

**Lhadma's Law**: While making the common case fast, do **not** make the uncommon case worse.

**Diminishing Returns**: After the easy optimizations are made, further improvements yield smaller and smaller speedups.

---

## Summary

**Key Takeaways**:
- Computer architects design for future trends (Moore's Law)
- Performance = speed (latency/throughput), but also power and cost
- Dynamic power: P = ½CV²fα — voltage is the most impactful lever
- Iron Law: CPU Time = Instructions × CPI × Cycle Time — all three must be optimized
- Amdahl's Law: focus improvements on the most frequently executed parts
- Use geometric mean when averaging speedup ratios

**See Also**: [[02-Pipelining-and-Hazards]]
**Next**: [[02-Pipelining-and-Hazards]]
