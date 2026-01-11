---
id: 202512170328
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - hardware
  - mechanism
related_topics: "[[Concept - Router Architecture Planes]]"
created: 2025-12-17
---

# Switching Fabrics

## 💡 The Core Idea
The switching fabric is the hardware interconnect that moves data from the input port to the output port. Its capacity determines the total throughput of the router.

## 🧠 Types of Fabrics

### 1. Memory (Generation 1)
* **Mechanism:** Packet is copied into the CPU's RAM, then copied out.
* **Limit:** Limited by memory bandwidth and bus speed. Behaves like a standard PC.

### 2. Shared Bus (Generation 2)
* **Mechanism:** All input ports share a single communication bus. Only one packet can cross at a time.
* **Limit:** Bus bandwidth.

### 3. Crossbar Switch (Generation 3/Modern)
* **Mechanism:** A grid of horizontal and vertical buses. Multiple packets can cross simultaneously (e.g., A->Y and B->X) as long as they don't share an output.
* **Challenge:** **Head-of-Line (HOL) Blocking** (if the first packet in the queue is blocked, subsequent packets are stuck).



## 🔗 Connections
- **Source:** [[Source - Router Architecture and Lookups]]