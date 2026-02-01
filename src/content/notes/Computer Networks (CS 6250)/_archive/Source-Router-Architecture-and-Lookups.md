---
id: 202512170326
type: source
subtype: lecture_note
course: "[[Computer Networks (CS 6250)]]"
module: "Module 5: Router Internals"
title: Router Architecture and Prefix Lookups
status: Finished
tags:
  - input
  - networks
  - hardware
  - algorithms
created: 2025-12-17
---

# Router Architecture and Prefix Lookups

## 📌 Context & Summary
> [!abstract] AI Context
> This lecture opens up the "black box" of the router. It explains the hardware architecture (Input/Output ports, Switching Fabrics) and the critical separation between the **Forwarding Plane** (Hardware/Fast) and **Control Plane** (Software/Slow). It concludes with a deep dive into the algorithms used for fast IP lookups: **Longest Prefix Match** and **Multibit Tries**.

## 📝 Notes & Highlights

### Router Architecture
- **Control Plane (The Brain):** Implemented in software (Routing Processor). Runs protocols like BGP/OSPF to build the routing table.
- **Forwarding Plane (The Muscle):** Implemented in hardware. Moves packets from Input -> Output interfaces in nanoseconds.

### Key Components
1.  **Input Ports:** Perform the physical termination, data link decapsulation, and the **Lookup** (FIB check).
2.  **Switching Fabric:** The interconnect that moves data.
    - *Types:* Memory (Slow/Old), Bus (Shared bandwidth), Crossbar (High speed/Parallel).
3.  **Output Ports:** Queues packets and transmits them to the next link.

### The Lookup Problem: Longest Prefix Match (LPM)
- Routers don't match exact IPs; they match **Prefixes** (CIDR).
- **Challenge:** Millions of routes, very high speeds (line rate).
- **Optimization:** We cannot use simple caching because traffic flows are short and diverse.

### Algorithms: Tries
- **Unibit Trie:** Checks 1 bit at a time. Efficient memory, but slow (32 lookups for a 32-bit address).
- **Multibit Trie:** Checks $k$ bits at a time (Stride).
    - **Controlled Prefix Expansion:** Expands prefixes to match the stride length.
    - **Trade-off:** Uses more memory to achieve fewer memory accesses (faster lookup).

## 🔗 Extracted Concepts
- [[Concept - Router Architecture Planes]]
- [[Concept - Switching Fabrics]]
- [[Concept - Longest Prefix Match]]
- [[Concept - Trie-Based Lookups]]