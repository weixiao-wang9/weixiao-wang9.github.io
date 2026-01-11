---
id: 202512170405
type: source
subtype: lecture_note
course: "[[Computer Networks (CS 6250)]]"
module: "Module 6: Packet Switching & QoS"
title: Packet Classification, Switching, and Traffic Shaping
status: Finished
tags:
  - input
  - networks
  - algorithms
  - qos
created: 2025-12-17
---

# Packet Classification, Switching, and Traffic Shaping

## 📌 Context & Summary
> [!abstract] AI Context
> This lecture moves beyond simple destination-based forwarding. It covers **Packet Classification** (handling traffic based on multiple criteria like TCP flags or Source IP for firewalls/QoS). It explores the hardware constraints of routers, specifically **Head-of-Line (HOL) Blocking** in crossbar switches, and concludes with **Traffic Scheduling** algorithms (Fair Queuing, Deficit Round Robin) and **Shaping** mechanisms (Token/Leaky Buckets).

## 📝 Notes & Highlights

### Packet Classification
- **Why it's needed:** Longest prefix matching isn't enough for firewalls, resource reservation (DiffServ), or traffic-type routing (e.g., video streams).
- **Simple Solutions:**
    - **Linear Search:** Good for few rules, prohibitive for thousands.
    - **Caching:** High hit rates (80-90%), but missed hits result in slow linear searches.
    - **MPLS:** Labels are assigned at the edge, avoiding re-classification at intermediate routers.



### Classification Algorithms (2D Rules)
- **Set-Pruning Tries:** Builds a destination trie where every leaf node contains a source trie compatible with that destination.
    - *Issue:* Memory explosion because source prefixes are replicated across multiple destination branches.
- **Backtracking:** Points destination prefixes to source tries; if a match fails, it backtracks up the destination trie.
    - *Issue:* High time cost due to backtracking steps.
- **Grid of Tries:** Uses "switch pointers" to jump directly to the next possible source trie, eliminating the need to backtrack.

### Switching & Scheduling
- **Crossbar Switches:** Connect $N$ inputs to $N$ outputs.
    - **Problem:** **Head-of-Line (HOL) Blocking**. If the packet at the head of an input queue is blocked, the entire queue waits, even if subsequent packets are destined for free outputs.
- **Solutions:**
    - **Knockout Scheme (Output Queuing):** Run the fabric $k$ times faster than input links.
    - **Parallel Iterative Matching:** Uses virtual queues and a Request-Grant-Accept phase to schedule non-blocked packets.

### Quality of Service (QoS)
- **FIFO with Tail Drop:** Simple but can drop important data and lacks fairness.
- **Bit-by-Bit Fair Queuing:** Serves flows bit-by-bit (simulated) to ensure fairness, but is computationally expensive ($O(\log(\text{flows}))$).
- **Deficit Round Robin (DRR):** A constant-time $O(1)$ approximation of fair queuing using "quantum" and "deficit" counters.

### Traffic Shaping & Policing
- **Token Bucket:** Allows bursts up to size $B$. Tokens accumulate at rate $R$.
- **Leaky Bucket:** Smoothes traffic into a constant output rate, acting like a water leak.

## 🔗 Extracted Concepts
- [[Concept - Packet Classification Algorithms]]
- [[Concept - Head of Line Blocking]]
- [[Concept - Deficit Round Robin]]
- [[Concept - Token vs Leaky Bucket]]