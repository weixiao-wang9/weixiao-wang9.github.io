---
id: 202512170407
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - hardware
  - failure-mode
related_topics: []
created: 2025-12-17
---

# Head-of-Line (HOL) Blocking

## 💡 The Core Idea
In a crossbar switch with input queuing, a packet at the front of the queue can block all packets behind it if its destination output is busy, even if the destinations for the subsequent packets are free.



## 🧠 Solutions

### 1. Output Queuing (The "Knockout" Scheme)
* Remove input queues entirely and queue at the output.
* **Requirement:** The switching fabric must run $N$ times faster than the input links (speedup) to handle simultaneous arrivals.
* **Practicality:** Uses a "Knockout" concentrator to handle $k$ simultaneous packets, dropping traffic only if it exceeds $k$.

### 2. Parallel Iterative Matching (Virtual Queues)
* Keep input queuing but break the single queue into **Virtual Queues** (one per output).
* **Algorithm:**
    1.  **Request:** Inputs send requests to all desired outputs.
    2.  **Grant:** Outputs randomly select one request.
    3.  **Accept:** Inputs accept one grant.

## 🔗 Connections
- **Source:** [[Source - Packet Classification and QoS]]