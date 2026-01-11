---
id: 202512170407
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - qos
  - algorithm
related_topics: []
created: 2025-12-17
---

# Deficit Round Robin (DRR)

## 💡 The Core Idea
A scheduling algorithm that provides **fair bandwidth allocation** and handles variable packet sizes with $O(1)$ complexity, avoiding the high cost of bit-by-bit fair queuing.



## 🧠 Mechanism
Each flow is assigned:
1.  **Quantum ($Q_i$):** The bandwidth share per round (e.g., 500 bits).
2.  **Deficit Counter ($D_i$):** Tracks unused credit.

### The Algorithm
* In each round, add $Q_i$ to $D_i$.
* Send packets from the flow as long as $\text{Packet Size} \le D_i$.
* Decrement $D_i$ by the size of sent packets.
* If the queue is empty, reset $D_i$ to 0.

## 🔗 Connections
- **Source:** [[Source - Packet Classification and QoS]]