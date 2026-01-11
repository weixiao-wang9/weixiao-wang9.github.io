---
id: 202512170408
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - qos
  - mechanism
related_topics: []
created: 2025-12-17
---

# Traffic Shaping: Token Bucket vs. Leaky Bucket

## 💡 The Core Idea
Mechanisms to limit the rate of traffic flow. They distinguish between **Policing** (dropping excess) and **Shaping** (delaying/smoothing excess).



## 🧠 Comparison

### Token Bucket (Allows Bursts)
* **Analogy:** A bucket fills with tokens at rate $R$. To send a packet, you must "pay" with tokens.
* **Behavior:** If the bucket is full of tokens, a flow can send a large **burst** of traffic immediately (up to bucket size $B$).
* **Use Case:** Limiting average rate while tolerating short spikes.

### Leaky Bucket (Smooth Output)
* **Analogy:** Water enters a bucket and leaks out of a hole at a constant rate $r$.
* **Behavior:** Regardless of how fast packets arrive (input), they leave the bucket at a constant, rigid rate.
* **Use Case:** Converting bursty traffic into a constant bit rate stream.

## 🔗 Connections
- **Source:** [[Source - Packet Classification and QoS]]