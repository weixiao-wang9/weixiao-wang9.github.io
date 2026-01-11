---
id: 202512170429
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - streaming
  - algorithm
related_topics: "[[Concept - DASH Architecture]]"
created: 2025-12-17
---

# Bitrate Adaptation Algorithms (Rate vs. Buffer)

## 💡 The Core Idea
In DASH, the client must decide which bitrate to request for the next chunk ($R_{next}$) to maximize Quality of Experience (QoE).



## 🧠 Strategies

### 1. Rate-Based Adaptation
* **Method:** Estimates future bandwidth based on the throughput of the last few chunks.
* **Flaw (Underestimation):** In the "Steady State," the client pauses (OFF period) when the buffer is full. This resets the TCP congestion window. When it restarts (ON period), TCP ramp-up is slow, causing the client to underestimate bandwidth and pick a lower quality than necessary.

### 2. Buffer-Based Adaptation
* **Method:** Selects bitrate based on buffer occupancy.
    * Low Buffer $\rightarrow$ Low Bitrate (to prevent stall).
    * High Buffer $\rightarrow$ High Bitrate.
    * $$R_{next} = f(buffer_{now})$$.
* **Flaw:** Can oscillate unnecessarily between bitrates if the steps are too aggressive.

## 🔗 Connections
- **Source:** [[Source - Multimedia Applications]]