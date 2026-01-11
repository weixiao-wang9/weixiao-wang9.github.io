---
id: 202512170053
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - networks
  - algorithm
related_topics: "[[Concept - TCP Flow Control]]"
created: 2025-12-17
---

# TCP Congestion Control (AIMD & CUBIC)

## 💡 The Core Idea
Congestion control prevents a sender from overwhelming the **network**. It uses a "probe-and-adapt" approach, increasing transmission rate until packet loss occurs, then backing off.



## 🧠 Mechanisms

### AIMD (Additive Increase, Multiplicative Decrease)
* **Increase:** If successful, increase window size (`cwnd`) linearly (add 1 packet per RTT).
* **Decrease:** If loss is detected (timeout or triple duplicate ACKs), cut `cwnd` in half.
* **Result:** Creates a "Sawtooth" throughput pattern.

### Slow Start
Used for new connections. Instead of linear growth, `cwnd` increases **exponentially** (doubling every RTT) until it hits a threshold (`ssthresh`), then switches to AIMD.

### TCP CUBIC (Modern Standard)
Designed for high-bandwidth networks. Instead of linear growth, it uses a cubic function:
$$W(t) = C(t-K)^3 + W_{max}$$
This makes window growth independent of RTT, improving fairness.

## 🔗 Connections
- **Source:** [[Source - Transport and Application Layers]]