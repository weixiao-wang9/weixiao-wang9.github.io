---
id: 202512170056
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - networks
  - mechanism
related_topics: "[[Concept - TCP Congestion Control]]"
created: 2025-12-17
---

# TCP Flow Control

## 💡 The Core Idea
Flow control matches the sender's rate against the **receiver's** reading rate to prevent the receiver's buffer from overflowing.

## 🧠 Mechanism
The receiver maintains a buffer ($RcvBuffer$). It calculates the free space available:
$$rwnd = RcvBuffer - [LastByteRcvd - LastByteRead]$$
This value, **Receive Window (`rwnd`)**, is sent to the sender in every ACK packet.

### The Rule
The sender ensures that the amount of unacknowledged data in flight never exceeds `rwnd`:
$$LastByteSent - LastByteAcked \le rwnd$$


* **Corner Case:** If `rwnd = 0`, the sender stops sending large packets but continues sending 1-byte probe segments to check when space opens up.

## 🔗 Connections
- **Source:** [[Source - Transport and Application Layers]]