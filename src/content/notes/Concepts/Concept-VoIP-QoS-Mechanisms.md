---
id: 202512170428
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags: [concept, voip, qos, mechanism]
related_topics: []
created: 2025-12-17
---

# VoIP QoS: Jitter and Packet Loss

## 💡 The Core Idea
Voice over IP (VoIP) runs on "best-effort" networks (UDP), meaning it must handle delay variation (jitter) and packet loss without asking for retransmissions (which would be too slow).

## 🧠 Mechanisms

### 1. Jitter Buffer
Jitter is the variation in packet delay.
* **Mechanism:** The receiver maintains a "play-out buffer" to smooth out arrival times.
* **Trade-off:** A larger buffer reduces dropped packets but increases end-to-end delay; a smaller buffer reduces delay but drops more late packets.



### 2. Handling Packet Loss
VoIP can tolerate 1-20% loss. It uses three main techniques:
* **FEC (Forward Error Correction):** Transmits redundant data (e.g., a lower quality stream alongside main stream) to reconstruct lost packets. Consumes more bandwidth.
* **Interleaving:** Scrambles audio chunks so a burst of loss results in small, non-consecutive gaps rather than one large glitch. Increases latency.
* **Error Concealment:** "Guesses" the lost packet by interpolating the audio wave from packets before and after the gap.

## 🔗 Connections
- **Source:** [[Source - Multimedia Applications]]