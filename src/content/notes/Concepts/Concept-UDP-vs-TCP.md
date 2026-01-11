---
id: 202512170052
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - networks
  - protocols
related_topics: []
created: 2025-12-17
---

# UDP vs. TCP

## 💡 The Core Idea
The Transport layer offers two main flavors of delivery: **UDP** (Fast, Fire-and-forget) and **TCP** (Reliable, Ordered, Heavy).

## 🧠 Comparison

### UDP (User Datagram Protocol)
* **Characteristics:** Unreliable, Connectionless.
* **Pros:** No connection establishment delay (no handshake), no congestion control throttling.
* **Use Cases:** DNS, Voice over IP, Real-time gaming.
* **Header:** Lightweight (64 bits) containing only Source/Dest ports, Length, and Checksum.


### TCP (Transmission Control Protocol)
* **Characteristics:** Reliable, Connection-Oriented.
* **Pros:** Guarantees delivery, order, and data integrity.
* **Cons:** Higher latency due to handshakes and ACKs.
* **Use Cases:** Web (HTTP), Email (SMTP), File Transfer (FTP).

## 🔗 Connections
- **Source:** [[Source - Transport and Application Layers]]