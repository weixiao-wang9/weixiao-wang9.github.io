---
id: 202512170051
type: source
subtype: lecture_note
course: "[[Computer Networks (CS 6250)]]"
module: "Module 2: Transport Layer"
status: Finished
tags:
  - input
  - networks
  - transport_layer
created: 2025-12-17
---

# Transport and Application Layers

## 📌 Context & Summary
> [!abstract] AI Context
> This lecture moves up the stack to the Transport Layer. It contrasts the two main protocols: UDP (unreliable, fast) and TCP (reliable, heavy). It details how connections are managed (3-way handshake), how data is directed to the right app (Multiplexing/Ports), and the complex algorithms TCP uses to prevent network collapse (Flow & Congestion Control).

## 📝 Notes & Highlights

### Transport Layer Fundamentals
- **Purpose:** Provides logical communication between application processes running on different hosts.
- **Encapsulation:** The transport layer wraps application messages into **segments** by appending a header.
- **The Gap:** The Network layer offers "best-effort" delivery; the Transport layer (specifically TCP) adds reliability so applications don't have to worry about packet loss.

### Multiplexing & Demultiplexing
- **Problem:** IP addresses only identify the host, not the specific app (e.g., Spotify vs. Facebook).
- **Solution:** Ports are used to direct traffic to the correct socket.
    - **Connectionless (UDP):** Identified by a 2-tuple (Dest IP, Dest Port).
    - **Connection-Oriented (TCP):** Identified by a 4-tuple (Src IP, Src Port, Dest IP, Dest Port).

### UDP (User Datagram Protocol)
- **Nature:** Unreliable and connectionless. No handshakes, no congestion control.
- **Use Cases:** Real-time apps sensitive to delay (DNS, Streaming).
- **Header:** 64 bits (Source Port, Dest Port, Length, Checksum).

### TCP (Transmission Control Protocol)
- **3-Way Handshake:** 1.  Client sends `SYN`.
    2.  Server sends `SYNACK`.
    3.  Client sends `ACK`.
- **Reliability:** Uses ARQ (Automatic Repeat Request), timeouts, and retransmissions to ensure data integrity.

### Control Mechanisms
1.  **Flow Control:** Prevents the sender from overflowing the *receiver's* buffer.
2.  **Congestion Control:** Prevents the sender from overwhelming the *network*.
    - **AIMD:** Additive Increase, Multiplicative Decrease (Sawtooth pattern).
    - **Slow Start:** Exponential increase at the start of a connection.
    - **Modern TCP:** **TCP CUBIC** uses a cubic function for window growth, independent of RTT.

## 🔗 Extracted Concepts
- [[Concept - Transport Layer Multiplexing]]
- [[Concept - UDP vs TCP]]
- [[Concept - TCP Connection Management]]
- [[Concept - TCP Congestion Control]]
- [[Concept - TCP Flow Control]]