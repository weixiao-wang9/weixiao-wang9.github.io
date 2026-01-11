---
id: 202512170052
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - networks
  - mechanism
related_topics: "[[Concept - Internet Protocol Stack]]"
created: 2025-12-17
---

# Transport Layer Multiplexing & Demultiplexing

## 💡 The Core Idea
Multiplexing allows a single host to run multiple network applications simultaneously. It uses **Ports** to ensure data arriving at an IP address is delivered to the correct process (socket).



## 🧠 Context & Details
* **Multiplexing (Sender):** Gathering data from different sockets and encapsulating it with header information (ports).
* **Demultiplexing (Receiver):** Delivering received segments to the correct socket by examining the header.

### Socket Identifiers
* **UDP Sockets:** Identified by a **2-tuple** `(Dest IP, Dest Port)`. If two packets have different source IPs but the same Dest IP/Port, they go to the same UDP socket.
* **TCP Sockets:** Identified by a **4-tuple** `(Source IP, Source Port, Dest IP, Dest Port)`. This allows a web server (Port 80) to distinguish between different clients simultaneously.

## 🔗 Connections
- **Source:** [[Source - Transport and Application Layers]]