---
id: 202512170025
type: atom
status: permanent
tags:
  - concept
  - networks
  - architecture
related_topics: "[[Concept - Data Encapsulation]]"
created: 2025/12/17 00:25:15
---

# Internet Protocol Stack (5-Layer Model)

## 💡 The Core Idea
The Internet architecture organizes protocols into five distinct layers. Each layer provides a service to the layer above it (abstraction) and relies on the layer below it, ensuring modularity and scalability.

## 🧠 Context & Details
Unlike the theoretical 7-layer OSI model, the practical Internet architecture consolidates the top three layers.

### The 5 Layers
1.  **Application Layer:** The interface for user applications (HTTP, SMTP, FTP, DNS). The data unit is a **Message**.
    * *Note:* Combines OSI's Session (stream management) and Presentation (formatting/encryption) layers.
2.  **Transport Layer:** Responsible for end-to-end communication between hosts. The data unit is a **Segment**.
    * *TCP:* Connection-oriented, reliable, flow control.
    * *UDP:* Connectionless, best-effort, no reliability.
3.  **Network Layer:** Moves data between different hosts across the internet ("The Glue"). The data unit is a **Datagram**.
    * *Key Protocol:* IP (Internet Protocol).
4.  **Data Link Layer:** Moves data between adjacent nodes (host-to-router or router-to-router). The data unit is a **Frame**.
    * *Examples:* Ethernet, WiFi, PPP.
5.  **Physical Layer:** Transfers raw bits over the physical medium (copper, fiber, radio).

## 🔗 Connections
- **Source:** [[Source - Intro to Internet Architecture]]
- **Relates to:** [[Concept - End-to-End Principle]] (Intelligence is at the top layer)