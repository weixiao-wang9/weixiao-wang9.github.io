---
id: 202512170053
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - networks
  - mechanism
related_topics: "[[Concept - UDP vs TCP]]"
created: 2025-12-17
---

# TCP Connection Management (3-Way Handshake)

## 💡 The Core Idea
Before data transfer begins, TCP establishes a logical connection using a **Three-Way Handshake** to synchronize sequence numbers and allocate resources.



## 🧠 Mechanism
### Establishment
1.  **SYN:** Client sends segment with `SYN=1` and a random initial sequence number ($client\_isn$).
2.  **SYNACK:** Server allocates buffers, sends `SYN=1`, `ACK=client_isn+1`, and its own sequence number.
3.  **ACK:** Client allocates buffers and acknowledges the server's sequence number.

### Teardown
To close a connection, a 4-step process occurs involving `FIN` bits. Both sides must independently close their half of the connection.

## 🔗 Connections
- **Source:** [[Source - Transport and Application Layers]]