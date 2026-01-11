---
id: 202512170413
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - sdn
  - architecture
related_topics:
  - - Concept - SDN Network Architecture
created: 2025-12-17
---

# Control Plane vs. Data Plane Separation

## 💡 The Core Idea
SDN decouples the system that decides *where* traffic goes (Control Plane) from the system that actually *moves* the traffic (Data Plane).



## 🧠 Comparison

### Data Plane (Forwarding)
* **Function:** Local decision making at the router level.
* **Action:** Consults the forwarding table to move a packet from input to output link.
* **Speed:** Nanoseconds (implemented in hardware).

### Control Plane (Routing)
* **Function:** End-to-end path determination.
* **Action:** Computes forwarding tables and distributes them to routers.
* **Speed:** Seconds (implemented in software).

### Why Separate Them?
1.  **Independent Evolution:** Hardware (switches) and Software (routing logic) can upgrade independently.
2.  **High-Level Control:** Allows global software programs to debug and manage the network easily.
3.  **Innovation:** Enables new applications in Security (DDoS mitigation), Data Center management, and Traffic Engineering.

## 🔗 Connections
- **Source:** [[Source - Software Defined Networking]]