---
id: 202512170416
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - sdn
  - architecture
related_topics: "[[Concept - SDN Controller Architecture]]"
created: 2025-12-17
---

# The SDN Landscape (Layers)

## 💡 The Core Idea
The SDN ecosystem is not just "Switches and Controllers"; it is a multi-layered stack that abstracts the hardware from the logic.



## 🧠 The Layers
1.  **Infrastructure:** Physical devices (routers/switches) that perform simple forwarding tasks based on rules from the controller.
2.  **Southbound Interface:** The bridge between the Control and Data planes.
    * *Standard:* **OpenFlow** (most popular), ForCES, OVSDB.
3.  **Network Operating System (NOS):** The controller (e.g., OpenDayLight). It hides low-level distribution details and provides a centralized view.
4.  **Northbound Interface:** The API used by applications to talk to the controller. Unlike Southbound, there is no single standard yet (e.g., REST APIs).
5.  **Network Applications:** The logic that implements control (e.g., Load Balancing, Firewalling, Routing).

## 🔗 Connections
- **Source:** [[Source - SDN Part 2 and Applications]]