---
id: 202512170140
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - networks
  - protocol
related_topics:
  - - Concept - Dijkstra's Algorithm
created: 2025-12-17
---

# OSPF (Open Shortest Path First)

## 💡 The Core Idea
OSPF is a widely used **Link State** protocol for intradomain routing. It improves on older protocols (like RIP) by using flooding, authentication, and hierarchy.



## 🧠 Architecture & Operation
* **Hierarchy:** The Autonomous System (AS) is divided into **Areas**.
    * **Backbone Area:** Connects all other areas. Traffic between areas *must* pass through the backbone.
* **LSA (Link State Advertisement):** Used to communicate local topology to all other routers. Flooded periodically (every 30 mins) or on change.

### Router Processing Steps
1.  **Receive:** LS Update packet arrives.
2.  **Database:** Update the Link-State Database (LSDB).
3.  **Compute:** Run **SPF (Shortest Path First)** calculation (Dijkstra).
4.  **Install:** Update the **FIB (Forwarding Information Base)** used for actual packet forwarding.

## 🔗 Connections
- **Source:** [[Source - Intradomain Routing]]