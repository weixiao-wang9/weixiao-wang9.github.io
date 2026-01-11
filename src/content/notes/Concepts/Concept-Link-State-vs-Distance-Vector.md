---
id: 202512170137
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - networks
  - algorithms
related_topics: "[[Concept - Distance Vector Routing]]"
created: 2025-12-17
---

# Link State vs. Distance Vector Routing

## 💡 The Core Idea
There are two fundamental approaches to network routing:
1.  **Link State (LS):** Every router has a complete map of the network (Global knowledge).
2.  **Distance Vector (DV):** Routers only know the distance to their neighbors and what their neighbors tell them (Local knowledge).



## 🧠 Comparison

| Feature | Link State (e.g., OSPF) | Distance Vector (e.g., RIP) |
| :--- | :--- | :--- |
| **Knowledge** | Global (Topology map) | Local (Neighbors' vectors) |
| **Algorithm** | Dijkstra's Algorithm | Bellman-Ford Algorithm |
| **Updates** | Broadcasts (Flooding) link status to *all* nodes | Exchanges distance vectors only with *neighbors* |
| **Convergence** | Fast, but computationally heavy ($O(n^2)$) | Slower, susceptible to routing loops (Count-to-Infinity) |

## 🔗 Connections
- **Source:** [[Source - Intradomain Routing]]