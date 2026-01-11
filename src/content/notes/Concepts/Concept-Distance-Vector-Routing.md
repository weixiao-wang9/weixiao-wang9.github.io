---
id: 202512170138
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - networks
  - algorithm
related_topics: "[[Concept - Count to Infinity Problem]]"
created: 2025-12-17
---

# Distance Vector Routing (Bellman-Ford)

## 💡 The Core Idea
In Distance Vector routing, nodes iteratively calculate the shortest path using the **Bellman-Ford equation**. It is distributed and asynchronous; nodes do not need a global map, they simply trust their neighbors' estimates.

## 🧠 The Equation
$$D_x(y) = min_v \{c(x,v) + D_v(y)\}$$
* $D_x(y)$: Cost from node $x$ to destination $y$.
* $c(x,v)$: Cost to neighbor $v$.
* $D_v(y)$: The neighbor $v$'s current cost to reach $y$.

## ⚙️ How it works
1.  **Wait:** Wait for a change in local link cost or a message from a neighbor.
2.  **Recompute:** Run the Bellman-Ford equation using the new data.
3.  **Notify:** If the least-cost path changes, send the new distance vector to neighbors.

## 🔗 Connections
- **Source:** [[Source - Intradomain Routing]]
- **Protocol Implementation:** [[Concept - RIP Protocol]] (Not explicitly extracted but mentioned in source as RIP)