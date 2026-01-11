---
id: 202512170138
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - networks
  - algorithm
related_topics: "[[Concept - OSPF Protocol]]"
created: 2025-12-17
---

# Dijkstra's Algorithm (Link State)

## 💡 The Core Idea
Dijkstra's algorithm calculates the least-cost path from a source node to *all* other nodes in a network by iteratively adding the closest unvisited node to a set of confirmed nodes ($N'$).



## 🧠 Mechanism
1.  **Initialization:** The source node knows the cost to its immediate neighbors. All non-neighbors are set to infinity.
2.  **Iteration:**
    * Select the node ($w$) outside of set $N'$ with the lowest cost.
    * Add $w$ to $N'$.
    * Update neighbors of $w$: Is it cheaper to go through $w$ to get to neighbor $v$?
        * $D(v) = min(D(v), D(w) + c(w,v))$.
3.  **Completion:** Algorithm stops when all nodes are in $N'$ (requires $n$ iterations).

### Complexity
The computational complexity is $O(n^2)$, where $n$ is the number of nodes.

## 🔗 Connections
- **Source:** [[Source - Intradomain Routing]]