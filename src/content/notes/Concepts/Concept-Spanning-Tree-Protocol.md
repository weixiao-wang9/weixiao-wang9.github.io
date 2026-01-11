---
id: 202512170045
type: atom
status: permanent
tags: [concept, networks, algorithm]
related_topics: []
created: 2025-12-17
---

# Spanning Tree Protocol (STP)

## 💡 The Core Idea
STP is a distributed algorithm used by Layer 2 Bridges/Switches to prevent **broadcast loops** (storms) in a network topology by logically disabling specific links.



## 🧠 Context & Mechanism
Bridges "learn" by observing source MAC addresses. However, if the topology has a cycle (loop), frames will circle forever. STP solves this by building a loop-free logical tree.

### The Algorithm
1.  **Elect a Root Bridge:** The bridge with the lowest ID (or priority) becomes the Root.
2.  **Calculate Paths:** All other bridges calculate the shortest path (cost) to the Root.
3.  **Disable Ports:** Links that are not part of the shortest path to the root are put into a "blocking" state.

### Learning Logic
Bridges exchange configuration messages: `<My ID, Root ID, Distance to Root>`.
* A bridge stops forwarding on a port if it receives a message indicating a better path to the root exists via a neighbor.

## 🔗 Connections
- **Source:** [[Source - Intro to Internet Architecture]]