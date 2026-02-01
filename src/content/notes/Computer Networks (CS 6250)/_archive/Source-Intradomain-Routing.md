---
id: 202512170137
type: source
subtype: lecture_note
course: "[[Computer Networks (CS401)]]"
module: "Module 3: Routing Algorithms"
title: Intradomain Routing
author: Course Instructor
status: finished
tags: [input, networks, routing]
created: 2025-12-17
---

# Intradomain Routing

## 📌 Context & Summary
> [!abstract] AI Context
> This lecture shifts from "Forwarding" (moving a packet within a router) to "Routing" (determining the path through the network). It contrasts the two fundamental algorithm classes: **Link State** (Global knowledge/Dijkstra) and **Distance Vector** (Local knowledge/Bellman-Ford). It also covers their implementation in protocols like **OSPF** and **RIP**, and discusses Traffic Engineering.

## 📝 Notes & Highlights

### Routing vs. Forwarding
- **Forwarding:** Transferring a packet from an incoming link to an outgoing link *within* a single router.
- **Routing:** Determining the "good paths" (routes) from source to destination using routing protocols.
- **Intradomain:** Routing within the same administrative domain (e.g., one ISP).

### Link State Algorithms (LS)
- Uses global knowledge. All nodes know the network topology and link costs.
- **Algorithm:** Dijkstra's Algorithm.
- **Complexity:** $O(n^2)$.
- **Protocol:** **OSPF** (Open Shortest Path First).
    - Uses Link State Advertisements (LSAs) to flood topology changes.
    - Hierarchical structure: Uses "Areas" and a "Backbone" to manage scale.

### Distance Vector Algorithms (DV)
- Iterative, asynchronous, and distributed. Nodes only talk to immediate neighbors.
- **Algorithm:** Bellman-Ford Equation ($D_x(y) = min_v \{c(x,v) + D_v(y)\}$).
- **Protocol:** **RIP** (Routing Information Protocol).
    - Uses "hop count" as the cost metric.
- **Issues:** Susceptible to the "Count-to-Infinity" problem when link costs increase.
    - **Mitigation:** Poison Reverse (advertising infinite distance to loops).

### Traffic Engineering & Optimization
- **Hot Potato Routing:** Choosing the closest egress point (exit) from the network to get rid of traffic as soon as possible, regardless of the external path length.
- **Framework:** Measure (topology/traffic) -> Model (predict flow) -> Control (update weights).

## 🔗 Extracted Concepts
- [[Concept - Link State vs Distance Vector]]
- [[Concept - Dijkstra's Algorithm]]
- [[Concept - Distance Vector Routing]]
- [[Concept - OSPF Protocol]]
- [[Concept - Count to Infinity Problem]]
- [[Concept - Hot Potato Routing]]