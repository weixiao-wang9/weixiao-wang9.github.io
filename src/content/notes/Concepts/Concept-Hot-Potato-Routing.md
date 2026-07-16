---
id: 202512170141
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags:
  - concept
  - networks
  - strategy
related_topics: []
created: 2025-12-17
title: "Hot Potato Routing"
---

# Hot Potato Routing

## 💡 The Core Idea
A routing strategy where a network tries to hand off traffic to another network (or exit point) as quickly as possible. The router chooses the **closest** egress point based on internal costs (IGP), ignoring the cost of the path outside its own network.

## 🧠 Why use it?
* **Simplicity:** Computations are based only on known internal path costs.
* **Resource Efficiency:** It reduces resource consumption within the local network by "getting rid" of the packet immediately.

## 🔗 Connections
- **Source:** Source - Intradomain Routing